"""HRET: a heterogeneous information network recommender.

The model keeps one embedding table per (meta-path, node type), fuses the
views of a node with :class:`~hret.models.fusion.MetaPathFusion`, and ranks
businesses with a BPR objective. The reference configuration uses attention
fusion; concat, hybrid and mean are available as ablations.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import MetaPathFusion


class HRET(nn.Module):
    """Meta-path aware heterogeneous recommender.

    Parameters
    ----------
    num_users, num_businesses:
        Vocabulary sizes.
    user_paths, business_paths:
        Names of the meta-paths that produce a view of a user / business.
    dim:
        Embedding size.
    scorer:
        ``"mlp"`` (default) or ``"dot"``.
    fusion:
        ``"attention"`` (default) or ``"mean"``.
    """

    def __init__(
        self,
        num_users: int,
        num_businesses: int,
        user_paths: Sequence[str],
        business_paths: Sequence[str],
        dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        scorer: str = "mlp",
        fusion: str = "attention",
        use_id_embedding: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if not user_paths or not business_paths:
            raise ValueError("HRET needs at least one meta-path for users and for businesses")
        if scorer not in {"mlp", "dot"}:
            raise ValueError(f"Unknown scorer: {scorer!r}")

        self.num_users = num_users
        self.num_businesses = num_businesses
        self.user_paths = list(user_paths)
        self.business_paths = list(business_paths)
        self.dim = dim
        self.scorer = scorer

        self.user_tables = nn.ModuleList(
            [nn.Embedding(num_users, dim) for _ in self.user_paths]
        )
        self.business_tables = nn.ModuleList(
            [nn.Embedding(num_businesses, dim) for _ in self.business_paths]
        )
        self.user_fusion = MetaPathFusion(
            dim, mode=fusion, dropout=dropout, num_paths=len(self.user_paths)
        )
        self.business_fusion = MetaPathFusion(
            dim, mode=fusion, dropout=dropout, num_paths=len(self.business_paths)
        )

        # A direct interaction view (classic MF) complements the meta-path
        # views. It is initialised close to zero so that training starts from
        # the pre-trained structural signal.
        self.use_id_embedding = use_id_embedding
        if use_id_embedding:
            self.user_id_embedding: nn.Embedding | None = nn.Embedding(num_users, dim)
            self.business_id_embedding: nn.Embedding | None = nn.Embedding(
                num_businesses, dim
            )
            self.user_norm = nn.LayerNorm(dim)
            self.business_norm = nn.LayerNorm(dim)
        else:
            self.user_id_embedding = None
            self.business_id_embedding = None

        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.business_bias = nn.Embedding(num_businesses, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.business_bias.weight)
        else:
            self.user_bias = None
            self.business_bias = None

        if scorer == "mlp":
            self.predictor: nn.Module = nn.Sequential(
                nn.Linear(3 * dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.predictor = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in list(self.user_tables) + list(self.business_tables):
            nn.init.xavier_uniform_(table.weight)
        if self.user_id_embedding is not None and self.business_id_embedding is not None:
            nn.init.normal_(self.user_id_embedding.weight, std=0.01)
            nn.init.normal_(self.business_id_embedding.weight, std=0.01)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_users(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = [table(user_ids) for table in self.user_tables]
        fused, attention = self.user_fusion(embeddings)
        if self.user_id_embedding is not None:
            fused = self.user_norm(fused + self.user_id_embedding(user_ids))
        return fused, attention

    def encode_businesses(
        self, business_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = [table(business_ids) for table in self.business_tables]
        fused, attention = self.business_fusion(embeddings)
        if self.business_id_embedding is not None:
            fused = self.business_norm(fused + self.business_id_embedding(business_ids))
        return fused, attention

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_from_embeddings(self, users: torch.Tensor, businesses: torch.Tensor) -> torch.Tensor:
        if self.scorer == "dot":
            return (users * businesses).sum(dim=-1)
        features = torch.cat([users, businesses, users * businesses], dim=-1)
        return self.predictor(features).squeeze(-1)

    def score_pairs(self, user_ids: torch.Tensor, business_ids: torch.Tensor) -> torch.Tensor:
        """Score aligned ``[B]`` user / business pairs."""
        user_vectors, _ = self.encode_users(user_ids)
        business_vectors, _ = self.encode_businesses(business_ids)
        scores = self._score_from_embeddings(user_vectors, business_vectors)
        if self.use_bias:
            scores = (
                scores
                + self.user_bias(user_ids).squeeze(-1)
                + self.business_bias(business_ids).squeeze(-1)
            )
        return scores

    def score_matrix(
        self, user_ids: torch.Tensor, business_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score every ``user`` against a business catalogue.

        Returns a ``[len(user_ids), num_businesses]`` matrix. The MLP scorer
        is evaluated in chunks to keep peak memory bounded.
        """
        if business_ids is None:
            business_ids = torch.arange(self.num_businesses, device=user_ids.device)
        user_vectors, _ = self.encode_users(user_ids)
        business_vectors, _ = self.encode_businesses(business_ids)

        if self.scorer == "dot":
            scores = user_vectors @ business_vectors.t()
            if self.use_bias:
                scores = scores + self.user_bias(user_ids)
                scores = scores + self.business_bias(business_ids).t()
            return scores

        chunk = 256
        outputs: List[torch.Tensor] = []
        for start in range(0, business_vectors.size(0), chunk):
            candidate = business_vectors[start : start + chunk]  # [C, D]
            num_candidates = candidate.size(0)
            users_expanded = user_vectors.unsqueeze(1).expand(-1, num_candidates, -1)
            candidates_expanded = candidate.unsqueeze(0).expand(user_vectors.size(0), -1, -1)
            features = torch.cat(
                [users_expanded, candidates_expanded, users_expanded * candidates_expanded],
                dim=-1,
            )
            outputs.append(self.predictor(features).squeeze(-1))
        scores = torch.cat(outputs, dim=1)
        if self.use_bias:
            scores = scores + self.user_bias(user_ids)
            scores = scores + self.business_bias(business_ids).t()
        return scores

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------
    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Bayesian personalised ranking loss."""
        positive = self.score_pairs(user_ids, positive_ids)
        negative = self.score_pairs(user_ids, negative_ids)
        return -F.logsigmoid(positive - negative).mean()

    def embedding_l2(self) -> torch.Tensor:
        """Mean squared norm of all embedding tables (for weight decay)."""
        total = torch.zeros((), device=self.user_tables[0].weight.device)
        for table in list(self.user_tables) + list(self.business_tables):
            total = total + table.weight.pow(2).sum()
        if self.user_id_embedding is not None:
            total = total + self.user_id_embedding.weight.pow(2).sum()
        if self.business_id_embedding is not None:
            total = total + self.business_id_embedding.weight.pow(2).sum()
        return total / max(1, self.num_users + self.num_businesses)

    # ------------------------------------------------------------------
    # Pre-trained initialisation
    # ------------------------------------------------------------------
    def load_pretrained(
        self, embeddings: Dict[Tuple[str, str], np.ndarray]
    ) -> None:
        """Initialise tables from ``{(meta_path, node_type): array}``."""
        with torch.no_grad():
            for path, table in zip(self.user_paths, self.user_tables):
                array = embeddings.get((path, "user"))
                if array is None:
                    raise KeyError(f"Missing pre-trained user embedding for meta-path {path!r}")
                if array.shape != tuple(table.weight.shape):
                    raise ValueError(
                        f"Shape mismatch for {(path, 'user')}: {array.shape} vs {tuple(table.weight.shape)}"
                    )
                table.weight.copy_(torch.as_tensor(array, dtype=table.weight.dtype))
            for path, table in zip(self.business_paths, self.business_tables):
                array = embeddings.get((path, "business"))
                if array is None:
                    raise KeyError(f"Missing pre-trained business embedding for meta-path {path!r}")
                if array.shape != tuple(table.weight.shape):
                    raise ValueError(
                        f"Shape mismatch for {(path, 'business')}: {array.shape} vs {tuple(table.weight.shape)}"
                    )
                table.weight.copy_(torch.as_tensor(array, dtype=table.weight.dtype))

    @classmethod
    def from_config(
        cls,
        num_users: int,
        num_businesses: int,
        config,
    ) -> "HRET":
        paths = list(config.model.meta_paths)
        user_paths = [name for name in paths if _path_center(name) == "user"]
        business_paths = [name for name in paths if _path_center(name) == "business"]
        return cls(
            num_users=num_users,
            num_businesses=num_businesses,
            user_paths=user_paths,
            business_paths=business_paths,
            dim=config.model.dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            scorer=config.model.scorer,
            fusion=config.model.fusion,
            use_id_embedding=config.model.use_id_embedding,
            use_bias=config.model.use_bias,
        )


def _path_center(name: str) -> str:
    from ..data.metapath import MetaPath

    return MetaPath.from_name(name).center_type
