"""Meta-path attention fusion.

A node has one embedding per meta-path it participates in. Rather than
concatenating them (which fixes the weight of every relation), HRET can learn
a soft importance weight per meta-path and aggregate the views::

    e_i'   = W_p_i e_i                    (per-path projection)
    q      = W_q * mean_i(e_i')
    alpha  = softmax_i( <q, W_k e_i'> / sqrt(d) )
    h      = LayerNorm( sum_i alpha_i * W_v e_i' + mean_i(e_i') )

Four modes are available:

``attention``
    The equation above. This is the reference HRET configuration.
``concat``
    Concatenate every view and project back to ``d`` dimensions.
``hybrid``
    Concatenate the projected views and add the attention-pooled view.
``mean``
    Uniform average, useful as a sanity-check ablation.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

MODES = ("attention", "concat", "hybrid", "mean")


class MetaPathFusion(nn.Module):
    """Aggregate meta-path specific embeddings into one vector."""

    def __init__(
        self,
        dim: int,
        mode: str = "attention",
        dropout: float = 0.2,
        num_paths: int = 1,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown fusion mode: {mode!r}")
        self.dim = dim
        self.mode = mode
        self.num_paths = num_paths
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        if mode in {"attention", "hybrid"}:
            # Views come from independently pre-trained metapath2vec spaces;
            # project each one into a shared space before comparing them.
            self.projections = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(num_paths)]
            )
            self.query = nn.Linear(dim, dim, bias=False)
            self.key = nn.Linear(dim, dim, bias=False)
            self.value = nn.Linear(dim, dim, bias=False)
        if mode in {"concat", "hybrid"}:
            self.concat_project = nn.Linear(dim * num_paths, dim)

    def _attention(
        self, stacked: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = stacked.mean(dim=1, keepdim=True)
        query = self.query(pooled)  # [B, 1, D]
        keys = self.key(stacked)  # [B, P, D]
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.dim)  # [B, P]
        weights = torch.softmax(scores, dim=-1)
        values = self.value(stacked)
        aggregated = (weights.unsqueeze(-1) * values).sum(dim=1)
        return aggregated, weights

    def forward(self, embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse ``[B, D]`` tensors.

        Returns the fused representation ``[B, D]`` and the meta-path weights
        ``[B, P]``. Non-attentive modes return uniform weights.
        """
        if not embeddings:
            raise ValueError("At least one embedding is required for fusion")
        if len(embeddings) != self.num_paths:
            raise ValueError(
                f"Expected {self.num_paths} views, received {len(embeddings)}"
            )

        if self.mode == "mean":
            stacked = torch.stack(embeddings, dim=1)
            pooled = stacked.mean(dim=1)
            fused = self.norm(self.dropout(pooled) + pooled)
            weights = torch.full(
                (stacked.size(0), stacked.size(1)),
                1.0 / stacked.size(1),
                device=stacked.device,
                dtype=stacked.dtype,
            )
            return fused, weights

        if self.mode == "concat":
            stacked = torch.stack(embeddings, dim=1)  # [B, P, D]
            concatenated = stacked.flatten(start_dim=1)
            pooled = self.concat_project(concatenated)
            fused = self.norm(self.dropout(pooled) + stacked.mean(dim=1))
            weights = torch.full(
                (stacked.size(0), stacked.size(1)),
                1.0 / stacked.size(1),
                device=stacked.device,
                dtype=stacked.dtype,
            )
            return fused, weights

        projected = [
            projection(embedding)
            for projection, embedding in zip(self.projections, embeddings)
        ]
        stacked = torch.stack(projected, dim=1)  # [B, P, D]
        aggregated, weights = self._attention(stacked)

        if self.mode == "attention":
            fused = self.norm(self.dropout(aggregated) + stacked.mean(dim=1))
            return fused, weights

        # hybrid
        concatenated = stacked.flatten(start_dim=1)
        combined = self.concat_project(concatenated) + aggregated
        fused = self.norm(self.dropout(combined) + stacked.mean(dim=1))
        return fused, weights
