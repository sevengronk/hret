"""Baselines used in the paper's comparison: popularity and BPR-MF."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PopularityRecommender(nn.Module):
    """Non-personalised baseline that ranks by training popularity."""

    def __init__(self, num_businesses: int, interaction_counts: Dict[int, int]) -> None:
        super().__init__()
        counts = torch.zeros(num_businesses)
        for business, count in interaction_counts.items():
            counts[business] = float(count)
        # log1p keeps the score scale comparable to learned models.
        self.register_buffer("scores", torch.log1p(counts))

    def score_matrix(
        self, user_ids: torch.Tensor, business_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.scores if business_ids is None else self.scores[business_ids]
        return scores.unsqueeze(0).expand(user_ids.size(0), -1)

    def bpr_loss(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover - not trained
        raise NotImplementedError("PopularityRecommender is not trainable")


class BPRMF(nn.Module):
    """Classic matrix factorisation trained with BPR (Rendle et al., 2009)."""

    def __init__(
        self,
        num_users: int,
        num_businesses: int,
        dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_businesses = num_businesses
        self.user_embedding = nn.Embedding(num_users, dim)
        self.business_embedding = nn.Embedding(num_businesses, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.business_bias = nn.Embedding(num_businesses, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.business_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.business_bias.weight)

    def score_pairs(self, user_ids: torch.Tensor, business_ids: torch.Tensor) -> torch.Tensor:
        user = self.dropout(self.user_embedding(user_ids))
        business = self.dropout(self.business_embedding(business_ids))
        return (
            (user * business).sum(dim=-1)
            + self.user_bias(user_ids).squeeze(-1)
            + self.business_bias(business_ids).squeeze(-1)
        )

    def score_matrix(
        self, user_ids: torch.Tensor, business_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if business_ids is None:
            business_ids = torch.arange(self.num_businesses, device=user_ids.device)
        user = self.user_embedding(user_ids)
        business = self.business_embedding(business_ids)
        scores = user @ business.t()
        scores = scores + self.user_bias(user_ids)
        scores = scores + self.business_bias(business_ids).t()
        return scores

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        positive = self.score_pairs(user_ids, positive_ids)
        negative = self.score_pairs(user_ids, negative_ids)
        return -F.logsigmoid(positive - negative).mean()

    def embedding_l2(self) -> torch.Tensor:
        total = (
            self.user_embedding.weight.pow(2).sum()
            + self.business_embedding.weight.pow(2).sum()
        )
        return total / max(1, self.num_users + self.num_businesses)


def interaction_counts(train_by_user: Dict[int, np.ndarray]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for items in train_by_user.values():
        for item in items:
            counts[int(item)] = counts.get(int(item), 0) + 1
    return counts
