"""Ranking evaluation for top-N business recommendation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence, Set

import numpy as np
import torch


@dataclass
class EvalResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    ranks: Dict[int, int] = field(default_factory=dict)
    num_users: int = 0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        body = " ".join(f"{key}={value:.4f}" for key, value in self.metrics.items())
        return f"EvalResult(users={self.num_users}, {body})"


def _ndcg_at_k(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def evaluate_ranking(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    targets: Mapping[int, int],
    seen_by_user: Callable[[int], Set[int]],
    num_items: int,
    ks: Sequence[int],
    candidate_pool: int = 0,
    batch_size: int = 256,
    seed: int = 0,
) -> EvalResult:
    """Evaluate a scoring function with the full-ranking protocol.

    ``score_fn`` receives a tensor of user ids and returns a
    ``[len(user_ids), num_items]`` score matrix. Already-seen items are
    masked before computing the rank of the held-out target.
    """
    if not targets:
        raise ValueError("No evaluation users: the dataset is too small for the split")

    users = sorted(targets)
    generator = np.random.default_rng(seed)
    ranks: Dict[int, int] = {}
    skipped = 0

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        user_tensor = torch.as_tensor(batch_users, dtype=torch.long)
        scores = score_fn(user_tensor).detach().float().cpu()
        if scores.size(0) != len(batch_users) or scores.size(1) != num_items:
            raise ValueError(
                f"score_fn returned shape {tuple(scores.shape)}, expected "
                f"({len(batch_users)}, {num_items})"
            )

        for row, user in enumerate(batch_users):
            seen = seen_by_user(user)
            target = targets[user]
            if target in seen:
                # Should not happen with the leave-one-out split, but a
                # defensive check keeps the metric honest.
                seen = set(seen) - {target}
            row_scores = scores[row].clone()
            if seen:
                seen_index = torch.as_tensor(sorted(seen), dtype=torch.long)
                row_scores[seen_index] = -torch.inf

            if candidate_pool and candidate_pool > 1:
                mask = torch.full((num_items,), -torch.inf)
                mask[target] = row_scores[target]
                available = torch.ones(num_items, dtype=torch.bool)
                if seen:
                    available[torch.as_tensor(sorted(seen), dtype=torch.long)] = False
                available[target] = False
                candidates = torch.nonzero(available, as_tuple=False).flatten().numpy()
                sample_size = min(candidate_pool - 1, len(candidates))
                if sample_size > 0:
                    chosen = generator.choice(candidates, size=sample_size, replace=False)
                    mask[torch.as_tensor(chosen, dtype=torch.long)] = row_scores[
                        torch.as_tensor(chosen, dtype=torch.long)
                    ]
                row_scores = mask

            target_score = row_scores[target]
            if not torch.isfinite(target_score):
                skipped += 1
                continue
            rank = int((row_scores > target_score).sum().item()) + 1
            ranks[user] = rank

    if not ranks:
        raise ValueError("Every candidate was masked; check the split or the candidate pool")

    metrics: Dict[str, float] = {}
    rank_values = np.fromiter(ranks.values(), dtype=np.float64)
    for k in ks:
        metrics[f"HR@{k}"] = float((rank_values <= k).mean())
        metrics[f"NDCG@{k}"] = float(
            np.mean([_ndcg_at_k(int(rank), k) for rank in rank_values])
        )
    metrics["MRR"] = float(np.mean(1.0 / rank_values))
    metrics["users"] = float(len(rank_values))

    return EvalResult(metrics=metrics, ranks=ranks, num_users=len(rank_values))


def format_metrics(metrics: Mapping[str, float]) -> str:
    return "  ".join(
        f"{key}: {value:.4f}" for key, value in metrics.items() if key != "users"
    )
