from __future__ import annotations

import math

import torch

from hret.evaluation import evaluate_ranking


def _fixed_scorer(num_items: int = 4):
    # Scores descend with the item index: item 0 is rank 1, item 1 rank 2, ...
    base = torch.tensor([-float(index) for index in range(num_items)])

    def score_fn(user_ids: torch.Tensor) -> torch.Tensor:
        return base.unsqueeze(0).expand(user_ids.size(0), -1)

    return score_fn


def test_metrics_on_known_ranks() -> None:
    targets = {0: 0, 1: 1, 2: 2}
    result = evaluate_ranking(
        _fixed_scorer(),
        targets=targets,
        seen_by_user=lambda user: set(),
        num_items=4,
        ks=[1, 3],
    )
    assert result.ranks == {0: 1, 1: 2, 2: 3}
    assert result.metrics["HR@1"] == 1 / 3
    assert result.metrics["HR@3"] == 1.0
    assert result.metrics["NDCG@1"] == 1 / 3
    expected_ndcg3 = (1 + 1 / math.log2(3) + 1 / 2) / 3
    assert abs(result.metrics["NDCG@3"] - expected_ndcg3) < 1e-9
    assert abs(result.metrics["MRR"] - (1 + 1 / 2 + 1 / 3) / 3) < 1e-9


def test_seen_items_are_masked() -> None:
    # Target item 1 would be rank 2, but masking the better item 0 makes it rank 1.
    result = evaluate_ranking(
        _fixed_scorer(),
        targets={0: 1},
        seen_by_user=lambda user: {0},
        num_items=4,
        ks=[1],
    )
    assert result.ranks == {0: 1}
    assert result.metrics["HR@1"] == 1.0


def test_candidate_pool_keeps_the_target() -> None:
    targets = {user: user % 4 for user in range(20)}
    result = evaluate_ranking(
        _fixed_scorer(),
        targets=targets,
        seen_by_user=lambda user: set(),
        num_items=4,
        ks=[1],
        candidate_pool=2,
        seed=0,
    )
    assert result.num_users > 0
    assert all(1 <= rank <= 4 for rank in result.ranks.values())


def test_score_shape_is_validated() -> None:
    def bad_scorer(user_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(user_ids.size(0), 2)

    try:
        evaluate_ranking(
            bad_scorer,
            targets={0: 0},
            seen_by_user=lambda user: set(),
            num_items=4,
            ks=[1],
        )
    except ValueError as error:
        assert "expected" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")
