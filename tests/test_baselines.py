from __future__ import annotations

import torch

from hret.baselines import BPRMF, PopularityRecommender, interaction_counts


def test_popularity_scores_are_constant_per_user() -> None:
    model = PopularityRecommender(5, {0: 10, 2: 1})
    scores = model.score_matrix(torch.tensor([0, 1, 2]))
    assert scores.shape == (3, 5)
    assert torch.allclose(scores[0], scores[1])
    assert torch.argmax(scores[0]).item() == 0  # most popular business


def test_bprmf_scores_and_loss() -> None:
    model = BPRMF(num_users=6, num_businesses=4, dim=8)
    scores = model.score_matrix(torch.arange(6))
    assert scores.shape == (6, 4)
    loss = model.bpr_loss(
        torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([0, 1])
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.user_embedding.weight.grad is not None


def test_interaction_counts() -> None:
    counts = interaction_counts({0: [1, 2], 1: [1]})
    assert counts == {1: 2, 2: 1}
