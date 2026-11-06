from __future__ import annotations

import numpy as np
import pytest
import torch

from hret.models.fusion import MetaPathFusion
from hret.models.hret import HRET
from hret.models.metapath2vec import SkipGram


def _model(scorer: str = "mlp", fusion: str = "attention") -> HRET:
    return HRET(
        num_users=10,
        num_businesses=8,
        user_paths=["UBU", "USU"],
        business_paths=["BUB", "BSB"],
        dim=16,
        hidden_dim=32,
        dropout=0.0,
        scorer=scorer,
        fusion=fusion,
    )


def test_score_shapes_and_attention_normalisation() -> None:
    model = _model()
    users = torch.tensor([0, 1, 2])
    businesses = torch.tensor([3, 4, 5])

    scores = model.score_pairs(users, businesses)
    assert scores.shape == (3,)

    matrix = model.score_matrix(torch.arange(10))
    assert matrix.shape == (10, 8)

    _, attention = model.encode_users(torch.arange(4))
    assert attention.shape == (4, 2)
    assert torch.allclose(attention.sum(dim=1), torch.ones(4), atol=1e-6)


def test_dot_and_mean_variants() -> None:
    dot_model = _model(scorer="dot", fusion="mean")
    matrix = dot_model.score_matrix(torch.arange(4))
    assert matrix.shape == (4, 8)

    _, attention = dot_model.encode_businesses(torch.arange(3))
    assert torch.allclose(attention, torch.full((3, 2), 0.5))


def test_bpr_loss_is_finite_and_positive() -> None:
    model = _model()
    loss = model.bpr_loss(
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4, 5]),
        torch.tensor([6, 7, 0]),
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    assert model.user_tables[0].weight.grad is not None


def test_load_pretrained_initialises_tables() -> None:
    model = _model()
    init = {}
    for path in ("UBU", "USU"):
        init[(path, "user")] = np.full((10, 16), 0.25, dtype=np.float32)
    for path in ("BUB", "BSB"):
        init[(path, "business")] = np.full((8, 16), 0.5, dtype=np.float32)

    model.load_pretrained(init)
    assert torch.allclose(model.user_tables[0].weight, torch.full((10, 16), 0.25))
    assert torch.allclose(model.business_tables[1].weight, torch.full((8, 16), 0.5))


def test_load_pretrained_checks_shape() -> None:
    model = _model()
    with pytest.raises(ValueError):
        model.load_pretrained(
            {
                ("UBU", "user"): np.zeros((10, 8), dtype=np.float32),
                ("USU", "user"): np.zeros((10, 16), dtype=np.float32),
                ("BUB", "business"): np.zeros((8, 16), dtype=np.float32),
                ("BSB", "business"): np.zeros((8, 16), dtype=np.float32),
            }
        )


def test_skipgram_loss_matches_manual_computation() -> None:
    torch.manual_seed(0)
    model = SkipGram(5, 4)
    centers = torch.tensor([0, 1])
    contexts = torch.tensor([2, 3])
    negatives = torch.tensor([[4, 0], [1, 2]])
    loss = model.loss(centers, contexts, negatives)

    center = model.center(centers)
    positive = (center * model.context(contexts)).sum(-1)
    negative = torch.bmm(model.context(negatives), center.unsqueeze(-1)).squeeze(-1)
    expected = (-torch.nn.functional.logsigmoid(positive)
                - torch.nn.functional.logsigmoid(-negative).sum(-1)).mean()
    assert torch.allclose(loss, expected)


def test_fusion_rejects_empty_input() -> None:
    fusion = MetaPathFusion(8)
    with pytest.raises(ValueError):
        fusion([])


def test_all_fusion_modes() -> None:
    for mode in ("attention", "mean", "concat", "hybrid"):
        model = _model(fusion=mode)
        matrix = model.score_matrix(torch.arange(5))
        assert matrix.shape == (5, 8), mode
        _, attention = model.encode_users(torch.arange(3))
        assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)


def test_fusion_rejects_wrong_view_count() -> None:
    fusion = MetaPathFusion(8, mode="concat", num_paths=3)
    with pytest.raises(ValueError):
        fusion([torch.zeros(2, 8), torch.zeros(2, 8)])
