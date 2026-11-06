from __future__ import annotations

import numpy as np

from hret.config import TrainConfig
from hret.models.hret import HRET
from hret.training.trainer import _sample_negatives, train_bpr


def test_negative_sampler_avoids_positives() -> None:
    generator = np.random.default_rng(0)
    negatives = _sample_negatives(
        users=[0, 1, 2],
        num_items=10,
        positives={0: [0, 1], 1: [2], 2: []},
        count=8,
        generator=generator,
    )
    assert negatives.shape == (3, 8)
    assert not set(negatives[0].tolist()) & {0, 1}
    assert 2 not in set(negatives[1].tolist())


def test_train_bpr_runs_and_records_history() -> None:
    model = HRET(
        num_users=12,
        num_businesses=6,
        user_paths=["UBU"],
        business_paths=["BUB"],
        dim=8,
        hidden_dim=8,
        scorer="dot",
    )
    train_by_user = {user: [user % 6] for user in range(12)}
    cfg = TrainConfig(epochs=2, batch_size=8, lr=0.01, patience=5, eval_every=1)

    history = train_bpr(
        model,
        train_by_user,
        num_items=6,
        cfg=cfg,
        device="cpu",
        val_fn=None,
        seed=0,
    )
    assert len(history.epochs) == 2
    assert all(np.isfinite(record["loss"]) for record in history.epochs)
    assert history.best_epoch == -1  # no validation function was provided
