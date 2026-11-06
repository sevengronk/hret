from __future__ import annotations

from pathlib import Path

import pytest

from hret.config import Config
from hret.runner import run_hret


@pytest.mark.slow
def test_end_to_end_pipeline(synthetic_dir: Path, tmp_path: Path) -> None:
    config = Config.from_dict(
        {
            "seed": 0,
            "device": "cpu",
            "data": {"dir": str(synthetic_dir), "min_interactions": 3},
            "model": {
                "dim": 8,
                "meta_paths": ["UBU", "USU", "BUB", "BSB"],
                "walks_per_node": 1,
                "walk_length": 8,
                "window_size": 2,
                "pretrain_epochs": 1,
                "pretrain_batch_size": 64,
                "neg_samples": 3,
                "dropout": 0.0,
                "hidden_dim": 16,
                "scorer": "dot",
                "fusion": "attention",
            },
            "train": {
                "epochs": 2,
                "batch_size": 32,
                "lr": 0.01,
                "patience": 2,
                "eval_every": 1,
            },
            "evaluation": {"ks": [5, 10]},
        }
    )
    output = tmp_path / "run"
    metrics = run_hret(config, output)

    assert (output / "hret.pt").exists()
    assert (output / "metrics.json").exists()
    assert (output / "top10_test.csv").exists()
    assert "HR@5" in metrics["test"]
    assert "NDCG@10" in metrics["test"]
    assert metrics["test"]["MRR"] > 0
    assert metrics["best_epoch"] >= 1
