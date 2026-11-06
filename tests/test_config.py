from __future__ import annotations

import pytest

from hret.config import Config


def test_defaults_and_nested_overrides() -> None:
    config = Config.from_dict({"seed": 1, "model": {"dim": 16}})
    assert config.seed == 1
    assert config.model.dim == 16
    assert config.data.split_strategy == "random"

    updated = config.apply_overrides(
        ["train.epochs=3", "model.meta_paths=[UBU, BUB]", "data.split_strategy=temporal"]
    )
    assert updated.train.epochs == 3
    assert updated.model.meta_paths == ["UBU", "BUB"]
    assert updated.data.split_strategy == "temporal"
    # The original config is not mutated.
    assert config.train.epochs == 40


def test_unknown_override_is_rejected() -> None:
    config = Config()
    with pytest.raises(KeyError):
        config.apply_overrides(["model.unknown=1"])


def test_round_trip() -> None:
    config = Config()
    assert Config.from_dict(config.to_dict()).to_dict() == config.to_dict()
