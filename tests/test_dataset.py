from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hret.data.dataset import build_hin, leave_one_out_split, load_dataset


def test_dataset_summary(dataset) -> None:
    summary = dataset.summary()
    assert summary["user"] == 40
    assert summary["business"] == 15
    assert summary["service"] == 12
    assert summary["interactions"] > 0


def test_leave_one_out_has_no_leakage(dataset) -> None:
    split = leave_one_out_split(dataset, min_interactions=3)
    assert split.eval_users, "expected at least one evaluable user"

    for user in split.eval_users:
        assert split.val[user] != split.test[user]
        assert split.val[user] not in split.train_by_user[user]
        assert split.test[user] not in split.train_by_user[user]
        assert split.test[user] not in split.seen_for_test(user)
        assert split.val[user] in split.seen_for_test(user)


def test_build_hin_uses_training_edges_only(dataset) -> None:
    split = leave_one_out_split(dataset, min_interactions=3)
    hin = build_hin(dataset, split.train)

    trained_users = {interaction.user for interaction in split.train}
    for user in trained_users:
        assert hin.degree("user", "business", user) > 0

    # Static schema edges are always present, even for cold users.
    assert sum(hin.edge_counts().values()) > 0


def test_missing_column_is_reported(tmp_path: Path) -> None:
    data = tmp_path / "broken"
    data.mkdir()
    for name in ("users.csv", "businesses.csv", "services.csv", "business_services.csv"):
        with open(data / name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["user_id" if name == "users.csv" else "business_id"])
            writer.writerow(["x"])
    with open(data / "interactions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "business_id"])  # missing service_id, timestamp
        writer.writerow(["x", "y"])

    with pytest.raises(ValueError):
        load_dataset(data)


def test_split_is_reproducible_and_supports_temporal_order(dataset) -> None:
    first = leave_one_out_split(dataset, 3, strategy="random", seed=11)
    second = leave_one_out_split(dataset, 3, strategy="random", seed=11)
    assert first.val == second.val
    assert first.test == second.test

    temporal = leave_one_out_split(dataset, 3, strategy="temporal", seed=11)
    assert temporal.eval_users
    for user in temporal.eval_users:
        assert temporal.val[user] != temporal.test[user]


def test_unknown_split_strategy(dataset) -> None:
    with pytest.raises(ValueError):
        leave_one_out_split(dataset, 3, strategy="nope")
