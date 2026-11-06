from __future__ import annotations

import random

import pytest

from hret.data.metapath import (
    MetaPath,
    generate_walks,
    skipgram_pairs,
    type_sequences,
)
from hret.data.schema import HIN


def _small_hin() -> HIN:
    hin = HIN({"user": 3, "business": 1, "service": 2, "standard": 1, "region": 1})
    for user in range(3):
        hin.add_edge("user", "business", user, 0)
    for user, service in ((0, 0), (1, 1), (2, 0)):
        hin.add_edge("user", "service", user, service)
    return hin


def test_meta_path_parsing() -> None:
    path = MetaPath.from_name("UBU")
    assert path.types == ("user", "business", "user")
    assert path.center_type == "user"

    custom = MetaPath.from_name("user-service-user")
    assert custom.types == ("user", "service", "user")


def test_meta_path_validation() -> None:
    with pytest.raises(ValueError):
        MetaPath(name="UB", types=("user", "business"))
    with pytest.raises(ValueError):
        MetaPath(name="UB", types=("user", "business", "service"))


def test_walks_follow_the_schema() -> None:
    hin = _small_hin()
    path = MetaPath.from_name("UBU")
    walks = generate_walks(hin, path, walks_per_node=2, walk_length=7, rng=random.Random(0))

    assert len(walks) == 6
    for walk in walks:
        assert len(walk) <= 7
        assert len(walk) % 2 == 1  # starts and ends with a user
        for position, node in enumerate(walk):
            expected_type = "user" if position % 2 == 0 else "business"
            limit = hin.num_nodes[expected_type]
            assert 0 <= node < limit


def test_typed_sequences() -> None:
    walks = [[0, 0, 1, 0, 2]]
    path = MetaPath.from_name("UBU")
    sequences = type_sequences(walks, path)
    assert sequences["user"] == [[0, 1, 2]]
    assert sequences["business"] == [[0, 0]]


def test_skipgram_pairs_window() -> None:
    pairs = skipgram_pairs([[1, 2, 3]], window_size=1)
    assert pairs == [(1, 2), (2, 1), (2, 3), (3, 2)]


def test_skipgram_pairs_drops_self_pairs() -> None:
    pairs = skipgram_pairs([[5, 5, 5]], window_size=1)
    assert pairs == []
