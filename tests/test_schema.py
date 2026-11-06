from __future__ import annotations

import pytest

from hret.data.schema import HIN


def test_hin_is_bidirectional_and_deduplicates() -> None:
    hin = HIN({"user": 2, "business": 1, "service": 1, "standard": 1, "region": 1})
    hin.add_edge("user", "business", 0, 0)
    hin.add_edge("user", "business", 0, 0)  # duplicate must be ignored

    assert hin.neighbors("user", "business", 0) == [0]
    assert hin.neighbors("business", "user", 0) == [0]
    assert hin.degree("user", "business", 0) == 1
    assert hin.edge_counts() == {"user->business": 1, "business->user": 1}


def test_hin_rejects_self_loops() -> None:
    hin = HIN({"user": 1, "business": 1, "service": 1, "standard": 1, "region": 1})
    hin.add_edge("user", "user", 0, 0)
    assert hin.num_edges() == 0


def test_hin_requires_all_node_types() -> None:
    with pytest.raises(ValueError):
        HIN({"user": 1})
