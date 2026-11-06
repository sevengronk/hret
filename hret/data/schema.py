"""Heterogeneous information network schema used by HRET.

The testing-and-inspection domain is modelled with five node types:

============  =========================================================
Node type     Meaning
============  =========================================================
``user``      An enterprise / organisation that requests a test.
``business``  A testing and inspection service provider.
``service``   A concrete inspection item (e.g. material analysis).
``standard``  The standard a service is accredited against.
``region``    Administrative region of a user or a business.
============  =========================================================

Edges are typed and always stored in both directions so that meta-path
walks can traverse the graph in either direction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

NODE_TYPES: Tuple[str, ...] = ("user", "business", "service", "standard", "region")


@dataclass(frozen=True)
class EdgeType:
    """A typed relation in the HIN schema."""

    src: str
    relation: str
    dst: str

    @property
    def name(self) -> str:
        return f"{self.src}-{self.relation}-{self.dst}"


EDGE_TYPES: Tuple[EdgeType, ...] = (
    EdgeType("user", "interacts_with", "business"),
    EdgeType("user", "demands", "service"),
    EdgeType("business", "provides", "service"),
    EdgeType("user", "located_in", "region"),
    EdgeType("business", "located_in", "region"),
    EdgeType("service", "governed_by", "standard"),
)

EDGE_TYPES_BY_NAME: Dict[str, EdgeType] = {edge.name: edge for edge in EDGE_TYPES}


class HIN:
    """A lightweight heterogeneous graph.

    Adjacency is stored as ``(src_type, dst_type) -> src_id -> [dst_id, ...]``.
    The schema used here has at most one relation per type pair, so the
    relation name does not have to be part of the lookup key.
    """

    def __init__(self, num_nodes: Dict[str, int]) -> None:
        missing = set(NODE_TYPES) - set(num_nodes)
        if missing:
            raise ValueError(f"num_nodes is missing node types: {sorted(missing)}")
        self.num_nodes: Dict[str, int] = {key: int(value) for key, value in num_nodes.items()}
        self._adj: Dict[Tuple[str, str], Dict[int, List[int]]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_edge(
        self,
        src_type: str,
        dst_type: str,
        src: int,
        dst: int,
        bidirectional: bool = True,
    ) -> None:
        if src_type == dst_type and src == dst:
            return
        self._add_directed(src_type, dst_type, src, dst)
        if bidirectional:
            self._add_directed(dst_type, src_type, dst, src)

    def _add_directed(self, src_type: str, dst_type: str, src: int, dst: int) -> None:
        key = (src_type, dst_type)
        if key not in self._adj:
            self._adj[key] = defaultdict(list)
        neighbours = self._adj[key][src]
        # Keep the graph simple: duplicate edges carry no extra information
        # for random walks and would bias the transition probabilities.
        if dst not in neighbours:
            neighbours.append(dst)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def neighbors(self, src_type: str, dst_type: str, node: int) -> List[int]:
        """Return the neighbours of ``node`` with the requested type."""
        return self._adj.get((src_type, dst_type), {}).get(node, [])

    def degree(self, src_type: str, dst_type: str, node: int) -> int:
        return len(self.neighbors(src_type, dst_type, node))

    def num_edges(self) -> int:
        """Number of directed edges (every undirected edge counts twice)."""
        return sum(len(values) for table in self._adj.values() for values in table.values())

    def edge_counts(self) -> Dict[str, int]:
        """Directed edge count grouped by ``src->dst`` type pair."""
        counts: Dict[str, int] = {}
        for (src_type, dst_type), table in self._adj.items():
            counts[f"{src_type}->{dst_type}"] = sum(len(v) for v in table.values())
        return counts

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nodes = ", ".join(f"{k}={v}" for k, v in self.num_nodes.items())
        return f"HIN({nodes}, directed_edges={self.num_edges()})"
