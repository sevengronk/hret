"""Meta-path definitions and meta-path guided random walks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .schema import HIN

#: Symmetric meta-paths used by the reference model.
#:
#: * ``UBU`` — users who consulted the same business provider.
#: * ``USU`` — users who requested the same inspection service.
#: * ``URU`` — users located in the same region.
#: * ``BUB`` — businesses serving the same users.
#: * ``BSB`` — businesses offering the same inspection service.
META_PATH_REGISTRY: Dict[str, Tuple[str, ...]] = {
    "UBU": ("user", "business", "user"),
    "USU": ("user", "service", "user"),
    "URU": ("user", "region", "user"),
    "BUB": ("business", "user", "business"),
    "BSB": ("business", "service", "business"),
    "BRB": ("business", "region", "business"),
    "USBU": ("user", "service", "business", "user"),
    "BSUB": ("business", "service", "user", "business"),
}


@dataclass(frozen=True)
class MetaPath:
    """A symmetric meta-path, e.g. ``user - business - user``."""

    name: str
    types: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.types) < 3:
            raise ValueError(f"Meta-path {self.name!r} needs at least 3 node types")
        if self.types[0] != self.types[-1]:
            raise ValueError(f"Meta-path {self.name!r} must start and end with the same type")

    @property
    def center_type(self) -> str:
        return self.types[0]

    @classmethod
    def from_name(cls, name: str) -> "MetaPath":
        key = name.upper()
        if key in META_PATH_REGISTRY:
            return cls(name=key, types=META_PATH_REGISTRY[key])
        # Allow an explicit type sequence such as "user-business-user".
        parts = tuple(part.strip() for part in name.lower().split("-"))
        if len(parts) >= 3:
            return cls(name="-".join(parts), types=parts)
        raise KeyError(
            f"Unknown meta-path {name!r}. Known names: {sorted(META_PATH_REGISTRY)} "
            "or pass a dash-separated type sequence."
        )

    @classmethod
    def parse_many(cls, names: Iterable[str]) -> List["MetaPath"]:
        parsed = [cls.from_name(name) for name in names]
        if len({path.name for path in parsed}) != len(parsed):
            raise ValueError("Duplicate meta-paths in configuration")
        return parsed


def generate_walks(
    hin: HIN,
    meta_path: MetaPath,
    walks_per_node: int,
    walk_length: int,
    rng: random.Random | None = None,
) -> List[List[int]]:
    """Generate typed random walks that follow ``meta_path``.

    Each walk starts at a node of ``meta_path.center_type`` and takes one
    edge per step, enforcing the next node type. Walks stop early when a
    node has no neighbour of the required type.
    """
    if walks_per_node <= 0:
        raise ValueError("walks_per_node must be positive")
    if walk_length < 2:
        raise ValueError("walk_length must be >= 2")
    rng = rng or random.Random(0)

    starts = range(hin.num_nodes[meta_path.center_type])
    walks: List[List[int]] = []
    for start in starts:
        for _ in range(walks_per_node):
            walk = [start]
            current = start
            period = len(meta_path.types) - 1
            for step in range(1, walk_length):
                src_type = meta_path.types[(step - 1) % period]
                dst_type = meta_path.types[step % period]
                neighbours = hin.neighbors(src_type, dst_type, current)
                if not neighbours:
                    break
                current = neighbours[rng.randrange(len(neighbours))]
                walk.append(current)
            if len(walk) >= 3:
                walks.append(walk)
    return walks


def type_sequences(
    walks: Sequence[Sequence[int]], meta_path: MetaPath
) -> Dict[str, List[List[int]]]:
    """Split typed walks into per-node-type sequences.

    A ``UBU`` walk ``[u0, b3, u7, b1, u2]`` yields ``[u0, u7, u2]`` for users
    and ``[b3, b1]`` for businesses. Skip-gram is then trained on each
    per-type sequence, which is the standard metapath2vec construction.
    """
    sequences: Dict[str, List[List[int]]] = {}
    for node_type in dict.fromkeys(meta_path.types):
        sequences[node_type] = []
    for walk in walks:
        period = len(meta_path.types) - 1
        for node_type in sequences:
            seq = [
                node
                for position, node in enumerate(walk)
                if meta_path.types[position % period] == node_type
            ]
            if len(seq) >= 2:
                sequences[node_type].append(seq)
    return {key: value for key, value in sequences.items() if value}


def skipgram_pairs(
    sequences: Sequence[Sequence[int]], window_size: int
) -> List[Tuple[int, int]]:
    """Enumerate ``(center, context)`` pairs within a sliding window."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    pairs: List[Tuple[int, int]] = []
    for sequence in sequences:
        for index, center in enumerate(sequence):
            left = max(0, index - window_size)
            right = min(len(sequence), index + window_size + 1)
            for other in range(left, right):
                if other != index and sequence[other] != center:
                    pairs.append((center, sequence[other]))
    return pairs


def path_summary(paths: Mapping[str, MetaPath]) -> List[Dict[str, str]]:
    """Human-readable description used by logs and docs."""
    return [
        {"name": path.name, "path": " -> ".join(path.types), "center": path.center_type}
        for path in paths.values()
    ]
