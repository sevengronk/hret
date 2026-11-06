"""Dataset loading and the leave-one-out ranking protocol.

Expected files inside ``data.dir``::

    users.csv              user_id,region
    businesses.csv         business_id,region
    services.csv           service_id,category,standard_id
    business_services.csv  business_id,service_id
    interactions.csv       user_id,business_id,service_id,timestamp

Only the columns listed above are required; extra columns are ignored.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .schema import HIN

REQUIRED_FILES = {
    "users.csv": ("user_id", "region"),
    "businesses.csv": ("business_id", "region"),
    "services.csv": ("service_id", "category", "standard_id"),
    "business_services.csv": ("business_id", "service_id"),
    "interactions.csv": ("user_id", "business_id", "service_id", "timestamp"),
}


@dataclass(frozen=True)
class Interaction:
    user: int
    business: int
    service: int
    timestamp: int = 0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.user, self.business, self.service, self.timestamp)


@dataclass
class Dataset:
    """Index-encoded testing-and-inspection dataset."""

    user_ids: List[str]
    business_ids: List[str]
    service_ids: List[str]
    standard_ids: List[str]
    region_ids: List[str]
    # Node attributes, aligned with the id lists above.
    user_region: List[int] = field(default_factory=list)
    business_region: List[int] = field(default_factory=list)
    service_standard: List[int] = field(default_factory=list)
    # Adjacency needed to instantiate the HIN schema.
    business_services: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    interactions: List[Interaction] = field(default_factory=list)

    @property
    def num_nodes(self) -> Dict[str, int]:
        return {
            "user": len(self.user_ids),
            "business": len(self.business_ids),
            "service": len(self.service_ids),
            "standard": len(self.standard_ids),
            "region": len(self.region_ids),
        }

    def summary(self) -> Dict[str, int]:
        return {
            **self.num_nodes,
            "interactions": len(self.interactions),
            "business_service_edges": sum(len(v) for v in self.business_services.values()),
        }


def _read_rows(path: Path, required: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return [dict(row) for row in reader]


def _index(values: Iterable[str], label: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for value in values:
        value = str(value).strip()
        if not value:
            raise ValueError(f"Empty {label} identifier found")
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def _as_int(value: str, column: str, path: Path) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} in {path} must be numeric, got {value!r}") from exc


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load and index a dataset directory.

    Identifier order follows first appearance in the CSV files, which makes
    the encoding deterministic for a fixed dataset.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_dir}")

    users = _read_rows(data_dir / "users.csv", REQUIRED_FILES["users.csv"])
    businesses = _read_rows(data_dir / "businesses.csv", REQUIRED_FILES["businesses.csv"])
    services = _read_rows(data_dir / "services.csv", REQUIRED_FILES["services.csv"])
    business_services = _read_rows(
        data_dir / "business_services.csv", REQUIRED_FILES["business_services.csv"]
    )
    interactions = _read_rows(
        data_dir / "interactions.csv", REQUIRED_FILES["interactions.csv"]
    )

    user_index = _index((row["user_id"] for row in users), "user")
    business_index = _index((row["business_id"] for row in businesses), "business")
    service_index = _index((row["service_id"] for row in services), "service")
    standard_index = _index((row["standard_id"] for row in services), "standard")
    region_index = _index(
        [row["region"] for row in users] + [row["region"] for row in businesses], "region"
    )

    def clean(value: str) -> str:
        return str(value).strip()

    user_region = [region_index[clean(row["region"])] for row in users]
    business_region = [region_index[clean(row["region"])] for row in businesses]
    service_standard = [standard_index[clean(row["standard_id"])] for row in services]

    business_service_map: Dict[int, List[int]] = defaultdict(list)
    for row in business_services:
        business = business_index.get(clean(row["business_id"]))
        service = service_index.get(clean(row["service_id"]))
        if business is None:
            raise ValueError(f"business_services.csv references unknown business {row['business_id']!r}")
        if service is None:
            raise ValueError(f"business_services.csv references unknown service {row['service_id']!r}")
        if service not in business_service_map[business]:
            business_service_map[business].append(service)

    encoded_interactions: List[Interaction] = []
    for line_no, row in enumerate(interactions, start=2):
        user = user_index.get(clean(row["user_id"]))
        business = business_index.get(clean(row["business_id"]))
        service = service_index.get(clean(row["service_id"]))
        if user is None:
            raise ValueError(f"interactions.csv:{line_no} references unknown user {row['user_id']!r}")
        if business is None:
            raise ValueError(
                f"interactions.csv:{line_no} references unknown business {row['business_id']!r}"
            )
        if service is None:
            raise ValueError(
                f"interactions.csv:{line_no} references unknown service {row['service_id']!r}"
            )
        encoded_interactions.append(
            Interaction(
                user=user,
                business=business,
                service=service,
                timestamp=_as_int(row["timestamp"], "timestamp", data_dir / "interactions.csv"),
            )
        )

    if not encoded_interactions:
        raise ValueError("interactions.csv contains no interactions")

    return Dataset(
        user_ids=list(user_index.keys()),
        business_ids=list(business_index.keys()),
        service_ids=list(service_index.keys()),
        standard_ids=list(standard_index.keys()),
        region_ids=list(region_index.keys()),
        user_region=user_region,
        business_region=business_region,
        service_standard=service_standard,
        business_services={k: sorted(v) for k, v in business_service_map.items()},
        interactions=encoded_interactions,
    )


@dataclass
class Split:
    """Leave-one-out split for top-N business recommendation."""

    train: List[Interaction]
    val: Dict[int, int]
    test: Dict[int, int]
    train_by_user: Dict[int, List[int]]

    @property
    def eval_users(self) -> List[int]:
        return sorted(set(self.val) & set(self.test))

    def seen_for_val(self, user: int) -> Set[int]:
        return set(self.train_by_user.get(user, []))

    def seen_for_test(self, user: int) -> Set[int]:
        seen = set(self.train_by_user.get(user, []))
        if user in self.val:
            seen.add(self.val[user])
        return seen


def leave_one_out_split(
    dataset: Dataset,
    min_interactions: int = 3,
    strategy: str = "random",
    seed: int = 0,
) -> Split:
    """Split each user's history into train / validation / test.

    Repeated interactions with the same business are collapsed to their latest
    occurrence, because the task is to rank *businesses*, not visits. Then:

    * ``strategy="random"`` shuffles the distinct businesses and holds out the
      last two for validation and test (default);
    * ``strategy="temporal"`` holds out the two most recent interactions.

    Both protocols are leave-one-out. Random is the default, matching the
    common HIN-recommendation protocol; temporal is available for deployment
    simulations where the time order matters.
    """
    if min_interactions < 2:
        raise ValueError("min_interactions must be >= 2")
    if strategy not in {"random", "temporal"}:
        raise ValueError(f"Unknown split strategy: {strategy!r}")
    rng = random.Random(seed)

    grouped: Dict[int, List[Interaction]] = defaultdict(list)
    for interaction in dataset.interactions:
        grouped[interaction.user].append(interaction)

    train: List[Interaction] = []
    val: Dict[int, int] = {}
    test: Dict[int, int] = {}
    train_by_user: Dict[int, List[int]] = defaultdict(list)

    for user, history in grouped.items():
        # Collapse duplicates: keep the latest record for each business.
        latest: Dict[int, Interaction] = {}
        for interaction in history:
            previous = latest.get(interaction.business)
            if previous is None or interaction.timestamp >= previous.timestamp:
                latest[interaction.business] = interaction
        ordered = list(latest.values())
        if strategy == "temporal":
            ordered.sort(key=lambda item: (item.timestamp, item.business))
        else:
            rng.shuffle(ordered)

        if len(ordered) < min_interactions:
            chosen_train = ordered
        else:
            chosen_train = ordered[:-2]
            val[user] = ordered[-2].business
            test[user] = ordered[-1].business

        for interaction in chosen_train:
            train.append(interaction)
            if interaction.business not in train_by_user[user]:
                train_by_user[user].append(interaction.business)

    train.sort(key=lambda item: (item.user, item.timestamp, item.business))
    return Split(train=train, val=val, test=test, train_by_user=dict(train_by_user))


def build_hin(dataset: Dataset, interactions: Optional[Sequence[Interaction]] = None) -> HIN:
    """Instantiate the HIN from the dataset.

    Pass the *training* interactions only when building a model input; the
    held-out interactions must not be used to construct the graph.
    """
    hin = HIN(dataset.num_nodes)
    edges = dataset.interactions if interactions is None else interactions

    for interaction in edges:
        hin.add_edge("user", "business", interaction.user, interaction.business)
        hin.add_edge("user", "service", interaction.user, interaction.service)

    for business, services in dataset.business_services.items():
        for service in services:
            hin.add_edge("business", "service", business, service)

    for user, region in enumerate(dataset.user_region):
        hin.add_edge("user", "region", user, region)
    for business, region in enumerate(dataset.business_region):
        hin.add_edge("business", "region", business, region)
    for service, standard in enumerate(dataset.service_standard):
        hin.add_edge("service", "standard", service, standard)

    return hin
