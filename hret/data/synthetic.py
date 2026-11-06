"""Synthetic testing-and-inspection data with a learnable preference signal.

The generator is intentionally simple but not trivial: every user has a
preferred inspection category, a region, and a varying preference strength.
Businesses offer services from a small set of categories and are located in
a region. Interactions are sampled with a weight that rewards category match,
region match and a small amount of noise, so a meta-path model can recover
structure that a popularity baseline cannot.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

CATEGORIES = (
    "material",
    "electrical",
    "environment",
    "food",
    "construction",
    "medical",
)


def _weighted_sample_without_replacement(
    rng: random.Random, population: Sequence[int], weights: Sequence[float], count: int
) -> List[int]:
    chosen: List[int] = []
    available = list(population)
    available_weights = list(weights)
    count = min(count, len(available))
    for _ in range(count):
        index = rng.choices(range(len(available)), weights=available_weights, k=1)[0]
        chosen.append(available.pop(index))
        available_weights.pop(index)
    return chosen


def generate_synthetic(
    out_dir: str | Path,
    seed: int = 42,
    num_users: int = 240,
    num_businesses: int = 80,
    num_services: int = 24,
    num_regions: int = 6,
    min_interactions: int = 8,
    max_interactions: int = 18,
) -> Dict[str, int]:
    """Write ``users.csv``, ``businesses.csv``, ... to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if num_services < len(CATEGORIES):
        raise ValueError(f"num_services must be >= {len(CATEGORIES)}")
    if num_regions < 2:
        raise ValueError("num_regions must be >= 2")

    regions = [f"R{index:02d}" for index in range(num_regions)]
    standards = [f"GB/T-{1000 + index}" for index in range(8)]

    services = []
    for index in range(num_services):
        services.append(
            {
                "service_id": f"S{index:04d}",
                "category": CATEGORIES[index % len(CATEGORIES)],
                "standard_id": standards[index % len(standards)],
            }
        )

    service_category = {row["service_id"]: row["category"] for row in services}

    businesses = []
    business_services: Dict[str, List[str]] = {}
    for index in range(num_businesses):
        business_id = f"B{index:04d}"
        focus = rng.choice(CATEGORIES)
        pool = [s for s in services if s["category"] == focus]
        others = [s for s in services if s["category"] != focus]
        count = rng.randint(3, 6)
        chosen = rng.sample(pool, k=min(len(pool), max(1, count // 2)))
        remaining = [s for s in others if s not in chosen]
        chosen += rng.sample(remaining, k=min(len(remaining), count - len(chosen)))
        businesses.append(
            {
                "business_id": business_id,
                "region": rng.choice(regions),
            }
        )
        business_services[business_id] = [s["service_id"] for s in chosen]

    users = []
    interactions = []
    for index in range(num_users):
        user_id = f"U{index:04d}"
        region = rng.choice(regions)
        preferred = rng.choice(CATEGORIES)
        strength = rng.uniform(0.45, 0.95)
        users.append(
            {
                "user_id": user_id,
                "region": region,
                "industry": preferred,
            }
        )

        weights = []
        for business in businesses:
            categories = {service_category[s] for s in business_services[business["business_id"]]}
            category_match = 1.0 if preferred in categories else 0.0
            region_match = 1.0 if business["region"] == region else 0.0
            weight = (
                0.02
                + 4.5 * strength * category_match
                + 1.5 * region_match
                + rng.random() * 0.2
            )
            weights.append(weight)

        count = rng.randint(min_interactions, max_interactions)
        chosen_businesses = _weighted_sample_without_replacement(
            rng, range(len(businesses)), weights, count
        )

        day = 0
        for business_index in chosen_businesses:
            business = businesses[business_index]
            offered = business_services[business["business_id"]]
            service_weights = [
                8.0 if service_category[service] == preferred else 1.0 for service in offered
            ]
            service = rng.choices(offered, weights=service_weights, k=1)[0]
            day += rng.randint(1, 14)
            interactions.append(
                {
                    "user_id": user_id,
                    "business_id": business["business_id"],
                    "service_id": service,
                    "timestamp": f"2022{1 + (day // 28) % 12:02d}{1 + day % 28:02d}",
                }
            )

    def write(name: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
        with open(out_dir / name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("users.csv", users, ["user_id", "region", "industry"])
    write("businesses.csv", businesses, ["business_id", "region"])
    write("services.csv", services, ["service_id", "category", "standard_id"])
    write(
        "business_services.csv",
        [
            {"business_id": business_id, "service_id": service_id}
            for business_id, services_ in business_services.items()
            for service_id in services_
        ],
        ["business_id", "service_id"],
    )
    write(
        "interactions.csv",
        interactions,
        ["user_id", "business_id", "service_id", "timestamp"],
    )

    return {
        "users": len(users),
        "businesses": len(businesses),
        "services": len(services),
        "regions": len(regions),
        "interactions": len(interactions),
    }
