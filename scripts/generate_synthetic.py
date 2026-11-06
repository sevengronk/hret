#!/usr/bin/env python
"""Generate the synthetic testing-and-inspection dataset shipped in data/sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hret.data.synthetic import generate_synthetic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic HRET data")
    parser.add_argument("--out", default="data/sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users", type=int, default=240)
    parser.add_argument("--businesses", type=int, default=80)
    parser.add_argument("--services", type=int, default=24)
    parser.add_argument("--regions", type=int, default=6)
    parser.add_argument("--min-interactions", type=int, default=8)
    parser.add_argument("--max-interactions", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_synthetic(
        args.out,
        seed=args.seed,
        num_users=args.users,
        num_businesses=args.businesses,
        num_services=args.services,
        num_regions=args.regions,
        min_interactions=args.min_interactions,
        max_interactions=args.max_interactions,
    )
    print(f"Wrote dataset to {args.out}: {summary}")


if __name__ == "__main__":
    main()
