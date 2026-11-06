#!/usr/bin/env python
"""Train HRET on a dataset directory.

Examples
--------
    python scripts/train.py --config configs/hret.yaml
    python scripts/train.py --config configs/hret.yaml --output runs/exp1 \
        --set train.epochs=60 model.dim=128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hret.config import Config  # noqa: E402
from hret.runner import run_hret  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HRET")
    parser.add_argument("--config", default="configs/hret.yaml", help="YAML configuration")
    parser.add_argument("--output", default="runs/hret", help="Output directory")
    parser.add_argument("--device", default=None, help="Override device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set train.epochs=20 model.dim=32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config).apply_overrides(args.set)
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    results = run_hret(config, args.output)
    print("\nTest metrics:")
    for key, value in results["test"].items():
        if key != "users":
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
