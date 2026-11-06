#!/usr/bin/env python
"""Run the HRET benchmark against popularity and BPR-MF baselines.

By default a single seed is used. Pass several seeds to report mean and
standard deviation, which is recommended before drawing conclusions:

    python scripts/benchmark.py --config configs/hret.yaml \
        --output runs/benchmark --seeds 42 7 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hret.config import Config  # noqa: E402
from hret.runner import run_benchmark  # noqa: E402
from hret.utils import save_json  # noqa: E402

MODELS = ("Popularity", "BPR-MF", "HRET")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HRET")
    parser.add_argument("--config", default="configs/hret.yaml")
    parser.add_argument("--output", default="runs/benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="One or more random seeds (default: 42)",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Config.from_yaml(args.config).apply_overrides(args.set)
    if args.device:
        base.device = args.device

    output = Path(args.output)
    per_seed = []
    for seed in args.seeds:
        config = base.copy()
        config.seed = seed
        per_seed.append(run_benchmark(config, output / f"seed_{seed}"))

    summary = {}
    print(f"\nTest metrics over {len(args.seeds)} seed(s): {args.seeds}")
    header = f"{'model':<12} {'HR@5':>14} {'HR@10':>14} {'NDCG@5':>14} {'NDCG@10':>14} {'MRR':>14}"
    print(header)
    for model in MODELS:
        metrics = {}
        for metric in ("HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR"):
            values = [run[model]["test"][metric] * 100 for run in per_seed]
            metrics[metric] = {"mean": mean(values), "std": pstdev(values)}
        summary[model] = metrics
        cells = " ".join(
            f"{metrics[metric]['mean']:6.2f}±{metrics[metric]['std']:5.2f}"
            for metric in ("HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR")
        )
        print(f"{model:<12} {cells}")

    save_json(
        {
            "seeds": args.seeds,
            "summary": summary,
            "per_seed": {str(seed): run for seed, run in zip(args.seeds, per_seed)},
        },
        output / "benchmark_summary.json",
    )
    print(f"\nWrote {output / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
