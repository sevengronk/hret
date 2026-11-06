#!/usr/bin/env python
"""Evaluate a trained HRET checkpoint on the validation and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from hret.config import Config  # noqa: E402
from hret.runner import build_hret, evaluate_split, prepare_bundle  # noqa: E402
from hret.utils import get_logger, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HRET checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to hret.pt")
    parser.add_argument("--output", default=None, help="Where to write metrics JSON")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config.from_dict(checkpoint["config"])
    if args.device:
        config.device = args.device

    device = torch.device(config.resolve_device())
    logger = get_logger("hret-eval")
    bundle = prepare_bundle(config, device=device, logger=logger, pretrain=False)

    for key, actual, expected in (
        ("user_ids", bundle.dataset.user_ids, checkpoint["user_ids"]),
        ("business_ids", bundle.dataset.business_ids, checkpoint["business_ids"]),
    ):
        if list(actual) != list(expected):
            raise ValueError(
                f"Checkpoint {key} do not match the dataset in {config.data.dir}. "
                "Evaluate with the data used for training."
            )

    model = build_hret(bundle, config, use_pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val = evaluate_split(model, bundle, config, which="val", device=device)
    test = evaluate_split(model, bundle, config, which="test", device=device)
    logger.info("val  | %s", val.metrics)
    logger.info("test | %s", test.metrics)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "val": val.metrics,
        "test": test.metrics,
    }
    output = args.output or str(Path(args.checkpoint).parent / "metrics_eval.json")
    save_json(payload, output)
    logger.info("metrics written to %s", output)


if __name__ == "__main__":
    main()
