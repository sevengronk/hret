#!/usr/bin/env python
"""Export meta-path attention weights from a trained checkpoint.

The output makes it possible to inspect *why* a user or a business received a
particular representation: for every test user we report the softmax weight of
each meta-path view, plus the weights of the business they actually chose.

Example
-------
    python scripts/explain.py --checkpoint runs/hret/hret.pt \
        --output runs/hret/attention
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from hret.config import Config  # noqa: E402
from hret.runner import build_hret, prepare_bundle  # noqa: E402
from hret.utils import get_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export meta-path attention weights")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="Output prefix (default: checkpoint dir)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config.from_dict(checkpoint["config"])
    if args.device:
        config.device = args.device
    device = torch.device(config.resolve_device())
    logger = get_logger("hret-explain")

    bundle = prepare_bundle(config, device=device, logger=logger, pretrain=False)
    model = build_hret(bundle, config, use_pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    prefix = Path(args.output) if args.output else Path(args.checkpoint).parent / "attention"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    users = sorted(bundle.split.test)
    user_rows = []
    business_rows = []
    sums = torch.zeros(len(bundle.user_paths))
    sums_business = torch.zeros(len(bundle.business_paths))

    with torch.no_grad():
        for start in range(0, len(users), 256):
            batch = users[start : start + 256]
            user_tensor = torch.as_tensor(batch, dtype=torch.long, device=device)
            _, attention = model.encode_users(user_tensor)
            sums += attention.sum(dim=0).cpu()
            for row, user in enumerate(batch):
                user_rows.append(
                    [bundle.dataset.user_ids[user]]
                    + [f"{value:.6f}" for value in attention[row].tolist()]
                )

                target = bundle.split.test[user]
                _, business_attention = model.encode_businesses(
                    torch.tensor([target], device=device)
                )
                sums_business += business_attention.sum(dim=0).cpu()
                business_rows.append(
                    [bundle.dataset.user_ids[user], bundle.dataset.business_ids[target]]
                    + [f"{value:.6f}" for value in business_attention[0].tolist()]
                )

    with open(f"{prefix}_users.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id"] + bundle.user_paths)
        writer.writerows(user_rows)

    with open(f"{prefix}_businesses.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "business_id"] + bundle.business_paths)
        writer.writerows(business_rows)

    logger.info("mean user attention: %s", dict(zip(bundle.user_paths, (sums / len(users)).tolist())))
    logger.info(
        "mean chosen-business attention: %s",
        dict(zip(bundle.business_paths, (sums_business / len(users)).tolist())),
    )
    logger.info("wrote %s_users.csv and %s_businesses.csv", prefix, prefix)


if __name__ == "__main__":
    main()
