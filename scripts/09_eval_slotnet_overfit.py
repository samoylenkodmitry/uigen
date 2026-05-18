#!/usr/bin/env python3
"""Evaluate direct SlotNet V3.4 atlas overfit MAE on a sample CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.dataset import RenderDataset
from atlas_ai.profiles import load_atlas_profile
from infer_skin import detect_base_channels, load_checkpoint
from models.slotnet_v34 import SlotNetV34


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = RenderDataset(args.samples)
    if args.limit is not None:
        rows.rows = rows.rows[: args.limit]
    if len(rows) == 0:
        raise SystemExit(f"no samples in {args.samples}")

    device = torch.device(args.device)
    atlas_profile = load_atlas_profile(args.atlas_profile)
    base_channels = args.base_channels or detect_base_channels(Path(args.slotnet))
    model = SlotNetV34(atlas_profile=atlas_profile, base_channels=base_channels).to(device)
    load_checkpoint(model, Path(args.slotnet))
    model.eval()

    total_abs = 0.0
    total_pixels = 0
    per_sample = []
    with torch.no_grad():
        for idx in range(len(rows)):
            item = rows[idx]
            view = item["view"].unsqueeze(0).to(device)
            target = item["atlas"].unsqueeze(0).to(device)
            pred = model(view)["prediction"].sigmoid().clamp(0, 1)
            abs_err = (pred - target).abs()
            mae = float(abs_err.mean().detach().cpu())
            per_sample.append(
                {
                    "skin_id": str(item["skin_id"]),
                    "variant_id": str(item["variant_id"]),
                    "rgb_mae": mae,
                }
            )
            total_abs += float(abs_err.sum().detach().cpu())
            total_pixels += abs_err.numel()

    result = {
        "samples": len(per_sample),
        "rgb_mae": total_abs / total_pixels,
        "per_sample": per_sample,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
