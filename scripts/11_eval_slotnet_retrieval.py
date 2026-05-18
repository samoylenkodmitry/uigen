#!/usr/bin/env python3
"""Evaluate whether SlotNet predictions retrieve the correct target skin.

For each input row, predict exported BMP tensors and compare them against one
target atlas per skin_id from the same CSV. The top-1 target by exported-pixel
MAE should be the row's own skin_id in a multi-skin memorization run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.dataset import RenderDataset, image_to_tensor
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, crop_export_target
from infer_skin import detect_checkpoint_info, load_checkpoint
from models.slotnet_v34 import SlotNetV34
from models.slotnet_v35 import SlotNetV35
from atlas_ai.profiles import load_atlas_profile


def build_target_bank(rows: RenderDataset, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    bank: dict[str, dict[str, torch.Tensor]] = {}
    for row in rows.rows:
        if row.skin_id in bank:
            continue
        atlas = image_to_tensor(row.atlas_png, "RGB").unsqueeze(0).to(device)
        bank[row.skin_id] = {
            spec.file_name: crop_export_target(atlas, spec)
            for spec in TRAINABLE_EXPORT_SPECS
        }
    return bank


def exported_mae(pred_files: dict[str, torch.Tensor], target_files: dict[str, torch.Tensor]) -> torch.Tensor:
    file_maes = []
    for spec in TRAINABLE_EXPORT_SPECS:
        file_maes.append((pred_files[spec.file_name] - target_files[spec.file_name]).abs().mean())
    return torch.stack(file_maes).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    rows = RenderDataset(args.samples)
    if args.limit is not None:
        rows.rows = rows.rows[: args.limit]
    if len(rows) == 0:
        raise SystemExit(f"no samples in {args.samples}")

    device = torch.device(args.device)
    atlas_profile = load_atlas_profile(args.atlas_profile)
    ckpt_info = detect_checkpoint_info(Path(args.slotnet))
    base_channels = args.base_channels or ckpt_info.base_channels
    if ckpt_info.version == 34:
        model = SlotNetV34(atlas_profile=atlas_profile, base_channels=base_channels).to(device)
    elif ckpt_info.version == 35:
        model = SlotNetV35(
            base_channels=base_channels,
            style_dim=ckpt_info.style_dim or 192,
            head_channels=ckpt_info.head_channels,
        ).to(device)
    else:
        raise SystemExit(f"unsupported SlotNet version {ckpt_info.version}")
    load_checkpoint(model, Path(args.slotnet))
    model.eval()

    target_bank = build_target_bank(rows, device)
    per_sample = []
    true_maes = []
    best_maes = []
    hits = 0
    with torch.no_grad():
        for idx in range(len(rows)):
            item = rows[idx]
            view = item["view"].unsqueeze(0).to(device)
            output = model(view)
            if ckpt_info.version == 34:
                pred = output["prediction"].sigmoid().clamp(0, 1)
                pred_files = {
                    spec.file_name: crop_export_target(pred, spec)
                    for spec in TRAINABLE_EXPORT_SPECS
                }
            else:
                pred_files = {
                    name: logits.sigmoid().clamp(0, 1)
                    for name, logits in output["files"].items()
                }

            distances = {
                skin_id: float(exported_mae(pred_files, target_files).detach().cpu())
                for skin_id, target_files in target_bank.items()
            }
            predicted_skin = min(distances, key=distances.get)
            true_skin = str(item["skin_id"])
            is_hit = predicted_skin == true_skin
            hits += int(is_hit)
            true_mae = distances[true_skin]
            best_mae = distances[predicted_skin]
            true_maes.append(true_mae)
            best_maes.append(best_mae)
            per_sample.append(
                {
                    "index": idx,
                    "skin_id": true_skin,
                    "variant_id": str(item["variant_id"]),
                    "predicted_skin_id": predicted_skin,
                    "top1_hit": is_hit,
                    "true_exported_pixels_mae": true_mae,
                    "best_exported_pixels_mae": best_mae,
                }
            )

    result = {
        "samples": len(per_sample),
        "target_skins": len(target_bank),
        "slotnet_version": ckpt_info.version,
        "top1_accuracy": hits / len(per_sample),
        "mean_true_exported_pixels_mae": float(np.mean(true_maes)),
        "median_true_exported_pixels_mae": float(np.median(true_maes)),
        "mean_best_exported_pixels_mae": float(np.mean(best_maes)),
        "median_best_exported_pixels_mae": float(np.median(best_maes)),
        "per_sample": per_sample,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
