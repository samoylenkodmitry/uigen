#!/usr/bin/env python3
"""Evaluate SlotNet overfit on exported Cranamp BMP pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.dataset import RenderDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, blank_atlas_like_files, crop_export_target
from atlas_ai.profiles import load_atlas_profile
from atlas_ai.support_mask import load_support_masks
from infer_skin import detect_checkpoint_info, load_checkpoint
from models.losses import sobel_edges
from models.slotnet_v34 import SlotNetV34
from models.slotnet_v35 import SlotNetV35


def file_stem(file_name: str) -> str:
    return file_name.lower().removesuffix(".bmp")


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
    supported_mask = torch.zeros((atlas_profile.canvas_h, atlas_profile.canvas_w), dtype=torch.bool, device=device)
    for slot in atlas_profile.slots:
        supported_mask[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w] = True
    file_support_masks = {
        name: mask.to(device=device)
        for name, mask in load_support_masks().items()
    }

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

    totals = {
        "full_atlas": [0.0, 0],
        "supported_slots": [0.0, 0],
    }
    file_totals = {
        spec.file_name: {"mae": 0.0, "sobel": 0.0, "hit5": 0.0, "samples": 0}
        for spec in TRAINABLE_EXPORT_SPECS
    }
    per_sample = []
    with torch.no_grad():
        for idx in range(len(rows)):
            item = rows[idx]
            view = item["view"].unsqueeze(0).to(device)
            target = item["atlas"].unsqueeze(0).to(device)
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
                pred = blank_atlas_like_files(pred_files)
            abs_err = (pred - target).abs()
            full_mae = float(abs_err.mean().detach().cpu())
            supported_mae = float(abs_err[:, :, supported_mask].mean().detach().cpu())
            per_file = {}
            sample_maes = []
            sample_sobels = []
            sample_hit5s = []
            for spec in TRAINABLE_EXPORT_SPECS:
                pred_file = pred_files[spec.file_name]
                target_file = crop_export_target(target, spec)
                file_abs = (pred_file - target_file).abs()
                support = file_support_masks[spec.file_name]
                support_f = support.to(target_file.dtype)
                support_pixels = float(support.sum().clamp(min=1))
                channels = pred_file.shape[1]
                batch = pred_file.shape[0]
                support_denom = support_pixels * channels * batch
                mae = float((file_abs * support_f).sum().detach().cpu() / support_denom)
                pred_for_edges = pred_file * support_f + target_file * (1 - support_f)
                edge_diff = (sobel_edges(pred_for_edges) - sobel_edges(target_file)).abs()
                sobel = float((edge_diff * support_f).sum().detach().cpu() / support_denom)
                hit5_per_chan = (file_abs <= (5.0 / 255.0)).to(target_file.dtype)
                hit5 = float((hit5_per_chan * support_f).sum().detach().cpu() / support_denom)
                stem = file_stem(spec.file_name)
                per_file[stem] = {
                    "mae": mae,
                    "sobel_mae": sobel,
                    "hit_5_255": hit5,
                }
                sample_maes.append(mae)
                sample_sobels.append(sobel)
                sample_hit5s.append(hit5)
                file_totals[spec.file_name]["mae"] += mae
                file_totals[spec.file_name]["sobel"] += sobel
                file_totals[spec.file_name]["hit5"] += hit5
                file_totals[spec.file_name]["samples"] += 1
            exported_mae = sum(sample_maes) / len(sample_maes)
            exported_sobel = sum(sample_sobels) / len(sample_sobels)
            exported_hit5 = sum(sample_hit5s) / len(sample_hit5s)
            per_sample.append(
                {
                    "skin_id": str(item["skin_id"]),
                    "variant_id": str(item["variant_id"]),
                    "full_atlas_mae": full_mae,
                    "supported_slots_mae": supported_mae,
                    "exported_pixels_mae": exported_mae,
                    "exported_pixels_sobel_mae": exported_sobel,
                    "exported_pixels_hit_5_255": exported_hit5,
                    "per_exported_file": per_file,
                }
            )
            masks = {
                "full_atlas": torch.ones_like(abs_err, dtype=torch.bool),
                "supported_slots": supported_mask[None, None, :, :].expand_as(abs_err),
            }
            for name, mask in masks.items():
                totals[name][0] += float(abs_err[mask].sum().detach().cpu())
                totals[name][1] += int(mask.sum().detach().cpu())

    per_file_result = {}
    for file_name, values in file_totals.items():
        count = values["samples"]
        stem = file_stem(file_name)
        per_file_result[stem] = {
            "mae": values["mae"] / count,
            "sobel_mae": values["sobel"] / count,
            "hit_5_255": values["hit5"] / count,
        }
    exported_maes = [values["mae"] for values in per_file_result.values()]
    exported_sobels = [values["sobel_mae"] for values in per_file_result.values()]
    exported_hit5s = [values["hit_5_255"] for values in per_file_result.values()]
    result = {
        "samples": len(per_sample),
        "slotnet_version": ckpt_info.version,
        "full_atlas_mae": totals["full_atlas"][0] / totals["full_atlas"][1],
        "supported_slots_mae": totals["supported_slots"][0] / totals["supported_slots"][1],
        "exported_pixels_mae": sum(exported_maes) / len(exported_maes),
        "exported_pixels_sobel_mae": sum(exported_sobels) / len(exported_sobels),
        "exported_pixels_hit_5_255": sum(exported_hit5s) / len(exported_hit5s),
        "per_exported_file": per_file_result,
        "per_sample": per_sample,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
