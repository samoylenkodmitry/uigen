#!/usr/bin/env python3
"""V3.1 diagnostics per GPT review.

A. Mean-atlas baseline   -- compute the per-pixel median over training atlases,
                            measure RGB MAE / Sobel MAE on a sample of val
                            atlases. The trained model must beat this.
B. Input sensitivity     -- run the model on (real_A, real_B, zeros, noise)
                            inputs; report output std and pairwise diffs.
                            Collapsed model gives diffs near 0.

Usage:
    python scripts/_v31_diagnostics.py \\
        --slotnet runs/slotnet_v31/best.safetensors \\
        --train data_v0/train.csv \\
        --val data_v0/val.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.dataset import RenderDataset, image_to_tensor
from atlas_ai.profiles import load_atlas_profile, load_json
from models.atlas import load_full_atlas_target, pack_default_atlas_tensor
from models.losses import sobel_edges
from models.slotnet_v31 import SlotNetV31


def compute_mean_atlas(rows, atlas_profile, sample_n: int = 128) -> torch.Tensor:
    n = min(sample_n, len(rows))
    acc = torch.zeros(3, atlas_profile.canvas_h, atlas_profile.canvas_w)
    for r in rows[:n]:
        acc += image_to_tensor(r.atlas_png, "RGB")
    return acc / n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--train", default="data_v0/train.csv")
    parser.add_argument("--val", default="data_v0/val.csv")
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--magenta-policy", default="configs/magenta_policy.json")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--n-mean", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ap = load_atlas_profile(args.atlas_profile)
    pol = load_json(args.magenta_policy)
    da = pack_default_atlas_tensor(args.default_skin, ap).to(device)

    train_ds = RenderDataset(args.train)
    val_ds = RenderDataset(args.val)
    mean_atlas = compute_mean_atlas(train_ds.rows, ap, sample_n=args.n_mean).to(device)

    model = SlotNetV31(atlas_profile=ap, default_atlas=da, base_channels=args.base_channels).to(device).eval()
    try:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(args.slotnet))
    except Exception:
        model.load_state_dict(torch.load(args.slotnet, map_location="cpu"))

    print("\n=== A. Mean-atlas baseline vs model ===")
    n = min(args.n_val, len(val_ds.rows))
    mean_rgb_mae_baseline = []
    mean_sobel_mae_baseline = []
    pred_rgb_mae_model = []
    pred_sobel_mae_model = []
    with torch.no_grad():
        for r in val_ds.rows[:n]:
            tgt = load_full_atlas_target(r.atlas_png, r.atlas_mask_png, r.visible_mask_png, r.slot_weight_f32, ap, pol, hidden_weight=0.03)
            atlas_mask = tgt["atlas_mask"][0].to(device)  # [H, W]
            tgt_rgb = tgt["target_rgb"].to(device)
            mb = atlas_mask > 0.5
            if mb.sum() == 0:
                continue
            # Mean-atlas baseline
            diff = (mean_atlas - tgt_rgb).abs()
            mae = (diff * atlas_mask[None]).sum() / (atlas_mask.sum() * 3 + 1e-8)
            mean_rgb_mae_baseline.append(float(mae))
            # Sobel for mean-atlas baseline
            se_mean = sobel_edges(mean_atlas.unsqueeze(0))
            se_tgt = sobel_edges(tgt_rgb.unsqueeze(0))
            sm = (se_mean - se_tgt).abs()
            mae_s = (sm * atlas_mask[None, None]).sum() / (atlas_mask.sum() * sm.shape[1] + 1e-8)
            mean_sobel_mae_baseline.append(float(mae_s))
            # Model
            view = image_to_tensor(r.view_png, "RGB").unsqueeze(0).to(device)
            out = model(view)
            pred_rgb = out["prediction"][0, :3].sigmoid()
            diff_m = (pred_rgb - tgt_rgb).abs()
            mae_m = (diff_m * atlas_mask[None]).sum() / (atlas_mask.sum() * 3 + 1e-8)
            pred_rgb_mae_model.append(float(mae_m))
            se_pred = sobel_edges(pred_rgb.unsqueeze(0))
            sm_m = (se_pred - se_tgt).abs()
            mae_sm = (sm_m * atlas_mask[None, None]).sum() / (atlas_mask.sum() * sm_m.shape[1] + 1e-8)
            pred_sobel_mae_model.append(float(mae_sm))
    print(f"  mean_atlas RGB MAE   = {np.mean(mean_rgb_mae_baseline):.4f}  Sobel MAE = {np.mean(mean_sobel_mae_baseline):.4f}")
    print(f"  model      RGB MAE   = {np.mean(pred_rgb_mae_model):.4f}  Sobel MAE = {np.mean(pred_sobel_mae_model):.4f}")
    if np.mean(pred_rgb_mae_model) >= np.mean(mean_rgb_mae_baseline):
        print("  WARNING: model does NOT beat mean-atlas baseline on RGB.")
    else:
        print("  OK: model beats mean-atlas baseline on RGB.")

    print("\n=== B. Input sensitivity ===")
    real_a = image_to_tensor(val_ds.rows[0].view_png, "RGB").unsqueeze(0).to(device)
    real_b = image_to_tensor(val_ds.rows[1 % n].view_png, "RGB").unsqueeze(0).to(device)
    zeros = torch.zeros_like(real_a)
    noise = torch.rand_like(real_a)
    with torch.no_grad():
        out_a = model(real_a)["prediction"][0, :3].sigmoid()
        out_b = model(real_b)["prediction"][0, :3].sigmoid()
        out_z = model(zeros)["prediction"][0, :3].sigmoid()
        out_n = model(noise)["prediction"][0, :3].sigmoid()
    def stat(name, t):
        return f"{name:12s} mean=({t.mean(dim=(1,2))[0]:.3f},{t.mean(dim=(1,2))[1]:.3f},{t.mean(dim=(1,2))[2]:.3f}) std={t.std():.4f}"
    print(f"  {stat('real_A', out_a)}")
    print(f"  {stat('real_B', out_b)}")
    print(f"  {stat('zeros', out_z)}")
    print(f"  {stat('noise', out_n)}")
    print(f"  diff(real_A, real_B) = {(out_a - out_b).abs().mean():.4f}")
    print(f"  diff(real_A, zeros)  = {(out_a - out_z).abs().mean():.4f}")
    print(f"  diff(real_A, noise)  = {(out_a - out_n).abs().mean():.4f}")
    if float((out_a - out_b).abs().mean()) < 0.05 and float(out_a.std()) < 0.05:
        print("  WARNING: looks collapsed (low diff + low std).")
    else:
        print("  OK: model output varies with input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
