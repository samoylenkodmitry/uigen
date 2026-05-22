#!/usr/bin/env python3
"""Targeted diagnostic for the Gate C tvxq failure mode.

For one bad skin (tvxq) and one good skin (control), dump:
  - the letterboxed input view the model actually sees
  - target BMP crop, predicted BMP crop, absolute-diff heatmap per trainable file
  - attention heatmap per file overlaid on the input view (encoder grid resolution)
  - per-file supported-pixel and full-rectangle MAE summary JSON

Used after Gate C primary-pass / strict-fail to decide whether tvxq is a data,
attention, or capacity issue.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from infer_skin import (
    CheckpointInfo,
    build_model_from_info,
    detect_checkpoint_info,
    image_to_tensor,
    letterbox_to_canvas,
    load_checkpoint,
    INPUT_H,
    INPUT_W,
)


def find_rows_for_skin(csv_path: Path, skin_id: str, n: int) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["skin_id"] == skin_id]
    if not rows:
        raise SystemExit(f"no rows for skin_id={skin_id} in {csv_path}")
    return rows[:n]


def load_target_atlas(atlas_png: Path) -> np.ndarray:
    with Image.open(atlas_png) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return arr


def crop_target_bmp(atlas: np.ndarray, spec) -> np.ndarray:
    return atlas[spec.y : spec.y + spec.h, spec.x : spec.x + spec.w].copy()


def diff_visual(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-pixel absolute diff, summed across channels, scaled to 0-255."""
    d = np.abs(target.astype(np.int16) - pred.astype(np.int16)).sum(axis=-1)
    return np.clip(d, 0, 255).astype(np.uint8)


def mae01(target: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean RGB absolute error in [0, 1], optionally over supported pixels only."""
    diff = np.abs(pred.astype(np.float32) - target.astype(np.float32)) / 255.0
    if mask is None:
        return float(diff.mean())
    if mask.shape != target.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {target.shape[:2]}")
    supported = diff[mask]
    if supported.size == 0:
        return 0.0
    return float(supported.mean())


def attention_overlay(attn: np.ndarray, view_rgb: np.ndarray) -> np.ndarray:
    """Resize attention to view size, render as red-channel overlay on the view."""
    h_view, w_view = view_rgb.shape[:2]
    attn01 = attn - attn.min()
    if attn01.max() > 0:
        attn01 = attn01 / attn01.max()
    attn_img = Image.fromarray((attn01 * 255).astype(np.uint8))
    attn_resized = np.asarray(
        attn_img.resize((w_view, h_view), Image.Resampling.BILINEAR)
    )
    overlay = view_rgb.copy()
    red = attn_resized.astype(np.float32)
    overlay = overlay.astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] * 0.4 + red * 0.8, 0, 255)
    overlay[..., 1] = overlay[..., 1] * 0.4
    overlay[..., 2] = overlay[..., 2] * 0.4
    return overlay.astype(np.uint8)


def run_one_skin(
    *,
    model,
    device,
    csv_path: Path,
    skin_id: str,
    out_dir: Path,
    n_views: int,
    attention_files: set[str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = find_rows_for_skin(csv_path, skin_id, n_views)

    # Use the first row's atlas as the canonical target (atlas is per-skin, not per-view).
    atlas_target = load_target_atlas(Path(rows[0]["atlas_png"]))

    support_masks = {
        name: mask.cpu().numpy().astype(bool)
        for name, mask in load_support_masks().items()
    }
    supported_maes: dict[str, list[float]] = {
        spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS
    }
    unmasked_maes: dict[str, list[float]] = {
        spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS
    }

    for view_idx, row in enumerate(rows):
        view_png = Path(row["view_png"])
        with Image.open(view_png) as src:
            canvas = letterbox_to_canvas(src.convert("RGB"), INPUT_W, INPUT_H)
        canvas_arr = np.asarray(canvas)
        if view_idx == 0:
            Image.fromarray(canvas_arr).save(out_dir / "00_input_letterboxed.png")
            with Image.open(view_png) as src:
                src.convert("RGB").save(out_dir / "00_input_raw.png")

        view_tensor = image_to_tensor(canvas, device)
        with torch.no_grad():
            output = model(view_tensor, return_attention=True)

        for spec in TRAINABLE_EXPORT_SPECS:
            logits = output["files"][spec.file_name][0].sigmoid().clamp(0, 1)
            pred_bmp = (
                logits.cpu().numpy().transpose(1, 2, 0) * 255.0
            ).clip(0, 255).astype(np.uint8)
            target_bmp = crop_target_bmp(atlas_target, spec)
            if pred_bmp.shape != target_bmp.shape:
                raise RuntimeError(
                    f"shape mismatch on {spec.file_name}: pred={pred_bmp.shape} target={target_bmp.shape}"
                )
            mask = support_masks[spec.file_name]
            supported_maes[spec.file_name].append(mae01(target_bmp, pred_bmp, mask))
            unmasked_maes[spec.file_name].append(mae01(target_bmp, pred_bmp))

            if view_idx == 0:
                Image.fromarray(target_bmp).save(out_dir / f"target_{spec.slot}.png")
                Image.fromarray(pred_bmp).save(out_dir / f"predicted_{spec.slot}.png")
                Image.fromarray(diff_visual(target_bmp, pred_bmp)).save(
                    out_dir / f"diff_{spec.slot}.png"
                )
                if spec.file_name in attention_files:
                    attn = output["attention"][spec.file_name][0].float().cpu().numpy()
                    np.save(out_dir / f"attn_raw_{spec.slot}.npy", attn)
                    overlay = attention_overlay(attn, canvas_arr)
                    Image.fromarray(overlay).save(out_dir / f"attn_overlay_{spec.slot}.png")

    # Aggregate per-file MAE across all examined views.
    summary = {
        "skin_id": skin_id,
        "n_views": len(rows),
        "view_pngs": [r["view_png"] for r in rows],
        "atlas_png": rows[0]["atlas_png"],
        "metric_note": (
            "supported_mean matches Gate C training/eval and is canonical. "
            "unmasked_mean is full exported BMP rectangle debug only."
        ),
        "per_file_mae": {
            name: {
                "supported_mean": float(np.mean(supported_maes[name])),
                "supported_values": supported_maes[name],
                "unmasked_mean": float(np.mean(unmasked_maes[name])),
                "unmasked_values": unmasked_maes[name],
                "supported_fraction": float(support_masks[name].mean()),
            }
            for name in supported_maes
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="train/eval CSV containing both skin ids")
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bad-skin", default="tvxq_winamp_skins_by_roseweedy_c379f7bd")
    parser.add_argument("--good-skin", default="minimalistic_black_145917e6")
    parser.add_argument("--n-views", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    csv_path = Path(args.samples)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt_info = detect_checkpoint_info(Path(args.slotnet))
    if ckpt_info.version != 50:
        raise SystemExit(f"need a V5 checkpoint, got version {ckpt_info.version}")
    model = build_model_from_info(ckpt_info, device=device)
    load_checkpoint(model, Path(args.slotnet))
    model.eval()

    attention_files = {"MAIN.bmp", "EQMAIN.bmp", "PLEDIT.bmp", "VOLUME.bmp", "BALANCE.bmp"}

    print(f"bad skin:  {args.bad_skin}")
    bad = run_one_skin(
        model=model,
        device=device,
        csv_path=csv_path,
        skin_id=args.bad_skin,
        out_dir=out_dir / "bad",
        n_views=args.n_views,
        attention_files=attention_files,
    )
    print(f"good skin: {args.good_skin}")
    good = run_one_skin(
        model=model,
        device=device,
        csv_path=csv_path,
        skin_id=args.good_skin,
        out_dir=out_dir / "good",
        n_views=args.n_views,
        attention_files=attention_files,
    )

    combined = {"bad": bad, "good": good, "checkpoint": str(args.slotnet)}
    (out_dir / "summary.json").write_text(json.dumps(combined, indent=2))

    print("per-file supported-pixel mean MAE (bad vs good):")
    print(f"  {'file':10s}  {'bad':>10s}  {'good':>10s}  {'ratio':>8s}  {'support':>8s}")
    for spec in TRAINABLE_EXPORT_SPECS:
        b = bad["per_file_mae"][spec.file_name]["supported_mean"]
        g = good["per_file_mae"][spec.file_name]["supported_mean"]
        ratio = b / g if g > 1e-6 else float("inf")
        support = bad["per_file_mae"][spec.file_name]["supported_fraction"]
        print(f"  {spec.file_name:10s}  {b:>10.6f}  {g:>10.6f}  {ratio:>8.2f}  {support:>7.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
