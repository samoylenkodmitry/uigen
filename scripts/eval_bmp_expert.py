#!/usr/bin/env python3
"""V10 BMP expert eval: load a checkpoint, score MAE / hit_5_255 / Sobel MAE
over all variants of the per-BMP dataset, and emit a predicted-vs-target grid.

Usage:

    python scripts/eval_bmp_expert.py \
        --data data_v10_gate1 --bmp MAIN.bmp \
        --checkpoint runs/v10/MAIN/last.safetensors --out runs/v10/MAIN/eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v10_bmp import BMPExpertDataset
from atlas_ai.v8_assets import tensor_to_image
from models.bmp_expert_net import BMPExpertNet


def _sobel(x: torch.Tensor) -> torch.Tensor:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _build_expert_from_state(state: dict) -> BMPExpertNet:
    g = lambda k: int(state[k].reshape(-1)[0].item())
    return BMPExpertNet(
        target_h=g("target_h_buf"), target_w=g("target_w_buf"),
        base=g("base_buf"), attn_dim=g("attn_dim_buf"),
        dec_ch=g("dec_ch_buf"), heads=g("heads_buf"),
        attn_layers=g("attn_layers_buf"),
    )


def _grid(pairs: list[tuple[Image.Image, Image.Image, str]], cell_h: int = 160) -> Image.Image:
    """Stack rows of (target | prediction) labeled by variant id."""
    tiles = []
    for tgt, pred, label in pairs:
        a = tgt.convert("RGB"); b = pred.convert("RGB")
        scale = cell_h / max(a.height, 1)
        sw = max(8, round(a.width * scale))
        a = a.resize((sw, cell_h), Image.NEAREST)
        b = b.resize((sw, cell_h), Image.NEAREST)
        tiles.append((a, b, label))
    if not tiles:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    pair_w = tiles[0][0].width + tiles[0][1].width + 4
    row_h = cell_h + 18
    out = Image.new("RGB", (pair_w, row_h * len(tiles)), (18, 18, 22))
    d = ImageDraw.Draw(out)
    for i, (a, b, label) in enumerate(tiles):
        y = i * row_h
        d.text((4, y + 2), f"{label}   (target | prediction)", fill=(235, 235, 235))
        out.paste(a, (0, y + 18))
        out.paste(b, (a.width + 4, y + 18))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--bmp", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grid-samples", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = BMPExpertDataset(args.data, args.bmp)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    from safetensors.torch import load_file
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        last = ckpt.with_name("last.safetensors")
        if last.exists():
            print(f"WARNING: {ckpt} missing -> falling back to {last}", flush=True)
            ckpt = last
        else:
            raise SystemExit(f"checkpoint not found: {args.checkpoint} (and no last.safetensors next to it)")
    state = load_file(str(ckpt))
    model = _build_expert_from_state(state).to(device).eval()
    model.load_state_dict(state)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_variant: list[dict] = []
    tot_mae = tot_hit = tot_sob = 0.0
    n = 0
    grid_pairs: list[tuple[Image.Image, Image.Image, str]] = []
    grid_idxs = set(int(round(i * (len(ds) - 1) / max(1, args.grid_samples - 1)))
                    for i in range(min(args.grid_samples, len(ds))))
    row_i = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["render"].to(device)
            y = batch["target"].to(device)
            p = model(x).clamp(0.0, 1.0)
            mae = (p - y).abs().mean(dim=(1, 2, 3)).cpu()
            hit = ((p - y).abs() * 255.0 <= 5.0).all(dim=1).float().mean(dim=(1, 2)).cpu()
            sob = (_sobel(p) - _sobel(y)).abs().mean(dim=(1, 2, 3)).cpu()
            for b in range(x.shape[0]):
                v_id = batch["variant_id"][b]
                row = {"variant_id": v_id, "mae": float(mae[b]),
                       "hit_5_255": float(hit[b]), "sobel_mae": float(sob[b])}
                per_variant.append(row)
                tot_mae += row["mae"]; tot_hit += row["hit_5_255"]; tot_sob += row["sobel_mae"]
                n += 1
                if row_i in grid_idxs:
                    grid_pairs.append((tensor_to_image(y[b].cpu()),
                                       tensor_to_image(p[b].cpu()),
                                       f"v{v_id}  mae={row['mae']:.4f}  hit5={row['hit_5_255']:.3f}"))
                row_i += 1

    agg = {"bmp": args.bmp, "n": n,
           "mae_mean": tot_mae / max(1, n),
           "hit_5_255_mean": tot_hit / max(1, n),
           "sobel_mae_mean": tot_sob / max(1, n)}
    spec_h, spec_w = model.target_h, model.target_w
    agg["gate1_pass"] = bool(agg["mae_mean"] < 0.01 and agg["hit_5_255_mean"] > 0.90)
    agg["target_hw"] = [spec_h, spec_w]
    (out / "metrics.json").write_text(json.dumps(agg, indent=2))
    with (out / "per_variant.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant_id", "mae", "hit_5_255", "sobel_mae"])
        w.writeheader(); w.writerows(per_variant)
    if grid_pairs:
        _grid(grid_pairs).save(out / "pred_vs_target_grid.png")

    print(f"eval {args.bmp}: n={n} mae={agg['mae_mean']:.4f} hit_5_255={agg['hit_5_255_mean']:.3f} "
          f"sobel={agg['sobel_mae_mean']:.4f}  gate1_pass={agg['gate1_pass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
