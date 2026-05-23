#!/usr/bin/env python3
"""Dump target/predicted/diff contact sheets for V7 EQMAIN-only probes.

For one V7 completer checkpoint, runs N forward passes with fresh masks of a
chosen mode and writes a horizontal strip per round:

    [observed_rgb | target | predicted | diff]

The diff is `|pred - target|` boosted x4 and clipped to [0, 1] so subtle errors
are visible. Saves PNGs into the checkpoint's directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.state_families import load_state_families
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_masks import V7MaskWeights, sample_v7_observed_mask
from models.v7_completer import V7Completer


FILE_TO_ID = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def load_state_dict(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _detect_v7_kwargs(state: dict) -> dict:
    return {
        "base_channels": int(state["base_channels_buffer"].reshape(-1)[0].item()),
        "file_embedding_dim": int(state["file_embedding_dim_buffer"].reshape(-1)[0].item()),
    }


def _to_png(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((arr.clip(0, 1) * 255.0).astype(np.uint8), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skin-source", required=True,
                        help="Directory containing the trainable BMPs.")
    parser.add_argument("--file-name", default="EQMAIN.bmp")
    parser.add_argument("--state-families",
                        default="configs/state_families_classic.yaml")
    parser.add_argument("--mode", choices=["whole_file", "state_family", "passthrough"],
                        default="state_family")
    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of fresh mask draws to render.")
    parser.add_argument("--out", required=True,
                        help="Output directory for the PNG strips.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load clean target.
    spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == args.file_name)
    with Image.open(Path(args.skin_source) / args.file_name) as im:
        clean = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
    target = torch.from_numpy(clean.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Load support and state families.
    support = load_support_masks()[args.file_name].cpu().numpy().astype(np.uint8)
    families = load_state_families(args.state_families)
    rects = families.get(args.file_name, [])

    # Build model.
    state = load_state_dict(Path(args.checkpoint))
    model = V7Completer(**_detect_v7_kwargs(state)).to(device).eval()
    model.load_state_dict(state)

    # Build mask weights for chosen mode.
    weight_kwargs = {"provenance": 0.0, "state_family": 0.0,
                     "random_rect": 0.0, "whole_file": 0.0, "passthrough": 0.0}
    weight_kwargs[args.mode] = 1.0
    weights = V7MaskWeights(**weight_kwargs)

    rng = np.random.default_rng(args.seed)
    file_id = torch.tensor([FILE_TO_ID[args.file_name]], dtype=torch.long, device=device)

    for r in range(args.rounds):
        mask_np, mode = sample_v7_observed_mask(
            rng, h=spec.h, w=spec.w,
            family_rects=rects, visible_masks=None, weights=weights,
        )
        observed_mask = (mask_np & support).astype(np.float32)
        observed_t = torch.from_numpy(observed_mask).unsqueeze(0).unsqueeze(0).to(device)
        observed_rgb = target * observed_t
        with torch.no_grad():
            final = model(observed_rgb, observed_t, file_id)
        pred = final[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        obs = observed_rgb[0].cpu().numpy().transpose(1, 2, 0)
        diff = np.abs(pred - clean) * 4.0
        # Stack horizontally: [observed | clean target | predicted | diff x4]
        strip = np.concatenate([obs, clean, pred, diff], axis=1)
        _to_png(strip).save(out_dir / f"{args.file_name.replace('.', '_')}_{mode}_round{r:02d}.png")
        # Aggregate support-masked metrics for this round.
        sup = torch.from_numpy(support.astype(np.float32)).to(device)
        diff_t = (final[0] - target[0]).abs() * sup
        mae = float(diff_t.sum().item() / sup.sum().item() / 3.0)
        print(f"  round {r}: mode={mode} mae={mae:.6f} hidden_pixels={int((1.0 - observed_mask).sum())}")

    print(f"wrote {args.rounds} contact strips to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
