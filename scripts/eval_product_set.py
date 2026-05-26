#!/usr/bin/env python3
"""Run V8 mockup->skin baseline on a fixed product-eval image set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.hidden_state_compiler import compile_hidden_states
from atlas_ai.torch_cranamp_renderer import render_visible
from atlas_ai.v8_assets import image_to_tensor, save_exported_tensors, tensor_to_image
from atlas_ai.v8_layout import default_layout, draw_layout_overlay, normalize_mockup_image, save_layout
from atlas_ai.visible_extractor import extract_visible_assets


def _sobel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    out = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (8, 8, 10))
    out.paste(a.convert("RGB"), (0, 0))
    out.paste(b.convert("RGB"), (a.width, 0))
    return out


def _iter_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mockups", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    mockups = _iter_images(Path(args.mockups))
    if args.limit is not None:
        mockups = mockups[: args.limit]
    if not mockups:
        raise SystemExit(f"no mockup images under {args.mockups}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in mockups:
        case = out / path.stem
        case.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            normalized, _scale, _offset = normalize_mockup_image(source)
        layout = default_layout(*normalized.size)
        normalized.save(case / "normalized.png")
        draw_layout_overlay(normalized, layout).save(case / "debug_overlay.png")
        save_layout(layout, case / "layout.json")
        visible = extract_visible_assets(normalized, layout, default_skin=args.default_skin)
        compiled = compile_hidden_states(visible, default_skin=args.default_skin)
        zip_path = save_exported_tensors(compiled, case / "skin", default_skin=args.default_skin, package=True)
        rendered_t = render_visible(compiled, layout)
        rendered = tensor_to_image(rendered_t)
        rendered.save(case / "render_preview.png")
        _side_by_side(normalized, rendered).save(case / "side_by_side.png")
        target = image_to_tensor(normalized)
        rgb_mae = float((rendered_t - target).abs().mean())
        edge_mae = float((_sobel(rendered_t) - _sobel(target)).abs().mean())
        row = {
            "mockup": str(path),
            "case_dir": str(case),
            "skin_wsz": str(zip_path),
            "load_success": bool(zip_path.exists()),
            "render_rgb_mae": rgb_mae,
            "render_sobel_mae": edge_mae,
            "human_similarity_1_5": "",
            "human_sharpness_1_5": "",
            "notes": "",
        }
        rows.append(row)
        print(f"{path.name}: rgb_mae={rgb_mae:.4f} edge_mae={edge_mae:.4f}")

    fields = list(rows[0].keys())
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
