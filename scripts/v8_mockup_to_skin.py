#!/usr/bin/env python3
"""V8 deterministic baseline: mockup image -> valid skin.wsz.

This is not the final neural converter. It is the product-loop baseline that
normalizes a mockup, extracts visible/default exported BMPs, compiles plausible
hidden states, renders a preview, and packages a loadable classic skin.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.hidden_state_compiler import compile_hidden_states
from atlas_ai.torch_cranamp_renderer import render_visible
from atlas_ai.v8_assets import save_exported_tensors, tensor_to_image
from atlas_ai.v8_layout import (
    NORMALIZED_SIZE,
    draw_layout_overlay,
    load_layout,
    load_rect_override,
    make_layout,
    normalize_mockup_image,
    save_layout,
)
from atlas_ai.visible_extractor import extract_visible_assets


def _side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    w = left.width + right.width
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h), (8, 8, 10))
    out.paste(left.convert("RGB"), (0, 0))
    out.paste(right.convert("RGB"), (left.width, 0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layout-json", default=None, help="Use an existing V8 layout.")
    parser.add_argument("--rects-json", default=None, help="Manual rect override when normalizing.")
    parser.add_argument("--rects-space", choices=["original", "normalized"], default="original")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--width", type=int, default=NORMALIZED_SIZE[0])
    parser.add_argument("--height", type=int, default=NORMALIZED_SIZE[1])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.layout_json:
        layout = load_layout(args.layout_json)
        with Image.open(args.input) as source:
            normalized = source.convert("RGB")
        if normalized.size != tuple(layout["normalized_size"]):
            normalized = normalized.resize(tuple(layout["normalized_size"]), Image.Resampling.LANCZOS)
    else:
        with Image.open(args.input) as source:
            normalized, scale, offset = normalize_mockup_image(
                source, size=(args.width, args.height)
            )
        rects = load_rect_override(args.rects_json) if args.rects_json else None
        layout = make_layout(
            normalized_size=(args.width, args.height),
            rect_override=rects,
            override_space=args.rects_space,
            letterbox_scale=scale,
            letterbox_offset=offset,
        )
        layout["input_path"] = str(args.input)
        layout["letterbox"] = {"scale": scale, "offset": [offset[0], offset[1]]}

    normalized.save(out / "normalized.png")
    draw_layout_overlay(normalized, layout).save(out / "debug_overlay.png")
    save_layout(layout, out / "layout.json")

    visible = extract_visible_assets(normalized, layout, default_skin=args.default_skin)
    compiled = compile_hidden_states(visible, default_skin=args.default_skin)
    skin_dir = out / "skin"
    zip_path = save_exported_tensors(compiled, skin_dir, default_skin=args.default_skin, package=True)

    rendered = tensor_to_image(render_visible(compiled, layout))
    rendered.save(out / "render_preview.png")
    _side_by_side(normalized, rendered).save(out / "side_by_side.png")
    print(f"wrote {zip_path}")
    print(f"wrote {out / 'render_preview.png'}")
    print(f"wrote {out / 'side_by_side.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
