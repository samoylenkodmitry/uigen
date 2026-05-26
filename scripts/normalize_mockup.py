#!/usr/bin/env python3
"""Normalize a generated Winamp mockup and record V8 window layout.

Manual rect overrides are JSON:

    {"rects": {"main": [x, y, w, h], "eq": [...], "playlist": [...]}}

By default rects are interpreted in the original input image space and are
scaled through the letterbox transform. Use --rects-space normalized when the
JSON already references the normalized canvas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.v8_layout import (
    NORMALIZED_SIZE,
    draw_layout_overlay,
    load_rect_override,
    make_layout,
    normalize_mockup_image,
    save_layout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Generated mockup image.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--width", type=int, default=NORMALIZED_SIZE[0])
    parser.add_argument("--height", type=int, default=NORMALIZED_SIZE[1])
    parser.add_argument("--rects-json", default=None, help="Manual main/eq/playlist rect override.")
    parser.add_argument(
        "--rects-space",
        choices=["original", "normalized"],
        default="original",
        help="Coordinate space for --rects-json.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
    print(f"wrote {out / 'normalized.png'}")
    print(f"wrote {out / 'layout.json'}")
    print(f"wrote {out / 'debug_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
