#!/usr/bin/env python3
"""Build the static Cranamp-supported-pixel profile.

Renders many variants of a reference skin while recording which source
rectangle every Cranamp blit reads from each supported BMP. The union of
those rectangles defines the set of pixels Cranamp can actually display,
so it is the only set the loss and metrics should care about.

The profile is deterministic: source rectangles in Cranamp are skin-agnostic
constants. The render loop only exists to exercise state-dependent code
paths (paused/playing, eq on/off, etc.).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from collections import defaultdict
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS


def load_cranamp_module():
    cli_module_path = REPO / "cranamp_cli/cranamp/tools/cranamp_cli.py"
    spec = importlib.util.spec_from_file_location("cranamp_cli_tool", cli_module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin", default="skins_raw/BlueCurve_Winamp.wsz",
                        help="Reference skin used to drive the renderer. "
                             "Source rectangles are skin-independent, so this only varies state.")
    parser.add_argument("--variants", type=int, default=128)
    parser.add_argument("--canvas-w", type=int, default=960)
    parser.add_argument("--canvas-h", type=int, default=1728)
    parser.add_argument("--out", default="configs/supported_pixels_classic.json")
    args = parser.parse_args()

    cranamp = load_cranamp_module()
    Renderer = cranamp.Renderer

    recorded: dict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
    original_blit = Renderer.blit

    def recording_blit(self, slot_name, file_name, src, dest, scale, component_id=None):
        sx, sy, sw, sh = (int(v) for v in src)
        recorded[file_name].add((sx, sy, sw, sh))
        return original_blit(self, slot_name, file_name, src, dest, scale, component_id)

    Renderer.blit = recording_blit
    try:
        skin_path = Path(args.skin)
        if not skin_path.exists():
            raise FileNotFoundError(args.skin)
        for variant in range(args.variants):
            seed = 1_000_003 * (variant + 1)
            params = cranamp.rand_params(seed=seed, canvas_w=args.canvas_w, canvas_h=args.canvas_h)
            cranamp.render_with_params(skin_path, params, canvas_w=args.canvas_w, canvas_h=args.canvas_h)
    finally:
        Renderer.blit = original_blit

    out: dict[str, list[list[int]]] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        rects = sorted(recorded.get(spec.file_name, set()))
        # Clip to the exported BMP dimensions so loss masks line up.
        clipped: list[list[int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for sx, sy, sw, sh in rects:
            x0 = max(0, sx)
            y0 = max(0, sy)
            x1 = min(spec.w, sx + sw)
            y1 = min(spec.h, sy + sh)
            if x0 >= x1 or y0 >= y1:
                continue
            key = (x0, y0, x1 - x0, y1 - y0)
            if key in seen:
                continue
            seen.add(key)
            clipped.append([x0, y0, x1 - x0, y1 - y0])
        out[spec.file_name] = clipped

    # Compute coverage stats for the report.
    stats = []
    for spec in TRAINABLE_EXPORT_SPECS:
        covered = _support_pixel_count(out[spec.file_name], spec.h, spec.w)
        total = spec.h * spec.w
        stats.append((spec.file_name, covered, total, covered / total))

    print(f"recorded {sum(len(v) for v in out.values())} unique rect(s) across "
          f"{len(out)} file(s) from {args.variants} variant(s)")
    print(f"{'file':14s} {'covered':>10s} {'total':>10s} {'frac':>6s}")
    for name, covered, total, frac in stats:
        print(f"{name:14s} {covered:10d} {total:10d} {frac:6.2%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


def _support_pixel_count(rects: list[list[int]], h: int, w: int) -> int:
    if not rects:
        return 0
    import numpy as np
    mask = np.zeros((h, w), dtype=bool)
    for x, y, rw, rh in rects:
        mask[y:y + rh, x:x + rw] = True
    return int(mask.sum())


if __name__ == "__main__":
    raise SystemExit(main())
