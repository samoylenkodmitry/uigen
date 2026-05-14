#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def count_magenta(path: Path) -> int:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    raw = rgb.tobytes()
    return sum(1 for idx in range(0, len(raw), 3) if raw[idx : idx + 3] == b"\xff\x00\xff")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin-dir", default="assets/default_skin")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    counts = {
        file.name: count_magenta(file)
        for file in sorted(Path(args.skin_dir).glob("*.bmp"), key=lambda p: p.name.lower())
    }
    report = {
        "skin_dir": args.skin_dir,
        "magenta_pixel_counts": counts,
        "cranamp_decode_behavior": "src/winamp/skin.rs keys exact RGB #FF00FF to transparent alpha for decoded BMPs",
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
