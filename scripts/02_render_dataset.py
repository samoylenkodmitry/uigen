#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import subprocess


def stable_seed(skin_id: str, variant_id: int) -> int:
    base = int(hashlib.sha256(skin_id.encode("utf-8")).hexdigest()[:12], 16)
    return base * 1_000_003 + variant_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-skins", default="data_v0/valid_skins.csv")
    parser.add_argument("--cranamp-cli", default="cranamp_cli/cranamp-cli")
    parser.add_argument("--out", default="data_v0")
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--canvas-w", type=int, default=941)
    parser.add_argument("--canvas-h", type=int, default=1672)
    parser.add_argument("--state-balanced", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    for dirname in ["views", "rects", "states", "visible_masks", "params"]:
        (out / dirname).mkdir(parents=True, exist_ok=True)

    with Path(args.valid_skins).open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]

    for row in rows:
        skin_id = row["skin_id"]
        for variant_id in range(args.variants):
            sample_id = f"{skin_id}_{variant_id:04d}"
            seed = stable_seed(skin_id, variant_id)
            cmd = [
                args.cranamp_cli,
                "render-random",
                "--skin-dir",
                row["source_path"],
                "--seed",
                str(seed),
                "--canvas-w",
                str(args.canvas_w),
                "--canvas-h",
                str(args.canvas_h),
                "--out-view",
                str(out / "views" / f"{sample_id}.png"),
                "--out-rects",
                str(out / "rects" / f"{sample_id}.f32"),
                "--out-state",
                str(out / "states" / f"{sample_id}.f32"),
                "--out-visible-atlas-mask",
                str(out / "visible_masks" / f"{sample_id}.png"),
                "--out-params",
                str(out / "params" / f"{sample_id}.json"),
                "--state-balanced",
                "true" if args.state_balanced else "false",
            ]
            subprocess.run(cmd, check=True)

    print(f"rendered {len(rows) * args.variants} sample(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
