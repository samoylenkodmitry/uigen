#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.profiles import load_atlas_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--valid-skins", default="data_v34/valid_skins.csv")
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    args = parser.parse_args()

    profile = load_atlas_profile(args.atlas_profile)

    if args.data:
        sample_rows = []
        for split in ["train", "val", "test"]:
            path = Path(args.data) / f"{split}.csv"
            if path.exists():
                with path.open("r", newline="", encoding="utf-8") as f:
                    sample_rows.extend(csv.DictReader(f))
        for row in sample_rows:
            check_sample_row(row, profile)
        print(f"checked {len(sample_rows)} dataset sample(s)")
        return 0

    with Path(args.valid_skins).open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]

    for row in rows:
        for key in ["atlas_path", "meta_path"]:
            if not Path(row[key]).exists():
                raise FileNotFoundError(row[key])
        with Image.open(row["atlas_path"]) as atlas:
            if atlas.size != (profile.canvas_w, profile.canvas_h):
                raise ValueError(f"bad atlas size for {row['skin_id']}: {atlas.size}")

    print(f"checked {len(rows)} packed skin(s)")
    return 0


def check_sample_row(row: dict[str, str], profile) -> None:
    for key in [
        "view_png",
        "atlas_png",
        "meta_json",
    ]:
        if not Path(row[key]).exists():
            raise FileNotFoundError(row[key])
    with Image.open(row["view_png"]) as view:
        if view.mode != "RGB" or view.size != (960, 1728):
            raise ValueError(f"bad view for {row['skin_id']}: mode={view.mode} size={view.size}")
    with Image.open(row["atlas_png"]) as atlas:
        if atlas.mode != "RGB" or atlas.size != (profile.canvas_w, profile.canvas_h):
            raise ValueError(f"bad atlas for {row['skin_id']}: mode={atlas.mode} size={atlas.size}")

if __name__ == "__main__":
    raise SystemExit(main())
