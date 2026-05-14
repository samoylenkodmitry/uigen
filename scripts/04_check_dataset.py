#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.profiles import load_atlas_profile, load_json


def check_state_regions(atlas_profile_path: str, state_regions_path: str) -> None:
    profile = load_atlas_profile(atlas_profile_path)
    slots = profile.slots_by_name
    regions = load_json(state_regions_path)
    for slot_name, groups in regions.items():
        slot = slots[slot_name]
        for group_name, rects in groups.items():
            for rect in rects:
                x0, y0, x1, y1 = rect
                if not (0 <= x0 < x1 <= slot.w and 0 <= y0 < y1 <= slot.h):
                    raise ValueError(f"{slot_name}.{group_name} rectangle outside slot: {rect}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-skins", default="data_v0/valid_skins.csv")
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--state-regions", default="configs/state_regions_v1.json")
    args = parser.parse_args()

    profile = load_atlas_profile(args.atlas_profile)
    check_state_regions(args.atlas_profile, args.state_regions)

    with Path(args.valid_skins).open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]

    for row in rows:
        for key in ["atlas_path", "mask_path", "slot_weight_path", "meta_path"]:
            if not Path(row[key]).exists():
                raise FileNotFoundError(row[key])
        with Image.open(row["atlas_path"]) as atlas:
            if atlas.size != (profile.canvas_w, profile.canvas_h):
                raise ValueError(f"bad atlas size for {row['skin_id']}: {atlas.size}")
        with Image.open(row["mask_path"]) as mask:
            if mask.mode != "L" or mask.size != (profile.canvas_w, profile.canvas_h):
                raise ValueError(f"bad mask for {row['skin_id']}: mode={mask.mode} size={mask.size}")
        weights = np.fromfile(row["slot_weight_path"], dtype="<f4")
        if weights.shape != (len(profile.slots),):
            raise ValueError(f"bad slot weights for {row['skin_id']}: {weights.shape}")

    print(f"checked {len(rows)} packed skin(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

