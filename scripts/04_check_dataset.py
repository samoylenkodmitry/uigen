#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

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
    parser.add_argument("--data", default=None)
    parser.add_argument("--valid-skins", default="data_v0/valid_skins.csv")
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--state-regions", default="configs/state_regions_v1.json")
    parser.add_argument("--debug-out", default=None)
    args = parser.parse_args()

    profile = load_atlas_profile(args.atlas_profile)
    check_state_regions(args.atlas_profile, args.state_regions)

    if args.data:
        sample_rows = []
        for split in ["train", "val", "test"]:
            path = Path(args.data) / f"{split}.csv"
            if path.exists():
                with path.open("r", newline="", encoding="utf-8") as f:
                    sample_rows.extend(csv.DictReader(f))
        for row in sample_rows:
            check_sample_row(row, profile)
        if args.debug_out and sample_rows:
            write_contact_sheet(sample_rows, Path(args.debug_out))
        print(f"checked {len(sample_rows)} dataset sample(s)")
        return 0

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


def check_sample_row(row: dict[str, str], profile) -> None:
    for key in [
        "view_png",
        "rects_f32",
        "state_f32",
        "visible_mask_png",
        "atlas_png",
        "atlas_mask_png",
        "slot_weight_f32",
        "meta_json",
        "params_json",
    ]:
        if not Path(row[key]).exists():
            raise FileNotFoundError(row[key])
    with Image.open(row["view_png"]) as view:
        if view.mode != "RGB" or view.size != (960, 1728):
            raise ValueError(f"bad view for {row['skin_id']}: mode={view.mode} size={view.size}")
    with Image.open(row["visible_mask_png"]) as mask:
        if mask.mode != "L" or mask.size != (profile.canvas_w, profile.canvas_h):
            raise ValueError(f"bad visible mask for {row['skin_id']}: mode={mask.mode} size={mask.size}")
    for key, mode, size in [
        ("atlas_png", "RGB", (profile.canvas_w, profile.canvas_h)),
        ("atlas_mask_png", "L", (profile.canvas_w, profile.canvas_h)),
    ]:
        with Image.open(row[key]) as image:
            if image.mode != mode or image.size != size:
                raise ValueError(f"bad {key} for {row['skin_id']}: mode={image.mode} size={image.size}")
    rects = np.fromfile(row["rects_f32"], dtype="<f4")
    if rects.shape != (80 * 5,):
        raise ValueError(f"bad rect shape for {row['skin_id']}: {rects.shape}")
    rects = rects.reshape(80, 5)
    if not np.all((rects[:, :4] >= 0.0) & (rects[:, :4] <= 1.0)):
        raise ValueError(f"rect coordinates outside [0,1] for {row['skin_id']}")
    if not np.all(np.isin(rects[:, 4], [0.0, 1.0])):
        raise ValueError(f"rect visibility flags must be 0/1 for {row['skin_id']}")
    states = np.fromfile(row["state_f32"], dtype="<f4")
    if states.shape != (32,):
        raise ValueError(f"bad state shape for {row['skin_id']}: {states.shape}")
    weights = np.fromfile(row["slot_weight_f32"], dtype="<f4")
    if weights.shape != (len(profile.slots),):
        raise ValueError(f"bad slot weights for {row['skin_id']}: {weights.shape}")


def write_contact_sheet(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    selected = rows if len(rows) <= 32 else rng.sample(rows, 32)
    thumbs = []
    for row in selected:
        with Image.open(row["view_png"]) as view:
            thumb = view.resize((192, 320))
        rects = np.fromfile(row["rects_f32"], dtype="<f4").reshape(80, 5)
        draw = ImageDraw.Draw(thumb)
        for x0, y0, x1, y1, visible in rects:
            if visible:
                draw.rectangle([x0 * 192, y0 * 320, x1 * 192, y1 * 320], outline=(255, 64, 64), width=1)
        thumbs.append(thumb)
    cols = min(4, len(thumbs))
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 192, rows_n * 320), (0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 192, (idx // cols) * 320))
    sheet.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
