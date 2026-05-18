#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re


def split_for_skin(skin_id: str, val_pct: float, test_pct: float) -> str:
    bucket = int(hashlib.sha1(skin_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < test_pct * 100:
        return "test"
    if bucket < (test_pct + val_pct) * 100:
        return "val"
    return "train"


def collect_samples(data: Path, valid_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows_by_skin = {row["skin_id"]: row for row in valid_rows}
    samples = []
    for view in sorted((data / "views").glob("*.png"), key=lambda p: p.name):
        match = re.match(r"(.+)_(\d{4})\.png$", view.name)
        if not match:
            continue
        skin_id, variant_id = match.groups()
        if skin_id not in rows_by_skin:
            continue
        packed = rows_by_skin[skin_id]
        sample = {
            "skin_id": skin_id,
            "variant_id": variant_id,
            "view_png": str(view),
            "atlas_png": packed["atlas_path"],
            "meta_json": packed["meta_path"],
        }
        samples.append(sample)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_v35")
    parser.add_argument("--valid-skins", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--split-by", default="skin_id", choices=["skin_id"])
    args = parser.parse_args()

    valid_skins = Path(args.valid_skins) if args.valid_skins else Path(args.data) / "valid_skins.csv"
    with valid_skins.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]

    if round(args.train + args.val + args.test, 6) != 1.0:
        raise SystemExit("--train + --val + --test must equal 1.0")

    data = Path(args.data)
    samples = collect_samples(data, rows)
    fieldnames = [
        "skin_id",
        "variant_id",
        "view_png",
        "atlas_png",
        "meta_json",
    ]
    split_by_skin = {}
    skin_ids = sorted({sample["skin_id"] for sample in samples})
    for skin_id in skin_ids:
        split_by_skin[skin_id] = "train" if len(skin_ids) < 3 else split_for_skin(skin_id, args.val, args.test)

    for split in ["train", "val", "test"]:
        out = data / f"{split}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([sample for sample in samples if split_by_skin[sample["skin_id"]] == split])

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["skin_id", "split"])
            writer.writeheader()
            for skin_id, split in sorted(split_by_skin.items()):
                writer.writerow({"skin_id": skin_id, "split": split})

    print(f"wrote splits for {len(samples)} sample(s) in {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
