#!/usr/bin/env python3
"""Split an existing sample CSV by variant_id within each skin."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


FIELDNAMES = ["skin_id", "variant_id", "view_png", "atlas_png", "meta_json"]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-before-variant", type=int, default=24)
    args = parser.parse_args()

    rows = read_rows(args.samples)
    train_rows = []
    val_rows = []
    for row in rows:
        try:
            variant = int(row["variant_id"])
        except ValueError as exc:
            raise SystemExit(f"variant_id must be numeric: {row['variant_id']}") from exc
        if variant < args.train_before_variant:
            train_rows.append(row)
        else:
            val_rows.append(row)

    out = Path(args.out_dir)
    write_rows(out / "train.csv", train_rows)
    write_rows(out / "val.csv", val_rows)

    train_skins = Counter(row["skin_id"] for row in train_rows)
    val_skins = Counter(row["skin_id"] for row in val_rows)
    print(
        f"wrote {len(train_rows)} train row(s) / {len(val_rows)} val row(s) "
        f"for {len(train_skins)} train skin(s) / {len(val_skins)} val skin(s)"
    )
    missing_val = sorted(set(train_skins) - set(val_skins))
    if missing_val:
        print("skins without validation variants:", ", ".join(missing_val))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
