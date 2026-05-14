#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def split_for_skin(skin_id: str, val_pct: int, test_pct: int) -> str:
    bucket = int(hashlib.sha1(skin_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-skins", default="data_v0/valid_skins.csv")
    parser.add_argument("--out", default="data_v0/splits.csv")
    parser.add_argument("--val-pct", type=int, default=10)
    parser.add_argument("--test-pct", type=int, default=10)
    args = parser.parse_args()

    with Path(args.valid_skins).open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["skin_id", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "skin_id": row["skin_id"],
                    "split": split_for_skin(row["skin_id"], args.val_pct, args.test_pct),
                }
            )

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

