#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.dataset import read_sample_csv
from atlas_ai.export import export_atlas_to_skin
from atlas_ai.profiles import load_atlas_profile, load_export_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data_v34/test.csv")
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    parser.add_argument("--export-profile", default="configs/export_profile_classic.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--out", default="eval/smoke")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    rows = read_sample_csv(args.samples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atlas_profile = load_atlas_profile(args.atlas_profile)
    export_profile = load_export_profile(args.export_profile)
    metrics = []
    for idx, row in enumerate(rows[: args.limit]):
        sample_out = out / f"{row.skin_id}_{row.variant_id}"
        zip_path = export_atlas_to_skin(row.atlas_png, atlas_profile, export_profile, args.default_skin, sample_out)
        shutil.copy2(row.view_png, sample_out / "target_view.png")
        metrics.append({"skin_id": row.skin_id, "variant_id": row.variant_id, "exported_wsz": str(zip_path), "export_ok": zip_path.exists()})
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["skin_id", "variant_id", "exported_wsz", "export_ok"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"evaluated/exported {len(metrics)} sample(s); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
