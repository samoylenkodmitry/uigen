#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.atlas import pack_skin_assets, save_packed_skin
from atlas_ai.profiles import assert_slots_fit, load_atlas_profile
from atlas_ai.skins import discover_skin_sources, load_default_assets, load_skin_assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skins-raw", default="skins_raw")
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--export-profile", default="configs/export_profile_classic.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--out", default="data_v0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    atlas_profile = load_atlas_profile(args.atlas_profile)
    assert_slots_fit(atlas_profile)
    default_assets = load_default_assets(args.default_skin)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    sources = discover_skin_sources(args.skins_raw)
    if args.limit is not None:
        sources = sources[: args.limit]

    for source in sources:
        try:
            assets = load_skin_assets(source)
            packed = pack_skin_assets(source, assets, default_assets, atlas_profile)
            if packed.rejected_reason:
                rows.append(
                    {
                        "skin_id": packed.skin_id,
                        "source_path": str(source),
                        "status": "reject",
                        "reason": packed.rejected_reason,
                        "atlas_path": "",
                        "mask_path": "",
                        "slot_weight_path": "",
                        "meta_path": "",
                    }
                )
                continue
            paths = save_packed_skin(packed, out)
            rows.append(
                {
                    "skin_id": packed.skin_id,
                    "source_path": str(source),
                    "status": "ok",
                    "reason": "",
                    **paths,
                }
            )
        except Exception as exc:  # pragma: no cover - protects long corpus runs
            rows.append(
                {
                    "skin_id": source.stem,
                    "source_path": str(source),
                    "status": "error",
                    "reason": str(exc),
                    "atlas_path": "",
                    "mask_path": "",
                    "slot_weight_path": "",
                    "meta_path": "",
                }
            )

    valid_csv = out / "valid_skins.csv"
    with valid_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "skin_id",
                "source_path",
                "status",
                "reason",
                "atlas_path",
                "mask_path",
                "slot_weight_path",
                "meta_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"packed {ok_count}/{len(rows)} skin source(s); wrote {valid_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

