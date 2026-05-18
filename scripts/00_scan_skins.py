#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.skins import discover_skin_sources, load_skin_assets, stable_skin_id


REQUIRED_FOR_PACK = {"main.bmp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skins-raw", default="skins_raw")
    parser.add_argument("--out", default="data_v35/skin_scan.csv")
    args = parser.parse_args()

    sources = discover_skin_sources(args.skins_raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in sources:
        try:
            assets = load_skin_assets(source)
            missing = sorted(REQUIRED_FOR_PACK - set(assets))
            status = "ok" if not missing else "reject"
            error = ""
        except Exception as exc:  # pragma: no cover - exercised by bad user files
            assets = {}
            missing = sorted(REQUIRED_FOR_PACK)
            status = "error"
            error = str(exc)
        rows.append(
            {
                "skin_id": stable_skin_id(source),
                "source_path": str(source),
                "source_type": "wsz" if source.suffix.lower() == ".wsz" else "directory",
                "status": status,
                "missing_required": json.dumps(missing),
                "present_files": json.dumps(sorted(assets)),
                "error": error,
            }
        )

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "skin_id",
                "source_path",
                "source_type",
                "status",
                "missing_required",
                "present_files",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"scanned {len(rows)} skin source(s); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
