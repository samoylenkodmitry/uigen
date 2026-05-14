#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.export import export_atlas_to_skin
from atlas_ai.profiles import load_atlas_profile, load_export_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--export-profile", default="configs/export_profile_classic.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    zip_path = export_atlas_to_skin(
        atlas_path=args.atlas,
        atlas_profile=load_atlas_profile(args.atlas_profile),
        export_profile=load_export_profile(args.export_profile),
        default_skin=args.default_skin,
        out_dir=args.out,
    )
    print(f"wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

