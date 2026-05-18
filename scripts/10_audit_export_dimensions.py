#!/usr/bin/env python3
"""Audit supported BMP export dimensions against a source skin.

This does not train or render distorted inputs. It checks the deterministic path:

    source skin BMPs -> packed atlas -> exported supported BMPs

The report shows whether each exported BMP matches the export profile and whether
the source BMP itself has the same dimensions or is being cropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.atlas import pack_skin_assets, save_packed_skin
from atlas_ai.export import export_atlas_to_skin
from atlas_ai.profiles import load_atlas_profile, load_export_profile
from atlas_ai.skins import canonical_display_name, load_default_assets, load_rgb_image, load_skin_assets, normalize_name


def image_mae(left: Image.Image, right: Image.Image) -> float:
    a = np.asarray(left.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(right.convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ValueError(f"image shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


def exported_padding_mae_to_black(exported: Image.Image, source: Image.Image) -> float | None:
    export_arr = np.asarray(exported.convert("RGB"), dtype=np.float32) / 255.0
    mask = np.zeros(export_arr.shape[:2], dtype=bool)
    mask[source.height:, :] = True
    mask[:, source.width:] = True
    if not mask.any():
        return None
    return float(np.abs(export_arr[mask]).mean())


def audit_skin_export(
    skin: str | Path,
    *,
    atlas_profile_path: str | Path,
    export_profile_path: str | Path,
    default_skin: str | Path,
) -> dict[str, Any]:
    atlas_profile = load_atlas_profile(atlas_profile_path)
    export_profile = load_export_profile(export_profile_path)
    assets = load_skin_assets(skin)
    default_assets = load_default_assets(default_skin)
    packed = pack_skin_assets(skin, assets, default_assets, atlas_profile)
    if packed.rejected_reason:
        raise ValueError(f"cannot pack {skin}: {packed.rejected_reason}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="uigen_export_audit_") as tmp:
        tmp_path = Path(tmp)
        packed_paths = save_packed_skin(packed, tmp_path / "packed")
        export_atlas_to_skin(
            atlas_path=packed_paths["atlas_path"],
            atlas_profile=atlas_profile,
            export_profile=export_profile,
            default_skin=default_skin,
            out_dir=tmp_path / "export",
        )
        for file_name, info in export_profile.items():
            key = normalize_name(file_name)
            asset = assets.get(key) or default_assets.get(key)
            source_status = "source" if key in assets else "default"
            if asset is None:
                rows.append(
                    {
                        "file_name": canonical_display_name(file_name),
                        "source_status": "missing",
                        "expected_size": [int(info["w"]), int(info["h"])],
                        "export_size": None,
                        "source_size": None,
                        "export_dimension_match": False,
                        "source_size_matches_export": False,
                        "crop_comparable": False,
                        "crop_mae": None,
                        "crop_exact_match": False,
                        "overlap_size": None,
                        "overlap_mae": None,
                        "overlap_exact_match": False,
                        "padding_mae_to_black": None,
                    }
                )
                continue

            out_name = canonical_display_name(file_name)
            export_path = tmp_path / "export" / out_name
            with Image.open(export_path) as exported:
                exported_rgb = exported.convert("RGB")
                export_size = [exported_rgb.width, exported_rgb.height]
                expected_size = [int(info["w"]), int(info["h"])]
                source_rgb = load_rgb_image(asset)
                source_size = [source_rgb.width, source_rgb.height]
                crop_comparable = (
                    export_size == expected_size
                    and source_rgb.width >= exported_rgb.width
                    and source_rgb.height >= exported_rgb.height
                )
                crop_mae = None
                crop_exact_match = False
                if crop_comparable:
                    source_crop = source_rgb.crop((0, 0, exported_rgb.width, exported_rgb.height))
                    crop_mae = image_mae(exported_rgb, source_crop)
                    crop_exact_match = crop_mae == 0.0
                overlap_w = min(source_rgb.width, exported_rgb.width)
                overlap_h = min(source_rgb.height, exported_rgb.height)
                overlap_mae = None
                overlap_exact_match = False
                if overlap_w > 0 and overlap_h > 0:
                    overlap_mae = image_mae(
                        exported_rgb.crop((0, 0, overlap_w, overlap_h)),
                        source_rgb.crop((0, 0, overlap_w, overlap_h)),
                    )
                    overlap_exact_match = overlap_mae == 0.0
                padding_mae_to_black = exported_padding_mae_to_black(exported_rgb, source_rgb)
                rows.append(
                    {
                        "file_name": out_name,
                        "source_status": source_status,
                        "expected_size": expected_size,
                        "export_size": export_size,
                        "source_size": source_size,
                        "export_dimension_match": export_size == expected_size,
                        "source_size_matches_export": source_size == export_size,
                        "crop_comparable": crop_comparable,
                        "crop_mae": crop_mae,
                        "crop_exact_match": crop_exact_match,
                        "overlap_size": [overlap_w, overlap_h],
                        "overlap_mae": overlap_mae,
                        "overlap_exact_match": overlap_exact_match,
                        "padding_mae_to_black": padding_mae_to_black,
                    }
                )

    return {
        "skin": str(skin),
        "atlas_profile": str(atlas_profile_path),
        "export_profile": str(export_profile_path),
        "default_skin": str(default_skin),
        "files": rows,
        "all_export_dimensions_match_profile": all(row["export_dimension_match"] for row in rows),
        "all_comparable_crops_exact": all(row["crop_exact_match"] for row in rows if row["crop_comparable"]),
        "all_overlaps_exact": all(row["overlap_exact_match"] for row in rows if row["overlap_mae"] is not None),
        "source_size_mismatches": [
            row["file_name"]
            for row in rows
            if row["source_status"] == "source" and not row["source_size_matches_export"]
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"skin: {report['skin']}")
    print("file        source    source_size  export_size  profile_ok  source_same  overlap_mae  pad_black")
    for row in report["files"]:
        source_size = "n/a" if row["source_size"] is None else f"{row['source_size'][0]}x{row['source_size'][1]}"
        export_size = "n/a" if row["export_size"] is None else f"{row['export_size'][0]}x{row['export_size'][1]}"
        overlap_mae = "n/a" if row["overlap_mae"] is None else f"{row['overlap_mae']:.8f}"
        pad_black = "n/a" if row["padding_mae_to_black"] is None else f"{row['padding_mae_to_black']:.8f}"
        print(
            f"{row['file_name']:<11} "
            f"{row['source_status']:<9} "
            f"{source_size:<12} "
            f"{export_size:<12} "
            f"{str(row['export_dimension_match']):<10} "
            f"{str(row['source_size_matches_export']):<11} "
            f"{overlap_mae:<12} "
            f"{pad_black}"
        )
    if report["source_size_mismatches"]:
        print("source size mismatches:", ", ".join(report["source_size_mismatches"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin", required=True)
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    parser.add_argument("--export-profile", default="configs/export_profile_classic.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--fail-on-source-size-mismatch", action="store_true")
    args = parser.parse_args()

    report = audit_skin_export(
        args.skin,
        atlas_profile_path=args.atlas_profile,
        export_profile_path=args.export_profile,
        default_skin=args.default_skin,
    )
    print_summary(report)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["all_export_dimensions_match_profile"]:
        return 2
    if args.fail_on_source_size_mismatch and report["source_size_mismatches"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
