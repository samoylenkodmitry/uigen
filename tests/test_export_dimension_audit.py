from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts/10_audit_export_dimensions.py"
SPEC = importlib.util.spec_from_file_location("audit_export_dimensions", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def test_export_audit_reports_main_source_crop_mismatch(tmp_path: Path):
    skin = tmp_path / "skin"
    skin.mkdir()
    pixels = np.zeros((116, 275, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(275, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = np.arange(116, dtype=np.uint8)[:, None]
    Image.fromarray(pixels, "RGB").save(skin / "MAIN.bmp")

    report = audit_module.audit_skin_export(
        skin,
        atlas_profile_path=ROOT / "configs/atlas_train_v1.json",
        export_profile_path=ROOT / "configs/export_profile_classic.json",
        default_skin=ROOT / "assets/default_skin",
    )
    rows = {row["file_name"]: row for row in report["files"]}
    main = rows["MAIN.bmp"]

    assert main["source_size"] == [275, 116]
    assert main["export_size"] == [275, 115]
    assert main["expected_size"] == [275, 115]
    assert main["export_dimension_match"] is True
    assert main["source_size_matches_export"] is False
    assert main["crop_exact_match"] is True
    assert main["crop_mae"] == 0.0
    assert report["source_size_mismatches"] == ["MAIN.bmp"]
