"""Tests for scripts/21_unpack_atlases_to_skin_dirs.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/21_unpack_atlases_to_skin_dirs.py"


def _make_fake_atlas(
    tmp_path: Path,
    skin_id: str,
    *,
    main_size: tuple[int, int] = (275, 115),       # canonical: (275, 115)
    titlebar_size: tuple[int, int] = (344, 87),    # canonical: (344, 87)
    cbuttons_size: tuple[int, int] = (136, 36),
    shufrep_size: tuple[int, int] = (92, 85),
    monoster_size: tuple[int, int] = (56, 24),
    playpaus_size: tuple[int, int] = (42, 9),
    eqmain_size: tuple[int, int] = (275, 315),
    pledit_size: tuple[int, int] = (280, 186),
    posbar_size: tuple[int, int] = (307, 10),
    volume_size: tuple[int, int] = (68, 433),
    balance_size: tuple[int, int] = (47, 433),
) -> tuple[Path, Path]:
    """Build a minimal data_v4_16skin-style atlas + meta_json for one skin.

    All slot rectangles are placed disjointly inside a 1024x1024 atlas. The
    sizes are caller-controllable so individual tests can simulate
    off-by-1 / smaller / degenerate cases.
    """
    atlases_dir = tmp_path / "atlases"
    atlases_dir.mkdir(parents=True, exist_ok=True)

    # 1024x1024 atlas with a distinctive color per slot for visual confidence.
    atlas = np.zeros((1024, 1024, 3), dtype=np.uint8)

    slot_specs = [
        ("BALANCE", "BALANCE.bmp", balance_size,  (736, 128)),
        ("CBUTTONS","CBUTTONS.bmp", cbuttons_size, (664,   0)),
        ("EQMAIN",  "EQMAIN.bmp",   eqmain_size,   (  0, 128)),
        ("MAIN",    "MAIN.bmp",     main_size,     (  0,   0)),
        ("MONOSTER","MONOSTER.bmp", monoster_size, (952,   0)),
        ("PLAYPAUS","PLAYPAUS.bmp", playpaus_size, (952,  32)),
        ("PLEDIT",  "PLEDIT.bmp",   pledit_size,   (320, 128)),
        ("POSBAR",  "POSBAR.bmp",   posbar_size,   (320, 384)),
        ("SHUFREP", "SHUFREP.bmp",  shufrep_size,  (824,   0)),
        ("TITLEBAR","TITLEBAR.bmp", titlebar_size, (320,   0)),
        ("VOLUME",  "VOLUME.bmp",   volume_size,   (640, 128)),
    ]

    slots: dict = {}
    for color_index, (slot, fname, (w, h), (x, y)) in enumerate(slot_specs):
        # Paint a fixed color so we can verify the content was extracted
        # from the right region.
        atlas[y:y+h, x:x+w, 0] = 30 + (color_index * 17) % 200
        atlas[y:y+h, x:x+w, 1] = 60 + (color_index * 23) % 180
        atlas[y:y+h, x:x+w, 2] = 90 + (color_index * 31) % 160
        slots[slot] = {
            "atlas_rect": [x, y, x + w, y + h],
            "capacity": [w, h],
            "file": fname,
            "pasted_rect": [x, y, x + w, y + h],
            "size": [w, h],
            "source_path": f"/fake/{skin_id}/{fname}",
            "status": "source",
        }

    atlas_path = atlases_dir / f"{skin_id}.png"
    meta_path = atlases_dir / f"{skin_id}.meta.json"
    Image.fromarray(atlas, "RGB").save(atlas_path)
    meta_path.write_text(json.dumps({"skin_id": skin_id, "slots": slots}, indent=2))
    return atlases_dir, meta_path


def _run_unpack(atlases_dir: Path, out_dir: Path, *extras: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--atlases-dir", str(atlases_dir),
         "--out-dir", str(out_dir),
         *extras],
        capture_output=True, text=True,
    )


def test_exact_size_skin_passes_strict(tmp_path):
    atlases_dir, _ = _make_fake_atlas(tmp_path, "exact_skin")
    out = tmp_path / "out"
    result = _run_unpack(atlases_dir, out, "--strict")
    assert result.returncode == 0, result.stderr
    assert (out / "exact_skin").is_dir()
    # All 11 files exist at their canonical dimensions.
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
    for spec in TRAINABLE_EXPORT_SPECS:
        path = out / "exact_skin" / spec.file_name
        assert path.exists(), path
        with Image.open(path) as im:
            assert im.size == (spec.w, spec.h), (spec.file_name, im.size)
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["accepted_count"] == 1
    assert manifest["skipped_count"] == 0


def test_off_by_one_larger_is_cropped(tmp_path):
    # Simulate MAIN with 1px extra row (the off-by-1 seen in real data).
    atlases_dir, _ = _make_fake_atlas(tmp_path, "off_by_one", main_size=(275, 116))
    out = tmp_path / "out"
    result = _run_unpack(atlases_dir, out)  # default: pad+crop on
    assert result.returncode == 0, result.stderr
    with Image.open(out / "off_by_one" / "MAIN.bmp") as im:
        assert im.size == (275, 115)
    manifest = json.loads((out / "_manifest.json").read_text())
    main_slot = next(s for s in manifest["skins"][0]["slots"] if s["file_name"] == "MAIN.bmp")
    assert "cropped" in main_slot["action"]


def test_smaller_content_is_padded(tmp_path):
    # Simulate VOLUME missing its tail slider frames (real-world pattern).
    atlases_dir, _ = _make_fake_atlas(tmp_path, "short_volume", volume_size=(68, 410))
    out = tmp_path / "out"
    result = _run_unpack(atlases_dir, out)
    assert result.returncode == 0, result.stderr
    with Image.open(out / "short_volume" / "VOLUME.bmp") as im:
        assert im.size == (68, 433)
    # The padded region must be all zeros.
    with Image.open(out / "short_volume" / "VOLUME.bmp") as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr[410:, :, :].sum() == 0, "pad region was not zeroed"
    manifest = json.loads((out / "_manifest.json").read_text())
    vol_slot = next(s for s in manifest["skins"][0]["slots"] if s["file_name"] == "VOLUME.bmp")
    assert "padded" in vol_slot["action"]


def test_strict_mode_rejects_off_by_one(tmp_path):
    atlases_dir, _ = _make_fake_atlas(tmp_path, "off_by_one", main_size=(275, 116))
    out = tmp_path / "out"
    result = _run_unpack(atlases_dir, out, "--strict")
    assert result.returncode == 0  # script exits 0 with skipped report
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_count"] == 1
    assert "policy" in manifest["skins"][0]["reason"]


def test_degenerate_slot_skips_whole_skin(tmp_path):
    # PLAYPAUS at 1x1 like the real darkside case.
    atlases_dir, _ = _make_fake_atlas(tmp_path, "tiny_pp", playpaus_size=(1, 1))
    out = tmp_path / "out"
    result = _run_unpack(atlases_dir, out)
    assert result.returncode == 0
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_count"] == 1
    assert "degenerate" in manifest["skins"][0]["reason"]
    # No skin directory should have been created.
    assert not (out / "tiny_pp").exists()


def test_min_content_fraction_threshold(tmp_path):
    # Slot is just under the 0.5 default threshold; with --min-content-fraction
    # 0.1 the skin should pass instead of being skipped.
    atlases_dir, _ = _make_fake_atlas(tmp_path, "small_eqmain", eqmain_size=(275, 100))
    out_default = tmp_path / "out_default"
    out_loose = tmp_path / "out_loose"
    _run_unpack(atlases_dir, out_default)
    _run_unpack(atlases_dir, out_loose, "--min-content-fraction", "0.1")
    m_default = json.loads((out_default / "_manifest.json").read_text())
    m_loose = json.loads((out_loose / "_manifest.json").read_text())
    assert m_default["accepted_count"] == 0
    assert m_loose["accepted_count"] == 1


def test_unpacked_skin_loads_in_v7_dataset(tmp_path):
    """The unpacked output must be ingestible by V7CompletionDataset."""
    atlases_dir, _ = _make_fake_atlas(tmp_path, "ok_skin")
    out = tmp_path / "out"
    _run_unpack(atlases_dir, out)
    from atlas_ai.dataset_v7_completion import V7CompletionDataset
    ds = V7CompletionDataset(
        skin_sources={"ok_skin": out / "ok_skin"},
        state_families_path=str(ROOT / "configs/state_families_classic.yaml"),
    )
    assert len(ds) == 11
    sample = ds[0]
    assert sample["skin_id"] == "ok_skin"
    assert sample["target_rgb"].dim() == 3


def test_multi_skin_unpack_writes_one_dir_per_accepted(tmp_path):
    # One ok, one with degenerate POSBAR -> ok accepted, degenerate skipped.
    _make_fake_atlas(tmp_path, "ok_skin")
    _make_fake_atlas(tmp_path, "bad_posbar", posbar_size=(3, 1))
    out = tmp_path / "out"
    _run_unpack(tmp_path / "atlases", out)
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["accepted_count"] == 1
    assert manifest["skipped_count"] == 1
    assert (out / "ok_skin").is_dir()
    assert not (out / "bad_posbar").exists()


def test_manifest_records_decisions_per_slot(tmp_path):
    atlases_dir, _ = _make_fake_atlas(tmp_path, "mixed",
                                       main_size=(275, 116),
                                       volume_size=(68, 410))
    out = tmp_path / "out"
    _run_unpack(atlases_dir, out)
    manifest = json.loads((out / "_manifest.json").read_text())
    slots = {s["file_name"]: s for s in manifest["skins"][0]["slots"]}
    assert "cropped" in slots["MAIN.bmp"]["action"]
    assert "padded" in slots["VOLUME.bmp"]["action"]
    assert slots["CBUTTONS.bmp"]["action"] == "exact"


def test_per_skin_meta_written(tmp_path):
    atlases_dir, _ = _make_fake_atlas(tmp_path, "ok_skin")
    out = tmp_path / "out"
    _run_unpack(atlases_dir, out)
    skin_meta = json.loads((out / "ok_skin" / "_meta.json").read_text())
    assert skin_meta["skin_id"] == "ok_skin"
    assert len(skin_meta["slots"]) == 11
