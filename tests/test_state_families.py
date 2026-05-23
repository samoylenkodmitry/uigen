"""State-family loader: schema, bounds, expansion correctness."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.state_families import (
    StateRect,
    group_by_family,
    load_state_families,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/state_families_classic.yaml"


def test_default_config_loads():
    families = load_state_families(CONFIG)
    assert len(families) == 11
    expected = {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    assert set(families) == expected


def test_every_file_has_at_least_one_rect():
    families = load_state_families(CONFIG)
    for file_name, rects in families.items():
        assert len(rects) >= 1, file_name


def test_every_rect_within_bounds():
    families = load_state_families(CONFIG)
    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    for file_name, rects in families.items():
        spec = spec_by_name[file_name]
        for r in rects:
            assert r.x >= 0
            assert r.y >= 0
            assert r.w > 0
            assert r.h > 0
            assert r.x + r.w <= spec.w, (file_name, r)
            assert r.y + r.h <= spec.h, (file_name, r)


def test_strip_expansion_volume_28_frames():
    families = load_state_families(CONFIG)
    volume = families["VOLUME.bmp"]
    grouped = group_by_family(volume)
    frames = grouped["slider_frames"]
    assert len(frames) == 28
    # First frame at (0, 0, 68, 13), last at (0, 27*15, 68, 13).
    assert frames[0] == StateRect("VOLUME.bmp", "slider_frames", "frame_00", 0, 0, 68, 13)
    last = frames[-1]
    assert (last.x, last.y, last.w, last.h) == (0, 27 * 15, 68, 13)
    # 'thumb' family is also present.
    assert "thumb" in grouped


def test_eqmain_slider_frames_are_one_28_frame_family():
    families = load_state_families(CONFIG)
    rects = families["EQMAIN.bmp"]
    grouped = group_by_family(rects)
    frames = grouped["slider_frames"]
    assert len(frames) == 28
    # frame_13 is the end of row 1; frame_27 is the end of row 2.
    assert (frames[13].x, frames[13].y, frames[13].w, frames[13].h) == (208, 164, 14, 63)
    assert (frames[27].x, frames[27].y, frames[27].w, frames[27].h) == (208, 229, 14, 63)


def test_cbuttons_six_buttons_two_states_each():
    families = load_state_families(CONFIG)
    cb = families["CBUTTONS.bmp"]
    grouped = group_by_family(cb)
    assert set(grouped) == {"prev", "play", "pause", "stop", "next", "eject"}
    for name, rects in grouped.items():
        assert len(rects) == 2, f"{name}: expected 2 states, got {len(rects)}"
        assert {r.name for r in rects} == {"unpressed", "pressed"}


def test_shufrep_toggles_present():
    families = load_state_families(CONFIG)
    sf = families["SHUFREP.bmp"]
    grouped = group_by_family(sf)
    assert {"repeat", "shuffle", "eq_toggle", "pl_toggle"} <= set(grouped)
    assert len(grouped["repeat"]) == 2
    assert len(grouped["shuffle"]) == 2


def test_rejects_unknown_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "schema": "state_families_classic_v1",
        "files": {"NOT_A_FILE.bmp": {"families": [{"name": "x", "kind": "sprite", "rects": [{"name": "r", "x": 0, "y": 0, "w": 1, "h": 1}]}]}},
    }))
    with pytest.raises(ValueError, match="not in TRAINABLE_EXPORT_SPECS"):
        load_state_families(bad)


def test_rejects_out_of_bounds_rect(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "schema": "state_families_classic_v1",
        "files": {"VOLUME.bmp": {"families": [{
            "name": "x", "kind": "sprite",
            "rects": [{"name": "r", "x": 0, "y": 0, "w": 999, "h": 999}],
        }]}},
    }))
    with pytest.raises(ValueError, match="escapes bounds"):
        load_state_families(bad)


def test_rejects_bad_schema_tag(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"schema": "wrong", "files": {}}))
    with pytest.raises(ValueError, match="schema=state_families_classic_v1"):
        load_state_families(bad)


def test_rejects_missing_trainable_file(tmp_path):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["files"].pop("MAIN.bmp")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="missing state-family declarations"):
        load_state_families(bad)


def test_rejects_empty_family(tmp_path):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["files"]["MAIN.bmp"]["families"] = [{"name": "empty", "kind": "sprite", "rects": []}]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="no rects declared"):
        load_state_families(bad)


def test_rejects_strip_axis_mismatch(tmp_path):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["files"]["VOLUME.bmp"]["families"] = [{
            "name": "x", "kind": "vertical_strip", "frame_axis": "x",
            "frame_count": 2, "frame_w": 4, "frame_h": 4,
            "x": 0, "y": 0, "pitch": 4,
        }]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="disagrees with frame_axis"):
        load_state_families(bad)


def test_rejects_zero_size(tmp_path):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["files"]["VOLUME.bmp"]["families"] = [{
            "name": "x", "kind": "sprite",
            "rects": [{"name": "r", "x": 0, "y": 0, "w": 0, "h": 10}],
        }]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="non-positive size"):
        load_state_families(bad)
