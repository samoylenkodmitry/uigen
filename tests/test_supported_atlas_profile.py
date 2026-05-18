"""Pin the atlas profile to Cranamp-supported training components only."""
from __future__ import annotations

from pathlib import Path

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.profiles import load_atlas_profile, load_export_profile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "atlas_train_v1.json"
EXPORT = ROOT / "configs" / "export_profile_classic.json"

SUPPORTED_SLOTS = {
    "MAIN", "TITLEBAR", "CBUTTONS", "SHUFREP", "MONOSTER", "PLAYPAUS",
    "EQMAIN", "PLEDIT", "POSBAR", "VOLUME", "BALANCE",
}

UNSUPPORTED_SLOTS = {
    "NUMBERS", "TEXT", "EQ_EX", "GEN", "VIDEO",
    "RESERVED_A", "RESERVED_B",
}


def test_atlas_profile_contains_only_supported_slots():
    ap = load_atlas_profile(str(CONFIG))
    names = {slot.name for slot in ap.slots}
    assert names == SUPPORTED_SLOTS
    assert not (names & UNSUPPORTED_SLOTS)
    assert [slot.id for slot in ap.slots] == list(range(len(ap.slots)))
    for slot in ap.slots:
        assert slot.file is not None
        assert slot.loss_weight == 1.0


def test_export_profile_contains_only_supported_slots():
    export = load_export_profile(str(EXPORT))
    exported_slots = {str(info["slot"]) for info in export.values()}
    assert exported_slots <= SUPPORTED_SLOTS
    assert not (exported_slots & UNSUPPORTED_SLOTS)


def test_static_trainable_export_spec_matches_profiles():
    atlas = load_atlas_profile(str(CONFIG))
    export = load_export_profile(str(EXPORT))
    slots = atlas.slots_by_name
    specs = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}

    assert set(specs) == set(export)
    for file_name, info in export.items():
        slot = slots[str(info["slot"])]
        spec = specs[file_name]
        assert (spec.x, spec.y) == (slot.x, slot.y)
        assert (spec.w, spec.h) == (int(info["w"]), int(info["h"]))
        assert spec.weight > 0
