from pathlib import Path

from atlas_ai.profiles import assert_slots_fit, load_atlas_profile, load_export_profile


ROOT = Path(__file__).resolve().parents[1]


def test_atlas_slots_fit_canvas_and_do_not_overlap():
    profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    assert_slots_fit(profile)

    slots = profile.slots
    for idx, a in enumerate(slots):
        for b in slots[idx + 1 :]:
            overlap_x = max(a.x, b.x) < min(a.x + a.w, b.x + b.w)
            overlap_y = max(a.y, b.y) < min(a.y + a.h, b.y + b.h)
            assert not (overlap_x and overlap_y), (a.name, b.name)


def test_export_profile_fits_referenced_atlas_slots():
    profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    export_profile = load_export_profile(ROOT / "configs/export_profile_classic.json")
    slots = profile.slots_by_name

    for file_name, info in export_profile.items():
        slot = slots[info["slot"]]
        assert info["w"] <= slot.w, file_name
        assert info["h"] <= slot.h, file_name
