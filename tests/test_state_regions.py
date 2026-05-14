from pathlib import Path

from atlas_ai.profiles import load_atlas_profile, load_json


ROOT = Path(__file__).resolve().parents[1]


def test_state_regions_are_inside_slot_capacity_rectangles():
    profile = load_atlas_profile(ROOT / "configs/atlas_v1.json")
    slots = profile.slots_by_name
    regions = load_json(ROOT / "configs/state_regions_v1.json")

    for slot_name, groups in regions.items():
        slot = slots[slot_name]
        for group_name, rects in groups.items():
            for rect in rects:
                x0, y0, x1, y1 = rect
                assert 0 <= x0 < x1 <= slot.w, (slot_name, group_name, rect)
                assert 0 <= y0 < y1 <= slot.h, (slot_name, group_name, rect)

