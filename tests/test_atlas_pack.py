from pathlib import Path

from PIL import Image

from atlas_ai.atlas import pack_skin_assets
from atlas_ai.profiles import assert_slots_fit, load_atlas_profile
from atlas_ai.skins import load_default_assets, load_skin_assets


ROOT = Path(__file__).resolve().parents[1]


def test_default_skin_packs_to_1024_rgb_atlas():
    profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    assert_slots_fit(profile)
    assets = load_skin_assets(ROOT / "assets/default_skin")

    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, profile)

    assert packed.rejected_reason is None
    assert packed.atlas.mode == "RGB"
    assert packed.atlas.size == (1024, 1024)


def test_packing_places_main_pixels_at_slot_origin():
    profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    assets = load_default_assets(ROOT / "assets/default_skin")
    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, profile)
    main_slot = profile.slots_by_name["MAIN"]

    with Image.open(ROOT / "assets/default_skin/MAIN.bmp") as source:
        source_rgb = source.convert("RGB")

    assert packed.atlas.getpixel((main_slot.x, main_slot.y)) == source_rgb.getpixel((0, 0))


def test_all_profile_slots_are_supported():
    profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")

    for slot in profile.slots:
        assert slot.loss_weight == 1.0
