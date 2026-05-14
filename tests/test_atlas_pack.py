from pathlib import Path

from PIL import Image

from atlas_ai.atlas import pack_skin_assets
from atlas_ai.profiles import assert_slots_fit, load_atlas_profile
from atlas_ai.skins import load_default_assets, load_skin_assets


ROOT = Path(__file__).resolve().parents[1]


def test_default_skin_packs_to_1024_rgb_atlas():
    profile = load_atlas_profile(ROOT / "configs/atlas_v1.json")
    assert_slots_fit(profile)
    assets = load_skin_assets(ROOT / "assets/default_skin")

    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, profile)

    assert packed.rejected_reason is None
    assert packed.atlas.mode == "RGB"
    assert packed.atlas.size == (1024, 1024)
    assert packed.mask.mode == "L"
    assert packed.mask.size == (1024, 1024)
    assert packed.slot_weights.shape == (len(profile.slots),)


def test_packing_places_main_pixels_and_mask_at_slot_origin():
    profile = load_atlas_profile(ROOT / "configs/atlas_v1.json")
    assets = load_default_assets(ROOT / "assets/default_skin")
    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, profile)
    main_slot = profile.slots_by_name["MAIN"]

    with Image.open(ROOT / "assets/default_skin/MAIN.bmp") as source:
        source_rgb = source.convert("RGB")

    assert packed.atlas.getpixel((main_slot.x, main_slot.y)) == source_rgb.getpixel((0, 0))
    assert packed.mask.getpixel((main_slot.x, main_slot.y)) == 255
    assert packed.mask.getpixel((main_slot.x + source_rgb.width, main_slot.y)) == 0


def test_zero_loss_slots_get_zero_weight():
    profile = load_atlas_profile(ROOT / "configs/atlas_v1.json")
    assets = load_default_assets(ROOT / "assets/default_skin")
    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, profile)

    assert packed.slot_weights[profile.slots_by_name["TITLEBAR"].id] == 1.0
    assert packed.slot_weights[profile.slots_by_name["NUMBERS"].id] == 0.0
    assert packed.slot_weights[profile.slots_by_name["TEXT"].id] == 0.0

