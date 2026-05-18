from pathlib import Path
import zipfile

from PIL import Image

from atlas_ai.atlas import pack_skin_assets, save_packed_skin
from atlas_ai.export import export_atlas_to_skin
from atlas_ai.profiles import load_atlas_profile, load_export_profile
from atlas_ai.skins import load_default_assets


ROOT = Path(__file__).resolve().parents[1]


def test_exported_skin_files_match_export_profile_dimensions(tmp_path):
    atlas_profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    export_profile = load_export_profile(ROOT / "configs/export_profile_classic.json")
    assets = load_default_assets(ROOT / "assets/default_skin")
    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, atlas_profile)
    paths = save_packed_skin(packed, tmp_path / "packed")
    assert set(paths) == {"atlas_path", "meta_path"}

    zip_path = export_atlas_to_skin(
        atlas_path=paths["atlas_path"],
        atlas_profile=atlas_profile,
        export_profile=export_profile,
        default_skin=ROOT / "assets/default_skin",
        out_dir=tmp_path / "skin",
    )

    assert zip_path.exists()
    for file_name, info in export_profile.items():
        path = tmp_path / "skin" / file_name
        assert path.exists(), file_name
        with Image.open(path) as image:
            assert image.size == (info["w"], info["h"])


def test_export_copies_default_text_files_and_writes_wsz(tmp_path):
    atlas_profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    export_profile = load_export_profile(ROOT / "configs/export_profile_classic.json")
    assets = load_default_assets(ROOT / "assets/default_skin")
    packed = pack_skin_assets(ROOT / "assets/default_skin", assets, assets, atlas_profile)
    paths = save_packed_skin(packed, tmp_path / "packed")

    zip_path = export_atlas_to_skin(
        atlas_path=paths["atlas_path"],
        atlas_profile=atlas_profile,
        export_profile=export_profile,
        default_skin=ROOT / "assets/default_skin",
        out_dir=tmp_path / "skin",
    )

    assert (tmp_path / "skin" / "PLEDIT.TXT").exists()
    assert (tmp_path / "skin" / "VISCOLOR.TXT").exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "MAIN.bmp" in names
    assert "NUMBERS.bmp" in names
    assert "PLEDIT.bmp" in names
    assert "TEXT.bmp" in names
    assert "PLEDIT.TXT" in names
    assert "VIDEO.bmp" not in names
    assert "atlas.png" not in names


def test_export_does_not_copy_unsupported_default_bitmaps(tmp_path):
    atlas_profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    export_profile = load_export_profile(ROOT / "configs/export_profile_classic.json")
    atlas_path = tmp_path / "atlas.png"
    Image.new("RGB", (atlas_profile.canvas_w, atlas_profile.canvas_h), "white").save(atlas_path)

    export_atlas_to_skin(
        atlas_path=atlas_path,
        atlas_profile=atlas_profile,
        export_profile=export_profile,
        default_skin=ROOT / "assets/default_skin",
        out_dir=tmp_path / "skin",
    )

    assert (tmp_path / "skin" / "NUMBERS.bmp").read_bytes() == (ROOT / "assets/default_skin" / "NUMBERS.bmp").read_bytes()
    assert (tmp_path / "skin" / "TEXT.bmp").read_bytes() == (ROOT / "assets/default_skin" / "TEXT.bmp").read_bytes()
    with Image.open(tmp_path / "skin" / "PLEDIT.bmp") as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
    assert not (tmp_path / "skin" / "VIDEO.bmp").exists()


def test_export_allows_atlas_already_in_output_dir(tmp_path):
    atlas_profile = load_atlas_profile(ROOT / "configs/atlas_train_v1.json")
    export_profile = load_export_profile(ROOT / "configs/export_profile_classic.json")
    out_dir = tmp_path / "skin"
    out_dir.mkdir()
    atlas_path = out_dir / "atlas.png"
    Image.new("RGB", (atlas_profile.canvas_w, atlas_profile.canvas_h), "white").save(atlas_path)
    (out_dir / "input_letterboxed.png").write_bytes(b"not a skin file")

    zip_path = export_atlas_to_skin(
        atlas_path=atlas_path,
        atlas_profile=atlas_profile,
        export_profile=export_profile,
        default_skin=ROOT / "assets/default_skin",
        out_dir=out_dir,
    )

    assert atlas_path.exists()
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "input_letterboxed.png" not in names
