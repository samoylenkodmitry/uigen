"""Final-frame provenance buffer tests for the Cranamp renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLI_MODULE_PATH = ROOT / "cranamp_cli/cranamp/tools/cranamp_cli.py"


@pytest.fixture(scope="module")
def cm():
    spec = importlib.util.spec_from_file_location("cranamp_cli_tool", CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_renderer(cm, canvas_w: int = 64, canvas_h: int = 64):
    return cm.Renderer(ROOT / "assets/default_skin", canvas_w, canvas_h)


def _inject(renderer, cm, file_name: str, rgba: np.ndarray) -> None:
    from atlas_ai.skins import normalize_name
    renderer.images[normalize_name(file_name)] = Image.fromarray(rgba, mode="RGBA")


def _solid_rgba(h: int, w: int, *, color=(255, 0, 0), alpha: int = 255) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = alpha
    return arr


def test_encode_decode_roundtrip(cm):
    for file_id, sy, sx in [(0, 0, 0), (10, 2047, 2047), (5, 314, 271), (7, 21, 13)]:
        packed = cm.encode_provenance(file_id, sy, sx)
        assert cm.decode_provenance(packed) == (file_id, sy, sx)
        assert packed != 0  # never collides with the "no source" sentinel


def test_initial_provenance_is_zero(cm):
    r = _make_renderer(cm, 32, 16)
    assert r.provenance.shape == (16, 32)
    assert r.provenance.dtype == np.uint32
    assert (r.provenance == 0).all()


def test_sprite_identity(cm):
    r = _make_renderer(cm, 64, 64)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (10, 20), 1.0)

    file_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    # Top-left of the blit lands at canvas (10, 20) -> src (0, 0).
    assert cm.decode_provenance(r.provenance[20, 10]) == (file_id, 0, 0)
    # Three pixels in maps to source (3, 3).
    assert cm.decode_provenance(r.provenance[23, 13]) == (file_id, 3, 3)
    # Pixels outside the blit remain zero.
    assert r.provenance[19, 10] == 0
    assert r.provenance[20, 9] == 0


def test_sprite_identity_with_src_offset(cm):
    r = _make_renderer(cm, 64, 64)
    _inject(r, cm, "TITLEBAR.bmp", _solid_rgba(32, 32))
    r.blit("TITLEBAR", "TITLEBAR.bmp", (5, 7, 4, 4), (0, 0), 1.0)
    file_id = cm.TRAINABLE_FILE_TO_ID["TITLEBAR.bmp"]
    # Canvas (0, 0) -> source coordinate (sx=5, sy=7).
    assert cm.decode_provenance(r.provenance[0, 0]) == (file_id, 7, 5)
    # Canvas (2, 3) -> source (sx + 3, sy + 2).
    assert cm.decode_provenance(r.provenance[2, 3]) == (file_id, 9, 8)


def test_transparent_pixels_do_not_write(cm):
    r = _make_renderer(cm, 16, 16)
    transparent = _solid_rgba(4, 4, alpha=0)
    _inject(r, cm, "MAIN.bmp", transparent)
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (2, 2), 1.0)
    assert r.provenance[2:6, 2:6].sum() == 0


def test_magenta_pixels_do_not_write(cm, tmp_path):
    # image_for converts magenta -> alpha 0 at load time. Round-trip through a
    # real file so we exercise that code path.
    skin = tmp_path / "skin"
    skin.mkdir()
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 2] = 255  # all-magenta sprite
    Image.fromarray(arr, mode="RGB").save(skin / "MAIN.bmp")
    # Provide stubs for the other required default-skin files so loader works.
    for fname in [
        "TITLEBAR.bmp", "CBUTTONS.bmp", "SHUFREP.bmp", "MONOSTER.bmp", "PLAYPAUS.bmp",
        "EQMAIN.bmp", "PLEDIT.bmp", "POSBAR.bmp", "VOLUME.bmp", "BALANCE.bmp",
        "NUMBERS.bmp", "TEXT.bmp", "VISCOLOR.TXT", "PLEDIT.TXT", "README.txt",
    ]:
        src = ROOT / "assets/default_skin" / fname
        if src.exists():
            (skin / fname).write_bytes(src.read_bytes())
    r = cm.Renderer(skin, 16, 16)
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (4, 4), 1.0)
    assert r.provenance[4:8, 4:8].sum() == 0


def test_overdraw_replaces_provenance(cm):
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(8, 8, color=(255, 0, 0)))
    _inject(r, cm, "CBUTTONS.bmp", _solid_rgba(4, 4, color=(0, 0, 255)))
    main_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    cb_id = cm.TRAINABLE_FILE_TO_ID["CBUTTONS.bmp"]

    r.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (0, 0), 1.0)
    assert cm.decode_provenance(r.provenance[2, 2])[0] == main_id
    assert cm.decode_provenance(r.provenance[6, 6])[0] == main_id

    r.blit("CBUTTONS", "CBUTTONS.bmp", (0, 0, 4, 4), (2, 2), 1.0)
    # CBUTTONS now owns (2, 2)..(5, 5).
    assert cm.decode_provenance(r.provenance[2, 2])[0] == cb_id
    assert cm.decode_provenance(r.provenance[5, 5])[0] == cb_id
    # MAIN provenance outside the overdraw region survives.
    assert cm.decode_provenance(r.provenance[0, 0])[0] == main_id
    assert cm.decode_provenance(r.provenance[6, 6])[0] == main_id


def test_fill_rect_zeros_provenance(cm):
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(8, 8))
    main_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    r.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (0, 0), 1.0)
    assert cm.decode_provenance(r.provenance[3, 3])[0] == main_id

    r.fill_rect((2, 2, 4, 4), (0, 0, 0, 255))
    # Filled region is cleared; pixels outside survive.
    assert (r.provenance[2:7, 2:7] == 0).all()
    assert cm.decode_provenance(r.provenance[0, 0])[0] == main_id
    assert cm.decode_provenance(r.provenance[7, 0])[0] == main_id


def test_erase_rect_zeros_provenance(cm):
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(8, 8))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (0, 0), 1.0)
    r.erase_rect_with_edge_average((1, 1, 4, 4))
    assert (r.provenance[1:5, 1:5] == 0).all()
    main_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    assert cm.decode_provenance(r.provenance[0, 0])[0] == main_id


def test_procedural_overlay_zeros_where_it_contributes(cm):
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(8, 8))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (0, 0), 1.0)
    main_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]

    layer = np.zeros((6, 6, 4), dtype=np.uint8)
    layer[2:4, 2:4, 3] = 255  # only the inner 2x2 contributes
    layer[2:4, 2:4, 0] = 99
    r.composite_procedural(Image.fromarray(layer, mode="RGBA"), (1, 1))

    # Pixels under the layer's opaque region were cleared.
    assert (r.provenance[3:5, 3:5] == 0).all()
    # Pixels under the layer's transparent region still belong to MAIN.
    assert cm.decode_provenance(r.provenance[1, 1])[0] == main_id
    assert cm.decode_provenance(r.provenance[6, 6])[0] == main_id


def test_scaled_blit_nearest_neighbor_mapping(cm):
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (0, 0), 2.0)
    file_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    # PIL nearest: dest pixel t -> src floor((t + 0.5) / scale). With scale=2:
    #   t=0 -> floor(0.25)=0
    #   t=1 -> floor(0.75)=0
    #   t=2 -> floor(1.25)=1
    expected = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0,
                (2, 2): 1, (2, 3): 1, (3, 2): 1, (3, 3): 1,
                (4, 4): 2, (6, 6): 3, (7, 7): 3}
    for (y, x), src_coord in expected.items():
        fid, sy, sx = cm.decode_provenance(r.provenance[y, x])
        assert fid == file_id, (y, x, fid)
        assert (sy, sx) == (src_coord, src_coord), (y, x, sy, sx)


def test_scaled_blit_back_maps_match_rendered_pixels(cm):
    """Every contributing canvas pixel must map back to a valid source pixel."""
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(3, 3))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 3, 3), (5, 5), 2.5)
    # rendered size = round(3 * 2.5) = 8
    file_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    region = r.provenance[5:13, 5:13]
    assert (region != 0).all()
    decoded = np.array([cm.decode_provenance(int(v)) for v in region.flatten()])
    assert (decoded[:, 0] == file_id).all()
    assert decoded[:, 1].min() >= 0 and decoded[:, 1].max() <= 2
    assert decoded[:, 2].min() >= 0 and decoded[:, 2].max() <= 2


def test_blit_clipping_outside_canvas(cm):
    r = _make_renderer(cm, 8, 8)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    # Negative offset: only the bottom-right 3x3 of the sprite lands on canvas.
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (-1, -1), 1.0)
    file_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    # Canvas (0, 0) <- source (1, 1) because canvas_x - rdx = 0 - (-1) = 1.
    assert cm.decode_provenance(r.provenance[0, 0]) == (file_id, 1, 1)
    assert cm.decode_provenance(r.provenance[2, 2]) == (file_id, 3, 3)
    # Canvas pixels outside the rendered region remain zero.
    assert r.provenance[3, 3] == 0


def test_window_pass_preserves_unrelated_provenance(cm):
    r = _make_renderer(cm, 64, 64)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    _inject(r, cm, "CBUTTONS.bmp", _solid_rgba(4, 4))

    def render_main(rr, _params):
        rr.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (10, 10), 1.0)

    def render_second(rr, _params):
        # Draws far from (10, 10); sub-canvas at (10, 10) is fully transparent.
        rr.blit("CBUTTONS", "CBUTTONS.bmp", (0, 0, 4, 4), (40, 40), 1.0)

    params = {"windows": {"main": [0, 0], "second": [0, 0]}}
    cm._render_window_pass(r, render_main, params, "main", (0, 0), 1.0, 64, 64)
    main_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    assert cm.decode_provenance(r.provenance[10, 10])[0] == main_id

    cm._render_window_pass(r, render_second, params, "second", (0, 0), 1.0, 64, 64)
    cb_id = cm.TRAINABLE_FILE_TO_ID["CBUTTONS.bmp"]
    # Earlier provenance under a transparent region of the second window survives.
    assert cm.decode_provenance(r.provenance[10, 10])[0] == main_id
    # Second window's own provenance is present.
    assert cm.decode_provenance(r.provenance[40, 40])[0] == cb_id


def test_window_pass_stretch_resizes_provenance(cm):
    r = _make_renderer(cm, 32, 32)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(2, 2))

    def render_fn(rr, _params):
        rr.blit("MAIN", "MAIN.bmp", (0, 0, 2, 2), (0, 0), 1.0)

    params = {"windows": {"w": [0, 0]}}
    # window_sy = 2.0: sub-canvas (32, 32) stretched to (32, 64), then
    # composited at origin (0, 0). The first 4 rows of the canvas should
    # contain MAIN provenance because rows 0..3 of stretched come from
    # sub_canvas rows 0..1.
    cm._render_window_pass(r, render_fn, params, "w", (0, 0), 2.0, 32, 32)
    file_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    # Source row 0 stretched to canvas rows 0..1, source row 1 -> rows 2..3.
    assert cm.decode_provenance(r.provenance[0, 0]) == (file_id, 0, 0)
    assert cm.decode_provenance(r.provenance[1, 0]) == (file_id, 0, 0)
    assert cm.decode_provenance(r.provenance[2, 0]) == (file_id, 1, 0)
    assert cm.decode_provenance(r.provenance[3, 0]) == (file_id, 1, 0)
    # Outside the stretched window region: no provenance.
    assert r.provenance[10, 10] == 0


def test_end_to_end_render_produces_provenance(cm):
    # Full pipeline produces provenance with all 11 trainable file ids present.
    params = cm.rand_params(seed=42, canvas_w=480, canvas_h=864)
    renderer = cm.render_with_params(ROOT / "assets/default_skin", params, 480, 864)
    nonzero = renderer.provenance[renderer.provenance != 0]
    assert nonzero.size > 0
    file_ids = set(int(x >> 22) for x in nonzero.tolist())
    expected = set(cm.TRAINABLE_FILE_TO_ID.values())
    assert file_ids == expected, f"missing file ids: {expected - file_ids}"
    # All decoded src coordinates must lie within the largest trainable BMP
    # bounds (well under the 2047 ceiling).
    src_y = (nonzero >> 11) & 0x7FF
    src_x = nonzero & 0x7FF
    assert int(src_y.max()) < 600
    assert int(src_x.max()) < 600
