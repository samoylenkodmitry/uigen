from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import zipfile

import torch
from PIL import Image

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.hidden_state_compiler import compile_hidden_states
from atlas_ai.torch_cranamp_renderer import render_visible
from atlas_ai.v8_assets import load_exported_tensors, save_exported_tensors, tensor_to_image
from atlas_ai.v8_layout import default_layout, normalize_mockup_image
from atlas_ai.visible_extractor import extract_visible_assets
from models.visible_skin_net import VisibleSkinNet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKIN = ROOT / "assets/default_skin"


def test_v8_layout_normalizes_and_defaults_to_stack():
    image = Image.new("RGB", (320, 240), (12, 34, 56))
    normalized, scale, offset = normalize_mockup_image(image)
    layout = default_layout(*normalized.size)

    assert normalized.size == (960, 1728)
    assert scale == 3.0
    assert offset == (0, 504)
    assert set(layout["rects"]) == {"main", "eq", "playlist"}
    assert layout["rects"]["main"][2] == 960


def test_hidden_compiler_and_renderer_produce_valid_skin(tmp_path):
    files = load_exported_tensors(DEFAULT_SKIN, default_skin=DEFAULT_SKIN)
    compiled = compile_hidden_states(files, default_skin=str(DEFAULT_SKIN))

    assert set(compiled) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    assert not torch.equal(compiled["CBUTTONS.bmp"][:, 0:18, 0:23],
                           compiled["CBUTTONS.bmp"][:, 18:36, 0:23])
    assert not torch.equal(compiled["VOLUME.bmp"][:, 0:13, 0:68],
                           compiled["VOLUME.bmp"][:, 405:418, 0:68])

    layout = default_layout()
    rendered = render_visible(compiled, layout)
    assert rendered.shape == (3, 1728, 960)
    assert 0.0 <= float(rendered.min()) <= float(rendered.max()) <= 1.0

    zip_path = save_exported_tensors(compiled, tmp_path / "skin", default_skin=DEFAULT_SKIN)
    assert zip_path and zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "MAIN.bmp" in names
    assert "VOLUME.bmp" in names
    assert "TEXT.bmp" in names


def test_visible_extractor_returns_exact_export_shapes():
    base = load_exported_tensors(DEFAULT_SKIN, default_skin=DEFAULT_SKIN)
    layout = default_layout()
    mockup = tensor_to_image(render_visible(base, layout))
    visible = extract_visible_assets(mockup, layout, default_skin=DEFAULT_SKIN)

    for spec in TRAINABLE_EXPORT_SPECS:
        assert visible[spec.file_name].shape == (3, spec.h, spec.w)


def test_visible_skin_net_forward_shapes():
    model = VisibleSkinNet(base_channels=8, style_dim=32, head_channels=16, head_divisor=16)
    out = model(torch.rand(1, 3, 128, 128))
    assert set(out) == {"files"}
    for spec in TRAINABLE_EXPORT_SPECS:
        assert out["files"][spec.file_name].shape == (1, 3, spec.h, spec.w)


def test_v8_mockup_to_skin_script_smoke(tmp_path):
    mockup = tmp_path / "mockup.png"
    Image.new("RGB", (320, 576), (30, 40, 50)).save(mockup)
    out = tmp_path / "out"
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/v8_mockup_to_skin.py"),
        "--input",
        str(mockup),
        "--out",
        str(out),
        "--default-skin",
        str(DEFAULT_SKIN),
    ], check=True)
    assert (out / "normalized.png").exists()
    assert (out / "layout.json").exists()
    assert (out / "render_preview.png").exists()
    assert (out / "skin/skin.wsz").exists()
