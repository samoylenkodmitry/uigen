from pathlib import Path
import importlib.util
import json
import shutil
import subprocess

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cranamp_cli/cranamp-cli"


def test_cranamp_cli_dump_classic_spec(tmp_path):
    out = tmp_path / "export_profile.json"

    subprocess.run([str(CLI), "dump-classic-spec", "--out", str(out)], check=True)

    profile = json.loads(out.read_text(encoding="utf-8"))
    assert profile["MAIN.bmp"] == {"h": 115, "slot": "MAIN", "w": 275}
    assert profile["TITLEBAR.bmp"] == {"h": 87, "slot": "TITLEBAR", "w": 344}


def test_cranamp_cli_render_random_outputs_full_view_png(tmp_path):
    view = tmp_path / "view.png"

    subprocess.run(
        [
            str(CLI),
            "render-random",
            "--skin-dir",
            str(ROOT / "assets/default_skin"),
            "--seed",
            "123",
            "--canvas-w",
            "960",
            "--canvas-h",
            "1728",
            "--out-view",
            str(view),
        ],
        check=True,
    )

    with Image.open(view) as image:
        assert image.mode == "RGB"
        assert image.size == (960, 1728)
        lower_view = image.crop((0, 700, 920, 1650))
        assert np.asarray(lower_view).std() > 2


def test_renderer_uses_playlist_footer_pixels(tmp_path):
    skin_dir = tmp_path / "skin"
    shutil.copytree(ROOT / "assets/default_skin", skin_dir)
    with Image.open(skin_dir / "PLEDIT.bmp") as source:
        pledit = source.convert("RGB")
    pledit.putpixel((10, 80), (255, 0, 0))
    pledit.save(skin_dir / "PLEDIT.bmp")

    cli_module_path = ROOT / "cranamp_cli/cranamp/tools/cranamp_cli.py"
    spec = importlib.util.spec_from_file_location("cranamp_cli_tool", cli_module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    params = module.rand_params(seed=1, canvas_w=960, canvas_h=1728)
    params["scale"] = 1.0
    params["windows"] = {"main": [0, 0], "eq": [0, 116], "playlist": [0, 232]}
    params["window_scales"] = {"main": 1.0, "eq": 1.0, "playlist": 1.0}
    params["component_transforms"] = {}
    renderer = module.render_with_params(skin_dir, params, canvas_w=960, canvas_h=1728)

    # PLEDIT source (10, 80) is inside the footer strip used for Add/Rem/Sel/Misc.
    assert renderer.canvas.convert("RGB").getpixel((10, 232 + 223 + 8)) == (255, 0, 0)
