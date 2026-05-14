from pathlib import Path
import json
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


def test_cranamp_cli_render_random_outputs_replayable_dataset_files(tmp_path):
    view = tmp_path / "view.png"
    rects = tmp_path / "rects.f32"
    state = tmp_path / "state.f32"
    mask = tmp_path / "mask.png"
    params = tmp_path / "params.json"
    replay = tmp_path / "replay.png"

    subprocess.run(
        [
            str(CLI),
            "render-random",
            "--skin-dir",
            str(ROOT / "assets/default_skin"),
            "--seed",
            "123",
            "--canvas-w",
            "768",
            "--canvas-h",
            "1280",
            "--out-view",
            str(view),
            "--out-rects",
            str(rects),
            "--out-state",
            str(state),
            "--out-visible-atlas-mask",
            str(mask),
            "--out-params",
            str(params),
            "--state-balanced",
            "false",
        ],
        check=True,
    )

    subprocess.run(
        [
            str(CLI),
            "render-with-params",
            "--skin-dir",
            str(ROOT / "assets/default_skin"),
            "--params",
            str(params),
            "--canvas-w",
            "768",
            "--canvas-h",
            "1280",
            "--out-view",
            str(replay),
        ],
        check=True,
    )

    with Image.open(view) as image:
        assert image.mode == "RGB"
        assert image.size == (768, 1280)
    with Image.open(mask) as image:
        assert image.mode == "L"
        assert image.size == (1024, 1024)
        assert image.getbbox() is not None
    assert np.fromfile(rects, dtype="<f4").shape == (80 * 5,)
    assert np.fromfile(state, dtype="<f4").shape == (32,)
    assert json.loads(params.read_text(encoding="utf-8"))["schema"] == "cranamp_cli_renderer_v1"
    assert view.read_bytes() == replay.read_bytes()

