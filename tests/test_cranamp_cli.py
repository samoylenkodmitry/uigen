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
            "960",
            "--canvas-h",
            "1728",
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
            "960",
            "--canvas-h",
            "1728",
            "--out-view",
            str(replay),
        ],
        check=True,
    )

    rendered_params = json.loads(params.read_text(encoding="utf-8"))
    with Image.open(view) as image:
        assert image.mode == "RGB"
        assert image.size == (960, 1728)
        pl_x, pl_y = rendered_params["windows"]["playlist"]
        scale = rendered_params["scale"]
        bottom_y = 261 - 38
        list_button = image.crop(
            (
                round(pl_x + 228 * scale),
                round(pl_y + (bottom_y + 7) * scale),
                round(pl_x + (228 + 28) * scale),
                round(pl_y + (bottom_y + 25) * scale),
            )
        )
        assert np.asarray(list_button).std() > 2
        first_song = image.crop(
            (
                round(pl_x + 16 * scale),
                round(pl_y + 22 * scale),
                round(pl_x + 150 * scale),
                round(pl_y + 32 * scale),
            )
        )
        assert np.asarray(first_song).std() > 2
        histogram = image.crop(
            (
                round(rendered_params["windows"]["main"][0] + 27 * scale),
                round(rendered_params["windows"]["main"][1] + 43 * scale),
                round(rendered_params["windows"]["main"][0] + 97 * scale),
                round(rendered_params["windows"]["main"][1] + 59 * scale),
            )
        )
        assert np.asarray(histogram).std() > 2
    with Image.open(mask) as image:
        assert image.mode == "L"
        assert image.size == (1024, 1024)
        assert image.getbbox() is not None
    assert np.fromfile(rects, dtype="<f4").shape == (80 * 5,)
    assert np.fromfile(state, dtype="<f4").shape == (32,)
    assert rendered_params["schema"] == "cranamp_cli_renderer_v3"
    assert rendered_params["scale"] >= 2.0
    main = rendered_params["windows"]["main"]
    eq = rendered_params["windows"]["eq"]
    playlist = rendered_params["windows"]["playlist"]
    assert main == [0, 0]
    assert main[0] == eq[0] == playlist[0]
    window_scales = rendered_params["window_scales"]
    main_sy = float(window_scales["main"])
    eq_sy = float(window_scales["eq"])
    assert eq[1] == main[1] + int(round(116 * rendered_params["scale"] * main_sy))
    assert playlist[1] == eq[1] + int(round(116 * rendered_params["scale"] * eq_sy))
    transforms = rendered_params["component_transforms"]
    assert {"posbar", "volume", "balance", "eq_sliders"} <= transforms.keys()
    # transport_*/shufrep group-mode may drop per-member transforms; require at least one transport_* OR a transport group.
    has_transport = any(k.startswith("transport") for k in transforms.keys())
    assert has_transport
    assert any(t["dx"] or t["dy"] or t["sx"] != 1.0 or t["sy"] != 1.0 for t in transforms.values())
    assert {t["mode"] for t in transforms.values()} - {"identity"}
    assert len(rendered_params["playlist_entries"]) >= 10
    assert len(rendered_params["state"]["histogram"]) == 16
    assert view.read_bytes() == replay.read_bytes()
