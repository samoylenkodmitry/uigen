"""V8 product-layout helpers for mockup -> skin conversion.

The product input is a generated Winamp-like screenshot, not an old atlas. V8
normalizes that screenshot to a fixed canvas and records where the main, EQ,
and playlist windows live. The first implementation supports manual rects and
a deterministic default stack; automatic detection can replace the default
later without changing downstream modules.
"""

from __future__ import annotations

from pathlib import Path
import json

from PIL import Image, ImageDraw


NORMALIZED_SIZE = (960, 1728)
WINDOW_UNITS = {
    "main": (275, 116),
    "eq": (275, 116),
    "playlist": (275, 261),
}


def _rect_list(rect) -> list[float]:
    if isinstance(rect, dict):
        return [float(rect[k]) for k in ("x", "y", "w", "h")]
    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        return [float(v) for v in rect]
    raise ValueError(f"rect must be [x,y,w,h] or mapping, got {rect!r}")


def default_layout(width: int = NORMALIZED_SIZE[0], height: int = NORMALIZED_SIZE[1]) -> dict:
    """Default stacked classic layout filling the normalized canvas width."""
    scale = width / 275.0
    main_h = round(116 * scale)
    eq_h = round(116 * scale)
    playlist_h = round(261 * scale)
    return {
        "schema": "uigen_v8_layout_v1",
        "normalized_size": [int(width), int(height)],
        "rects": {
            "main": [0, 0, int(width), main_h],
            "eq": [0, main_h, int(width), eq_h],
            "playlist": [0, main_h + eq_h, int(width), playlist_h],
        },
    }


def scale_rects_for_letterbox(rects: dict, *, scale: float, offset: tuple[int, int]) -> dict:
    ox, oy = offset
    out = {}
    for key in ("main", "eq", "playlist"):
        x, y, w, h = _rect_list(rects[key])
        out[key] = [
            round(x * scale + ox, 3),
            round(y * scale + oy, 3),
            round(w * scale, 3),
            round(h * scale, 3),
        ]
    return out


def normalize_mockup_image(
    image: Image.Image,
    *,
    size: tuple[int, int] = NORMALIZED_SIZE,
    fill: tuple[int, int, int] = (8, 8, 10),
) -> tuple[Image.Image, float, tuple[int, int]]:
    """Letterbox an arbitrary input into the V8 normalized canvas."""
    image = image.convert("RGB")
    out_w, out_h = size
    scale = min(out_w / image.width, out_h / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    offset = ((out_w - new_w) // 2, (out_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas, scale, offset


def load_rect_override(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rects = data.get("rects", data)
    missing = [k for k in ("main", "eq", "playlist") if k not in rects]
    if missing:
        raise ValueError(f"rect override missing: {', '.join(missing)}")
    return {k: _rect_list(rects[k]) for k in ("main", "eq", "playlist")}


def make_layout(
    *,
    normalized_size: tuple[int, int],
    rect_override: dict | None = None,
    override_space: str = "original",
    letterbox_scale: float = 1.0,
    letterbox_offset: tuple[int, int] = (0, 0),
) -> dict:
    if rect_override is None:
        return default_layout(*normalized_size)
    if override_space == "original":
        rects = scale_rects_for_letterbox(
            rect_override, scale=letterbox_scale, offset=letterbox_offset
        )
    elif override_space == "normalized":
        rects = {k: _rect_list(v) for k, v in rect_override.items()}
    else:
        raise ValueError("override_space must be 'original' or 'normalized'")
    return {
        "schema": "uigen_v8_layout_v1",
        "normalized_size": [int(normalized_size[0]), int(normalized_size[1])],
        "rects": rects,
    }


def product_render_params(layout: dict) -> dict:
    """Neutral, deterministic Cranamp render params derived from a V8 layout.

    The product gate must render the skin AS DESIGNED, so this produces params
    with NO jitter: identity component transforms, window_scales=1, scale and
    window origins matching the layout, and neutral playback/EQ state. Dynamic
    app content (spectrum, playback indicator, playlist text) is zeroed/fixed so
    it does not pollute skin-chrome comparison — what remains is the skin.

    The schema matches cranamp tools/cranamp_cli.py render-params expectations.
    """
    rects = layout["rects"]
    mx, my, mw, _mh = _rect_list(rects["main"])
    ex, ey = _rect_list(rects["eq"])[:2]
    px, py = _rect_list(rects["playlist"])[:2]
    scale = float(mw) / 275.0  # window width = mw; heights = 116/261 * scale
    return {
        "schema": "cranamp_cli_renderer_v3",
        "scale": round(scale, 6),
        "windows": {
            "main": [round(mx), round(my)],
            "eq": [round(ex), round(ey)],
            "playlist": [round(px), round(py)],
        },
        "window_scales": {"main": 1.0, "eq": 1.0, "playlist": 1.0},
        "component_transforms": {},                     # identity fallback => no jitter
        "group_modes": {"transport": False, "shufrep": False},
        "playlist_entries": [f"{i + 1:02d}. Track {i + 1:02d}" for i in range(8)],
        "state": {
            "pressed_transport_button": -1,             # nothing pressed
            "volume": 0.5,
            "balance": 0.5,
            "posbar": 0.0,
            "shuffle": False,
            "repeat": False,
            "eq_on": True,
            "eq_auto": False,
            "eq_values": [0.5] * 11,                    # flat 0 dB (neutral)
            "playlist_scroll": 0.0,
            "playlist_selected_row": 0,
            "playback": "stopped",                      # no play indicator
            "histogram": [0.0] * 16,                    # no spectrum bars
        },
    }


# A fixed, deterministic spread of slider positions for DEMO/inspection renders.
# Neutral params hide the per-frame picture (every thumb at center); this exercises
# the EQ band frames, volume/balance/posbar at distinct positions so the render
# shows whether the skin's slider sprites are reproduced across their range.
_DEMO_EQ_CURVE = [0.05, 0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 0.75, 0.50, 0.30, 0.10]


def demo_render_params(layout: dict) -> dict:
    """product_render_params but with EQ/volume/balance/posbar at varied (still
    deterministic) positions, so a demo render shows the slider sprites in use
    across their range rather than all at neutral center."""
    p = product_render_params(layout)
    p["state"]["eq_values"] = list(_DEMO_EQ_CURVE)
    p["state"]["volume"] = 0.72
    p["state"]["balance"] = 0.38
    p["state"]["posbar"] = 0.45
    return p


def draw_layout_overlay(image: Image.Image, layout: dict) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = {"main": (255, 80, 80), "eq": (80, 255, 120), "playlist": (80, 160, 255)}
    for name, rect in layout["rects"].items():
        x, y, w, h = _rect_list(rect)
        color = colors.get(name, (255, 255, 0))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x + 6, y + 6), name, fill=color)
    return overlay


def save_layout(layout: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_layout(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "uigen_v8_layout_v1":
        raise ValueError(f"not a V8 layout: {path}")
    return data


__all__ = [
    "NORMALIZED_SIZE",
    "WINDOW_UNITS",
    "default_layout",
    "draw_layout_overlay",
    "load_layout",
    "load_rect_override",
    "make_layout",
    "normalize_mockup_image",
    "save_layout",
    "scale_rects_for_letterbox",
]
