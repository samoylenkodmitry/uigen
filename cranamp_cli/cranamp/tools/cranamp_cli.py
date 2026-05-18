#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.profiles import load_export_profile
from atlas_ai.skins import load_default_assets, load_skin_assets, normalize_name


CANVAS_DEFAULT = (960, 1728)
EXPORT_PROFILE = REPO_ROOT / "configs/export_profile_classic.json"
DEFAULT_SKIN = REPO_ROOT / "assets/default_skin"


def scale_pair(scale: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(scale, tuple):
        return scale
    return (scale, scale)


class Renderer:
    """Render randomized Cranamp input PNGs."""

    def __init__(self, skin_source: Path, canvas_w: int, canvas_h: int):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.assets = load_skin_assets(skin_source)
        self.default_assets = load_default_assets(DEFAULT_SKIN)
        self.images: dict[str, Image.Image] = {}
        self.canvas = Image.new("RGBA", (canvas_w, canvas_h), (14, 14, 18, 255))

    def image_for(self, file_name: str) -> Image.Image:
        key = normalize_name(file_name)
        if key not in self.images:
            asset = self.assets.get(key) or self.default_assets.get(key)
            if asset is None:
                raise FileNotFoundError(file_name)
            from io import BytesIO

            with Image.open(BytesIO(asset.data)) as source:
                rgba = source.convert("RGBA")
            data = np.array(rgba)
            magenta = (data[:, :, 0] == 255) & (data[:, :, 1] == 0) & (data[:, :, 2] == 255)
            data[:, :, 3] = np.where(magenta, 0, data[:, :, 3])
            self.images[key] = Image.fromarray(data)
        return self.images[key]

    def mark_rect(
        self,
        component_id: int,
        rect: tuple[float, float, float, float],
        scale: float | tuple[float, float] = 1.0,
    ) -> None:
        return None

    def mark_abs_rect(self, component_id: int, rect: tuple[float, float, float, float]) -> None:
        return None

    def blit(
        self,
        slot_name: str,
        file_name: str,
        src: tuple[int, int, int, int],
        dest: tuple[float, float],
        scale: float | tuple[float, float],
        component_id: int | None = None,
    ) -> None:
        sx, sy, sw, sh = src
        dx, dy = dest
        scale_x, scale_y = scale_pair(scale)
        source = self.image_for(file_name)
        crop = source.crop((sx, sy, sx + sw, sy + sh))
        rendered_w = max(1, round(sw * scale_x))
        rendered_h = max(1, round(sh * scale_y))
        if rendered_w != sw or rendered_h != sh:
            rendered_crop = crop.resize((rendered_w, rendered_h), Image.Resampling.NEAREST)
        else:
            rendered_crop = crop
        rdx, rdy = round(dx), round(dy)
        self.canvas.alpha_composite(rendered_crop, (rdx, rdy))
        if component_id is not None:
            self.mark_rect(component_id, (dx, dy, sw, sh), scale)

    def composite_procedural(self, layer: Image.Image, dest: tuple[int, int]) -> None:
        """Alpha-composite procedurally-painted content onto the canvas."""
        dx, dy = int(dest[0]), int(dest[1])
        self.canvas.alpha_composite(layer, (dx, dy))

    def fill_rect(self, rect: tuple[float, float, float, float], color: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        x0, y0 = round(x), round(y)
        x1, y1 = round(x + w), round(y + h)
        ImageDraw.Draw(self.canvas).rectangle([x0, y0, x1, y1], fill=color)

    def erase_rect_with_edge_average(
        self,
        rect: tuple[float, float, float, float],
        scale: float | tuple[float, float] = 1.0,
    ) -> None:
        probe_offset = 3
        x, y, w, h = rect
        scale_x, scale_y = scale_pair(scale)
        x0 = max(0, min(self.canvas_w - 1, round(x)))
        y0 = max(0, min(self.canvas_h - 1, round(y)))
        x1 = max(x0 + 1, min(self.canvas_w, round(x + w * scale_x)))
        y1 = max(y0 + 1, min(self.canvas_h, round(y + h * scale_y)))
        cx = max(0, min(self.canvas_w - 1, (x0 + x1 - 1) // 2))
        cy = max(0, min(self.canvas_h - 1, (y0 + y1 - 1) // 2))
        pixels = self.canvas.load()
        samples = []
        if y0 >= probe_offset:
            samples.append(pixels[cx, y0 - probe_offset])
        if y1 + probe_offset - 1 < self.canvas_h:
            samples.append(pixels[cx, y1 + probe_offset - 1])
        if x0 >= probe_offset:
            samples.append(pixels[x0 - probe_offset, cy])
        if x1 + probe_offset - 1 < self.canvas_w:
            samples.append(pixels[x1 + probe_offset - 1, cy])
        if not samples:
            samples.append((14, 14, 18, 255))
        avg = tuple(int(round(sum(sample[channel] for sample in samples) / len(samples))) for channel in range(4))
        ImageDraw.Draw(self.canvas).rectangle([x0, y0, x1 - 1, y1 - 1], fill=avg)


_ARTIST_POOL = [
    "Cranamp", "Night Bus", "Small Grid", "Frame Step", "Raster Kids", "Hidden Tab",
    "Null Track", "Palette Lab", "Mono Deck", "Pixel Sort", "Old Skin", "Sample Bus",
    "Track Mask", "Synth Log", "Locator", "Blue Metal", "Checksum", "Final Slot",
    "Quiet Engine", "Vector Field", "Buffer Boys", "Loop Theory", "Phase Lock", "Glitch Salon",
    "Stride Mode", "Dither", "FFT Choir", "Saturated", "Skyfade", "808 Garden", "Edge Case",
    "Static Bloom", "Halftone", "Modal Crash", "Tile Pump", "Rasterizer", "Render Boys",
    "Voxel Cult", "Trace Bus", "Lo Bit", "Hi Bit", "Subpixel", "Anti Alias", "Bitplane",
    "Magenta Key", "Alpha Crew", "RGB Drift", "Atlas Five", "Reseed", "Cold Slot",
]
_TITLE_POOL = [
    "Cold Start", "Status Line", "Blue Window", "Seek Position", "Button State",
    "Equalized", "Stub Name", "Gamma Drift", "Stereo Flag", "List Mode", "Footer Menu",
    "Render Pass", "Visible Atlas", "Histogram", "Top Left", "Playlist Row", "Replay",
    "Rect Label", "Repaint", "Pressed", "Released", "Half Step", "Quarter Note",
    "Beta Slider", "Margin Pull", "Drag State", "Drop Frame", "Dead Pixel", "Sprite Tear",
    "Init Boot", "Long Press", "Tap Hold", "Side Channel", "Mono Mix", "Pan Right",
    "Pan Left", "Track Index", "Scrollbar", "Toggle On", "Toggle Off", "Auto EQ",
    "Manual Gain", "Soft Knee", "Hard Clip", "Click Free", "Limit Push", "Tail Out",
    "Cue In", "Fade Curve", "Repeat One", "Shuffle Bag",
]
_GLYPH_GARBLE = "!@#$%^&*()_+-=[]{};:,.<>/?\\|~`"


def _random_song(rng: random.Random) -> str:
    """Generate a "NN. Artist - Title M:SS" line with mild perturbation."""
    artist = rng.choice(_ARTIST_POOL)
    title = rng.choice(_TITLE_POOL)
    if rng.random() < 0.15:
        # Extra suffix on title (Remix/Live/Demo/Edit/etc.).
        title = title + " (" + rng.choice(["Remix", "Live", "Demo", "Edit", "Reprise", "vJ", "B-Side"]) + ")"
    if rng.random() < 0.08:
        # Light glyph garble — simulates imagegen mistyped chars.
        chars = list(title)
        for i in rng.sample(range(len(chars)), min(2, len(chars))):
            chars[i] = rng.choice(_GLYPH_GARBLE)
        title = "".join(chars)
    minutes = rng.randint(0, 9)
    seconds = rng.randint(0, 59)
    duration = f"{minutes}:{seconds:02d}"
    return f"{artist} - {title} {duration}"


def _random_playlist_entries(rng: random.Random) -> list[str]:
    n = rng.randint(10, 22)
    entries = []
    for idx in range(n):
        if rng.random() < 0.04:
            # Occasionally a malformed/short line — drop the number prefix or duration.
            line = _random_song(rng).split(" - ", 1)[-1]
        elif rng.random() < 0.04:
            # Very long title gets truncated by the renderer's [:38] clip — fine.
            line = f"{idx + 1:02d}. " + _random_song(rng) + " " + _random_song(rng)
        else:
            line = f"{idx + 1:02d}. {_random_song(rng)}"
        entries.append(line)
    return entries


def rand_params(seed: int, canvas_w: int, canvas_h: int) -> dict:
    rng = random.Random(seed)
    stack_units_h = 116 + 116 + 261
    fit_scale = min(canvas_w / 275, canvas_h / stack_units_h)
    max_scale = min(3.4, fit_scale)
    min_scale = min(2.8, max_scale)
    if max_scale > min_scale and rng.random() < 0.55:
        scale = round(max_scale, 4)
    elif max_scale > min_scale:
        scale = round(rng.uniform(min_scale, max_scale), 4)
    else:
        scale = round(max_scale, 4)
    main_x = 0
    main_y = 0

    pressed = rng.choice([-1, -1, 0, 1, 2, 3, 4, 5])
    volume = rng.random()
    balance = rng.random()
    posbar = rng.random()

    def signed_int(limit: int) -> int:
        if limit <= 0:
            return 0
        value = 0
        while value == 0:
            value = rng.randint(-limit, limit)
        return value

    def scale_value(delta: float) -> float:
        return round(1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.08, delta), 3)

    def transform(dx: int, dy: int, sx: float, sy: float) -> dict[str, float]:
        mode = rng.choice(
            [
                "move",
                "scalex",
                "scaley",
                "scalexy",
                "move_scalex",
                "move_scaley",
                "move_scalexy",
            ]
        )
        move = mode.startswith("move") or mode == "move"
        scale_x = "scalex" in mode
        scale_y = "scaley" in mode or mode.endswith("scalexy")
        return {
            "mode": mode,
            "dx": signed_int(dx) if move else 0,
            "dy": signed_int(dy) if move else 0,
            "sx": scale_value(sx) if scale_x else 1.0,
            "sy": scale_value(sy) if scale_y else 1.0,
        }

    playlist_entries = _random_playlist_entries(rng)

    component_transforms = {
        "playback_indicator": transform(6, 6, 0.18, 0.18),
        "mono_stereo": transform(6, 6, 0.18, 0.18),
        "posbar": transform(5, 5, 0.16, 0.16),
        "transport": transform(5, 5, 0.16, 0.16),
        "transport_prev": transform(6, 6, 0.18, 0.18),
        "transport_play": transform(6, 6, 0.18, 0.18),
        "transport_pause": transform(6, 6, 0.18, 0.18),
        "transport_stop": transform(6, 6, 0.18, 0.18),
        "transport_next": transform(6, 6, 0.18, 0.18),
        "transport_eject": transform(6, 6, 0.18, 0.18),
        "volume": transform(5, 5, 0.16, 0.16),
        "balance": transform(5, 5, 0.16, 0.16),
        "shufrep": transform(6, 6, 0.18, 0.18),
        "shuffle": transform(6, 6, 0.18, 0.18),
        "repeat": transform(6, 6, 0.18, 0.18),
        "eq_toggle": transform(6, 6, 0.18, 0.18),
        "pl_toggle": transform(6, 6, 0.18, 0.18),
        "eq_sliders": transform(8, 8, 0.22, 0.22),
        "playlist_scrollbar": transform(8, 8, 0.22, 0.22),
    }

    # Group mode: with some probability, drop the per-member transforms so the
    # renderer falls back to the parent group transform and members shift as one.
    groups = {
        "transport": ["transport_prev", "transport_play", "transport_pause",
                      "transport_stop", "transport_next", "transport_eject"],
        "shufrep": ["shuffle", "repeat", "eq_toggle", "pl_toggle"],
    }
    group_modes = {}
    for group_name, members in groups.items():
        in_group_mode = rng.random() < 0.35
        group_modes[group_name] = in_group_mode
        if in_group_mode:
            for member in members:
                component_transforms.pop(member, None)

    window_scales = {
        "main": round(rng.uniform(0.94, 1.10), 3),
        "eq": round(rng.uniform(0.94, 1.10), 3),
        "playlist": round(rng.uniform(0.94, 1.10), 3),
    }
    eq_y = main_y + int(round(116 * scale * window_scales["main"]))
    playlist_y = eq_y + int(round(116 * scale * window_scales["eq"]))

    return {
        "schema": "cranamp_cli_renderer_v3",
        "seed": seed,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "scale": scale,
        "windows": {
            "main": [main_x, main_y],
            "eq": [main_x, eq_y],
            "playlist": [main_x, playlist_y],
        },
        "window_scales": window_scales,
        "component_transforms": component_transforms,
        "group_modes": group_modes,
        "playlist_entries": playlist_entries,
        "state": {
            "pressed_transport_button": pressed,
            "volume": volume,
            "balance": balance,
            "posbar": posbar,
            "shuffle": bool(rng.getrandbits(1)),
            "repeat": bool(rng.getrandbits(1)),
            "eq_on": bool(rng.getrandbits(1)),
            "eq_auto": bool(rng.getrandbits(1)),
            "eq_values": [rng.random() for _ in range(11)],
            "playlist_scroll": rng.random(),
            "playlist_selected_row": rng.randint(0, 17),
            "playback": rng.choice(["playing", "paused", "stopped"]),
            "histogram": [rng.random() for _ in range(16)],
        },
    }


def scaled_xy(origin: list[int], local: tuple[float, float], scale: float) -> tuple[float, float]:
    return (origin[0] + local[0] * scale, origin[1] + local[1] * scale)


def component_transform(params: dict, name: str, fallback: str | None = None) -> dict[str, float]:
    transforms = params.get("component_transforms", {})
    return transforms.get(
        name,
        transforms.get(
            fallback,
            {"mode": "identity", "dx": 0.0, "dy": 0.0, "sx": 1.0, "sy": 1.0},
        ),
    )


def component_scale(scale: float, transform: dict[str, float]) -> tuple[float, float]:
    return (scale * float(transform["sx"]), scale * float(transform["sy"]))


def transformed_xy(
    origin: list[int],
    local: tuple[float, float],
    scale: float,
    transform: dict[str, float],
) -> tuple[float, float]:
    return (
        origin[0] + (local[0] + float(transform["dx"])) * scale,
        origin[1] + (local[1] + float(transform["dy"])) * scale,
    )


def transformed_group_xy(
    origin: list[int],
    base: tuple[float, float],
    offset: tuple[float, float],
    scale: float,
    transform: dict[str, float],
) -> tuple[float, float]:
    return (
        origin[0] + (base[0] + float(transform["dx"]) + offset[0] * float(transform["sx"])) * scale,
        origin[1] + (base[1] + float(transform["dy"]) + offset[1] * float(transform["sy"])) * scale,
    )


def erase_control(
    renderer: Renderer,
    origin: list[int],
    rect: tuple[float, float, float, float],
    scale: float,
) -> None:
    x, y, w, h = rect
    renderer.erase_rect_with_edge_average((*scaled_xy(origin, (x, y), scale), w, h), scale)


def abs_rect(dest: tuple[float, float], src: tuple[int, int, int, int], scale: float | tuple[float, float]) -> tuple[float, float, float, float]:
    _, _, w, h = src
    scale_x, scale_y = scale_pair(scale)
    return (dest[0], dest[1], dest[0] + w * scale_x, dest[1] + h * scale_y)


def union_abs_rects(rects: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def draw_main_histogram(renderer: Renderer, origin: list[int], scale: float, values: list[float]) -> None:
    values = values or [0.0] * 16
    area_x, area_y, area_w, area_h = 27, 43, 70, 16
    renderer.fill_rect((*scaled_xy(origin, (area_x, area_y), scale), area_w * scale, area_h * scale), (0, 0, 0, 255))
    bar_w = area_w / len(values)
    for idx, value in enumerate(values):
        height = max(1.0, min(area_h - 2, value * (area_h - 2)))
        x = area_x + idx * bar_w + 0.75
        y = area_y + area_h - height - 1
        color = (44, 196, 184, 255) if idx % 3 else (226, 126, 39, 255)
        renderer.fill_rect((*scaled_xy(origin, (x, y), scale), max(1.0, (bar_w - 1.4) * scale), height * scale), color)


def draw_playlist_entries(
    renderer: Renderer,
    origin: list[int],
    scale: float,
    entries: list[str],
    selected_row: int,
    list_h: int,
) -> None:
    list_w = 243
    layer = Image.new("RGBA", (list_w, list_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default()
    for row, entry in enumerate(entries[: list_h // 11]):
        color = (232, 238, 220, 255) if row == selected_row else (93, 184, 207, 255)
        draw.text((4, 2 + row * 11), entry[:38], font=font, fill=color)
    resized = layer.resize((max(1, round(list_w * scale)), max(1, round(list_h * scale))), Image.Resampling.NEAREST)
    renderer.composite_procedural(resized, (round(origin[0] + 12 * scale), round(origin[1] + 20 * scale)))


def render_main(renderer: Renderer, params: dict) -> None:
    scale = params["scale"]
    origin = params["windows"]["main"]
    state = params["state"]
    ox, oy = origin

    renderer.blit("MAIN", "MAIN.bmp", (0, 0, 275, 115), (ox, oy), scale, 0)
    renderer.mark_rect(1, (ox, oy, 275, 14), scale)
    renderer.blit("TITLEBAR", "TITLEBAR.bmp", (27, 0, 275, 14), (ox, oy), scale)
    renderer.blit("TITLEBAR", "TITLEBAR.bmp", (0, 0, 9, 9), scaled_xy(origin, (6, 3), scale), scale)
    renderer.blit("TITLEBAR", "TITLEBAR.bmp", (9, 0, 9, 9), scaled_xy(origin, (244, 3), scale), scale)
    renderer.blit("TITLEBAR", "TITLEBAR.bmp", (0, 18, 9, 9), scaled_xy(origin, (254, 3), scale), scale)
    renderer.blit("TITLEBAR", "TITLEBAR.bmp", (18, 0, 9, 9), scaled_xy(origin, (264, 3), scale), scale)
    renderer.mark_rect(23, (*scaled_xy(origin, (244, 3), scale), 29, 9), scale)

    renderer.mark_rect(2, (*scaled_xy(origin, (24, 26), scale), 77, 38), scale)
    renderer.mark_rect(3, (*scaled_xy(origin, (111, 27), scale), 150, 12), scale)
    renderer.mark_rect(4, (*scaled_xy(origin, (24, 43), scale), 76, 16), scale)
    draw_main_histogram(renderer, origin, scale, state.get("histogram", [0.0] * 16))
    for rect in [
        (26, 28, 9, 9),
        (212, 41, 56, 12),
        (17, 72, 248, 10),
        (16, 88, 23, 18),
        (39, 88, 23, 18),
        (62, 88, 23, 18),
        (85, 88, 23, 18),
        (108, 88, 22, 18),
        (136, 89, 22, 16),
        (107, 57, 68, 13),
        (177, 57, 38, 13),
        (164, 89, 47, 15),
        (210, 89, 28, 15),
        (219, 58, 23, 12),
        (242, 58, 23, 12),
    ]:
        erase_control(renderer, origin, rect, scale)

    playback = state["playback"]
    status_src = {"playing": (0, 0, 9, 9), "paused": (9, 0, 9, 9), "stopped": (18, 0, 9, 9)}[playback]
    status_t = component_transform(params, "playback_indicator")
    renderer.blit(
        "PLAYPAUS",
        "PLAYPAUS.bmp",
        status_src,
        transformed_xy(origin, (26, 28), scale, status_t),
        component_scale(scale, status_t),
        22,
    )
    mono_t = component_transform(params, "mono_stereo")
    renderer.blit(
        "MONOSTER",
        "MONOSTER.bmp",
        (0, 0, 56, 12),
        transformed_xy(origin, (212, 41), scale, mono_t),
        component_scale(scale, mono_t),
        21,
    )

    posbar_t = component_transform(params, "posbar")
    posbar_base = (17, 72)
    posbar_scale = component_scale(scale, posbar_t)
    posbar_dest = transformed_xy(origin, posbar_base, scale, posbar_t)
    renderer.blit(
        "POSBAR",
        "POSBAR.bmp",
        (0, 0, 248, 10),
        posbar_dest,
        posbar_scale,
        5,
    )
    renderer.fill_rect(
        (
            posbar_dest[0] + 5 * posbar_scale[0],
            posbar_dest[1] + 4 * posbar_scale[1],
            max(1.0, (state["posbar"] * 224) * posbar_scale[0]),
            max(1.0, 2 * posbar_scale[1]),
        ),
        (226, 126, 39, 255),
    )
    pos_thumb_x = 17 + state["posbar"] * (248 - 29)
    renderer.blit(
        "POSBAR",
        "POSBAR.bmp",
        (248, 0, 29, 10),
        transformed_group_xy(origin, posbar_base, (pos_thumb_x - 17, 0), scale, posbar_t),
        posbar_scale,
    )

    pressed = state["pressed_transport_button"]
    buttons = [
        (7, 0, 0, 23, 18, 16, 88),
        (8, 23, 0, 23, 18, 39, 88),
        (9, 46, 0, 23, 18, 62, 88),
        (10, 69, 0, 23, 18, 85, 88),
        (11, 92, 0, 22, 18, 108, 88),
        (12, 114, 0, 22, 16, 136, 89),
    ]
    transport_names = ["prev", "play", "pause", "stop", "next", "eject"]
    transport_rects = []
    for (idx, sx, sy, sw, sh, dx, dy), name in zip(buttons, transport_names):
        src_y = sy + (18 if pressed == idx - 7 and idx != 12 else 0)
        if idx == 12 and pressed == 5:
            src_y = 16
        transport_t = component_transform(params, f"transport_{name}", "transport")
        transport_scale = component_scale(scale, transport_t)
        dest = transformed_xy(origin, (dx, dy), scale, transport_t)
        renderer.blit(
            "CBUTTONS",
            "CBUTTONS.bmp",
            (sx, src_y, sw, sh),
            dest,
            transport_scale,
            idx,
        )
        transport_rects.append(abs_rect(dest, (sx, src_y, sw, sh), transport_scale))
    renderer.mark_abs_rect(6, union_abs_rects(transport_rects))

    volume_frame = min(27, max(0, round(state["volume"] * 27)))
    volume_t = component_transform(params, "volume")
    volume_base = (107, 57)
    renderer.blit(
        "VOLUME",
        "VOLUME.bmp",
        (0, volume_frame * 15, 68, 13),
        transformed_xy(origin, volume_base, scale, volume_t),
        component_scale(scale, volume_t),
        13,
    )
    renderer.blit(
        "VOLUME",
        "VOLUME.bmp",
        (15, 422, 14, 11),
        transformed_group_xy(origin, volume_base, (state["volume"] * 54, 1), scale, volume_t),
        component_scale(scale, volume_t),
        14,
    )

    balance_frame = min(27, max(0, round(state["balance"] * 27)))
    balance_t = component_transform(params, "balance")
    balance_base = (177, 57)
    renderer.blit(
        "BALANCE",
        "BALANCE.bmp",
        (9, balance_frame * 15, 38, 13),
        transformed_xy(origin, balance_base, scale, balance_t),
        component_scale(scale, balance_t),
        15,
    )
    renderer.blit(
        "BALANCE",
        "BALANCE.bmp",
        (15, 422, 14, 11),
        transformed_group_xy(origin, balance_base, (state["balance"] * 24, 1), scale, balance_t),
        component_scale(scale, balance_t),
        16,
    )

    shuffle_src = (28, 30 if state["shuffle"] else 0, 47, 15)
    repeat_src = (0, 30 if state["repeat"] else 0, 28, 15)
    eq_src = (0, 73 if state["eq_on"] else 61, 23, 12)
    pl_src = (23, 73, 23, 12)
    shuffle_t = component_transform(params, "shuffle", "shufrep")
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        shuffle_src,
        transformed_xy(origin, (164, 89), scale, shuffle_t),
        component_scale(scale, shuffle_t),
        17,
    )
    repeat_t = component_transform(params, "repeat", "shufrep")
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        repeat_src,
        transformed_xy(origin, (210, 89), scale, repeat_t),
        component_scale(scale, repeat_t),
        18,
    )
    eq_toggle_t = component_transform(params, "eq_toggle", "shufrep")
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        eq_src,
        transformed_xy(origin, (219, 58), scale, eq_toggle_t),
        component_scale(scale, eq_toggle_t),
        19,
    )
    pl_toggle_t = component_transform(params, "pl_toggle", "shufrep")
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        pl_src,
        transformed_xy(origin, (242, 58), scale, pl_toggle_t),
        component_scale(scale, pl_toggle_t),
        20,
    )


def render_eq(renderer: Renderer, params: dict) -> None:
    scale = params["scale"]
    origin = params["windows"]["eq"]
    state = params["state"]
    ox, oy = origin

    renderer.blit("EQMAIN", "EQMAIN.bmp", (0, 0, 275, 116), (ox, oy), scale, 24)
    renderer.blit("EQMAIN", "EQMAIN.bmp", (0, 134, 275, 14), (ox, oy), scale, 25)
    renderer.blit("EQMAIN", "EQMAIN.bmp", (0, 116, 9, 9), scaled_xy(origin, (264, 3), scale), scale, 41)
    renderer.blit("EQMAIN", "EQMAIN.bmp", (86, 17, 113, 19), scaled_xy(origin, (86, 17), scale), scale, 26)

    on_src = (69 if state["eq_on"] else 10, 119, 26, 12)
    auto_src = (95 if state["eq_auto"] else 36, 119, 32, 12)
    renderer.blit("EQMAIN", "EQMAIN.bmp", on_src, scaled_xy(origin, (14, 18), scale), scale)
    renderer.blit("EQMAIN", "EQMAIN.bmp", auto_src, scaled_xy(origin, (40, 18), scale), scale)
    renderer.mark_rect(27, (*scaled_xy(origin, (14, 18), scale), 58, 12), scale)
    renderer.blit("EQMAIN", "EQMAIN.bmp", (224, 164, 44, 12), scaled_xy(origin, (217, 18), scale), scale, 40)

    slider_xs = [21, 78, 96, 114, 132, 150, 168, 186, 204, 222, 240]
    thumb_xs = [22, 79, 97, 115, 133, 151, 169, 187, 205, 223, 241]
    values = state["eq_values"]
    eq_slider_t = component_transform(params, "eq_sliders")
    for slider_x in slider_xs:
        erase_control(renderer, origin, (slider_x, 38, 14, 63), scale)
    for idx, value in enumerate(values):
        frame = min(27, max(0, round(value * 27)))
        src_x = 13 + (frame % 14) * 15
        src_y = 164 if frame < 14 else 229
        comp = 28 if idx == 0 else None
        renderer.blit(
            "EQMAIN",
            "EQMAIN.bmp",
            (src_x, src_y, 14, 63),
            transformed_xy(origin, (slider_xs[idx], 38), scale, eq_slider_t),
            component_scale(scale, eq_slider_t),
            comp,
        )
        thumb_y = 38 + value * (63 - 11)
        thumb_dest = (
            origin[0] + (thumb_xs[idx] + float(eq_slider_t["dx"])) * scale,
            origin[1] + (38 + float(eq_slider_t["dy"]) + (thumb_y - 38) * float(eq_slider_t["sy"])) * scale,
        )
        renderer.blit(
            "EQMAIN",
            "EQMAIN.bmp",
            (0, 164, 11, 11),
            thumb_dest,
            component_scale(scale, eq_slider_t),
        )

    group = (*transformed_xy(origin, (78, 38), scale, eq_slider_t), 176, 63)
    renderer.mark_rect(29, group, (scale, scale * float(eq_slider_t["sy"])))


def render_playlist(renderer: Renderer, params: dict) -> None:
    scale = params["scale"]
    origin = params["windows"]["playlist"]
    state = params["state"]
    ox, oy = origin
    width = 275
    height = 261
    bottom_y = height - 38
    right_x = width - 20
    list_h = bottom_y - 20

    renderer.mark_rect(42, (ox, oy, width, height), scale)
    renderer.fill_rect((ox, oy, width * scale, height * scale), (5, 5, 5, 255))
    renderer.fill_rect((*scaled_xy(origin, (12, 20), scale), 243 * scale, list_h * scale), (0, 0, 0, 255))

    renderer.blit("PLEDIT", "PLEDIT.bmp", (0, 21, 25, 20), (ox, oy), scale)
    for tx in [25, 50, 75, 175, 200, 225]:
        renderer.blit("PLEDIT", "PLEDIT.bmp", (127, 21, 25, 20), scaled_xy(origin, (tx, 0), scale), scale)
    renderer.blit("PLEDIT", "PLEDIT.bmp", (26, 21, 100, 20), scaled_xy(origin, (87, 0), scale), scale, 43)
    renderer.blit("PLEDIT", "PLEDIT.bmp", (153, 21, 25, 20), scaled_xy(origin, (250, 0), scale), scale)

    for y in range(20, bottom_y, 29):
        h = min(29, bottom_y - y)
        renderer.blit("PLEDIT", "PLEDIT.bmp", (0, 42, 12, h), scaled_xy(origin, (0, y), scale), scale)
        renderer.blit("PLEDIT", "PLEDIT.bmp", (31, 42, 20, h), scaled_xy(origin, (right_x, y), scale), scale)

    renderer.blit("PLEDIT", "PLEDIT.bmp", (0, 72, 125, 38), scaled_xy(origin, (0, bottom_y), scale), scale)
    renderer.blit("PLEDIT", "PLEDIT.bmp", (126, 72, 150, 38), scaled_xy(origin, (125, bottom_y), scale), scale)

    selected_y = 22 + min(17, int(state["playlist_selected_row"])) * 11
    renderer.fill_rect((*scaled_xy(origin, (12, selected_y), scale), 243 * scale, 9 * scale), (66, 53, 30, 255))
    draw_playlist_entries(
        renderer,
        origin,
        scale,
        params.get("playlist_entries", []),
        min(17, int(state["playlist_selected_row"])),
        list_h,
    )
    renderer.mark_rect(44, (*scaled_xy(origin, (12, 20), scale), 243, list_h), scale)
    renderer.mark_rect(45, (*scaled_xy(origin, (12, selected_y), scale), 243, 9), scale)
    scroll_t = component_transform(params, "playlist_scrollbar")
    scroll_base = (260, 20)
    renderer.mark_rect(46, (*transformed_xy(origin, scroll_base, scale, scroll_t), 8, list_h), component_scale(scale, scroll_t))
    thumb_y = 20 + state["playlist_scroll"] * max(1, list_h - 18)
    erase_control(renderer, origin, (260, thumb_y, 8, 18), scale)
    renderer.blit(
        "PLEDIT",
        "PLEDIT.bmp",
        (52, 53, 8, 18),
        transformed_group_xy(origin, scroll_base, (0, thumb_y - 20), scale, scroll_t),
        component_scale(scale, scroll_t),
        47,
    )
    renderer.mark_rect(48, (*scaled_xy(origin, (0, bottom_y), scale), 275, 38), scale)
    footer_rects = {
        49: (10, bottom_y + 7, 28, 18),
        50: (39, bottom_y + 7, 28, 18),
        51: (69, bottom_y + 7, 28, 18),
        52: (99, bottom_y + 7, 36, 18),
        53: (228, bottom_y + 7, 28, 18),
        54: (139, bottom_y + 25, 58, 8),
        55: (132, bottom_y + 10, 89, 23),
        56: (255, bottom_y + 18, 20, 20),
        57: (250, 0, 25, 20),
        58: (250, 0, 25, 20),
        59: (250, 0, 25, 20),
    }
    for comp_id, rect in footer_rects.items():
        renderer.mark_rect(comp_id, (*scaled_xy(origin, (rect[0], rect[1]), scale), rect[2], rect[3]), scale)


def _render_window_pass(
    renderer: Renderer,
    render_fn,
    params: dict,
    window_name: str,
    final_origin: tuple[int, int],
    window_sy: float,
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Render one window onto a transparent sub-canvas, then composite it."""
    sub_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    saved_canvas = renderer.canvas

    sub_params = {**params, "windows": {**params["windows"], window_name: [0, 0]}}
    renderer.canvas = sub_canvas
    render_fn(renderer, sub_params)
    renderer.canvas = saved_canvas

    if abs(window_sy - 1.0) > 1e-6:
        stretched_h = max(1, int(round(canvas_h * window_sy)))
        stretched = sub_canvas.resize((canvas_w, stretched_h), Image.Resampling.NEAREST)
    else:
        stretched = sub_canvas

    ox, oy = final_origin
    renderer.canvas.alpha_composite(stretched, (int(ox), int(oy)))


def render_with_params(skin_source: Path, params: dict, canvas_w: int | None = None, canvas_h: int | None = None) -> Renderer:
    canvas_w = canvas_w or int(params.get("canvas_w", CANVAS_DEFAULT[0]))
    canvas_h = canvas_h or int(params.get("canvas_h", CANVAS_DEFAULT[1]))
    renderer = Renderer(skin_source, canvas_w, canvas_h)

    window_scales = params.get("window_scales", {"main": 1.0, "eq": 1.0, "playlist": 1.0})
    windows = params["windows"]
    render_passes = [
        ("main", render_main, tuple(windows["main"]), float(window_scales.get("main", 1.0))),
        ("eq", render_eq, tuple(windows["eq"]), float(window_scales.get("eq", 1.0))),
        ("playlist", render_playlist, tuple(windows["playlist"]), float(window_scales.get("playlist", 1.0))),
    ]
    for name, fn, final_origin, sy in render_passes:
        _render_window_pass(renderer, fn, params, name, final_origin, sy, canvas_w, canvas_h)

    return renderer


def write_outputs(renderer: Renderer, args: argparse.Namespace) -> None:
    renderer.canvas.convert("RGB").save(args.out_view)


def command_dump_classic_spec(args: argparse.Namespace) -> int:
    profile = load_export_profile(EXPORT_PROFILE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def command_render_random(args: argparse.Namespace) -> int:
    params = rand_params(args.seed, args.canvas_w, args.canvas_h)
    renderer = render_with_params(Path(args.skin_dir), params)
    write_outputs(renderer, args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cranamp-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump = subparsers.add_parser("dump-classic-spec")
    dump.add_argument("--out", required=True)
    dump.set_defaults(func=command_dump_classic_spec)

    random_render = subparsers.add_parser("render-random")
    random_render.add_argument("--skin-dir", required=True)
    random_render.add_argument("--seed", required=True, type=int)
    random_render.add_argument("--canvas-w", required=True, type=int)
    random_render.add_argument("--canvas-h", required=True, type=int)
    random_render.add_argument("--out-view", required=True)
    random_render.set_defaults(func=command_render_random)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for attr in ["out_view", "out"]:
        path = getattr(args, attr, None)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
