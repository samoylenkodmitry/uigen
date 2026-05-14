#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.profiles import load_atlas_profile, load_export_profile
from atlas_ai.rects import derive_eq_band_rects, encode_rect
from atlas_ai.skins import load_default_assets, load_skin_assets, normalize_name


CANVAS_DEFAULT = (768, 1280)
ATLAS_PROFILE = REPO_ROOT / "configs/atlas_v1.json"
EXPORT_PROFILE = REPO_ROOT / "configs/export_profile_classic.json"
DEFAULT_SKIN = REPO_ROOT / "assets/default_skin"


def scale_pair(scale: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(scale, tuple):
        return scale
    return (scale, scale)


class Renderer:
    def __init__(self, skin_source: Path, canvas_w: int, canvas_h: int):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.atlas_profile = load_atlas_profile(ATLAS_PROFILE)
        self.slots = self.atlas_profile.slots_by_name
        self.assets = load_skin_assets(skin_source)
        self.default_assets = load_default_assets(DEFAULT_SKIN)
        self.images: dict[str, Image.Image] = {}
        self.canvas = Image.new("RGBA", (canvas_w, canvas_h), (14, 14, 18, 255))
        self.visible_mask = Image.new("L", (self.atlas_profile.canvas_w, self.atlas_profile.canvas_h), 0)
        self.rects = np.zeros((80, 5), dtype="<f4")
        self.state = np.zeros((32,), dtype="<f4")

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
        x, y, w, h = rect
        scale_x, scale_y = scale_pair(scale)
        self.rects[component_id] = encode_rect(
            x,
            y,
            x + w * scale_x,
            y + h * scale_y,
            canvas_w=self.canvas_w,
            canvas_h=self.canvas_h,
        )

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
        if scale_x != 1.0 or scale_y != 1.0:
            crop = crop.resize(
                (max(1, round(sw * scale_x)), max(1, round(sh * scale_y))),
                Image.Resampling.NEAREST,
            )
        self.canvas.alpha_composite(crop, (round(dx), round(dy)))

        slot = self.slots[slot_name]
        alpha = np.array(source.crop((sx, sy, sx + sw, sy + sh)).getchannel("A"))
        mask = np.array(self.visible_mask)
        visible = alpha > 0
        if visible.any():
            y0 = slot.y + sy
            x0 = slot.x + sx
            mask[y0 : y0 + sh, x0 : x0 + sw] = np.where(
                visible,
                255,
                mask[y0 : y0 + sh, x0 : x0 + sw],
            )
            self.visible_mask = Image.fromarray(mask, "L")

        if component_id is not None:
            self.mark_rect(component_id, (dx, dy, sw, sh), scale)

    def fill_rect(self, rect: tuple[float, float, float, float], color: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        ImageDraw.Draw(self.canvas).rectangle(
            [round(x), round(y), round(x + w), round(y + h)],
            fill=color,
        )


def rand_params(seed: int, canvas_w: int, canvas_h: int, state_balanced: bool) -> dict:
    rng = random.Random(seed)
    stack_units_h = 116 + 116 + 261
    fit_scale = min((canvas_w - 32) / 275, (canvas_h - 32) / stack_units_h)
    max_scale = min(2.35, fit_scale)
    scale = round(rng.uniform(2.0, max_scale), 4) if max_scale >= 2.0 else round(max_scale, 4)
    total_h = int(stack_units_h * scale)
    main_w = int(275 * scale)
    x_space = max(0, canvas_w - main_w)
    y_space = max(0, canvas_h - total_h)
    main_x = rng.randint(0, x_space) if x_space else 0
    main_y = rng.randint(0, y_space) if y_space else 0

    pressed_options = [-1, 0, 1, 2, 3, 4, 5]
    pressed = pressed_options[seed % len(pressed_options)] if state_balanced else rng.choice([-1, -1, 0, 1, 2, 3, 4, 5])
    volume = (seed % 28) / 27.0 if state_balanced else rng.random()
    balance = ((seed // 3) % 28) / 27.0 if state_balanced else rng.random()
    posbar = ((seed // 5) % 20) / 19.0 if state_balanced else rng.random()

    def transform(dx: int, dy: int, sx: float, sy: float) -> dict[str, float]:
        return {
            "dx": rng.randint(-dx, dx),
            "dy": rng.randint(-dy, dy),
            "sx": round(rng.uniform(1.0 - sx, 1.0 + sx), 3),
            "sy": round(rng.uniform(1.0 - sy, 1.0 + sy), 3),
        }

    return {
        "schema": "cranamp_cli_renderer_v2",
        "seed": seed,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "scale": scale,
        "windows": {
            "main": [main_x, main_y],
            "eq": [main_x, main_y + int(116 * scale)],
            "playlist": [main_x, main_y + int((116 + 116) * scale)],
        },
        "component_transforms": {
            "playback_indicator": transform(7, 5, 0.12, 0.14),
            "mono_stereo": transform(8, 5, 0.12, 0.14),
            "posbar": transform(12, 7, 0.12, 0.18),
            "transport": transform(12, 9, 0.14, 0.16),
            "volume": transform(8, 6, 0.13, 0.16),
            "balance": transform(8, 6, 0.13, 0.16),
            "shufrep": transform(10, 8, 0.14, 0.16),
            "eq_sliders": transform(9, 8, 0.10, 0.14),
            "playlist_scrollbar": transform(4, 10, 0.08, 0.12),
        },
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
        },
    }


def scaled_xy(origin: list[int], local: tuple[float, float], scale: float) -> tuple[float, float]:
    return (origin[0] + local[0] * scale, origin[1] + local[1] * scale)


def component_transform(params: dict, name: str) -> dict[str, float]:
    return params.get("component_transforms", {}).get(
        name,
        {"dx": 0.0, "dy": 0.0, "sx": 1.0, "sy": 1.0},
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
    renderer.blit(
        "POSBAR",
        "POSBAR.bmp",
        (0, 0, 248, 10),
        transformed_xy(origin, posbar_base, scale, posbar_t),
        component_scale(scale, posbar_t),
        5,
    )
    pos_thumb_x = 17 + state["posbar"] * (248 - 29)
    renderer.blit(
        "POSBAR",
        "POSBAR.bmp",
        (248, 0, 29, 10),
        transformed_group_xy(origin, posbar_base, (pos_thumb_x - 17, 0), scale, posbar_t),
        component_scale(scale, posbar_t),
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
    transport_t = component_transform(params, "transport")
    transport_base = (16, 88)
    for idx, sx, sy, sw, sh, dx, dy in buttons:
        src_y = sy + (18 if pressed == idx - 7 and idx != 12 else 0)
        if idx == 12 and pressed == 5:
            src_y = 16
        renderer.blit(
            "CBUTTONS",
            "CBUTTONS.bmp",
            (sx, src_y, sw, sh),
            transformed_group_xy(origin, transport_base, (dx - 16, dy - 88), scale, transport_t),
            component_scale(scale, transport_t),
            idx,
        )
    renderer.mark_rect(
        6,
        (*transformed_xy(origin, transport_base, scale, transport_t), 142, 18),
        component_scale(scale, transport_t),
    )

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
    shufrep_t = component_transform(params, "shufrep")
    shufrep_base = (164, 58)
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        shuffle_src,
        transformed_group_xy(origin, shufrep_base, (0, 31), scale, shufrep_t),
        component_scale(scale, shufrep_t),
        17,
    )
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        repeat_src,
        transformed_group_xy(origin, shufrep_base, (46, 31), scale, shufrep_t),
        component_scale(scale, shufrep_t),
        18,
    )
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        eq_src,
        transformed_group_xy(origin, shufrep_base, (55, 0), scale, shufrep_t),
        component_scale(scale, shufrep_t),
        19,
    )
    renderer.blit(
        "SHUFREP",
        "SHUFREP.bmp",
        pl_src,
        transformed_group_xy(origin, shufrep_base, (78, 0), scale, shufrep_t),
        component_scale(scale, shufrep_t),
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
    eq_slider_base = (21, 38)
    for idx, value in enumerate(values):
        frame = min(27, max(0, round(value * 27)))
        src_x = 13 + (frame % 14) * 15
        src_y = 164 if frame < 14 else 229
        comp = 28 if idx == 0 else None
        renderer.blit(
            "EQMAIN",
            "EQMAIN.bmp",
            (src_x, src_y, 14, 63),
            transformed_group_xy(origin, eq_slider_base, (slider_xs[idx] - 21, 0), scale, eq_slider_t),
            component_scale(scale, eq_slider_t),
            comp,
        )
        thumb_y = 38 + value * (63 - 11)
        renderer.blit(
            "EQMAIN",
            "EQMAIN.bmp",
            (0, 164, 11, 11),
            transformed_group_xy(
                origin,
                eq_slider_base,
                (thumb_xs[idx] - 21, thumb_y - 38),
                scale,
                eq_slider_t,
            ),
            component_scale(scale, eq_slider_t),
        )

    group = (*transformed_group_xy(origin, eq_slider_base, (78 - 21, 0), scale, eq_slider_t), 176, 63)
    renderer.mark_rect(29, group, component_scale(scale, eq_slider_t))
    for offset, rect in enumerate(derive_eq_band_rects(tuple(renderer.rects[29]))):
        renderer.rects[30 + offset] = rect


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
    renderer.mark_rect(44, (*scaled_xy(origin, (12, 20), scale), 243, list_h), scale)
    renderer.mark_rect(45, (*scaled_xy(origin, (12, selected_y), scale), 243, 9), scale)
    scroll_t = component_transform(params, "playlist_scrollbar")
    scroll_base = (260, 20)
    renderer.mark_rect(46, (*transformed_xy(origin, scroll_base, scale, scroll_t), 8, list_h), component_scale(scale, scroll_t))
    thumb_y = 20 + state["playlist_scroll"] * max(1, list_h - 18)
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


def render_with_params(skin_source: Path, params: dict, canvas_w: int | None = None, canvas_h: int | None = None) -> Renderer:
    canvas_w = canvas_w or int(params.get("canvas_w", CANVAS_DEFAULT[0]))
    canvas_h = canvas_h or int(params.get("canvas_h", CANVAS_DEFAULT[1]))
    renderer = Renderer(skin_source, canvas_w, canvas_h)
    render_main(renderer, params)
    render_eq(renderer, params)
    render_playlist(renderer, params)

    state = params["state"]
    renderer.state[0] = (state["pressed_transport_button"] + 1) / 6.0
    renderer.state[1] = state["volume"]
    renderer.state[2] = state["balance"]
    renderer.state[3] = state["posbar"]
    renderer.state[4] = float(state["shuffle"])
    renderer.state[5] = float(state["repeat"])
    renderer.state[6] = float(state["eq_on"])
    renderer.state[7] = float(state["eq_auto"])
    renderer.state[8:19] = np.array(state["eq_values"], dtype="<f4")
    renderer.state[19] = state["playlist_scroll"]
    renderer.state[20] = state["playlist_selected_row"] / 17.0
    renderer.state[21:27] = params["scale"]
    return renderer


def write_outputs(renderer: Renderer, args: argparse.Namespace, params: dict | None = None) -> None:
    renderer.canvas.convert("RGB").save(args.out_view)
    if getattr(args, "out_rects", None):
        renderer.rects.astype("<f4").tofile(args.out_rects)
    if getattr(args, "out_state", None):
        renderer.state.astype("<f4").tofile(args.out_state)
    if getattr(args, "out_visible_atlas_mask", None):
        renderer.visible_mask.save(args.out_visible_atlas_mask)
    if getattr(args, "out_params", None) and params is not None:
        Path(args.out_params).write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_dump_classic_spec(args: argparse.Namespace) -> int:
    profile = load_export_profile(EXPORT_PROFILE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def command_render_random(args: argparse.Namespace) -> int:
    params = rand_params(args.seed, args.canvas_w, args.canvas_h, args.state_balanced.lower() == "true")
    renderer = render_with_params(Path(args.skin_dir), params)
    write_outputs(renderer, args, params)
    return 0


def command_render_with_params(args: argparse.Namespace) -> int:
    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    renderer = render_with_params(Path(args.skin_dir), params, args.canvas_w, args.canvas_h)
    write_outputs(renderer, args, None)
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
    random_render.add_argument("--out-rects", required=True)
    random_render.add_argument("--out-state", required=True)
    random_render.add_argument("--out-visible-atlas-mask", required=True)
    random_render.add_argument("--out-params", required=True)
    random_render.add_argument("--state-balanced", default="false", choices=["true", "false"])
    random_render.set_defaults(func=command_render_random)

    replay = subparsers.add_parser("render-with-params")
    replay.add_argument("--skin-dir", required=True)
    replay.add_argument("--params", required=True)
    replay.add_argument("--canvas-w", required=True, type=int)
    replay.add_argument("--canvas-h", required=True, type=int)
    replay.add_argument("--out-view", required=True)
    replay.add_argument("--out-rects")
    replay.add_argument("--out-state")
    replay.add_argument("--out-visible-atlas-mask")
    replay.set_defaults(func=command_render_with_params)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    for attr in ["out_view", "out_rects", "out_state", "out_visible_atlas_mask", "out_params", "out"]:
        path = getattr(args, attr, None)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
