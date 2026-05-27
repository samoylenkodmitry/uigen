"""Differentiable Cranamp-lite renderer for V8 product matching.

This renderer intentionally covers the visible three-window product view rather
than full Cranamp behavior. It renders the classic main/EQ/playlist layout from
exported BMP tensors into the normalized mockup layout, so refinement can
optimize `render(predicted_skin) ~= input_mockup`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batch(files: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    batched = {}
    was_batched = True
    for name, tensor in files.items():
        if tensor.dim() == 3:
            was_batched = False
            batched[name] = tensor.unsqueeze(0)
        elif tensor.dim() == 4:
            batched[name] = tensor
        else:
            raise ValueError(f"{name}: expected [3,H,W] or [B,3,H,W], got {tuple(tensor.shape)}")
    return batched, was_batched


def _layout_rect(layout: dict, name: str) -> tuple[int, int, int, int]:
    x, y, w, h = layout["rects"][name]
    return round(float(x)), round(float(y)), round(float(w)), round(float(h))


def _crop(img: torch.Tensor, src: tuple[int, int, int, int]) -> torch.Tensor:
    x, y, w, h = src
    return img[:, :, y : y + h, x : x + w]


def _transparent_mask(patch: torch.Tensor) -> torch.Tensor:
    # Classic Winamp magenta transparency key. Keep this mask constant for
    # refinement; it should act as an export constraint, not an optimized alpha.
    r, g, b = patch[:, 0:1], patch[:, 1:2], patch[:, 2:3]
    return ~((r > 0.98) & (g < 0.02) & (b > 0.98))


def _paste(
    canvas: torch.Tensor,
    patch: torch.Tensor,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    transparent: bool = True,
) -> None:
    if w <= 0 or h <= 0:
        return
    if patch.shape[-2:] != (h, w):
        patch = F.interpolate(patch, size=(h, w), mode="nearest")
    _, _, ch, cw = canvas.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + w), min(ch, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    px0, py0 = x0 - x, y0 - y
    px1, py1 = px0 + (x1 - x0), py0 + (y1 - y0)
    src = patch[:, :, py0:py1, px0:px1]
    dst = canvas[:, :, y0:y1, x0:x1]
    if transparent:
        mask = _transparent_mask(src).to(src.dtype)
        canvas[:, :, y0:y1, x0:x1] = src * mask + dst * (1.0 - mask)
    else:
        canvas[:, :, y0:y1, x0:x1] = src


def _paste_src(
    canvas: torch.Tensor,
    files: dict[str, torch.Tensor],
    file_name: str,
    src: tuple[int, int, int, int],
    dest: tuple[float, float],
    scale: tuple[float, float],
    *,
    transparent: bool = True,
) -> None:
    sx, sy, sw, sh = src
    patch = _crop(files[file_name], (sx, sy, sw, sh))
    _paste(
        canvas,
        patch,
        x=round(dest[0]),
        y=round(dest[1]),
        w=max(1, round(sw * scale[0])),
        h=max(1, round(sh * scale[1])),
        transparent=transparent,
    )


def _main(files: dict[str, torch.Tensor], canvas: torch.Tensor, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    sx, sy = w / 275.0, h / 116.0
    scale = (sx, sy)
    _paste_src(canvas, files, "MAIN.bmp", (0, 0, 275, 115), (x, y), scale, transparent=False)
    _paste_src(canvas, files, "TITLEBAR.bmp", (27, 0, 275, 14), (x, y), scale)
    for src, local in [
        ((0, 0, 9, 9), (6, 3)),
        ((9, 0, 9, 9), (244, 3)),
        ((0, 18, 9, 9), (254, 3)),
        ((18, 0, 9, 9), (264, 3)),
    ]:
        _paste_src(canvas, files, "TITLEBAR.bmp", src, (x + local[0] * sx, y + local[1] * sy), scale)
    _paste_src(canvas, files, "PLAYPAUS.bmp", (0, 0, 9, 9), (x + 26 * sx, y + 28 * sy), scale)
    _paste_src(canvas, files, "MONOSTER.bmp", (0, 0, 56, 12), (x + 212 * sx, y + 41 * sy), scale)
    _paste_src(canvas, files, "POSBAR.bmp", (0, 0, 248, 10), (x + 17 * sx, y + 72 * sy), scale)
    _paste_src(canvas, files, "POSBAR.bmp", (248, 0, 29, 10), (x + 126 * sx, y + 72 * sy), scale)
    buttons = [
        ((0, 0, 23, 18), (16, 88)),
        ((23, 0, 23, 18), (39, 88)),
        ((46, 0, 23, 18), (62, 88)),
        ((69, 0, 23, 18), (85, 88)),
        ((92, 0, 22, 18), (108, 88)),
        ((114, 0, 22, 16), (136, 89)),
    ]
    for src, local in buttons:
        _paste_src(canvas, files, "CBUTTONS.bmp", src, (x + local[0] * sx, y + local[1] * sy), scale)
    _paste_src(canvas, files, "VOLUME.bmp", (0, 195, 68, 13), (x + 107 * sx, y + 57 * sy), scale)
    _paste_src(canvas, files, "BALANCE.bmp", (9, 195, 38, 13), (x + 177 * sx, y + 57 * sy), scale)
    _paste_src(canvas, files, "SHUFREP.bmp", (28, 0, 47, 15), (x + 164 * sx, y + 89 * sy), scale)
    _paste_src(canvas, files, "SHUFREP.bmp", (0, 0, 28, 15), (x + 210 * sx, y + 89 * sy), scale)
    _paste_src(canvas, files, "SHUFREP.bmp", (0, 61, 23, 12), (x + 219 * sx, y + 58 * sy), scale)
    _paste_src(canvas, files, "SHUFREP.bmp", (23, 73, 23, 12), (x + 242 * sx, y + 58 * sy), scale)


def _eq(files: dict[str, torch.Tensor], canvas: torch.Tensor, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    sx, sy = w / 275.0, h / 116.0
    scale = (sx, sy)
    _paste_src(canvas, files, "EQMAIN.bmp", (0, 0, 275, 116), (x, y), scale, transparent=False)
    _paste_src(canvas, files, "EQMAIN.bmp", (0, 134, 275, 14), (x, y), scale)
    _paste_src(canvas, files, "EQMAIN.bmp", (69, 119, 26, 12), (x + 14 * sx, y + 18 * sy), scale)
    _paste_src(canvas, files, "EQMAIN.bmp", (95, 119, 32, 12), (x + 40 * sx, y + 18 * sy), scale)
    # Proxy renderer (optimization only; real Cranamp is the product gate). Mirror
    # the real engine's EQ slider model: a thumb-free groove frame + a separate
    # thumb sprite EQMAIN(0,164,11,11) on top. The compiler writes the groove to
    # the frame grid and the skin thumb to (0,164). The proxy draws all thumbs at
    # a uniform mid height (it has no per-band EQ state).
    slider_xs = [21, 78, 96, 114, 132, 150, 168, 186, 204, 222, 240]
    thumb_xs = [22, 79, 97, 115, 133, 151, 169, 187, 205, 223, 241]
    thumb_y = 38 + round(0.5 * (63 - 11))
    for lx in slider_xs:
        _paste_src(canvas, files, "EQMAIN.bmp", (13, 164, 14, 63), (x + lx * sx, y + 38 * sy), scale)
    for tx in thumb_xs:
        _paste_src(canvas, files, "EQMAIN.bmp", (0, 164, 11, 11), (x + tx * sx, y + thumb_y * sy), scale)


def _playlist(files: dict[str, torch.Tensor], canvas: torch.Tensor, rect: tuple[int, int, int, int]) -> None:
    # Proxy: compose the playlist chrome from the same PLEDIT sub-sprites the
    # real engine samples (titlebar left/fill/title/right, L/R borders, footer),
    # mirroring render_playlist. The list body (dynamic text) is left as the dark
    # canvas. Product gate is real Cranamp; this keeps the proxy approximately
    # faithful now that PLEDIT.bmp holds sub-sprites, not a whole-window dump.
    x, y, w, h = rect
    sx, sy = w / 275.0, h / 261.0
    sc = (sx, sy)
    bottom_y, right_x = 223, 255
    _paste_src(canvas, files, "PLEDIT.bmp", (0, 21, 25, 20), (x, y), sc, transparent=False)
    for tx in (25, 50, 75, 175, 200, 225):
        _paste_src(canvas, files, "PLEDIT.bmp", (127, 21, 25, 20), (x + tx * sx, y), sc, transparent=False)
    _paste_src(canvas, files, "PLEDIT.bmp", (26, 21, 100, 20), (x + 87 * sx, y), sc, transparent=False)
    _paste_src(canvas, files, "PLEDIT.bmp", (153, 21, 25, 20), (x + 250 * sx, y), sc, transparent=False)
    for ly in range(20, bottom_y, 29):
        hh = min(29, bottom_y - ly)
        _paste_src(canvas, files, "PLEDIT.bmp", (0, 42, 12, hh), (x, y + ly * sy), sc, transparent=False)
        _paste_src(canvas, files, "PLEDIT.bmp", (31, 42, 20, hh), (x + right_x * sx, y + ly * sy), sc, transparent=False)
    _paste_src(canvas, files, "PLEDIT.bmp", (0, 72, 125, 38), (x, y + bottom_y * sy), sc, transparent=False)
    _paste_src(canvas, files, "PLEDIT.bmp", (126, 72, 150, 38), (x + 125 * sx, y + bottom_y * sy), sc, transparent=False)


def render_visible(files: dict[str, torch.Tensor], layout: dict, *, background: float = 0.03) -> torch.Tensor:
    """Render exported BMP tensors into the normalized V8 layout."""
    bfiles, was_batched = _as_batch(files)
    sample = next(iter(bfiles.values()))
    width, height = [int(v) for v in layout["normalized_size"]]
    canvas = sample.new_full((sample.shape[0], 3, height, width), float(background))
    _main(bfiles, canvas, _layout_rect(layout, "main"))
    _eq(bfiles, canvas, _layout_rect(layout, "eq"))
    _playlist(bfiles, canvas, _layout_rect(layout, "playlist"))
    return canvas if was_batched else canvas[0]


__all__ = ["render_visible"]
