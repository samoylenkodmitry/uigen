"""Deterministic V8 hidden-state compiler.

A single mockup only shows one transport/slider/toggle state. The V8 product
contract therefore asks for plausible hidden states, not recovery of unknown
historical atlas pixels. This module starts from visible/default exported BMP
tensors and synthesizes the extra classic frames using fixed Winamp geometry.
"""

from __future__ import annotations

import torch

from atlas_ai.v8_assets import load_exported_tensors


def _shift(img: torch.Tensor, dx: int = 1, dy: int = 1) -> torch.Tensor:
    out = img.clone()
    fill = img.mean(dim=(1, 2), keepdim=True)
    out[:] = fill
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    w = img.shape[2] - abs(dx)
    h = img.shape[1] - abs(dy)
    if w > 0 and h > 0:
        out[:, dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = img[
            :, src_y0 : src_y0 + h, src_x0 : src_x0 + w
        ]
    return out


def _pressed(img: torch.Tensor) -> torch.Tensor:
    shifted = _shift(img, dx=1, dy=1)
    return (shifted * 0.82 + 0.04).clamp(0.0, 1.0)


def _glow(img: torch.Tensor) -> torch.Tensor:
    return (img * 1.12 + 0.05).clamp(0.0, 1.0)


def _dim(img: torch.Tensor) -> torch.Tensor:
    return (img * 0.72).clamp(0.0, 1.0)


def _paste(dst: torch.Tensor, src: torch.Tensor, x: int, y: int) -> None:
    h, w = src.shape[-2:]
    dst[:, y : y + h, x : x + w] = src


def _compile_slider(
    file: torch.Tensor,
    *,
    frame_x: int,
    visible_y: int,
    frame_w: int,
    frame_h: int,
    pitch: int,
    count: int,
    thumb: torch.Tensor,
    travel: int,
) -> None:
    base = file[:, visible_y : visible_y + frame_h, frame_x : frame_x + frame_w].clone()
    if base.numel() == 0:
        return
    for idx in range(count):
        frame = base.clone()
        thumb_x = round((idx / max(1, count - 1)) * travel)
        y = idx * pitch
        _paste(frame, thumb, thumb_x, min(1, max(0, frame_h - thumb.shape[1])))
        file[:, y : y + frame_h, frame_x : frame_x + frame_w] = frame


def _compile_cbuttons(files: dict[str, torch.Tensor]) -> None:
    cb = files["CBUTTONS.bmp"]
    specs = [
        (0, 23, 18, 18),
        (23, 23, 18, 18),
        (46, 23, 18, 18),
        (69, 23, 18, 18),
        (92, 22, 18, 18),
        (114, 22, 16, 16),
    ]
    for x, w, h, pressed_y in specs:
        normal = cb[:, 0:h, x : x + w].clone()
        cb[:, pressed_y : pressed_y + h, x : x + w] = _pressed(normal)


def _compile_shufrep(files: dict[str, torch.Tensor]) -> None:
    sh = files["SHUFREP.bmp"]
    repeat_off = sh[:, 0:15, 0:28].clone()
    shuffle_off = sh[:, 0:15, 28:75].clone()
    eq_off = sh[:, 61:73, 0:23].clone()
    sh[:, 30:45, 0:28] = _glow(repeat_off)
    sh[:, 30:45, 28:75] = _glow(shuffle_off)
    sh[:, 73:85, 0:23] = _glow(eq_off)


def _compile_playpaus_monoster(files: dict[str, torch.Tensor]) -> None:
    pp = files["PLAYPAUS.bmp"]
    base = pp[:, 0:9, 0:9].clone()
    pp[:, 0:9, 9:18] = _pressed(base)
    pp[:, 0:9, 18:27] = _dim(base)

    mono = files["MONOSTER.bmp"]
    stereo = mono[:, 0:12, 0:56].clone()
    mono[:, 12:24, 0:56] = _dim(stereo)


_EQ_SLIDER_XS = (21, 78, 96, 114, 132, 150, 168, 186, 204, 222, 240)
_EQ_VIS_Y0, _EQ_FH, _EQ_FW = 38, 63, 14
_EQ_THUMB = 11


def _eq_groove_and_thumb(eq: torch.Tensor):
    """Separate the skin's EQ slider into a thumb-free groove + a thumb sprite.

    The 11 EQ band columns (visible window, local (slider_x, 38, 14, 63)) share
    one groove but each has its thumb at a different height, so the per-pixel
    MEDIAN across the 11 columns is the thumb-free groove (each band's thumb is
    a per-row outlier). The thumb is the max-deviation-from-groove region of the
    band whose thumb sits centrally enough to crop an 11x11 sprite.
    Returns (groove[3,63,14], thumb[3,11,11]) or (None, None).
    """
    cols = []
    for lx in _EQ_SLIDER_XS:
        c = eq[:, _EQ_VIS_Y0 : _EQ_VIS_Y0 + _EQ_FH, lx : lx + _EQ_FW]
        if c.shape[-2:] == (_EQ_FH, _EQ_FW):
            cols.append(c)
    if len(cols) < 3:
        return None, None
    stack = torch.stack(cols, dim=0)                       # [N,3,63,14]
    groove = stack.median(dim=0).values.clone()            # [3,63,14] thumb-free
    dev = (stack - groove.unsqueeze(0)).abs().sum(dim=(1, 3))  # [N,63] per-row deviation
    cx = (_EQ_FW - _EQ_THUMB) // 2
    best = None
    for i in range(stack.shape[0]):
        row = int(torch.argmax(dev[i]).item())
        if 5 <= row <= _EQ_FH - 6:                         # 11-row crop fits
            mag = float(dev[i, row])
            if best is None or mag > best[2]:
                best = (i, row, mag)
    if best is None:
        return groove, None
    i, row, _ = best
    thumb = stack[i][:, row - 5 : row + 6, cx : cx + _EQ_THUMB].clone()  # [3,11,11]
    return groove, thumb


def _compile_eq(files: dict[str, torch.Tensor]) -> None:
    """Source EQ sliders + ON/AUTO from the SKIN's visible window, not defaults.

    The extractor writes the visible EQ window (with the skin's own sliders and
    buttons) into EQMAIN rows 0:116. The renderer samples every EQ slider from
    EQMAIN(208,164,14,63), ON from (69,119,26,12) and AUTO from (95,119,32,12) —
    sprite rows the extractor never touches, so they stayed default classic-
    orange. Copy the skin widgets from where the renderer draws them in the
    window (slider local (78,38,14,63); ON (14,18,26,12); AUTO (40,18,32,12))
    into those sprite rects so the product render shows the skin's EQ widgets.
    """
    eq = files["EQMAIN.bmp"]
    # Write the EXACT sprite rects the real Cranamp engine samples (see
    # cranamp tools/cranamp_cli.py render_eq). The engine draws, per band, a
    # background groove frame EQMAIN(13+(f%14)*15, 164|229, 14,63) indexed by EQ
    # VALUE, then a SEPARATE thumb sprite EQMAIN(0,164,11,11) on top. So we must
    # supply (a) a thumb-FREE groove and (b) a thumb sprite — not a baked
    # composite. Source them from the skin's visible EQ window (rows 0:116),
    # where the engine draws sliders at window-local (slider_x, 38, 14, 63).
    groove, thumb = _eq_groove_and_thumb(eq)
    if groove is not None:
        for f in range(28):
            fx = 13 + (f % 14) * 15
            fy = 164 if f < 14 else 229
            eq[:, fy : fy + 63, fx : fx + 14] = groove
    if thumb is not None:
        eq[:, 164:175, 0:11] = thumb        # EQMAIN(0,164,11,11) slider thumb
    # ON/AUTO buttons (both on/off state rects) + PRESETS, from the visible
    # window. ON drawn from (14,18), AUTO (40,18), PRESETS (217,18) (size 44x12).
    vis_on = eq[:, 18:30, 14:40].clone()        # 26x12
    vis_auto = eq[:, 18:30, 40:72].clone()      # 32x12
    vis_presets = eq[:, 18:30, 217:261].clone()  # 44x12
    if vis_on.shape[-2:] == (12, 26):
        eq[:, 119:131, 69:95] = vis_on          # on state
        eq[:, 119:131, 10:36] = vis_on          # off state (best-effort same art)
    if vis_auto.shape[-2:] == (12, 32):
        eq[:, 119:131, 95:127] = vis_auto       # auto on
        eq[:, 119:131, 36:68] = vis_auto        # auto off
    if vis_presets.shape[-2:] == (12, 44):
        eq[:, 164:176, 224:268] = vis_presets   # EQMAIN(224,164,44,12) PRESETS


def compile_hidden_states(
    files: dict[str, torch.Tensor],
    *,
    default_skin: str = "assets/default_skin",
) -> dict[str, torch.Tensor]:
    """Return a valid full classic BMP tensor set with plausible hidden states."""
    defaults = load_exported_tensors(default_skin, default_skin=default_skin)
    out = {name: tensor.clone() for name, tensor in defaults.items()}
    for name, tensor in files.items():
        out[name] = tensor.clone()

    _compile_cbuttons(out)
    _compile_shufrep(out)
    _compile_playpaus_monoster(out)

    vol_thumb = out["VOLUME.bmp"][:, 422:433, 15:29].clone()
    _compile_slider(
        out["VOLUME.bmp"],
        frame_x=0,
        visible_y=195,
        frame_w=68,
        frame_h=13,
        pitch=15,
        count=28,
        thumb=vol_thumb,
        travel=54,
    )
    bal_thumb = out["BALANCE.bmp"][:, 422:433, 15:29].clone()
    _compile_slider(
        out["BALANCE.bmp"],
        frame_x=9,
        visible_y=195,
        frame_w=38,
        frame_h=13,
        pitch=15,
        count=28,
        thumb=bal_thumb,
        travel=24,
    )
    _compile_eq(out)
    return out


__all__ = ["compile_hidden_states"]
