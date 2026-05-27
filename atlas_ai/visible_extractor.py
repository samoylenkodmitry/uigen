"""Deterministic V8 visible asset extractor.

This is the product baseline and data contract for future VisibleSkinNet
outputs. It does not try to recover hidden states; it extracts the visible
default composition from a normalized mockup into exact exported BMP tensors,
then `hidden_state_compiler` fills plausible hidden frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from atlas_ai.v8_assets import image_to_tensor, load_exported_tensors
from atlas_ai.v8_layout import WINDOW_UNITS


def _rect(layout: dict, name: str) -> tuple[float, float, float, float]:
    x, y, w, h = layout["rects"][name]
    return float(x), float(y), float(w), float(h)


def _crop_local(
    image: Image.Image,
    layout: dict,
    window: str,
    local: tuple[float, float, float, float],
    out_size: tuple[int, int],
) -> Image.Image:
    wx, wy, ww, wh = _rect(layout, window)
    unit_w, unit_h = WINDOW_UNITS[window]
    lx, ly, lw, lh = local
    box = (
        wx + (lx / unit_w) * ww,
        wy + (ly / unit_h) * wh,
        wx + ((lx + lw) / unit_w) * ww,
        wy + ((ly + lh) / unit_h) * wh,
    )
    return image.crop(box).resize(out_size, Image.Resampling.LANCZOS)


def _paste_crop(
    files: dict[str, torch.Tensor],
    image: Image.Image,
    layout: dict,
    *,
    file_name: str,
    dst: tuple[int, int],
    window: str,
    local: tuple[float, float, float, float],
    out_size: tuple[int, int],
) -> None:
    crop = image_to_tensor(_crop_local(image, layout, window, local, out_size))
    x, y = dst
    files[file_name][:, y : y + crop.shape[1], x : x + crop.shape[2]] = crop


def extract_visible_assets(
    normalized_image: Image.Image,
    layout: dict,
    *,
    default_skin: str | Path = "assets/default_skin",
) -> dict[str, torch.Tensor]:
    """Extract a visible-state BMP set from a normalized mockup.

    The output is exact-size exported tensors. It is intentionally default-
    backed: uncertain hidden/padded regions remain valid instead of black.
    """
    image = normalized_image.convert("RGB")
    files = load_exported_tensors(default_skin, default_skin=default_skin)

    # Main window: visible body + titlebar/control sprites.
    main_116 = image_to_tensor(_crop_local(image, layout, "main", (0, 0, 275, 116), (275, 116)))
    files["MAIN.bmp"][:, :, :] = main_116[:, :115, :]
    files["TITLEBAR.bmp"][:, 0:14, 27:302] = main_116[:, 0:14, :]
    _paste_crop(files, image, layout, file_name="TITLEBAR.bmp", dst=(0, 0),
                window="main", local=(6, 3, 9, 9), out_size=(9, 9))
    _paste_crop(files, image, layout, file_name="TITLEBAR.bmp", dst=(9, 0),
                window="main", local=(244, 3, 9, 9), out_size=(9, 9))
    _paste_crop(files, image, layout, file_name="TITLEBAR.bmp", dst=(18, 0),
                window="main", local=(264, 3, 9, 9), out_size=(9, 9))
    _paste_crop(files, image, layout, file_name="TITLEBAR.bmp", dst=(0, 18),
                window="main", local=(254, 3, 9, 9), out_size=(9, 9))

    _paste_crop(files, image, layout, file_name="PLAYPAUS.bmp", dst=(0, 0),
                window="main", local=(26, 28, 9, 9), out_size=(9, 9))
    _paste_crop(files, image, layout, file_name="MONOSTER.bmp", dst=(0, 0),
                window="main", local=(212, 41, 56, 12), out_size=(56, 12))
    _paste_crop(files, image, layout, file_name="POSBAR.bmp", dst=(0, 0),
                window="main", local=(17, 72, 248, 10), out_size=(248, 10))
    _paste_crop(files, image, layout, file_name="POSBAR.bmp", dst=(248, 0),
                window="main", local=(126, 72, 29, 10), out_size=(29, 10))

    transport = [
        ((16, 88, 23, 18), (0, 0), (23, 18)),
        ((39, 88, 23, 18), (23, 0), (23, 18)),
        ((62, 88, 23, 18), (46, 0), (23, 18)),
        ((85, 88, 23, 18), (69, 0), (23, 18)),
        ((108, 88, 22, 18), (92, 0), (22, 18)),
        ((136, 89, 22, 16), (114, 0), (22, 16)),
    ]
    for local, dst, size in transport:
        _paste_crop(files, image, layout, file_name="CBUTTONS.bmp", dst=dst,
                    window="main", local=local, out_size=size)

    _paste_crop(files, image, layout, file_name="VOLUME.bmp", dst=(0, 195),
                window="main", local=(107, 57, 68, 13), out_size=(68, 13))
    _paste_crop(files, image, layout, file_name="VOLUME.bmp", dst=(15, 422),
                window="main", local=(134, 58, 14, 11), out_size=(14, 11))
    _paste_crop(files, image, layout, file_name="BALANCE.bmp", dst=(9, 195),
                window="main", local=(177, 57, 38, 13), out_size=(38, 13))
    _paste_crop(files, image, layout, file_name="BALANCE.bmp", dst=(15, 422),
                window="main", local=(189, 58, 14, 11), out_size=(14, 11))

    _paste_crop(files, image, layout, file_name="SHUFREP.bmp", dst=(28, 0),
                window="main", local=(164, 89, 47, 15), out_size=(47, 15))
    _paste_crop(files, image, layout, file_name="SHUFREP.bmp", dst=(0, 0),
                window="main", local=(210, 89, 28, 15), out_size=(28, 15))
    _paste_crop(files, image, layout, file_name="SHUFREP.bmp", dst=(0, 61),
                window="main", local=(219, 58, 23, 12), out_size=(23, 12))
    _paste_crop(files, image, layout, file_name="SHUFREP.bmp", dst=(23, 73),
                window="main", local=(242, 58, 23, 12), out_size=(23, 12))

    # EQ visible panel.
    eq_116 = image_to_tensor(_crop_local(image, layout, "eq", (0, 0, 275, 116), (275, 116)))
    files["EQMAIN.bmp"][:, 0:116, :] = eq_116
    files["EQMAIN.bmp"][:, 134:148, :] = eq_116[:, 0:14, :]

    # Playlist: reconstruct the exact PLEDIT sub-sprites the real engine samples
    # in render_playlist, instead of dumping the whole window (which made the
    # engine tile the wrong pixels -> garbled "PLAYLAYLIST" title + right-edge
    # text repeat). Each sprite is extracted from the window-local position the
    # engine DRAWS it at and written to its PLEDIT source rect. The list body is
    # procedural (black fill + text), not sampled from PLEDIT, so it is skipped.
    #   (window-local draw rect) -> dst in PLEDIT.bmp
    pledit_sprites = [
        ((0, 0, 25, 20), (0, 21)),       # titlebar left corner
        ((50, 0, 25, 20), (127, 21)),    # titlebar fill tile (repeated by engine)
        ((87, 0, 100, 20), (26, 21)),    # titlebar centered title text
        ((250, 0, 25, 20), (153, 21)),   # titlebar right corner
        ((0, 20, 12, 29), (0, 42)),      # left border (tiled vertically)
        ((255, 20, 20, 29), (31, 42)),   # right border (tiled vertically)
        ((0, 223, 125, 38), (0, 72)),    # footer left
        ((125, 223, 150, 38), (126, 72)),  # footer right
        ((260, 30, 8, 18), (52, 53)),    # scrollbar thumb (best-effort position)
    ]
    for (lx, ly, lw, lh), dst in pledit_sprites:
        _paste_crop(files, image, layout, file_name="PLEDIT.bmp", dst=dst,
                    window="playlist", local=(lx, ly, lw, lh), out_size=(lw, lh))

    return files


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(255.0, v)))) for v in c)


def playlist_pledit_text(
    normalized_image: Image.Image,
    layout: dict,
    *,
    default_skin: str | Path = "assets/default_skin",
) -> str:
    """Derive a PLEDIT.TXT (playlist colors) from the mockup playlist body.

    The engine reads Normal (entry text), Current (selected text), NormalBG
    (list background) and SelectedBG (selected-row bg) from PLEDIT.TXT. Sample
    them from the body region (window-local 12,20,243,203): the median is the
    background; the pixels most distant from it in luminance are the text.
    """
    image = normalized_image.convert("RGB")
    body = _crop_local(image, layout, "playlist", (12, 20, 243, 203), (243, 203))
    arr = np.asarray(body, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return (Path(default_skin) / "PLEDIT.TXT").read_text(encoding="latin-1")
    coeff = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum = arr @ coeff
    bg = np.median(arr, axis=0)
    bg_lum = float(bg @ coeff)
    dist = np.abs(lum - bg_lum)
    thr = max(20.0, float(np.percentile(dist, 92)))
    textpix = arr[dist >= thr]
    text = textpix.mean(axis=0) if len(textpix) else np.array([255.0, 200.0, 108.0])
    selbg = bg * 0.6 + text * 0.4                       # muted selected-row bg
    current = np.clip(text * 0.45 + 255.0 * 0.55, 0, 255)  # brighter selected text
    return (
        "[Text]\n"
        f"Normal={_hex(text)}\n"
        f"Current={_hex(current)}\n"
        f"NormalBG={_hex(bg)}\n"
        f"SelectedBG={_hex(selbg)}\n"
        f"MbFG={_hex(text)}\n"
        f"MbBG={_hex(bg)}\n"
        "Font=Arial\n"
    )


__all__ = ["extract_visible_assets", "playlist_pledit_text"]
