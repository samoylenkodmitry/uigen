from __future__ import annotations


def encode_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    canvas_w: float = 768.0,
    canvas_h: float = 1280.0,
    min_visible_area: float = 4.0,
) -> tuple[float, float, float, float, float]:
    cx0 = min(max(x0, 0.0), canvas_w)
    cy0 = min(max(y0, 0.0), canvas_h)
    cx1 = min(max(x1, 0.0), canvas_w)
    cy1 = min(max(y1, 0.0), canvas_h)
    if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < min_visible_area:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (cx0 / canvas_w, cy0 / canvas_h, cx1 / canvas_w, cy1 / canvas_h, 1.0)


def derive_eq_band_rects(group_rect: tuple[float, float, float, float, float]) -> list[tuple[float, float, float, float, float]]:
    x0, y0, x1, y1, visible = group_rect
    if visible <= 0.0 or x1 <= x0 or y1 <= y0:
        return [(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(10)]
    band_w = (x1 - x0) / 10.0
    return [
        (round(x0 + band_w * idx, 10), y0, round(x0 + band_w * (idx + 1), 10), y1, 1.0)
        for idx in range(10)
    ]
