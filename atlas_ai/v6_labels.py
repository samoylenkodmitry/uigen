"""V6 source-preserving training labels derived from a final-frame provenance buffer.

Per the V6 plan, every synthetic Cranamp render emits a uint32 provenance buffer
shaped (canvas_h, canvas_w) that encodes the clean exported BMP source pixel that
survives at each canvas position. This module converts that buffer into per-file
training labels:

    visible_mask  uint8   [H, W]      1 where the file's exported pixel survives
    uv_target     float32 [2, H, W]   align_corners=False grid_sample coords
                                       channel 0 = u (x), channel 1 = v (y)

The model later predicts uv_grid and feeds it to:

    grid_sample(input_view, uv_grid.permute(0, 2, 3, 1), align_corners=False)

so the same coordinate convention is used here:

    u = 2.0 * ((screen_x + 0.5) / canvas_w) - 1.0
    v = 2.0 * ((screen_y + 0.5) / canvas_h) - 1.0

If multiple canvas pixels map to the same source BMP pixel (scale > 1), the
first one in row-major canvas order is used as the representative coordinate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .export_spec import TRAINABLE_EXPORT_SPECS


PROVENANCE_FILE_ID_SHIFT = 22
PROVENANCE_ROW_SHIFT = 11
PROVENANCE_COORD_MASK = 0x7FF  # 11 bits, holds src coords up to 2047


def build_v6_labels(
    provenance: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Convert a final-frame provenance buffer into per-file V6 labels.

    Args:
        provenance: uint32 array shaped (canvas_h, canvas_w). 0 means no
            clean exported BMP source survives at that canvas pixel; non-zero
            encodes (file_id, src_y, src_x) via the Cranamp renderer encoding.
        canvas_w, canvas_h: canvas dimensions used to render the view. UV
            normalization uses these as the grid_sample reference size.

    Returns:
        dict keyed by exported BMP file name (e.g. "MAIN.bmp"). Each value:
            visible_mask: uint8 [H, W]
            uv_target:    float32 [2, H, W]
    """
    if provenance.dtype != np.uint32:
        raise TypeError(f"provenance must be uint32, got {provenance.dtype}")
    if provenance.shape != (canvas_h, canvas_w):
        raise ValueError(
            f"provenance shape {provenance.shape} does not match "
            f"({canvas_h}, {canvas_w})"
        )

    labels: dict[str, dict[str, np.ndarray]] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        labels[spec.file_name] = {
            "visible_mask": np.zeros((spec.h, spec.w), dtype=np.uint8),
            "uv_target": np.zeros((2, spec.h, spec.w), dtype=np.float32),
        }

    nonzero = provenance != 0
    if not nonzero.any():
        return labels

    canvas_y, canvas_x = np.where(nonzero)  # row-major order
    packed = provenance[canvas_y, canvas_x].astype(np.int64) - 1
    src_x = packed & PROVENANCE_COORD_MASK
    src_y = (packed >> PROVENANCE_ROW_SHIFT) & PROVENANCE_COORD_MASK
    file_id = packed >> PROVENANCE_FILE_ID_SHIFT
    invalid_file = (file_id < 0) | (file_id >= len(TRAINABLE_EXPORT_SPECS))
    if invalid_file.any():
        bad = sorted(set(int(v) for v in file_id[invalid_file]))
        raise ValueError(f"provenance contains unknown trainable file id(s): {bad}")

    for fid, spec in enumerate(TRAINABLE_EXPORT_SPECS):
        on_file = file_id == fid
        if not on_file.any():
            continue
        fy = src_y[on_file]
        fx = src_x[on_file]
        cy = canvas_y[on_file]
        cx = canvas_x[on_file]
        in_bounds = (fy < spec.h) & (fx < spec.w)
        if not in_bounds.all():
            bad = np.where(~in_bounds)[0][0]
            raise ValueError(
                f"provenance for {spec.file_name} contains out-of-bounds source "
                f"coordinate (x={int(fx[bad])}, y={int(fy[bad])}); "
                f"expected width={spec.w}, height={spec.h}"
            )
        fy, fx, cy, cx = fy[in_bounds], fx[in_bounds], cy[in_bounds], cx[in_bounds]
        # First canvas pixel wins on ties. np.unique with return_index returns
        # the index of the first occurrence of each unique key.
        flat_keys = fy.astype(np.int64) * (PROVENANCE_COORD_MASK + 1) + fx.astype(np.int64)
        _, first_idx = np.unique(flat_keys, return_index=True)
        chosen_fy = fy[first_idx]
        chosen_fx = fx[first_idx]
        chosen_cy = cy[first_idx]
        chosen_cx = cx[first_idx]

        visible = labels[spec.file_name]["visible_mask"]
        uv = labels[spec.file_name]["uv_target"]
        visible[chosen_fy, chosen_fx] = 1
        u = 2.0 * (chosen_cx.astype(np.float32) + 0.5) / canvas_w - 1.0
        v = 2.0 * (chosen_cy.astype(np.float32) + 0.5) / canvas_h - 1.0
        uv[0, chosen_fy, chosen_fx] = u
        uv[1, chosen_fy, chosen_fx] = v

    return labels


def save_v6_labels(labels: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    """Save labels to a compressed .npz with keys 'visible_<SLOT>' / 'uv_<SLOT>'."""
    arrays: dict[str, np.ndarray] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        entry = labels.get(spec.file_name)
        if entry is None:
            continue
        arrays[f"visible_{spec.slot}"] = entry["visible_mask"]
        arrays[f"uv_{spec.slot}"] = entry["uv_target"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def load_v6_labels(in_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Inverse of save_v6_labels."""
    labels: dict[str, dict[str, np.ndarray]] = {}
    with np.load(in_path) as raw:
        for spec in TRAINABLE_EXPORT_SPECS:
            v_key = f"visible_{spec.slot}"
            u_key = f"uv_{spec.slot}"
            if v_key not in raw.files or u_key not in raw.files:
                continue
            labels[spec.file_name] = {
                "visible_mask": raw[v_key],
                "uv_target": raw[u_key],
            }
    return labels


def labels_summary(labels: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, int | float]]:
    """Per-file diagnostics: visible_count, visible_fraction. Useful for sanity output."""
    out: dict[str, dict[str, int | float]] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        entry = labels.get(spec.file_name)
        if entry is None:
            continue
        visible_count = int(entry["visible_mask"].sum())
        total = int(entry["visible_mask"].size)
        out[spec.file_name] = {
            "visible_count": visible_count,
            "visible_fraction": visible_count / total if total else 0.0,
        }
    return out


__all__ = [
    "build_v6_labels",
    "save_v6_labels",
    "load_v6_labels",
    "labels_summary",
    "PROVENANCE_FILE_ID_SHIFT",
    "PROVENANCE_ROW_SHIFT",
    "PROVENANCE_COORD_MASK",
]
