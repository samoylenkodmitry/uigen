"""Static per-file masks for Cranamp-supported pixels.

The set of pixels Cranamp actually renders from each supported BMP is
defined by `configs/supported_pixels_classic.json` (built by
`scripts/13_build_support_profile.py`). The loss and metrics only score
pixels inside the union of those source rectangles; the rest is
exported but never displayed and so must not influence training.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, ExportFileSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "supported_pixels_classic.json"


def _build_mask(rects: Iterable[Iterable[int]], h: int, w: int) -> torch.Tensor:
    mask = torch.zeros((h, w), dtype=torch.bool)
    for rect in rects:
        x, y, rw, rh = (int(v) for v in rect)
        mask[y:y + rh, x:x + rw] = True
    return mask


@lru_cache(maxsize=4)
def load_support_masks(
    profile_path: str | Path = DEFAULT_PROFILE,
) -> dict[str, torch.Tensor]:
    """Load per-file boolean masks shaped (h, w) from a profile JSON."""
    path = Path(profile_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    masks: dict[str, torch.Tensor] = {}
    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    for file_name, rects in data.items():
        if file_name not in spec_by_name:
            continue
        spec = spec_by_name[file_name]
        masks[file_name] = _build_mask(rects, spec.h, spec.w)
    for spec in TRAINABLE_EXPORT_SPECS:
        if spec.file_name not in masks:
            raise KeyError(f"support profile {path} is missing {spec.file_name}")
    return masks


def support_mask_for(spec: ExportFileSpec) -> torch.Tensor:
    return load_support_masks()[spec.file_name]
