"""V7 asset-completion observed-mask generator.

Per the V7 plan, the completer is trained on partial-evidence inputs:

    observed_rgb = clean_target * observed_mask
    observed_mask
    -> full clean target_rgb

The mask is drawn from one of four families with the V7 plan's mix:

    40%   provenance        true render masks from final-frame provenance
                            (sampled from a caller-provided pool of visible_mask
                            arrays)
    35%   state_family      reveal one state/frame of a state family in this
                            file, hide its siblings, leave everything else
                            observed
    20%   random_rect       a random rectangle, random aspect, zeroed in the
                            otherwise-observed mask. Includes stripes via
                            extreme aspects.
     5%   whole_file        observe nothing (mask all zeros) - rare, exercises
                            the "infer from style/coords only" pathway

`sample_v7_observed_mask` selects a mode by weight and falls back gracefully:

    - If no `visible_masks` pool is supplied, the provenance mode's weight is
      redistributed proportionally over the other modes.
    - If a file has no usable state families (single rectangle covering the
      whole BMP), state_family weight is redistributed similarly.

Returns uint8 [H, W] arrays in {0, 1}. 1 means the pixel is observed/given to
the completer; 0 means hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlas_ai.state_families import StateRect, group_by_family


@dataclass(frozen=True)
class V7MaskWeights:
    """Mix probability for the four observed-mask families. Sum is normalized
    to 1.0; absolute scale does not matter."""
    provenance: float = 0.40
    state_family: float = 0.35
    random_rect: float = 0.20
    whole_file: float = 0.05


# Modes in a fixed order so weight redistribution is deterministic.
MASK_MODES = ("provenance", "state_family", "random_rect", "whole_file")


def make_provenance_mask(
    rng: np.random.Generator,
    visible_masks: list[np.ndarray],
    h: int,
    w: int,
) -> np.ndarray:
    """Sample one mask from a pool of pre-recorded provenance visible_masks.

    The pool entries must already match (H, W). Caller guarantees this.
    """
    if not visible_masks:
        raise ValueError("provenance pool is empty")
    idx = int(rng.integers(0, len(visible_masks)))
    mask = visible_masks[idx]
    if mask.shape != (h, w):
        raise ValueError(f"provenance mask shape {mask.shape} != ({h}, {w})")
    return mask.astype(np.uint8)


def make_state_family_mask(
    rng: np.random.Generator,
    rects: list[StateRect],
    h: int,
    w: int,
) -> np.ndarray:
    """Reveal one rect of a randomly-picked family, hide its siblings.

    Pixels outside all of the family's rectangles stay observed (1). This is
    "you can see the rest of the BMP, but I'm hiding sibling state frames
    of the chosen one."
    """
    mask = np.ones((h, w), dtype=np.uint8)
    grouped = group_by_family(rects)
    families = [name for name, rs in grouped.items() if len(rs) >= 2]
    if not families:
        # Family with only one rect contributes nothing to hide; fall back to
        # an empty hide (returns all-ones, equivalent to a "trivial" sample).
        return mask
    family = rng.choice(families)
    siblings = grouped[family]
    revealed_idx = int(rng.integers(0, len(siblings)))
    for i, r in enumerate(siblings):
        if i == revealed_idx:
            continue
        mask[r.y : r.y + r.h, r.x : r.x + r.w] = 0
    return mask


def make_random_rect_mask(
    rng: np.random.Generator,
    h: int,
    w: int,
    min_frac: float = 0.05,
    max_frac: float = 0.5,
) -> np.ndarray:
    """Zero out a random rectangle of width/height between min_frac and
    max_frac of the BMP dimensions. Wide-and-thin or tall-and-thin
    distributions produce stripe-like masks."""
    mask = np.ones((h, w), dtype=np.uint8)
    rh = max(1, int(rng.uniform(min_frac, max_frac) * h))
    rw = max(1, int(rng.uniform(min_frac, max_frac) * w))
    if rng.random() < 0.5:
        # Occasionally collapse one dimension to produce a stripe-like rect.
        if rng.random() < 0.5:
            rh = max(1, int(rh * 0.2))
        else:
            rw = max(1, int(rw * 0.2))
    y0 = int(rng.integers(0, max(1, h - rh + 1)))
    x0 = int(rng.integers(0, max(1, w - rw + 1)))
    mask[y0 : y0 + rh, x0 : x0 + rw] = 0
    return mask


def make_whole_file_mask(h: int, w: int) -> np.ndarray:
    """Observe nothing (mask is all zeros)."""
    return np.zeros((h, w), dtype=np.uint8)


def _normalize_weights(
    weights: V7MaskWeights,
    have_provenance: bool,
    have_state_family: bool,
) -> dict[str, float]:
    raw = {
        "provenance": weights.provenance if have_provenance else 0.0,
        "state_family": weights.state_family if have_state_family else 0.0,
        "random_rect": weights.random_rect,
        "whole_file": weights.whole_file,
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("no mask modes are available; all weights resolved to 0")
    return {k: v / total for k, v in raw.items()}


def sample_v7_observed_mask(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    family_rects: list[StateRect] | None = None,
    visible_masks: list[np.ndarray] | None = None,
    weights: V7MaskWeights = V7MaskWeights(),
) -> tuple[np.ndarray, str]:
    """Pick a mode and produce an observed_mask.

    Returns (mask uint8 [h, w], mode_name). The mode_name is useful for
    logging the actual draw distribution during training.
    """
    have_provenance = visible_masks is not None and len(visible_masks) > 0
    grouped = group_by_family(family_rects) if family_rects else {}
    have_state_family = any(len(rs) >= 2 for rs in grouped.values())
    probs = _normalize_weights(weights, have_provenance, have_state_family)
    mode = str(rng.choice(MASK_MODES, p=[probs[m] for m in MASK_MODES]))
    if mode == "provenance":
        return make_provenance_mask(rng, visible_masks, h, w), mode
    if mode == "state_family":
        return make_state_family_mask(rng, family_rects, h, w), mode
    if mode == "random_rect":
        return make_random_rect_mask(rng, h, w), mode
    if mode == "whole_file":
        return make_whole_file_mask(h, w), mode
    raise ValueError(f"impossible mode {mode!r}")  # pragma: no cover


def apply_observed_mask(target_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply target_rgb [H, W, 3] by the observed mask [H, W]. Hidden
    pixels become zeros - the completer sees the zero as a "no information"
    sentinel and must reproduce the original via context."""
    if target_rgb.ndim != 3 or target_rgb.shape[-1] != 3:
        raise ValueError(f"target_rgb must be [H, W, 3], got {target_rgb.shape}")
    if mask.shape != target_rgb.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != target {target_rgb.shape[:2]}")
    return target_rgb * mask[..., None]


__all__ = [
    "V7MaskWeights",
    "MASK_MODES",
    "make_provenance_mask",
    "make_state_family_mask",
    "make_random_rect_mask",
    "make_whole_file_mask",
    "sample_v7_observed_mask",
    "apply_observed_mask",
]
