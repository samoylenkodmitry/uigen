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
    """Mix probability for the observed-mask families. Sum is normalized
    to 1.0; absolute scale does not matter.

    `passthrough` (mask=ones) defaults to 0 so the V7 plan default mix is
    unchanged. Set it >0 to run the all-observed copy-through diagnostic
    where the model sees the target everywhere within the support mask.
    """
    provenance: float = 0.40
    state_family: float = 0.35
    random_rect: float = 0.20
    whole_file: float = 0.05
    passthrough: float = 0.0


# Modes in a fixed order so weight redistribution is deterministic.
MASK_MODES = ("provenance", "state_family", "random_rect", "whole_file", "passthrough")


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


def _alternatives_families(rects: list[StateRect]) -> dict[str, list[StateRect]]:
    """Families eligible for sibling-hiding: mask_role == 'alternatives' with
    two or more rects. `components`/`single` families are excluded so we never
    treat complementary parts (POSBAR track+thumb, EQMAIN chrome, TITLEBAR
    corner buttons) as mutually exclusive states."""
    grouped = group_by_family(rects)
    return {
        name: rs for name, rs in grouped.items()
        if len(rs) >= 2 and rs[0].mask_role == "alternatives"
    }


def make_state_family_mask(
    rng: np.random.Generator,
    rects: list[StateRect],
    h: int,
    w: int,
) -> np.ndarray:
    """Reveal one rect of a randomly-picked `alternatives` family, hide its
    siblings.

    Pixels outside all of the family's rectangles stay observed (1). This is
    "you can see the rest of the BMP, but I'm hiding sibling state frames
    of the chosen one." Only families whose mask_role is 'alternatives' are
    eligible; complementary `components` parts are never hidden from one
    another.
    """
    mask = np.ones((h, w), dtype=np.uint8)
    families = _alternatives_families(rects)
    if not families:
        # No alternatives to hide; fall back to an all-ones (passthrough)
        # sample. The hidden-normalized loss treats this as a no-op, and
        # sample_v7_observed_mask redistributes the weight away when it can
        # tell up front that a file has no alternatives.
        return mask
    family = rng.choice(list(families.keys()))
    siblings = families[family]
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


def make_passthrough_mask(h: int, w: int) -> np.ndarray:
    """Observe everywhere (mask is all ones). Diagnostic mode: when used as
    the only mask source, the completer is given the target as input and
    must learn a near-identity inside the support region."""
    return np.ones((h, w), dtype=np.uint8)


def has_alternatives(rects: list[StateRect] | None) -> bool:
    """True if any family in `rects` is eligible for sibling-hiding."""
    return bool(_alternatives_families(rects)) if rects else False


def _raw_weights(
    weights: V7MaskWeights,
    have_provenance: bool,
    have_state_family: bool,
) -> dict[str, float]:
    """Mask-mode weights after zeroing modes whose prerequisite is missing.

    Two prerequisites:
      - provenance requires a non-empty visible_masks pool;
      - state_family requires the file to have at least one `alternatives`
        family (POSBAR, TITLEBAR, etc. have none, so state_family on them
        would only ever produce no-op all-ones masks).
    """
    return {
        "provenance": weights.provenance if have_provenance else 0.0,
        "state_family": weights.state_family if have_state_family else 0.0,
        "random_rect": weights.random_rect,
        "whole_file": weights.whole_file,
        "passthrough": weights.passthrough,
    }


def has_available_mask_mode(
    weights: V7MaskWeights,
    *,
    have_provenance: bool,
    have_state_family: bool,
) -> bool:
    """True if at least one mask mode survives the prerequisite filter.

    When this is False, `sample_v7_observed_mask` would raise — callers (e.g.
    the eval) should skip the file rather than sample it.
    """
    return sum(_raw_weights(weights, have_provenance, have_state_family).values()) > 0


def _normalize_weights(
    weights: V7MaskWeights,
    have_provenance: bool,
    have_state_family: bool,
) -> dict[str, float]:
    """Normalize the surviving mask-mode weights to a probability distribution.

    A mode whose prerequisite is missing gets its weight zeroed, and the
    remaining modes are renormalized (so the other shares are not silently
    inflated relative to each other). Raises when nothing survives.
    """
    raw = _raw_weights(weights, have_provenance, have_state_family)
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
    have_state_family = has_alternatives(family_rects)
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
    if mode == "passthrough":
        return make_passthrough_mask(h, w), mode
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
    "has_alternatives",
    "has_available_mask_mode",
    "make_provenance_mask",
    "make_state_family_mask",
    "make_random_rect_mask",
    "make_whole_file_mask",
    "make_passthrough_mask",
    "sample_v7_observed_mask",
    "apply_observed_mask",
]
