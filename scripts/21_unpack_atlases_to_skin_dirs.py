#!/usr/bin/env python3
"""Unpack data_v4_16skin atlases into per-skin BMP directories for V7 Gate B.

Reads each `data_v4_16skin/atlases/<skin>.png` together with its
`<skin>.meta.json`, crops the slot content at `pasted_rect`, normalizes
the content to the canonical TRAINABLE_EXPORT_SPECS dimensions, and writes:

    <out_dir>/<skin_id>/<FILE>.bmp     for every trainable file
    <out_dir>/_manifest.json           per-skin outcome (pad/crop/skip)
    <out_dir>/<skin_id>/_meta.json     per-file decision trace

The V7CompletionDataset can then point at `<out_dir>` as a multi-skin
source: each subdirectory is one skin's BMP set.

Policy flags
============

The user's V7 plan calls for: "do not silently pad; report/skip invalid
skins unless padding policy is explicitly implemented and tested."

This script implements four explicit knobs:

    --pad-smaller       (default: enabled)
        If a slot's content is SMALLER than canonical on either axis,
        pad the missing right/bottom with zeros (matches how the atlas
        was already storing unused trailing pixels).

    --crop-larger       (default: enabled)
        If a slot's content is LARGER than canonical on either axis,
        crop from the top-left so the canonical (w, h) is preserved.
        Targets the off-by-1 cases like MAIN 275x116 vs canonical
        275x115.

    --min-content-fraction FLOAT      (default: 0.5)
        If the content area is less than this fraction of the canonical
        area, the slot is treated as DEGENERATE (truly missing source
        pixels, not just a different size). Degenerate slots cause the
        whole skin to be SKIPPED so Gate B never silently trains against
        a 1x1 stub.

    --strict
        Disable all pad/crop; only accept exact-size content. Useful to
        confirm how many skins are "clean" before any normalization.

The manifest records every decision so downstream work can see which
skins were padded, which were cropped, and which were skipped (and why).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS


@dataclass
class SlotOutcome:
    file_name: str
    canonical_wh: tuple[int, int]
    content_wh: tuple[int, int]
    action: str    # "exact" | "padded" | "cropped" | "padded+cropped" | "degenerate" | "missing"
    notes: str = ""


@dataclass
class SkinOutcome:
    skin_id: str
    status: str    # "accepted" | "skipped"
    reason: str
    slots: list[SlotOutcome]


def _crop_pasted(atlas_arr: np.ndarray, pasted_rect: list[int]) -> np.ndarray:
    """Crop atlas (H, W, 3) at pasted_rect = [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = pasted_rect
    return atlas_arr[y0:y1, x0:x1, :]


def _normalize_to_canonical(
    content: np.ndarray,
    canonical_w: int,
    canonical_h: int,
    pad_smaller: bool,
    crop_larger: bool,
) -> tuple[np.ndarray | None, str]:
    """Normalize content array to (canonical_h, canonical_w, 3).

    Returns (normalized_or_None, action_label). `None` means the content
    cannot be brought to canonical under the current policy and the caller
    should reject the slot.
    """
    ch, cw, _ = content.shape
    actions: list[str] = []

    # Crop larger axes first so we are working with content <= canonical.
    if ch > canonical_h or cw > canonical_w:
        if not crop_larger:
            return None, "needs crop but --crop-larger disabled"
        new_h = min(ch, canonical_h)
        new_w = min(cw, canonical_w)
        content = content[:new_h, :new_w, :]
        ch, cw, _ = content.shape
        actions.append("cropped")

    # Pad smaller axes with zeros (right/bottom).
    if ch < canonical_h or cw < canonical_w:
        if not pad_smaller:
            return None, "needs pad but --pad-smaller disabled"
        out = np.zeros((canonical_h, canonical_w, 3), dtype=content.dtype)
        out[:ch, :cw, :] = content
        content = out
        actions.append("padded")

    if not actions:
        return content, "exact"
    return content, "+".join(actions)


def _content_area_fraction(content_wh: tuple[int, int], canonical_wh: tuple[int, int]) -> float:
    cw, ch = content_wh
    nw, nh = canonical_wh
    if nw <= 0 or nh <= 0:
        return 0.0
    return max(0.0, min(1.0, (cw * ch) / (nw * nh)))


def unpack_skin(
    atlas_path: Path,
    meta_path: Path,
    out_dir: Path,
    *,
    pad_smaller: bool,
    crop_larger: bool,
    min_content_fraction: float,
) -> SkinOutcome:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    skin_id = str(meta["skin_id"])
    with Image.open(atlas_path) as im:
        atlas_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)

    slot_outcomes: list[SlotOutcome] = []
    must_skip = False
    skip_reason = ""

    spec_by_slot = {spec.slot: spec for spec in TRAINABLE_EXPORT_SPECS}
    slots_node = meta.get("slots", {}) or {}

    # First pass: decide actions per slot without writing anything.
    decisions: dict[str, tuple[np.ndarray | None, SlotOutcome]] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        slot_info = slots_node.get(spec.slot)
        if slot_info is None:
            decisions[spec.file_name] = (None, SlotOutcome(
                file_name=spec.file_name,
                canonical_wh=(spec.w, spec.h),
                content_wh=(0, 0),
                action="missing",
                notes="meta_json had no entry for this slot",
            ))
            must_skip = True
            skip_reason = skip_reason or f"missing slot: {spec.slot}"
            continue
        content_wh = tuple(int(v) for v in slot_info["size"])  # (w, h)
        frac = _content_area_fraction(content_wh, (spec.w, spec.h))
        if frac < min_content_fraction:
            decisions[spec.file_name] = (None, SlotOutcome(
                file_name=spec.file_name,
                canonical_wh=(spec.w, spec.h),
                content_wh=content_wh,
                action="degenerate",
                notes=f"content fraction {frac:.3f} < min {min_content_fraction:.3f}",
            ))
            must_skip = True
            skip_reason = skip_reason or f"degenerate slot: {spec.slot} ({content_wh[0]}x{content_wh[1]})"
            continue
        pasted_rect = slot_info["pasted_rect"]
        content = _crop_pasted(atlas_arr, pasted_rect)
        # Defensive check: pasted_rect implied size should match meta `size`.
        ph, pw, _ = content.shape
        if (pw, ph) != content_wh:
            decisions[spec.file_name] = (None, SlotOutcome(
                file_name=spec.file_name,
                canonical_wh=(spec.w, spec.h),
                content_wh=content_wh,
                action="degenerate",
                notes=f"pasted_rect crop {pw}x{ph} != meta size {content_wh}",
            ))
            must_skip = True
            skip_reason = skip_reason or f"meta inconsistency: {spec.slot}"
            continue
        normalized, action = _normalize_to_canonical(
            content, spec.w, spec.h, pad_smaller, crop_larger,
        )
        if normalized is None:
            decisions[spec.file_name] = (None, SlotOutcome(
                file_name=spec.file_name,
                canonical_wh=(spec.w, spec.h),
                content_wh=content_wh,
                action="rejected",
                notes=action,
            ))
            must_skip = True
            skip_reason = skip_reason or f"policy rejected: {spec.slot} ({action})"
            continue
        decisions[spec.file_name] = (normalized, SlotOutcome(
            file_name=spec.file_name,
            canonical_wh=(spec.w, spec.h),
            content_wh=content_wh,
            action=action,
        ))
        slot_outcomes.append(decisions[spec.file_name][1])

    if must_skip:
        return SkinOutcome(
            skin_id=skin_id,
            status="skipped",
            reason=skip_reason,
            slots=[outcome for _, outcome in decisions.values()],
        )

    # Second pass: write files.
    skin_dir = out_dir / skin_id
    skin_dir.mkdir(parents=True, exist_ok=True)
    for spec in TRAINABLE_EXPORT_SPECS:
        normalized, outcome = decisions[spec.file_name]
        assert normalized is not None  # must_skip would have been True
        Image.fromarray(normalized, mode="RGB").save(skin_dir / spec.file_name)
    per_skin_meta = {
        "skin_id": skin_id,
        "atlas": str(atlas_path),
        "slots": [asdict(o) for o in slot_outcomes],
    }
    (skin_dir / "_meta.json").write_text(json.dumps(per_skin_meta, indent=2, sort_keys=True))
    return SkinOutcome(skin_id=skin_id, status="accepted", reason="", slots=slot_outcomes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlases-dir", default="data_v4_16skin/atlases",
                        help="Directory containing *.png + *.meta.json pairs.")
    parser.add_argument("--out-dir", required=True,
                        help="Output root; one subdirectory per accepted skin.")
    parser.add_argument("--pad-smaller", action=argparse.BooleanOptionalAction,
                        default=True, help="Pad smaller content with zeros.")
    parser.add_argument("--crop-larger", action=argparse.BooleanOptionalAction,
                        default=True, help="Crop larger content from top-left.")
    parser.add_argument("--min-content-fraction", type=float, default=0.5,
                        help="Slot is degenerate if content_wh area < this fraction "
                             "of canonical area. Triggers skip-whole-skin.")
    parser.add_argument("--strict", action="store_true",
                        help="Override pad/crop flags off; accept exact-size only.")
    args = parser.parse_args()

    if args.strict:
        args.pad_smaller = False
        args.crop_larger = False

    atlases_dir = Path(args.atlases_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = sorted(atlases_dir.glob("*.meta.json"))
    if not metas:
        raise SystemExit(f"no meta files found under {atlases_dir}")

    skin_outcomes: list[SkinOutcome] = []
    for meta_path in metas:
        skin_id = meta_path.stem.removesuffix(".meta")
        atlas_path = atlases_dir / f"{skin_id}.png"
        if not atlas_path.exists():
            print(f"[skip] {skin_id}: atlas {atlas_path} missing")
            skin_outcomes.append(SkinOutcome(
                skin_id=skin_id, status="skipped",
                reason="atlas png missing", slots=[],
            ))
            continue
        outcome = unpack_skin(
            atlas_path, meta_path, out_dir,
            pad_smaller=args.pad_smaller,
            crop_larger=args.crop_larger,
            min_content_fraction=args.min_content_fraction,
        )
        skin_outcomes.append(outcome)
        if outcome.status == "accepted":
            actions = sorted(set(s.action for s in outcome.slots))
            print(f"[ok]   {outcome.skin_id}: actions={actions}")
        else:
            print(f"[skip] {outcome.skin_id}: {outcome.reason}")

    manifest = {
        "args": vars(args),
        "atlases_dir": str(atlases_dir),
        "out_dir": str(out_dir),
        "skins": [
            {
                "skin_id": o.skin_id,
                "status": o.status,
                "reason": o.reason,
                "slots": [asdict(s) for s in o.slots],
            }
            for o in skin_outcomes
        ],
        "accepted_count": sum(1 for o in skin_outcomes if o.status == "accepted"),
        "skipped_count": sum(1 for o in skin_outcomes if o.status == "skipped"),
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print()
    print(f"accepted: {manifest['accepted_count']}/{len(metas)}")
    print(f"skipped:  {manifest['skipped_count']}")
    print(f"manifest: {out_dir / '_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
