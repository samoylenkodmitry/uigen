"""State-family geometry for Cranamp classic exported BMPs.

A state family is a group of sub-rectangles inside one exported BMP that share
a "pick one of these at a time" semantic in the source UI. The V7 completer
uses these to build state-family masks: reveal one frame of a family, hide
the siblings, and teach the model to infer the hidden sibling pixels.

The YAML schema is documented in configs/state_families_classic.yaml.

Two family kinds expand into rectangle lists:

    sprite              explicit list of (x, y, w, h) rectangles
    button_state_sheet  explicit list (same shape as sprite, named for clarity)
    toggle_state_sheet  explicit list (same shape as sprite, named for clarity)
    horizontal_strip    frame_count rectangles at (x + k*pitch, y), frame_w x frame_h
    vertical_strip      frame_count rectangles at (x, y + k*pitch), frame_w x frame_h

`load_state_families(path)` returns a flat list of `StateRect`s for each file,
suitable for sampling masks during training. Bounds are validated against
`atlas_ai.export_spec.TRAINABLE_EXPORT_SPECS` at load time; any rectangle
that escapes its file's bounds raises ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS


EXPLICIT_KINDS = {"sprite", "button_state_sheet", "toggle_state_sheet"}
STRIP_KINDS = {"horizontal_strip", "vertical_strip"}

# How a family's rectangles relate to each other, which decides whether
# state_family masking may hide siblings:
#   alternatives - mutually exclusive states; reveal one, hide the rest
#                  (CBUTTONS pressed/unpressed, slider frames, on/off toggles)
#   components   - complementary parts of one assembled image; never hide a
#                  sibling (POSBAR track+thumb, EQMAIN chrome, PLEDIT footer)
#   single       - one rectangle, nothing to alternate
MASK_ROLES = ("alternatives", "components", "single")


@dataclass(frozen=True)
class StateRect:
    """One sub-rectangle inside an exported BMP."""
    file_name: str
    family: str
    name: str
    x: int
    y: int
    w: int
    h: int
    mask_role: str = "single"


def _expand_family(file_name: str, family: dict) -> list[StateRect]:
    family_name = family["name"]
    kind = family["kind"]
    mask_role = family.get("mask_role")
    if mask_role not in MASK_ROLES:
        raise ValueError(
            f"{file_name}/{family_name}: mask_role must be one of "
            f"{'|'.join(MASK_ROLES)}, got {mask_role!r}"
        )
    if kind in EXPLICIT_KINDS:
        rects = family.get("rects") or []
        if not rects:
            raise ValueError(f"{file_name}/{family_name}: no rects declared")
        return [
            StateRect(
                file_name=file_name,
                family=family_name,
                name=r["name"],
                x=int(r["x"]),
                y=int(r["y"]),
                w=int(r["w"]),
                h=int(r["h"]),
                mask_role=mask_role,
            )
            for r in rects
        ]
    if kind in STRIP_KINDS:
        axis = family["frame_axis"]
        if (kind == "horizontal_strip" and axis != "x") or (kind == "vertical_strip" and axis != "y"):
            raise ValueError(
                f"{file_name}/{family_name}: kind={kind} disagrees with frame_axis={axis}"
            )
        count = int(family["frame_count"])
        frame_w = int(family["frame_w"])
        frame_h = int(family["frame_h"])
        x0 = int(family["x"])
        y0 = int(family["y"])
        pitch = int(family["pitch"])
        if count <= 0:
            raise ValueError(f"{file_name}/{family_name}: frame_count must be positive")
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"{file_name}/{family_name}: frame size must be positive")
        if pitch <= 0:
            raise ValueError(f"{file_name}/{family_name}: pitch must be positive")
        out: list[StateRect] = []
        for k in range(count):
            if axis == "x":
                rx, ry = x0 + k * pitch, y0
            else:
                rx, ry = x0, y0 + k * pitch
            out.append(StateRect(
                file_name=file_name,
                family=family_name,
                name=f"frame_{k:02d}",
                x=rx, y=ry, w=frame_w, h=frame_h,
                mask_role=mask_role,
            ))
        return out
    raise ValueError(f"{file_name}/{family_name}: unknown kind {kind!r}")


def load_state_families(path: str | Path) -> dict[str, list[StateRect]]:
    """Load and validate state families. Returns dict[file_name -> list[StateRect]].

    Validation:
        - schema must be 'state_families_classic_v1'
        - every file referenced must be in TRAINABLE_EXPORT_SPECS
        - every rectangle must lie inside its file's (w, h) bounds
        - rectangle w, h must be > 0
        - rectangle names must be unique within a (file, family)
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    schema = data.get("schema")
    if schema != "state_families_classic_v1":
        raise ValueError(f"{path}: expected schema=state_families_classic_v1, got {schema!r}")
    files_node = data.get("files") or {}
    if not isinstance(files_node, dict):
        raise ValueError(f"{path}: 'files' must be a mapping")

    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    out: dict[str, list[StateRect]] = {}
    for file_name, entry in files_node.items():
        spec = spec_by_name.get(file_name)
        if spec is None:
            raise ValueError(f"{file_name}: not in TRAINABLE_EXPORT_SPECS")
        families = (entry or {}).get("families") or []
        if not families:
            raise ValueError(f"{file_name}: no families declared")
        seen_names: set[tuple[str, str]] = set()
        rects: list[StateRect] = []
        for family in families:
            expanded = _expand_family(file_name, family)
            for r in expanded:
                if r.w <= 0 or r.h <= 0:
                    raise ValueError(f"{file_name}/{r.family}/{r.name}: non-positive size")
                if r.x < 0 or r.y < 0:
                    raise ValueError(f"{file_name}/{r.family}/{r.name}: negative origin")
                if r.x + r.w > spec.w or r.y + r.h > spec.h:
                    raise ValueError(
                        f"{file_name}/{r.family}/{r.name}: rect ({r.x},{r.y},{r.w},{r.h}) "
                        f"escapes bounds ({spec.w}x{spec.h})"
                    )
                key = (r.family, r.name)
                if key in seen_names:
                    raise ValueError(
                        f"{file_name}: duplicate name {r.family}/{r.name}"
                    )
                seen_names.add(key)
                rects.append(r)
        out[file_name] = rects
    missing = sorted(set(spec_by_name) - set(out))
    if missing:
        raise ValueError(
            "missing state-family declarations for trainable files: "
            + ", ".join(missing)
        )
    return out


def group_by_family(rects: list[StateRect]) -> dict[str, list[StateRect]]:
    """Group rectangles by family name within a single file."""
    out: dict[str, list[StateRect]] = {}
    for r in rects:
        out.setdefault(r.family, []).append(r)
    return out


__all__ = ["StateRect", "load_state_families", "group_by_family",
           "EXPLICIT_KINDS", "STRIP_KINDS", "MASK_ROLES"]
