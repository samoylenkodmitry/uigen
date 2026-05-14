from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Slot:
    id: int
    name: str
    file: str | None
    x: int
    y: int
    w: int
    h: int
    loss_weight: float


@dataclass(frozen=True)
class AtlasProfile:
    canvas_w: int
    canvas_h: int
    slots: tuple[Slot, ...]

    @property
    def slots_by_name(self) -> dict[str, Slot]:
        return {slot.name: slot for slot in self.slots}

    @property
    def slots_by_file(self) -> dict[str, Slot]:
        return {slot.file.lower(): slot for slot in self.slots if slot.file}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_atlas_profile(path: str | Path) -> AtlasProfile:
    raw = load_json(path)
    slots = tuple(
        Slot(
            id=int(slot["id"]),
            name=str(slot["name"]),
            file=slot.get("file"),
            x=int(slot["x"]),
            y=int(slot["y"]),
            w=int(slot["w"]),
            h=int(slot["h"]),
            loss_weight=float(slot["loss_weight"]),
        )
        for slot in raw["slots"]
    )
    return AtlasProfile(canvas_w=int(raw["canvas_w"]), canvas_h=int(raw["canvas_h"]), slots=slots)


def load_export_profile(path: str | Path) -> dict[str, dict[str, int | str]]:
    raw = load_json(path)
    if "files" in raw:
        raw = raw["files"]
    return raw


def assert_slots_fit(atlas: AtlasProfile) -> None:
    for slot in atlas.slots:
        if slot.x < 0 or slot.y < 0 or slot.w <= 0 or slot.h <= 0:
            raise ValueError(f"invalid slot geometry: {slot.name}")
        if slot.x + slot.w > atlas.canvas_w or slot.y + slot.h > atlas.canvas_h:
            raise ValueError(f"slot exceeds atlas bounds: {slot.name}")

