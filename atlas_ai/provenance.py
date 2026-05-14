from __future__ import annotations

import numpy as np


def encode_provenance(slot_id: int, atlas_x: int, atlas_y: int) -> int:
    if not (0 <= atlas_x < 1024 and 0 <= atlas_y < 1024):
        raise ValueError("atlas_x and atlas_y must be < 1024")
    return 1 + (slot_id << 20) + (atlas_y << 10) + atlas_x


def decode_provenance(value: int) -> tuple[int, int, int] | None:
    if value == 0:
        return None
    raw = value - 1
    slot_id = raw >> 20
    atlas_y = (raw >> 10) & 0x3FF
    atlas_x = raw & 0x3FF
    return slot_id, atlas_x, atlas_y


def provenance_to_visible_mask(buffer: np.ndarray) -> np.ndarray:
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    for value in np.unique(buffer):
        decoded = decode_provenance(int(value))
        if decoded is None:
            continue
        _, atlas_x, atlas_y = decoded
        mask[atlas_y, atlas_x] = 255
    return mask

