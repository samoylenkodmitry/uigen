from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExportFileSpec:
    file_name: str
    slot: str
    x: int
    y: int
    w: int
    h: int
    weight: float


# Static Cranamp-supported BMPs. These are the exact pixels exported into the
# generated skin, not the larger padded atlas slot capacities.
TRAINABLE_EXPORT_SPECS: tuple[ExportFileSpec, ...] = (
    ExportFileSpec("MAIN.bmp", "MAIN", 0, 0, 275, 115, 1.0),
    ExportFileSpec("TITLEBAR.bmp", "TITLEBAR", 320, 0, 344, 87, 2.0),
    ExportFileSpec("CBUTTONS.bmp", "CBUTTONS", 664, 0, 136, 36, 4.0),
    ExportFileSpec("SHUFREP.bmp", "SHUFREP", 824, 0, 92, 85, 4.0),
    ExportFileSpec("MONOSTER.bmp", "MONOSTER", 952, 0, 56, 24, 4.0),
    ExportFileSpec("PLAYPAUS.bmp", "PLAYPAUS", 952, 32, 42, 9, 4.0),
    ExportFileSpec("EQMAIN.bmp", "EQMAIN", 0, 128, 275, 315, 1.0),
    ExportFileSpec("PLEDIT.bmp", "PLEDIT", 320, 128, 280, 186, 1.0),
    ExportFileSpec("POSBAR.bmp", "POSBAR", 320, 384, 307, 10, 4.0),
    ExportFileSpec("VOLUME.bmp", "VOLUME", 640, 128, 68, 433, 4.0),
    ExportFileSpec("BALANCE.bmp", "BALANCE", 736, 128, 47, 433, 4.0),
)

TRAINABLE_EXPORT_FILES: tuple[str, ...] = tuple(spec.file_name for spec in TRAINABLE_EXPORT_SPECS)


def spec_by_file() -> dict[str, ExportFileSpec]:
    return {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}


def crop_export_target(target_rgb: torch.Tensor, spec: ExportFileSpec) -> torch.Tensor:
    return target_rgb[..., spec.y:spec.y + spec.h, spec.x:spec.x + spec.w]


def blank_atlas_like_files(
    files: dict[str, torch.Tensor],
    *,
    canvas_h: int = 1024,
    canvas_w: int = 1024,
) -> torch.Tensor:
    sample = next(iter(files.values()))
    if sample.ndim == 4:
        batch = sample.shape[0]
        atlas = sample.new_zeros((batch, 3, canvas_h, canvas_w))
    elif sample.ndim == 3:
        atlas = sample.new_zeros((3, canvas_h, canvas_w))
    else:
        raise ValueError(f"expected 3D or 4D file tensor, got {tuple(sample.shape)}")

    for spec in TRAINABLE_EXPORT_SPECS:
        file_rgb = files[spec.file_name]
        if file_rgb.shape[-2:] != (spec.h, spec.w):
            raise ValueError(
                f"{spec.file_name} has shape {tuple(file_rgb.shape[-2:])}, "
                f"expected {(spec.h, spec.w)}"
            )
        atlas[..., spec.y:spec.y + spec.h, spec.x:spec.x + spec.w] = file_rgb
    return atlas

