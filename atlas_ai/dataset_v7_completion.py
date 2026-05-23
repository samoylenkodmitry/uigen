"""V7 asset-completion dataset.

Yields per-(skin, file) items for training the V7 completer branch on
partial-evidence inputs:

    target_rgb     [3, H, W]      clean exported BMP pixels in [0, 1]
    observed_mask  [1, H, W]      1 where the model is told the value, 0 hidden
    observed_rgb   [3, H, W]      target_rgb * observed_mask
    mode           str            which mask family produced this item

Masks are drawn fresh on every __getitem__ via
`atlas_ai.v7_masks.sample_v7_observed_mask`, seeded deterministically by
`(self.seed, self.epoch, index)` so the same index reproduces the same item
within an epoch. Call `set_epoch()` between epochs to draw fresh masks.

Observed masks are intersected with the static Cranamp support mask. The
completion branch must not receive unsupported BMP pixels as evidence because
the deployed observer/copy path can never see them.

The dataset is multi-skin: construct it with a dict[skin_id, skin_source]
where each skin_source is a directory containing the 11 trainable BMPs.
The 11 files are enumerated from `TRAINABLE_EXPORT_SPECS`; non-trainable
files (TEXT.bmp, NUMBERS.bmp, etc.) are never yielded.

Optional `provenance_pools` is a dict[file_name, list[np.ndarray]] of
visible_mask buffers from V6 labels. When provided, the dataset draws
provenance-mode masks from these pools; when absent, the provenance weight
is redistributed proportionally over the other mask modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.state_families import StateRect, load_state_families
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_masks import V7MaskWeights, sample_v7_observed_mask


def _load_skin_targets(skin_source: Path) -> dict[str, torch.Tensor]:
    """Load every trainable exported BMP from a single skin source directory.

    Returns dict[file_name -> Tensor[3, H, W] in [0, 1]]. Raises FileNotFoundError
    when a file is missing and ValueError when a BMP's dimensions disagree with
    TRAINABLE_EXPORT_SPECS.
    """
    skin_source = Path(skin_source)
    targets: dict[str, torch.Tensor] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        path = skin_source / spec.file_name
        if not path.exists():
            raise FileNotFoundError(
                f"missing trainable BMP for skin source {skin_source}: {spec.file_name}"
            )
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        if arr.shape[:2] != (spec.h, spec.w):
            raise ValueError(
                f"{path} shape {arr.shape[:2]} != spec ({spec.h}, {spec.w})"
            )
        targets[spec.file_name] = (
            torch.from_numpy(arr.transpose(2, 0, 1).copy()).contiguous()
        )
    return targets


class V7CompletionDataset(Dataset):
    """Asset-completion dataset over (skin_id, file_name) pairs.

    Args:
        skin_sources: dict[skin_id, Path] mapping each skin id to a directory
            containing its 11 trainable exported BMPs.
        state_families_path: path to a state-family YAML config (see
            configs/state_families_classic.yaml).
        provenance_pools: optional dict[file_name, list[np.ndarray]] of
            pre-recorded visible_mask buffers for the provenance mask mode.
            Each entry must match the corresponding file's (H, W) shape.
        mask_weights: V7MaskWeights mix (default: V7 plan weights).
        seed: base seed for per-item RNGs. Items at the same index reproduce
            the same mask across runs.
    """

    def __init__(
        self,
        skin_sources: dict[str, str | Path],
        state_families_path: str | Path,
        provenance_pools: dict[str, list[np.ndarray]] | None = None,
        mask_weights: V7MaskWeights = V7MaskWeights(),
        seed: int = 0,
    ):
        if not skin_sources:
            raise ValueError("V7CompletionDataset requires at least one skin source")
        self.skin_ids: list[str] = sorted(skin_sources.keys())
        self.targets: dict[str, dict[str, torch.Tensor]] = {
            skin_id: _load_skin_targets(Path(skin_sources[skin_id]))
            for skin_id in self.skin_ids
        }
        self.state_families: dict[str, list[StateRect]] = load_state_families(
            state_families_path
        )
        # Sanity: every trainable file should have a state-family entry.
        for spec in TRAINABLE_EXPORT_SPECS:
            if spec.file_name not in self.state_families:
                raise ValueError(
                    f"state_families config missing {spec.file_name}"
                )
        self.provenance_pools = self._validate_provenance(provenance_pools or {})
        self.mask_weights = mask_weights
        self.seed = int(seed)
        self.epoch = 0
        self.spec_by_file = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
        self.support_masks: dict[str, np.ndarray] = {
            file_name: mask.cpu().numpy().astype(np.uint8)
            for file_name, mask in load_support_masks().items()
        }
        # Item order is (skin, file) in TRAINABLE_EXPORT_SPECS sequence so the
        # dataset is reproducible and indices are stable.
        self.items: list[tuple[str, str]] = [
            (skin_id, spec.file_name)
            for skin_id in self.skin_ids
            for spec in TRAINABLE_EXPORT_SPECS
        ]

    @staticmethod
    def _validate_provenance(
        pools: dict[str, list[np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
        out: dict[str, list[np.ndarray]] = {}
        for file_name, masks in pools.items():
            spec = spec_by_name.get(file_name)
            if spec is None:
                raise ValueError(
                    f"provenance_pools references unknown file {file_name!r}"
                )
            cleaned: list[np.ndarray] = []
            for mask in masks:
                if mask.shape != (spec.h, spec.w):
                    raise ValueError(
                        f"provenance mask for {file_name} has shape {mask.shape}, "
                        f"expected ({spec.h}, {spec.w})"
                    )
                cleaned.append(np.ascontiguousarray(mask.astype(np.uint8)))
            if cleaned:
                out[file_name] = cleaned
        return out

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic mask stream for the next epoch."""
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        skin_id, file_name = self.items[index]
        spec = self.spec_by_file[file_name]
        target_rgb = self.targets[skin_id][file_name]
        rects = self.state_families.get(file_name, [])
        pool = self.provenance_pools.get(file_name)
        rng = np.random.default_rng((self.seed, self.epoch, index))
        mask_np, mode = sample_v7_observed_mask(
            rng,
            h=spec.h,
            w=spec.w,
            family_rects=rects,
            visible_masks=pool,
            weights=self.mask_weights,
        )
        mask_np = (
            mask_np.astype(np.uint8, copy=False) & self.support_masks[file_name]
        )
        mask = torch.from_numpy(
            np.ascontiguousarray(mask_np.astype(np.float32))
        ).unsqueeze(0)
        observed_rgb = target_rgb * mask
        return {
            "skin_id": skin_id,
            "file_name": file_name,
            "target_rgb": target_rgb,
            "observed_mask": mask,
            "observed_rgb": observed_rgb,
            "mode": mode,
        }


__all__ = ["V7CompletionDataset"]
