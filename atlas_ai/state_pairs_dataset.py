"""State-pair dataset for the V7.1 StateFamilyExpander.

The StateFamilyExpander is the split-by-task replacement for the generic
completer's `state_family` mode. Instead of hiding sibling frames and asking a
U-Net to inpaint them from coords, we pose the well-defined task:

    given the SOURCE frame/state of an `alternatives` family, plus which frame
    it is (source_idx), which frame we want (target_idx), the family id and the
    file id, produce the TARGET frame/state.

Only families with `mask_role == "alternatives"` participate (POSBAR/MAIN/
TITLEBAR/PLEDIT components are excluded — they are not interchangeable states).
Within an alternatives family every rect is the same size (that is what makes
the frames interchangeable), so source/target crops batch cleanly per family.

Items are all ordered (source_idx, target_idx) pairs within each alternatives
family, per skin. Identity pairs (i == i) are included by default so the model
also learns to preserve unchanged content; the eval reports off-diagonal
(transition) metrics separately so the gate is not inflated by trivial copies.

Oracle conditioning note: `skin_index` is an ORACLE skin id. It is fine for a
capacity test (Gate S1/S2) but is not deployable conditioning — a shipping
model must derive skin context from observed assets / the render, not a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from atlas_ai.dataset_v7_completion import _load_skin_targets
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.state_families import StateRect, group_by_family, load_state_families
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_masks import has_alternatives


FILE_TO_ID: dict[str, int] = {s.file_name: i for i, s in enumerate(TRAINABLE_EXPORT_SPECS)}
FILE_DIMS: dict[str, tuple[int, int]] = {s.file_name: (s.w, s.h) for s in TRAINABLE_EXPORT_SPECS}

# Deployable, skin-INDEPENDENT pair geometry returned per (i, j) pair as
# `pair_geom`. In classic skins the per-family sprite layout is fixed across
# skins, so this vector is the same for a given (family, i, j) regardless of
# style — exactly the signal a geometry-only gate prior can use to localize the
# change on an UNSEEN skin (the S2a failure was the gate never opening there).
# All values normalized to ~[0,1] (delta index in [-1,1]):
#   src rect  (x,y,w,h)/file_dims          [4]
#   tgt rect  (x,y,w,h)/file_dims          [4]
#   src idx, tgt idx, delta idx / (n-1)    [3]
#   file (w,h) / GEOM_SIZE_NORM            [2]
GEOM_SIZE_NORM = 512.0
PAIR_GEOM_DIM = 13


def pair_geom_vector(src: StateRect, tgt: StateRect, i: int, j: int,
                     num_frames: int, file_w: int, file_h: int) -> list[float]:
    """13-d normalized geometry vector for the (src->tgt) pair. See PAIR_GEOM_DIM."""
    W = float(max(1, file_w))
    H = float(max(1, file_h))
    D = float(max(1, num_frames - 1))
    return [
        src.x / W, src.y / H, src.w / W, src.h / H,
        tgt.x / W, tgt.y / H, tgt.w / W, tgt.h / H,
        i / D, j / D, (j - i) / D,
        file_w / GEOM_SIZE_NORM, file_h / GEOM_SIZE_NORM,
    ]


@dataclass(frozen=True)
class AltFamily:
    """One alternatives family: its file, name, global id, and sibling rects
    (all the same size)."""
    file_name: str
    family: str
    family_id: int
    rects: tuple[StateRect, ...]

    @property
    def num_frames(self) -> int:
        return len(self.rects)

    @property
    def key(self) -> str:
        """Globally unique, human-readable family key. Family names alone are
        not unique ('slider_frames' is in VOLUME/BALANCE/EQMAIN)."""
        return f"{self.file_name.removesuffix('.bmp')}/{self.family}"


def collect_alt_families(state_families_path: str | Path) -> list[AltFamily]:
    """All `alternatives` families across trainable files, in a stable order
    (file order from TRAINABLE_EXPORT_SPECS, then family name sorted). The
    index in the returned list is the global family_id."""
    families = load_state_families(state_families_path)
    out: list[AltFamily] = []
    for spec in TRAINABLE_EXPORT_SPECS:
        grouped = group_by_family(families[spec.file_name])
        for fam_name in sorted(grouped):
            rects = grouped[fam_name]
            if not (len(rects) >= 2 and rects[0].mask_role == "alternatives"):
                continue
            sizes = {(r.w, r.h) for r in rects}
            if len(sizes) != 1:
                raise ValueError(
                    f"{spec.file_name}/{fam_name}: alternatives family must have "
                    f"uniform rect size, got {sorted(sizes)}"
                )
            out.append(AltFamily(spec.file_name, fam_name, len(out), tuple(rects)))
    return out


class StatePairsDataset(Dataset):
    """(skin, family, source_idx, target_idx) -> source/target frame crops.

    Args:
        skin_sources: dict[skin_id, dir] of the 11 trainable BMPs per skin.
        state_families_path: state-family YAML (alternatives families only used).
        include_identity: include i==j pairs (model learns to preserve). The
            eval still separates off-diagonal metrics for gating.
    """

    def __init__(
        self,
        skin_sources: dict[str, str | Path],
        state_families_path: str | Path,
        *,
        include_identity: bool = True,
        heldout_skins: list[str] | set[str] | None = None,
        heldout_pair_fraction: float = 0.0,
        pair_holdout_min_pairs: int = 20,
        local_delta: int = 2,
        split_seed: int = 0,
    ):
        """Args (S2 split additions, all optional — defaults reproduce S1):
            heldout_skins: skins whose every pair is split 'heldout' (unseen
                style; never trained). The deployability test.
            heldout_pair_fraction: fraction of each large family's off-diagonal
                (i,j) pairs held out across ALL train skins for the
                seen-skin / unseen-pair split. Only applied to families with
                >= pair_holdout_min_pairs off-diagonal pairs (2-frame buttons
                keep all pairs).
            local_delta: |i-j| <= local_delta is a 'local' transition (used by
                the mixed local/global pair sampler for big frame strips).
            split_seed: deterministic held-out-pair selection.
        """
        if not skin_sources:
            raise ValueError("StatePairsDataset requires at least one skin source")
        self.skin_ids: list[str] = sorted(skin_sources.keys())
        self.skin_id_to_index: dict[str, int] = {
            sid: i for i, sid in enumerate(self.skin_ids)
        }
        self.targets: dict[str, dict[str, torch.Tensor]] = {
            sid: _load_skin_targets(Path(skin_sources[sid])) for sid in self.skin_ids
        }
        self.alt_families: list[AltFamily] = collect_alt_families(state_families_path)
        if not self.alt_families:
            raise ValueError("no alternatives families found in state-family config")
        self.num_families = len(self.alt_families)
        self.max_frames = max(f.num_frames for f in self.alt_families)
        self.support_masks: dict[str, np.ndarray] = {
            fn: m.cpu().numpy().astype(np.uint8) for fn, m in load_support_masks().items()
        }
        self.include_identity = bool(include_identity)
        self.heldout_skins: set[str] = set(heldout_skins or [])
        self.local_delta = int(local_delta)
        # Held-out (i,j) pairs per family for the seen-skin/unseen-pair split,
        # consistent across all train skins. Only for big families.
        import random as _random
        rng = _random.Random(split_seed)
        self.heldout_pairs: dict[str, set[tuple[int, int]]] = {}
        for fam in self.alt_families:
            offdiag = [(i, j) for i in range(fam.num_frames)
                       for j in range(fam.num_frames) if i != j]
            if heldout_pair_fraction > 0 and len(offdiag) >= pair_holdout_min_pairs:
                k = max(1, int(round(heldout_pair_fraction * len(offdiag))))
                self.heldout_pairs[fam.key] = set(rng.sample(offdiag, k))
            else:
                self.heldout_pairs[fam.key] = set()
        # items: (skin_id, family_id, source_idx, target_idx) + parallel split /
        # locality labels.
        self.items: list[tuple[str, int, int, int]] = []
        self.split_of: list[str] = []
        self.is_local: list[bool] = []
        for sid in self.skin_ids:
            for fam in self.alt_families:
                n = fam.num_frames
                for i in range(n):
                    for j in range(n):
                        if not self.include_identity and i == j:
                            continue
                        self.items.append((sid, fam.family_id, i, j))
                        if sid in self.heldout_skins:
                            split = "heldout"
                        elif (i, j) in self.heldout_pairs[fam.key]:
                            split = "seen_val"
                        else:
                            split = "train"
                        self.split_of.append(split)
                        self.is_local.append(abs(i - j) <= self.local_delta)
        # Batch grouping key: unique family_key (file/family) so every batch
        # shares one frame size and the key matches family_key used in eval /
        # --only-family filters.
        self.group_keys: list[str] = [
            self.alt_families[fid].key for (_sid, fid, _i, _j) in self.items
        ]

    def training_item_weights(self, *, local_pair_prob: float = 0.5) -> list[float]:
        """Per-index within-family sampling weights for TRAIN items only.

        Non-train items (seen_val / heldout) get weight 0 so the sampler never
        draws them. Within each family's train pairs, local and global pairs
        are balanced to local_pair_prob vs (1 - local_pair_prob) so a big frame
        strip learns both neighbour moves and long jumps. (Per-family difficulty
        weighting is applied separately via the sampler's key_weights.)
        """
        from collections import defaultdict
        n_local: dict[str, int] = defaultdict(int)
        n_global: dict[str, int] = defaultdict(int)
        for idx, (_sid, fid, _i, _j) in enumerate(self.items):
            if self.split_of[idx] != "train":
                continue
            key = self.alt_families[fid].key
            if self.is_local[idx]:
                n_local[key] += 1
            else:
                n_global[key] += 1
        weights = [0.0] * len(self.items)
        for idx, (_sid, fid, _i, _j) in enumerate(self.items):
            if self.split_of[idx] != "train":
                continue
            key = self.alt_families[fid].key
            nl, ng = n_local[key], n_global[key]
            if nl and ng:
                weights[idx] = (local_pair_prob / nl) if self.is_local[idx] \
                    else ((1.0 - local_pair_prob) / ng)
            else:  # only one locality class present -> uniform
                denom = nl if self.is_local[idx] else ng
                weights[idx] = 1.0 / max(1, denom)
        return weights

    def __len__(self) -> int:
        return len(self.items)

    def _crop(self, full: torch.Tensor, r: StateRect) -> torch.Tensor:
        return full[:, r.y : r.y + r.h, r.x : r.x + r.w].contiguous()

    def __getitem__(self, index: int) -> dict:
        sid, fid, i, j = self.items[index]
        fam = self.alt_families[fid]
        src_rect, tgt_rect = fam.rects[i], fam.rects[j]
        full = self.targets[sid][fam.file_name]
        source_rgb = self._crop(full, src_rect)
        target_rgb = self._crop(full, tgt_rect)
        sup = self.support_masks[fam.file_name]
        tgt_sup = sup[tgt_rect.y : tgt_rect.y + tgt_rect.h, tgt_rect.x : tgt_rect.x + tgt_rect.w]
        target_support = torch.from_numpy(tgt_sup.astype(np.float32)).unsqueeze(0).contiguous()
        file_w, file_h = FILE_DIMS[fam.file_name]
        pair_geom = torch.tensor(
            pair_geom_vector(src_rect, tgt_rect, i, j, fam.num_frames, file_w, file_h),
            dtype=torch.float32,
        )  # [PAIR_GEOM_DIM]
        return {
            "source_rgb": source_rgb,          # [3, fh, fw]
            "target_rgb": target_rgb,          # [3, fh, fw]
            "target_support": target_support,  # [1, fh, fw] in {0,1}
            "pair_geom": pair_geom,            # [PAIR_GEOM_DIM] skin-independent geometry
            "source_idx": i,
            "target_idx": j,
            "family_id": fid,
            "family": fam.family,
            "family_key": fam.key,
            "file_name": fam.file_name,
            "file_id": FILE_TO_ID[fam.file_name],
            "skin_id": sid,
            "skin_index": self.skin_id_to_index[sid],
            "is_identity": int(i == j),
            "split": self.split_of[index],
            "is_local": int(self.is_local[index]),
        }


__all__ = ["StatePairsDataset", "AltFamily", "collect_alt_families", "FILE_TO_ID",
           "FILE_DIMS", "PAIR_GEOM_DIM", "pair_geom_vector"]
