"""V7 same-file batch samplers.

V7CompletionDataset yields items with per-file spatial shapes (e.g.
MAIN.bmp is 115x275, PLAYPAUS.bmp is 9x42). Default DataLoader collate
stacks tensors, which fails on mixed shapes. Every batch must therefore
contain items from a single file_name.

Two samplers are provided:

  - SameFileBatchSampler        round-robin: every item appears in exactly
                                one batch per epoch. Cheap. Easy files get
                                the same gradient share as hard files.

  - WeightedSameFileBatchSampler probability-weighted: each step samples
                                a file by configured weight, then samples
                                B items from that file with replacement
                                (necessary for one-skin Gate A where every
                                file has exactly one item). Total step
                                count is set explicitly by `num_batches`.

The weighted variant is the right tool when one file's representation is
strictly harder than another's. EQMAIN's slider sheet needs more gradient
exposure than PLAYPAUS' three 9x9 sprites; round-robin gives both the same
share. The weighted sampler fixes that without changing per-step compute.
"""

from __future__ import annotations

from typing import Container, Hashable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Sampler


class SameFileBatchSampler(Sampler[list[int]]):
    """Yields batches of dataset indices that share file_name.

    Args:
        items: same-length-as-dataset sequence of (skin_id, file_name) pairs.
            Pulled from `V7CompletionDataset.items` in practice.
        batch_size: target batch size per yielded list. Final batch per file
            may be smaller if the file's group is not a multiple of batch_size.
        shuffle: shuffle within each file group and the batch order. When
            False, behaves deterministically (batches sorted by file then
            skin appearance order).
        generator: optional torch.Generator for reproducible shuffles.
        drop_last: when True, drops the final under-sized batch per file
            instead of yielding it. Default False so every item is covered.
        include_files: optional set of file_names to keep. When given, items
            whose file is not in the set are dropped (their indices are never
            yielded). Used by the eval to skip files that have no eligible
            mask mode under the requested mask weights. The yielded indices
            still address the original dataset.
    """

    def __init__(
        self,
        items: Sequence[tuple[str, str]],
        batch_size: int,
        *,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
        include_files: Container[str] | None = None,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        self.drop_last = bool(drop_last)
        # Build groups in file_name order for stable defaults. idx is the
        # index into `items`, i.e. the dataset index, so filtering by file
        # never shifts the remaining files' indices.
        self.groups: dict[str, list[int]] = {}
        for idx, (_skin, file_name) in enumerate(items):
            if include_files is not None and file_name not in include_files:
                continue
            self.groups.setdefault(file_name, []).append(idx)
        self._file_order = list(self.groups.keys())

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            n = len(indices)
            full = n // self.batch_size
            rem = n - full * self.batch_size
            total += full + (0 if (self.drop_last or rem == 0) else 1)
        return total

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for file_name in self._file_order:
            indices = list(self.groups[file_name])
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[i] for i in perm]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in perm]
        for b in batches:
            yield b


class SameKeyBatchSampler(Sampler[list[int]]):
    """Same-shape batching: deterministic single-pass batches whose items all
    share one group key.

    This is *same-shape* batching, NOT family-balanced exposure. Each group is
    covered once per epoch, so a group with more items yields more batches —
    item count controls gradient share. For the StateFamilyExpander that means
    a 28-frame family (784 pairs) gets ~200x the batches of a 2-frame family
    (4 pairs). Use it for eval (every pair once) and use
    WeightedSameKeyBatchSampler for balanced training exposure.

    Generalizes SameFileBatchSampler to any per-index grouping key (e.g.
    (file, family) for the StateFamilyExpander, where every batch must share a
    frame size). `group_keys[i]` is the group of dataset index i.

    Args:
        group_keys: same length as the dataset; hashable group key per index.
        batch_size: target items per yielded batch (final per-group batch may
            be smaller unless drop_last).
        shuffle: shuffle within each group and across batches.
        generator: optional torch.Generator for reproducible shuffles.
        drop_last: drop the final under-sized batch per group.
    """

    def __init__(
        self,
        group_keys: Sequence[Hashable],
        batch_size: int,
        *,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        self.drop_last = bool(drop_last)
        self.groups: dict[Hashable, list[int]] = {}
        for idx, key in enumerate(group_keys):
            self.groups.setdefault(key, []).append(idx)
        self._key_order = list(self.groups.keys())

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            n = len(indices)
            full = n // self.batch_size
            rem = n - full * self.batch_size
            total += full + (0 if (self.drop_last or rem == 0) else 1)
        return total

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for key in self._key_order:
            indices = list(self.groups[key])
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[i] for i in perm]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in perm]
        for b in batches:
            yield b


class WeightedSameKeyBatchSampler(Sampler[list[int]]):
    """Family-balanced batches: yields exactly `num_batches` same-key batches,
    keys drawn by probability, items within a key drawn (optionally weighted)
    with replacement.

    Fixes the same-shape sampler's exposure bias: with equal `key_weights`
    every group (e.g. each alternatives family) gets the same number of
    gradient steps regardless of how many items it has — so eject's 4 pairs and
    VOLUME's 784 pairs train equally. `item_weights` lets a caller downweight a
    subset within each key (e.g. identity frame pairs) without dropping them.

    Args:
        group_keys: per-index hashable group key (len == dataset).
        batch_size: items per yielded batch.
        num_batches: number of batches to yield (the optimizer step count).
        key_weights: mapping group_key -> weight. Missing/<=0 keys are dropped.
            Default: equal weight per present key.
        item_weights: optional per-index non-negative weight for within-key
            sampling. Default: uniform within each key.
        within_key_replacement: sample items within a key with replacement
            (default True; required when a key has fewer items than batch_size).
        generator: optional torch.Generator for reproducibility.
    """

    def __init__(
        self,
        group_keys: Sequence[Hashable],
        batch_size: int,
        num_batches: int,
        *,
        key_weights: Mapping[Hashable, float] | None = None,
        item_weights: Sequence[float] | None = None,
        within_key_replacement: bool = True,
        generator: torch.Generator | None = None,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.within_key_replacement = bool(within_key_replacement)
        self.generator = generator
        groups: dict[Hashable, list[int]] = {}
        for idx, key in enumerate(group_keys):
            groups.setdefault(key, []).append(idx)
        keys = list(groups.keys())
        kw = [float(key_weights.get(k, 1.0)) if key_weights is not None else 1.0 for k in keys]
        kept = [(k, w) for k, w in zip(keys, kw) if w > 0]
        if not kept:
            raise ValueError("no group keys with positive weight")
        self.keys = [k for k, _ in kept]
        self.key_groups = [groups[k] for k in self.keys]
        kwt = torch.tensor([w for _, w in kept], dtype=torch.float64)
        self.key_probs = (kwt / kwt.sum()).to(torch.float32)
        # Per-key within-group item probabilities.
        self.item_probs: list[torch.Tensor | None] = []
        for g in self.key_groups:
            if item_weights is None:
                self.item_probs.append(None)  # uniform
                continue
            w = torch.tensor([max(0.0, float(item_weights[i])) for i in g], dtype=torch.float64)
            if float(w.sum()) <= 0:
                w = torch.ones(len(g), dtype=torch.float64)
            self.item_probs.append((w / w.sum()).to(torch.float32))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            ki = int(torch.multinomial(self.key_probs, 1, generator=self.generator).item())
            group = self.key_groups[ki]
            probs = self.item_probs[ki]
            replace = self.within_key_replacement or len(group) < self.batch_size
            if probs is not None:
                pick = torch.multinomial(probs, self.batch_size, replacement=replace,
                                         generator=self.generator).tolist()
            elif replace:
                pick = torch.randint(0, len(group), (self.batch_size,),
                                     generator=self.generator).tolist()
            else:
                perm = torch.randperm(len(group), generator=self.generator).tolist()
                pick = perm[: self.batch_size]
            yield [group[i] for i in pick]


class WeightedSameFileBatchSampler(Sampler[list[int]]):
    """Yields exactly `num_batches` same-file batches, with file groups drawn
    by configured weights.

    Args:
        items: same-length-as-dataset sequence of (skin_id, file_name) pairs.
        batch_size: number of dataset indices per yielded list.
        file_weights: mapping[file_name -> non-negative weight]. Weights are
            normalized to a probability distribution. Files absent from the
            mapping are skipped. Any file present in `items` but with weight
            <= 0 is also skipped.
        num_batches: number of batches to yield per iteration. This is the
            "true optimizer step count" - the trainer no longer thinks in
            terms of epochs over the dataset.
        generator: optional torch.Generator for reproducible sampling.
        within_file_replacement: if True (default), samples within the file
            group with replacement. This is the only viable mode for
            one-skin Gate A where every file has exactly one item.

    Yields:
        list[int] of length `batch_size`, all from the same file group.
    """

    def __init__(
        self,
        items: Sequence[tuple[str, str]],
        batch_size: int,
        file_weights: Mapping[str, float],
        num_batches: int,
        *,
        generator: torch.Generator | None = None,
        within_file_replacement: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.generator = generator
        self.within_file_replacement = bool(within_file_replacement)
        groups: dict[str, list[int]] = {}
        for idx, (_skin, file_name) in enumerate(items):
            groups.setdefault(file_name, []).append(idx)
        kept_files = []
        kept_weights = []
        for file_name, group in groups.items():
            w = float(file_weights.get(file_name, 0.0))
            if w > 0 and group:
                kept_files.append(file_name)
                kept_weights.append(w)
        if not kept_files:
            raise ValueError(
                "no files with positive weight present in dataset items "
                f"(items contain {sorted(groups)}, weights {dict(file_weights)})"
            )
        self.groups = {name: groups[name] for name in kept_files}
        self.files = kept_files
        weights_t = torch.tensor(kept_weights, dtype=torch.float64)
        self.probs = (weights_t / weights_t.sum()).to(torch.float32)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            file_idx = int(torch.multinomial(
                self.probs, num_samples=1, generator=self.generator,
            ).item())
            group = self.groups[self.files[file_idx]]
            if self.within_file_replacement or len(group) < self.batch_size:
                pick = torch.randint(
                    0, len(group), (self.batch_size,), generator=self.generator,
                ).tolist()
                yield [group[i] for i in pick]
            else:
                perm = torch.randperm(len(group), generator=self.generator).tolist()
                yield [group[perm[i]] for i in range(self.batch_size)]


__all__ = [
    "SameFileBatchSampler",
    "SameKeyBatchSampler",
    "WeightedSameKeyBatchSampler",
    "WeightedSameFileBatchSampler",
]
