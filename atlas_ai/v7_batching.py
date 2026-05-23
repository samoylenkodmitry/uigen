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

from typing import Iterator, Mapping, Sequence

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
    """

    def __init__(
        self,
        items: Sequence[tuple[str, str]],
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
        # Build groups in file_name order for stable defaults.
        self.groups: dict[str, list[int]] = {}
        for idx, (_skin, file_name) in enumerate(items):
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


__all__ = ["SameFileBatchSampler", "WeightedSameFileBatchSampler"]
