"""V7 same-file batch sampler.

V7CompletionDataset yields items with per-file spatial shapes (e.g.
MAIN.bmp is 115x275, PLAYPAUS.bmp is 9x42). Default DataLoader collate
stacks tensors, which fails on mixed shapes - the user-side caveat is:
"normal mixed-file DataLoader(batch_size > 1, shuffle=True) will fail. The
trainer should use batch size 1, same-file batching, or a custom
sampler/collate."

`SameFileBatchSampler` is that sampler: it groups dataset indices by their
file_name (read from `dataset.items`), shuffles within each group, slices
into batches of the requested size, then shuffles the order in which the
batches are emitted. Every batch only contains items from one file, so
default collate works.

For full epoch coverage: every item appears in exactly one batch per epoch,
even if a group is smaller than the batch size (it produces a short batch).
"""

from __future__ import annotations

from typing import Iterator, Sequence

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


__all__ = ["SameFileBatchSampler"]
