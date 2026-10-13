"""Random-crop tile dataset over in-memory (before, after, label) scenes. Nothing is read from disk."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..indices import normalize


class ArrayPairDataset(Dataset):
    def __init__(self, items, tile: int = 64, length: int = 2000, augment: bool = True):
        """items: list of (before[6,H,W], after[6,H,W], label[H,W])."""
        self.items = [i for i in items if min(i[2].shape) >= tile]
        if not self.items:
            raise ValueError("No scene is at least as large as the training tile.")
        self.tile, self.length, self.augment = tile, length, augment

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        a, b, y = self.items[np.random.randint(len(self.items))]
        h, w = y.shape
        r, c = np.random.randint(0, h - self.tile + 1), np.random.randint(0, w - self.tile + 1)
        sl = (slice(r, r + self.tile), slice(c, c + self.tile))
        a, b, y = a[(slice(None), *sl)], b[(slice(None), *sl)], y[sl]
        if self.augment:
            k = np.random.randint(4)
            a, b, y = (np.rot90(t, k, axes=(-2, -1)) for t in (a, b, y))
            if np.random.rand() < 0.5:
                a, b, y = (np.flip(t, axis=-1) for t in (a, b, y))
        return (
            torch.from_numpy(normalize(np.ascontiguousarray(a))),
            torch.from_numpy(normalize(np.ascontiguousarray(b))),
            torch.from_numpy(np.ascontiguousarray(y)).long(),
        )
