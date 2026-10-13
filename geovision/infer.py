"""Tiled, overlap-averaged inference over arbitrarily large scenes."""
from __future__ import annotations

import numpy as np
import torch

from .indices import normalize


def pad_min(arr: np.ndarray, tile: int) -> np.ndarray:
    """Pad [..., H, W] so both spatial dims are at least `tile` (edge replicate)."""
    h, w = arr.shape[-2:]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)], mode="edge")


def tile_origins(h: int, w: int, tile: int, overlap: int):
    stride = max(tile - overlap, 1)

    def starts(n):
        s = list(range(0, max(n - tile, 0) + 1, stride))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s

    return [(y, x) for y in starts(h) for x in starts(w)]


@torch.no_grad()
def predict_full(model, before, after, tile: int = 256, overlap: int = 32, device: str = "cpu") -> np.ndarray:
    """Class probabilities [C, H, W] for a before/after pair of [6, H, W] reflectance arrays."""
    assert tile % 8 == 0, "tile must be divisible by 8"
    model.eval().to(device)
    h, w = before.shape[-2:]
    a, b = normalize(pad_min(before, tile)), normalize(pad_min(after, tile))
    H, W = a.shape[-2:]
    acc, cnt = None, np.zeros((H, W), dtype="float32")
    for y, x in tile_origins(H, W, tile, overlap):
        ta = torch.from_numpy(a[:, y:y + tile, x:x + tile])[None].to(device)
        tb = torch.from_numpy(b[:, y:y + tile, x:x + tile])[None].to(device)
        p = torch.softmax(model(ta, tb), dim=1)[0].cpu().numpy()
        if acc is None:
            acc = np.zeros((p.shape[0], H, W), dtype="float32")
        acc[:, y:y + tile, x:x + tile] += p
        cnt[y:y + tile, x:x + tile] += 1
    return (acc / cnt)[:, :h, :w]
