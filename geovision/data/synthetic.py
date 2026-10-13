"""Synthetic bi-temporal Sentinel-2-like scenes with exact change labels.

Lets the whole project (training, inference, explainability, tests, the app) run offline.
Land cover -> spectral signatures in config.BANDS order (blue, green, red, nir, swir16, swir22).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

LAND_COVER = ["water", "forest", "cropland", "bare", "urban"]
SIGNATURES = np.array(
    [
        [0.06, 0.07, 0.04, 0.02, 0.01, 0.01],  # water
        [0.03, 0.06, 0.03, 0.42, 0.17, 0.08],  # forest
        [0.05, 0.08, 0.08, 0.30, 0.22, 0.12],  # cropland
        [0.09, 0.13, 0.17, 0.26, 0.36, 0.30],  # bare soil
        [0.18, 0.19, 0.20, 0.22, 0.30, 0.28],  # urban / impervious
    ],
    dtype="float32",
)
W, F, C, B, U = range(5)
# (source cover, new cover, change class id)  1 urban, 2 deforestation, 3 flooding, 4 land-use
TRANSITIONS = [
    (C, U, 1),
    (F, B, 2),
    (C, W, 3), (F, W, 3), (B, W, 3),
    (C, B, 4), (B, C, 4), (F, C, 4),
]


def _render(lc: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    refl = SIGNATURES[lc].transpose(2, 0, 1).copy()  # [6,H,W]
    gain = rng.uniform(0.92, 1.08)  # illumination / atmosphere difference between dates
    texture = 1 + 0.03 * gaussian_filter(rng.standard_normal(lc.shape), 1.5) / 0.1
    refl = refl * gain * texture + rng.normal(0, noise, refl.shape)
    return np.clip(refl, 0.001, 1).astype("float32")


def synthetic_scene(size: int = 256, seed: int = 0, noise: float = 0.012):
    """Return (before [6,H,W], after [6,H,W], label [H,W] int64)."""
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.standard_normal((size, size)), size / 20)
    edges = np.quantile(base, [0.08, 0.45, 0.75, 0.92])
    lc_before = np.digitize(base, edges)
    lc_after = lc_before.copy()
    label = np.zeros((size, size), dtype=np.int64)

    for src, dst, cls in TRANSITIONS:
        field = gaussian_filter(rng.standard_normal((size, size)), size / 32)
        blob = field > np.quantile(field, 1 - 0.07)
        m = blob & (lc_before == src) & (label == 0)
        lc_after[m] = dst
        label[m] = cls

    return _render(lc_before, rng, noise), _render(lc_after, rng, noise), label
