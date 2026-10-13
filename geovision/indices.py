"""Spectral indices and preprocessing helpers on numpy arrays shaped [6, H, W] (config.BANDS order)."""
from __future__ import annotations

import numpy as np

from .config import NORM_MEAN, NORM_STD

EPS = 1e-6


def _nd(x, y):
    return (x - y) / (x + y + EPS)


def spectral_indices(arr: np.ndarray) -> dict[str, np.ndarray]:
    """NDVI vegetation, NDWI water, NDBI built-up/bare, NBR burn/clear-cut, BUI = NDBI - NDVI."""
    _, green, red, nir, swir1, swir2 = arr
    ndvi, ndbi = _nd(nir, red), _nd(swir1, nir)
    return {
        "ndvi": ndvi,
        "ndwi": _nd(green, nir),
        "ndbi": ndbi,
        "nbr": _nd(nir, swir2),
        "bui": ndbi - ndvi,
    }


def valid_mask(arr: np.ndarray) -> np.ndarray:
    """Pixels with any signal (cloud-masked / no-data pixels were set to 0)."""
    return arr.sum(axis=0) > 0


def normalize(arr: np.ndarray) -> np.ndarray:
    return ((arr - NORM_MEAN) / NORM_STD).astype("float32")
