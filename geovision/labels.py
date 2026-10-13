"""Rule-based change labels from spectral-index differencing.

Used (a) as the no-training baseline in the app/CLI and (b) as *weak supervision* to train the
deep model on real streamed scenes where no hand-labelled change masks exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from .indices import spectral_indices, valid_mask


@dataclass(frozen=True)
class Thresholds:
    smooth: int = 3            # box filter size applied to indices (suppresses speckle)
    water_after: float = 0.10  # NDWI after > this and ...
    water_before: float = -0.05  # ... NDWI before < this  -> newly flooded
    forest_ndvi: float = 0.70  # dense vegetation before
    cleared_ndvi: float = 0.35  # sparse / cleared after
    bui_after: float = 0.04    # built-up index after
    bui_gain: float = 0.15     # and increase of built-up index
    landuse_ndvi: float = 0.20  # any other large NDVI shift


def _smooth(d: dict, k: int) -> dict:
    return {n: (uniform_filter(v, size=k) if k > 1 else v) for n, v in d.items()}


def rule_based_labels(before: np.ndarray, after: np.ndarray, thr: Thresholds = Thresholds()) -> np.ndarray:
    """Return int64 [H,W]: 0 none, 1 urban, 2 deforestation, 3 flooding, 4 land-use change."""
    ib, ia = _smooth(spectral_indices(before), thr.smooth), _smooth(spectral_indices(after), thr.smooth)
    valid = valid_mask(before) & valid_mask(after)

    flood = (ia["ndwi"] > thr.water_after) & (ib["ndwi"] < thr.water_before)
    urban = (
        (ia["bui"] > thr.bui_after)
        & (ia["bui"] - ib["bui"] > thr.bui_gain)
        & (ia["ndvi"] < thr.cleared_ndvi)
        & ~flood
    )
    defor = (ib["ndvi"] >= thr.forest_ndvi) & (ia["ndvi"] < thr.cleared_ndvi) & ~flood & ~urban
    landuse = (np.abs(ia["ndvi"] - ib["ndvi"]) > thr.landuse_ndvi) & ~(flood | urban | defor)

    out = np.zeros(before.shape[1:], dtype=np.int64)
    out[landuse] = 4
    out[defor] = 2
    out[urban] = 1
    out[flood] = 3
    out[~valid] = 0
    return out
