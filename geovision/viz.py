"""Matplotlib report: before/after, change map, Grad-CAM, area chart, spectral evidence."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import CLASS_COLORS, CLASS_NAMES
from .indices import spectral_indices


def rgb(arr: np.ndarray, gain: float = 3.0) -> np.ndarray:
    return np.clip(np.stack([arr[2], arr[1], arr[0]], -1) * gain, 0, 1)


def overlay(base: np.ndarray, change_map: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    out = base.copy()
    for c in range(1, len(CLASS_NAMES)):
        m = change_map == c
        out[m] = (1 - alpha) * out[m] + alpha * (np.array(CLASS_COLORS[c]) / 255.0)
    return out


def make_report(pair, result, path: str | None = None, dpi: int = 120):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    ax = ax.ravel()
    ax[0].imshow(rgb(pair.before)); ax[0].set_title("Before")
    ax[1].imshow(rgb(pair.after)); ax[1].set_title("After")
    ax[2].imshow(overlay(rgb(pair.after) * 0.8, result.change_map))
    ax[2].set_title(f"Detected change ({result.method})")
    ax[2].legend(handles=[Patch(color=np.array(CLASS_COLORS[c]) / 255, label=CLASS_NAMES[c])
                          for c in range(1, len(CLASS_NAMES))], loc="lower right", fontsize=8, framealpha=0.9)

    if result.cam is not None:
        ax[3].imshow(rgb(pair.after) * 0.6)
        im = ax[3].imshow(result.cam, cmap="jet", alpha=0.55, vmin=0, vmax=1)
        ax[3].set_title("Grad-CAM: where the model looked")
        plt.colorbar(im, ax=ax[3], fraction=0.046)
    else:
        ib, ia = spectral_indices(pair.before), spectral_indices(pair.after)
        im = ax[3].imshow(ia["ndvi"] - ib["ndvi"], cmap="RdYlGn", vmin=-1, vmax=1)
        ax[3].set_title("ΔNDVI (after - before)")
        plt.colorbar(im, ax=ax[3], fraction=0.046)

    names = [e["name"] for e in result.insights["classes"]]
    ha = [e["hectares"] for e in result.insights["classes"]]
    ax[4].barh(names, ha, color=[np.array(CLASS_COLORS[i + 1]) / 255 for i in range(len(names))])
    ax[4].set_xlabel("hectares"); ax[4].set_title("Area changed"); ax[4].invert_yaxis()

    if pair.label is not None:
        ax[5].imshow(overlay(rgb(pair.after) * 0.8, pair.label)); ax[5].set_title("Ground truth (synthetic)")
    else:
        ib, ia = spectral_indices(pair.before), spectral_indices(pair.after)
        im = ax[5].imshow(ia["ndwi"] - ib["ndwi"], cmap="PuOr", vmin=-1, vmax=1)
        ax[5].set_title("ΔNDWI (after - before)"); plt.colorbar(im, ax=ax[5], fraction=0.046)
    for i in (0, 1, 2, 3, 5):
        ax[i].axis("off")
    fig.suptitle(f"GeoVision AI - {pair.name or 'analysis'}: {result.insights['headline']}", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi)
    return fig
