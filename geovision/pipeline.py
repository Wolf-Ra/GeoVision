"""High-level API: fetch (stream) a before/after pair, run change detection, explain, export."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .data.synthetic import synthetic_scene
from .explain import explain_map, summarize
from .indices import valid_mask
from .infer import predict_full
from .labels import rule_based_labels


@dataclass
class Pair:
    before: np.ndarray            # [6, H, W] reflectance 0..1 (0 = masked / no data)
    after: np.ndarray
    res: float                    # metres per pixel
    name: str = ""
    bbox: Optional[tuple] = None
    label: Optional[np.ndarray] = None   # ground truth (synthetic only)
    crs: Optional[str] = None
    x: Optional[np.ndarray] = None       # pixel-centre coordinates for georeferenced export
    y: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    change_map: np.ndarray
    method: str
    probs: Optional[np.ndarray]
    cam: Optional[np.ndarray]
    insights: dict


def fetch_pair(bbox, before: str, after: str, res: int = 20, max_cloud: float = 30.0,
               provider: str = "earth-search", name: str = "") -> Pair:
    """Stream two cloud-free Sentinel-2 composites straight into memory (no files written)."""
    import dask

    from .data.stac_loader import composite, to_array

    kw = dict(max_cloud=max_cloud, resolution=res, provider=provider, with_indices=False)
    t0, t1 = composite(bbox, before, **kw), composite(bbox, after, **kw)
    t0, t1 = dask.compute(t0, t1)  # both composites stream in parallel here
    h, w = min(t0.sizes["y"], t1.sizes["y"]), min(t0.sizes["x"], t1.sizes["x"])
    a, b = to_array(t0)[:, :h, :w], to_array(t1)[:, :h, :w]
    crs = None
    try:
        crs = str(t0.odc.crs)
    except Exception:
        pass
    return Pair(a, b, float(res), name=name, bbox=tuple(bbox), crs=crs,
                x=t0.x.values[:w], y=t0.y.values[:h],
                meta={"scenes_before": t0.attrs.get("n_scenes"), "scenes_after": t1.attrs.get("n_scenes"),
                      "before": before, "after": after})


def synthetic_pair(size: int = 256, seed: int = 123) -> Pair:
    a, b, y = synthetic_scene(size, seed)
    return Pair(a, b, 10.0, name=f"synthetic-{seed}", label=y, meta={"synthetic": True})


def run_analysis(pair: Pair, method: str = "rules", model=None, device: str = "cpu",
                 explain: bool = True, tile: int = 256, overlap: int = 32) -> Result:
    if method == "model":
        if model is None:
            raise ValueError("method='model' needs a loaded model")
        probs = predict_full(model, pair.before, pair.after, tile, overlap, device)
        change = probs.argmax(0).astype(np.int64)
        change[~(valid_mask(pair.before) & valid_mask(pair.after))] = 0
        cam = explain_map(model, pair.before, pair.after, change, tile, overlap, device) if explain else None
    else:
        probs, cam = None, None
        change = rule_based_labels(pair.before, pair.after)
    return Result(change, method, probs, cam, summarize(change, pair.before, pair.after, pair.res, probs))


def export_geotiff(path: str, change_map: np.ndarray, pair: Pair) -> None:
    """Write the change map as a georeferenced GeoTIFF (only for streamed, georeferenced pairs)."""
    import rasterio
    from rasterio.transform import from_origin

    if pair.x is None or pair.crs is None:
        raise ValueError("This pair has no georeferencing (synthetic data).")
    dx, dy = abs(pair.x[1] - pair.x[0]), abs(pair.y[1] - pair.y[0])
    transform = from_origin(pair.x[0] - dx / 2, max(pair.y[0], pair.y[-1]) + dy / 2, dx, dy)
    data = change_map.astype("uint8")
    if pair.y[0] < pair.y[-1]:  # south-up input -> flip to north-up
        data = data[::-1]
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype="uint8", crs=pair.crs, transform=transform, compress="deflate") as dst:
        dst.write(data, 1)
