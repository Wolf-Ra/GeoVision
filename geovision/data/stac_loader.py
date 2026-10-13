"""
GeoVision AI - streaming satellite data access.

Searches public STAC catalogs and builds *lazy* xarray cubes backed by
Cloud-Optimized GeoTIFFs (COGs). Nothing is downloaded up front: when you call
.compute() / .values, only the HTTP byte-ranges covering your AOI and bands are
fetched, straight into memory.

Providers (no account needed for the default):
  - "earth-search"  : Element84 Earth Search on AWS (Sentinel-2 L2A, Landsat C2 L2). Open, no key.
  - "planetary"     : Microsoft Planetary Computer (needs `planetary-computer` for URL signing; free).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import xarray as xr
from odc.stac import configure_rio, load as odc_load
from pystac_client import Client

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "earth-search": "https://earth-search.aws.element84.com/v1",
    "planetary": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

# Sentinel-2 L2A band keys (same names on both providers except where noted)
S2_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
# Planetary Computer uses B02/B03/... asset keys instead of common names
_PC_BAND_MAP = {
    "blue": "B02", "green": "B03", "red": "B04", "nir": "B08",
    "swir16": "B11", "swir22": "B12", "scl": "SCL",
}

# Sentinel-2 Scene Classification Layer values to treat as invalid
# 0 no-data, 1 saturated, 3 cloud shadow, 8/9 cloud (med/high), 10 cirrus, 11 snow/ice
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)

# Sentinel-2 processing baseline 04.00 (2022-01-25 onward) added a +1000 DN offset
_BASELINE_04_DATE = np.datetime64("2022-01-25")


@dataclass(frozen=True)
class AOI:
    """Area of interest as lon/lat bbox (west, south, east, north)."""
    name: str
    bbox: tuple[float, float, float, float]
    description: str = ""


# Ready-made AOIs matching GeoVision's four change classes
PRESETS = {
    "urban_bengaluru": AOI("Bengaluru fringe", (77.62, 12.86, 77.74, 12.97), "Urban expansion"),
    "deforest_rondonia": AOI("Rondonia, Brazil", (-62.90, -10.20, -62.70, -10.00), "Deforestation"),
    "flood_assam": AOI("Brahmaputra, Assam", (91.60, 26.10, 91.85, 26.30), "Flooding (monsoon vs dry)"),
    "landuse_coimbatore": AOI("Coimbatore outskirts", (76.90, 10.95, 77.05, 11.08), "Land-use change"),
}


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def open_catalog(provider: Literal["earth-search", "planetary"] = "earth-search") -> Client:
    if provider == "planetary":
        try:
            import planetary_computer as pc
            return Client.open(PROVIDERS[provider], modifier=pc.sign_inplace)
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install planetary-computer for the 'planetary' provider") from e
    return Client.open(PROVIDERS[provider])


def search_scenes(
    bbox: Sequence[float],
    date_range: str,
    *,
    max_cloud: float = 30.0,
    provider: Literal["earth-search", "planetary"] = "earth-search",
    collection: str = "sentinel-2-l2a",
    limit: int = 200,
):
    """Return a list of STAC items. `date_range` like '2024-01-01/2024-03-31'."""
    cat = open_catalog(provider)
    search = cat.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No scenes for bbox={bbox}, {date_range}, cloud<{max_cloud}%. "
            "Widen the date range or raise max_cloud."
        )
    return items


# --------------------------------------------------------------------------- #
# Lazy load + preprocessing
# --------------------------------------------------------------------------- #
def load_cube(
    items,
    bbox: Sequence[float],
    *,
    bands: Sequence[str] = tuple(S2_BANDS),
    resolution: int = 10,
    provider: str = "earth-search",
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Build a lazy (dask-backed) cube [time, y, x] for the bbox.
    Only metadata is touched here; pixels stream when computed.
    """
    # Efficient remote COG reads (HTTP range requests, no disk cache)
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

    wanted = list(bands) + ["scl"]
    if provider == "planetary":
        wanted = [_PC_BAND_MAP[b] for b in wanted]

    ds = odc_load(
        items,
        bands=wanted,
        bbox=tuple(bbox),
        resolution=resolution,
        chunks=chunks or {"x": 1024, "y": 1024, "time": 1},
        groupby="solar_day",      # merge same-day tiles
        resampling={"scl": "nearest", "*": "bilinear"},
        dtype="uint16",
        nodata=0,
    )
    if provider == "planetary":
        ds = ds.rename({v: k for k, v in _PC_BAND_MAP.items() if v in ds})
    return ds


def to_reflectance(ds: xr.Dataset, bands: Sequence[str] = tuple(S2_BANDS)) -> xr.Dataset:
    """DN -> surface reflectance [0,1], handling the post-2022 +1000 offset."""
    offset = xr.where(ds.time >= _BASELINE_04_DATE, 1000.0, 0.0)
    out = ds.copy()
    for b in bands:
        out[b] = ((ds[b].astype("float32") - offset) / 10000.0).clip(0, 1)
    return out


def mask_clouds(ds: xr.Dataset) -> xr.Dataset:
    """Set invalid pixels (cloud, shadow, cirrus, snow, no-data) to NaN using the SCL band."""
    bad = ds["scl"].isin(SCL_INVALID)
    bands = [b for b in ds.data_vars if b != "scl"]
    return ds[bands].where(~bad)


def add_indices(ds: xr.Dataset) -> xr.Dataset:
    """NDVI (vegetation), NDWI (water), NDBI (built-up), NBR (burn/clear-cut)."""
    eps = 1e-6
    ds = ds.copy()
    ds["ndvi"] = (ds.nir - ds.red) / (ds.nir + ds.red + eps)
    ds["ndwi"] = (ds.green - ds.nir) / (ds.green + ds.nir + eps)
    ds["ndbi"] = (ds.swir16 - ds.nir) / (ds.swir16 + ds.nir + eps)
    ds["nbr"] = (ds.nir - ds.swir22) / (ds.nir + ds.swir22 + eps)
    return ds


def composite(
    bbox: Sequence[float],
    date_range: str,
    *,
    max_cloud: float = 30.0,
    resolution: int = 10,
    provider: Literal["earth-search", "planetary"] = "earth-search",
    with_indices: bool = True,
) -> xr.Dataset:
    """
    Cloud-free median composite for one time window. Returns a lazy 2D Dataset
    (y, x) - call .compute() to stream the pixels.
    """
    items = search_scenes(bbox, date_range, max_cloud=max_cloud, provider=provider)
    cube = load_cube(items, bbox, resolution=resolution, provider=provider)
    cube = mask_clouds(to_reflectance(cube, S2_BANDS).assign(scl=cube["scl"]))
    comp = cube.median(dim="time", skipna=True)
    comp.attrs.update(n_scenes=len(items), date_range=date_range, bbox=tuple(bbox))
    return add_indices(comp) if with_indices else comp


def load_pair(
    bbox: Sequence[float],
    before: str,
    after: str,
    **kwargs,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Bi-temporal pair on an identical pixel grid, ready for change detection."""
    t0 = composite(bbox, before, **kwargs)
    t1 = composite(bbox, after, **kwargs)
    t1 = t1.interp_like(t0, method="nearest") if t0.sizes != t1.sizes else t1
    return t0, t1


# --------------------------------------------------------------------------- #
# Model-ready export (still no files on disk)
# --------------------------------------------------------------------------- #
def to_array(ds: xr.Dataset, bands: Sequence[str] = tuple(S2_BANDS)) -> np.ndarray:
    """Stack bands into a [C, H, W] float32 array (NaN -> 0) for a segmentation model."""
    arr = np.stack([ds[b].values for b in bands]).astype("float32")
    return np.nan_to_num(arr, nan=0.0)


def tile(arr: np.ndarray, size: int = 256, stride: int | None = None):
    """Yield (row, col, patch[C,size,size]) tiles for model inference."""
    stride = stride or size
    _, h, w = arr.shape
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            yield r, c, arr[:, r:r + size, c:c + size]
