"""Phase 4b: align criterion scores onto one grid and combine them with weights.

QGIS's Raster Calculator silently resampled the shifted 10 m proximity grid and the
12.5 m DEM grid onto the Sentinel 10 m grid. Here the alignment is explicit:
each layer is reprojected onto the Sentinel grid with nearest-neighbour resampling
(scores are categorical 1-10 values, so they must not be interpolated).
"""

from __future__ import annotations

import math

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling

from stp.config import Grid

NODATA = -9999.0


def to_dataarray(data: np.ndarray, transform: Affine, crs, nodata: float) -> xr.DataArray:
    """Wrap a 2-D array as a georeferenced DataArray with NaN for NoData."""
    rows, cols = data.shape
    xs = transform.c + transform.a * (np.arange(cols) + 0.5)
    ys = transform.f + transform.e * (np.arange(rows) + 0.5)
    arr = data.astype("float64")
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(arr < -1e30, np.nan, arr)  # QGIS float NoData sentinel
    da = xr.DataArray(arr, coords={"y": ys, "x": xs}, dims=("y", "x"))
    return da.rio.write_crs(crs).rio.write_transform(transform).rio.write_nodata(np.nan)


def grid_template(grid: Grid) -> xr.DataArray:
    return to_dataarray(np.zeros(grid.shape), grid.transform, grid.crs, None)


def align(da: xr.DataArray, template: xr.DataArray) -> np.ndarray:
    """Reproject `da` onto `template`'s grid (nearest neighbour); NaN = NoData."""
    out = da.rio.reproject_match(template, resampling=Resampling.nearest, nodata=np.nan)
    if out.shape != template.shape:
        raise ValueError(f"Aligned shape {out.shape} != template {template.shape}")
    return out.values


def weighted_sum(layers: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Sum of weight x score; NaN wherever any layer is NaN. Weights must sum to 1."""
    if set(layers) != set(weights):
        raise ValueError(f"Layers {sorted(layers)} do not match weights {sorted(weights)}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"Weights sum to {total}, expected 1")
    out = np.zeros(next(iter(layers.values())).shape, dtype="float64")
    for name, w in weights.items():
        # Cast first: float32 products would put exact 8.0 ties a hair either side of 8.
        out += w * np.asarray(layers[name], dtype="float64")  # NaN propagates
    return out
