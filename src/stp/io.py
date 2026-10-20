"""Raster and vector I/O helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


def write_cog(
    path: str | Path, data: np.ndarray, transform: Affine, crs: CRS, nodata: float | int | None
) -> Path:
    """Write a (bands, rows, cols) or (rows, cols) array as a Cloud-Optimized GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = {
        "driver": "COG",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def read_raster(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """Read one band; returns (array, profile)."""
    with rasterio.open(path) as src:
        return src.read(band), src.profile


def read_boundary(path: str | Path, crs: CRS | str) -> gpd.GeoDataFrame:
    """Read the study-area boundary and reproject it to the working CRS."""
    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    return gdf.to_crs(crs)
