"""Phase 2: distance-to-feature rasters (roads, drainage, residential, water).

Reproduces docs/qgis_workflow.md section 4: clip each OSM layer to the boundary,
rasterise it (burn 1, pixel-centre rule) on the proximity grid, compute the
Euclidean distance in metres to the nearest burned pixel, and set pixels outside
the boundary to NoData.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry.base import BaseGeometry

from stp.config import Grid
from stp.preprocess import outside_mask

DIST_NODATA = -9999.0


def load_clipped(
    vectors_dir: str | Path, layer: str, boundary: BaseGeometry, crs
) -> gpd.GeoDataFrame:
    """Read a shapefile's geometry only (no attribute columns) and clip it to `boundary`."""
    path = Path(vectors_dir) / f"{layer}.shp"
    gdf = gpd.read_file(path, engine="pyogrio", columns=[])
    if gdf.crs is None:
        raise ValueError(f"{path.name} has no CRS")
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    clipped = gpd.clip(gdf, boundary)
    clipped = clipped[~clipped.is_empty]
    if clipped.empty:
        raise ValueError(f"{layer}: no features inside the boundary")
    return clipped


def burn(gdf: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    """Rasterise geometries onto `grid`: 1 where a feature is present, else 0."""
    return rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )


def distance_to(target: np.ndarray, res: float) -> np.ndarray:
    """Euclidean distance (map units) from each pixel centre to the nearest target pixel."""
    if not target.any():
        raise ValueError("No target pixels to measure distance to")
    # EDT measures distance to the nearest zero, so targets must be the zeros.
    return distance_transform_edt(target == 0, sampling=res).astype("float32")


def distance_raster(
    gdf: gpd.GeoDataFrame, grid: Grid, boundary: BaseGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distance raster with NoData outside the boundary, burned target mask)."""
    target = burn(gdf, grid)
    dist = distance_to(target, grid.res)
    dist[outside_mask(boundary, grid.shape, grid.transform)] = DIST_NODATA
    return dist, target
