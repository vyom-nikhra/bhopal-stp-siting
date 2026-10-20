"""Phase 5: from the suitability surface to final candidate sites.

Reproduces docs/qgis_workflow.md section 8:
  Query 1  suitability >= 8
  Query 2  Query 1 AND land-cover score == 10 (barren)
  Query 3  Query 2 AND roads score >= 7
  -> polygonize (4-connected) -> ellipsoidal area
  -> subtract 500 m buffers around schools and hospitals -> keep sites >= 5000 m2

QGIS applied the area filter *before* the buffer difference; `order` selects that
("filter_then_difference") or the corrected order ("difference_then_filter"), which
re-measures each site after the buffers are cut out.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import Geod
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

GEOD = Geod(ellps="WGS84")


def queries(
    suitability: np.ndarray,
    lulc_score: np.ndarray,
    roads_score: np.ndarray,
    suitability_min: float,
    lulc_required: float,
    roads_min: float,
) -> dict[str, np.ndarray]:
    """Boolean query masks (False where the suitability surface is NoData)."""
    valid = np.isfinite(suitability)
    q1 = valid & (suitability >= suitability_min)
    q2 = q1 & (lulc_score == lulc_required)
    q3 = q2 & (roads_score >= roads_min)
    return {"query_01": q1, "query_02": q2, "query_03": q3, "valid": valid}


def polygonize(mask: np.ndarray, transform: Affine, crs, connectivity: int = 4) -> gpd.GeoDataFrame:
    """Connected regions of True pixels as polygons (like gdal:polygonize, value 1 only)."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    geoms = [
        shape(geom)
        for geom, value in shapes(
            mask.astype("uint8"), mask=mask, connectivity=connectivity, transform=transform
        )
        if value == 1
    ]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def ellipsoidal_area(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Area in m2 on the WGS84 ellipsoid (QGIS's default `$area`), not the planar UTM area."""
    lonlat = gdf.to_crs(4326)
    return np.array([abs(GEOD.geometry_area_perimeter(g)[0]) for g in lonlat.geometry])


def exclusion_zones(
    points: list[gpd.GeoDataFrame], buffer_m: float, segments: int
) -> gpd.GeoDataFrame:
    """Buffer each layer's features and merge them (like native:buffer + mergevectorlayers)."""
    crs = points[0].crs
    if any(p.crs != crs for p in points):
        raise ValueError("Exclusion layers must share one CRS")
    if not crs.is_projected:
        raise ValueError(f"Buffering needs a projected CRS in metres, got {crs}")
    buffers = [
        gpd.GeoDataFrame(geometry=p.geometry.buffer(buffer_m, quad_segs=segments), crs=crs)
        for p in points
    ]
    return gpd.GeoDataFrame(geometry=np.concatenate([b.geometry.values for b in buffers]), crs=crs)


def select_sites(
    candidates: gpd.GeoDataFrame,
    cut: BaseGeometry,
    min_area: float,
    order: str,
) -> gpd.GeoDataFrame:
    """Apply the buffer difference and the minimum-area filter in the given order.

    `cut` is the merged exclusion geometry (e.g. `exclusion_zones(...).union_all()`).
    Returns sites with `area_sqm_before` (ellipsoidal area before the difference) and
    `area_sqm` (after it). The QGIS order filters on `area_sqm_before`.
    """
    if order not in ("filter_then_difference", "difference_then_filter"):
        raise ValueError(f"Unknown order {order!r}")
    sites = candidates.copy()
    sites["area_sqm_before"] = ellipsoidal_area(sites) if len(sites) else []
    if order == "filter_then_difference":
        sites = sites[np.round(sites["area_sqm_before"]) >= min_area].copy()
        sites["geometry"] = sites.geometry.difference(cut)
        sites = sites[~sites.is_empty].copy()
        sites["area_sqm"] = ellipsoidal_area(sites)
    else:
        sites["geometry"] = sites.geometry.difference(cut)
        sites = sites[~sites.is_empty].copy()
        sites["area_sqm"] = ellipsoidal_area(sites)
        sites = sites[sites["area_sqm"] >= min_area].copy()
    return sites.reset_index(drop=True)


def sites_from_surface(
    suitability: np.ndarray,
    lulc_score: np.ndarray,
    roads_score: np.ndarray,
    cut: BaseGeometry,
    sel: dict,
    transform: Affine,
    crs,
    order: str = "difference_then_filter",
) -> tuple[dict[str, np.ndarray], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """The whole Phase 5 chain on aligned arrays: (query masks, all polygons, final sites)."""
    q = queries(
        suitability,
        lulc_score,
        roads_score,
        sel["suitability_min"],
        sel["lulc_required"],
        sel["roads_score_min"],
    )
    polygons = polygonize(q["query_03"], transform, crs, sel["polygonize_connectivity"])
    if len(polygons) == 0:
        empty = gpd.GeoDataFrame({"area_sqm": []}, geometry=[], crs=crs)
        return q, empty, empty
    polygons["area_sqm"] = ellipsoidal_area(polygons)
    sites = select_sites(polygons, cut, sel["min_area_sqm"], order)
    return q, polygons, sites
