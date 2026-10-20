"""Phase 6b: describe and rank the final sites.

For each site: raster statistics inside it (rasterstats zonal statistics), the
distance to the nearest school, hospital, lake and residential area (geopandas
sjoin_nearest), its centroid in lon/lat, and its weight-sensitivity persistence.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats


def raster_stats(
    sites: gpd.GeoDataFrame, raster: str | Path, stats: list[str], prefix: str, nodata=None
) -> gpd.GeoDataFrame:
    """Add `<prefix>_<stat>` columns with zonal statistics of `raster` inside each site.

    Pixels count when their centre is inside the site (rasterstats default).
    """
    rows = zonal_stats(list(sites.geometry), str(raster), stats=stats, nodata=nodata)
    out = sites.copy()
    for stat in stats:
        out[f"{prefix}_{stat}"] = [r[stat] for r in rows]
    missing = out[[f"{prefix}_{s}" for s in stats]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{prefix}: no valid pixels inside sites {list(out.index[missing])}")
    return out


def nearest_distance(
    sites: gpd.GeoDataFrame, features: gpd.GeoDataFrame, column: str
) -> gpd.GeoDataFrame:
    """Add `column` = distance (CRS units) from each site's edge to the nearest feature."""
    if sites.crs != features.crs:
        raise ValueError(f"CRS mismatch: {sites.crs} vs {features.crs}")
    joined = gpd.sjoin_nearest(
        sites[["geometry"]], features[["geometry"]], how="left", distance_col=column
    )
    # Ties produce duplicate rows; keep one distance per site.
    dist = joined.groupby(level=0)[column].min()
    out = sites.copy()
    out[column] = dist.reindex(out.index).to_numpy()
    return out


def rank_sites(sites: gpd.GeoDataFrame, robust_threshold: float) -> gpd.GeoDataFrame:
    """Rank by mean suitability (desc), then area (desc); flag weight-robust sites."""
    out = sites.sort_values(["suit_mean", "area_sqm"], ascending=[False, False]).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["robust"] = out["persistence"] >= robust_threshold
    return out.reset_index(drop=True)
