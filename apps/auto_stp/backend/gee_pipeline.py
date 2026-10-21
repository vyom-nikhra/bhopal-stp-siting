"""Auto-STP: candidate land from Google Earth Engine.

Changes from the original app (see apps/auto_stp/README.md):
- Earth Engine is initialised once at startup (`init_earth_engine`), never inside a
  request, and never with the interactive `ee.Authenticate()`.
- Candidate land uses the final model's land-cover rule instead of NDVI < 0.2 and
  NDBI > 0 on a summer composite, which selects fallow farmland: the Dynamic World
  12-month mode must be grass, scrub or bare, and the pixel must be built in < 50%
  of observations. Slope < 15 degrees (SRTM) is kept.
- Results are never silently degraded: no `bestEffort` (which coarsens the scale) and
  no silent truncation at 5,000 polygons; both raise instead.
"""

from __future__ import annotations

import logging
from typing import Any

import ee

logger = logging.getLogger(__name__)

ELIGIBLE_CLASSES = [2, 5, 7]  # Dynamic World: grass, shrub_and_scrub, bare
BUILT_CLASS = 6
MAX_POLYGONS = 5000


def init_earth_engine(project: str) -> None:
    """Initialise Earth Engine with existing credentials; raise with instructions if absent."""
    try:
        ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialise Earth Engine for project {project!r}. Run "
            "`earthengine authenticate` once, or configure a service account."
        ) from exc
    logger.info("Earth Engine initialised (project %s).", project)


def get_suitable_pixels(
    ring: list[list[float]],
    end_date: str,
    settlement_built_pct: float = 50,
    max_slope_deg: float = 15,
) -> dict[str, Any]:
    """Candidate polygons (GeoJSON, EPSG:4326) inside the polygon `ring` ([[lng, lat], ...])."""
    aoi = ee.Geometry.Polygon([ring])
    end = ee.Date(end_date)
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(aoi)
        .filterDate(end.advance(-1, "year"), end)
        .select("label")
    )
    n_images = dw.size().getInfo()
    if n_images == 0:
        raise RuntimeError(
            f"No Dynamic World images for this area in the 12 months before {end_date}"
        )
    mode = dw.reduce(ee.Reducer.mode())
    built_pct = dw.map(lambda i: i.eq(BUILT_CLASS)).sum().divide(dw.count()).multiply(100)
    open_land = mode.remap(ELIGIBLE_CLASSES, [1] * len(ELIGIBLE_CLASSES), 0)
    not_settled = built_pct.lt(settlement_built_pct)
    slope_ok = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).lt(max_slope_deg)
    suitable = open_land.And(not_settled).And(slope_ok).selfMask().clip(aoi)

    vectors = suitable.reduceToVectors(
        geometryType="polygon",
        reducer=ee.Reducer.countEvery(),
        scale=10,
        maxPixels=1e9,
        geometry=aoi,
        eightConnected=False,
    )
    n = vectors.size().getInfo()
    if n > MAX_POLYGONS:
        raise ValueError(f"{n} candidate polygons (> {MAX_POLYGONS}); draw a smaller area")
    logger.info("Dynamic World images: %d; candidate polygons: %d", n_images, n)
    return vectors.getInfo() if n else {"type": "FeatureCollection", "features": []}
