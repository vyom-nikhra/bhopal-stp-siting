"""Auto-STP: exclusion zones from OpenStreetMap (schools, hospitals, clinics, water).

Changes from the original app:
- A failed download now raises. The original caught every exception and returned
  "no exclusion zones", so an Overpass outage silently approved sites next to schools.
  Only "nothing matches these tags here" (InsufficientResponseError) means "no features".
- osmnx 2.x takes `bbox=(left, bottom, right, top)`; the original passed
  (north, south, east, west), which osmnx 2 reads as a different box.
- The UTM zone comes from geopandas' `estimate_utm_crs()` instead of a hand-written helper.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import geopandas as gpd
import osmnx as ox
import pandas as pd
from osmnx._errors import InsufficientResponseError

logger = logging.getLogger(__name__)

TAG_SETS = ({"amenity": ["school", "hospital", "clinic"]}, {"natural": "water"})
BUFFER_M = 500


def use_overpass(url: str) -> str:
    """Point osmnx at an Overpass API endpoint (e.g. a mirror or your own instance).

    Returns the endpoint's host name. The public overpass-api.de is often at capacity
    and asks automated users to run their own instance.
    """
    host = urlparse(url).hostname
    if not host:
        raise ValueError(f"Not a URL: {url!r} (expected e.g. https://overpass-api.de/api)")
    ox.settings.overpass_url = url.rstrip("/")
    logger.info("Overpass endpoint: %s", ox.settings.overpass_url)
    return host


def _fetch(bbox: tuple[float, float, float, float], tags: dict) -> gpd.GeoDataFrame | None:
    try:
        gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
    except InsufficientResponseError:
        logger.info("No OSM features for %s in this area.", tags)
        return None
    return gdf[["geometry"]].copy()


def get_exclusion_zones(ring: list[list[float]]) -> gpd.GeoDataFrame:
    """Merged 500 m buffers around sensitive features inside the ring's bounding box."""
    lngs, lats = [c[0] for c in ring], [c[1] for c in ring]
    bbox = (min(lngs), min(lats), max(lngs), max(lats))  # left, bottom, right, top
    frames = [f for tags in TAG_SETS if (f := _fetch(bbox, tags)) is not None]
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    utm = combined.estimate_utm_crs()
    buffered = combined.to_crs(utm).buffer(BUFFER_M)
    merged = gpd.GeoDataFrame(geometry=[buffered.union_all()], crs=utm).to_crs("EPSG:4326")
    logger.info("Exclusion zone from %d features (%d m buffer).", len(combined), BUFFER_M)
    return merged
