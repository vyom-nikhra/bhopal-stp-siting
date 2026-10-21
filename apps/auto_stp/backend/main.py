"""Auto-STP FastAPI backend: POST /analyze_region with a GeoJSON polygon.

Run from the repo root (needs Earth Engine credentials and Overpass access):
    STP_EE_PROJECT=<your-project> uvicorn apps.auto_stp.backend.main:app
Optional: AUTO_STP_OVERPASS_URL=<Overpass API base URL> (default https://overpass-api.de/api).

Changes from the original app:
- Earth Engine is initialised once at startup.
- The endpoint is a plain `def`, so FastAPI runs the slow Earth Engine / Overpass calls in
  a worker thread instead of blocking the event loop (the original was `async def`).
- Areas are ellipsoidal (as QGIS's $area and the pipeline), the UTM zone comes from
  `estimate_utm_crs()`, and the minimum area is the final model's 0.56 ha plant footprint.
- Each returned site carries its area and centroid as GeoJSON properties.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apps.auto_stp.backend import gee_pipeline, osm_pipeline
from stp.config import load_config
from stp.net import prefer_reachable
from stp.selection import ellipsoidal_area

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("auto_stp")

_model = load_config()["model"]
MIN_AREA_SQM = _model["plant_capacity_mld"] * _model["land_ha_per_mld"] * 1e4
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api"
overpass_host = ""  # set at startup from AUTO_STP_OVERPASS_URL


@asynccontextmanager
async def lifespan(_app: FastAPI):
    project = os.environ.get("STP_EE_PROJECT")
    if not project:
        raise RuntimeError("Set STP_EE_PROJECT to your Earth Engine project id")
    global overpass_host
    overpass_host = osm_pipeline.use_overpass(
        os.environ.get("AUTO_STP_OVERPASS_URL", DEFAULT_OVERPASS_URL)
    )
    gee_pipeline.init_earth_engine(project)
    yield


app = FastAPI(title="Auto-STP API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    type: str = Field(default="Polygon")
    coordinates: list[list[list[float]]] = Field(..., description="[[[lng, lat], ...]]")


class SiteSummary(BaseModel):
    area_sqm: float
    centroid_lng: float
    centroid_lat: float


class AnalyzeResponse(BaseModel):
    status: str
    total_sites: int
    total_usable_area_sqm: float
    geojson: dict[str, Any]
    sites: list[SiteSummary]


def _empty(status: str) -> AnalyzeResponse:
    return AnalyzeResponse(
        status=status,
        total_sites=0,
        total_usable_area_sqm=0.0,
        geojson={"type": "FeatureCollection", "features": []},
        sites=[],
    )


@app.post("/analyze_region", response_model=AnalyzeResponse)
def analyze_region(payload: AnalyzeRequest) -> AnalyzeResponse:
    ring = payload.coordinates[0]
    if len(ring) < 4:
        raise HTTPException(status_code=422, detail="Polygon ring needs at least 4 points")

    try:
        candidate_fc = gee_pipeline.get_suitable_pixels(ring, end_date=date.today().isoformat())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Earth Engine step failed")
        raise HTTPException(status_code=502, detail=f"Earth Engine error: {exc}") from exc
    if not candidate_fc.get("features"):
        return _empty("no_candidates")
    candidates = gpd.GeoDataFrame.from_features(candidate_fc["features"], crs="EPSG:4326")

    try:
        prefer_reachable(overpass_host)
        exclusion = osm_pipeline.get_exclusion_zones(ring)
    except Exception as exc:
        logger.exception("OSM step failed")
        raise HTTPException(status_code=502, detail=f"OSM error: {exc}") from exc

    sites = candidates[["geometry"]].copy()
    if not exclusion.empty:
        sites["geometry"] = sites.geometry.difference(exclusion.union_all())
        sites = sites[~sites.is_empty]
    if sites.empty:
        return _empty("all_excluded")

    sites["area_sqm"] = ellipsoidal_area(sites)
    sites = sites[sites["area_sqm"] >= MIN_AREA_SQM].reset_index(drop=True)
    if sites.empty:
        return _empty("below_min_area")

    cent = sites.to_crs(sites.estimate_utm_crs()).centroid.to_crs(4326)
    sites["centroid_lng"], sites["centroid_lat"] = cent.x.round(6), cent.y.round(6)
    sites["area_sqm"] = sites["area_sqm"].round(1)
    summaries = [
        SiteSummary(area_sqm=r.area_sqm, centroid_lng=r.centroid_lng, centroid_lat=r.centroid_lat)
        for r in sites.itertuples()
    ]
    return AnalyzeResponse(
        status="success",
        total_sites=len(sites),
        total_usable_area_sqm=round(float(sites["area_sqm"].sum()), 1),
        geojson=sites.__geo_interface__,
        sites=summaries,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
