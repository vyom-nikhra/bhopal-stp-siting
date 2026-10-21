"""Auto-STP backend with Earth Engine and OSM mocked (no network)."""

from pathlib import Path

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box

from apps.auto_stp.backend import main, osm_pipeline

RING = [[77.39, 23.24], [77.42, 23.24], [77.42, 23.27], [77.39, 23.27], [77.39, 23.24]]
BIG = box(77.400, 23.250, 77.401, 23.251)  # ~1.1 ha
SMALL = box(77.410, 23.250, 77.4105, 23.2505)  # ~0.28 ha, below the 0.56 ha footprint
NEAR_SCHOOL = box(77.402, 23.260, 77.403, 23.261)


def _fc(*geoms):
    return gpd.GeoDataFrame(geometry=list(geoms), crs="EPSG:4326").__geo_interface__


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("STP_EE_PROJECT", "test-project")
    monkeypatch.setattr(main.gee_pipeline, "init_earth_engine", lambda project: None)
    monkeypatch.setattr(main, "prefer_reachable", lambda host: None)
    with TestClient(main.app) as c:
        yield c


def test_overpass_url_from_env(monkeypatch):
    import osmnx as ox

    monkeypatch.setattr(ox.settings, "overpass_url", ox.settings.overpass_url)  # restore after
    monkeypatch.setenv("STP_EE_PROJECT", "test-project")
    monkeypatch.setenv("AUTO_STP_OVERPASS_URL", "https://overpass.example.org/api/")
    monkeypatch.setattr(main.gee_pipeline, "init_earth_engine", lambda project: None)
    probed = []
    monkeypatch.setattr(main, "prefer_reachable", probed.append)
    monkeypatch.setattr(main.gee_pipeline, "get_suitable_pixels", lambda ring, end_date: _fc(BIG))
    monkeypatch.setattr(
        main.osm_pipeline,
        "get_exclusion_zones",
        lambda ring: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    )
    with TestClient(main.app) as c:
        assert c.post("/analyze_region", json={"coordinates": [RING]}).status_code == 200
    assert ox.settings.overpass_url == "https://overpass.example.org/api"
    assert probed == ["overpass.example.org"]


def test_bad_overpass_url_fails_startup(monkeypatch):
    monkeypatch.setenv("STP_EE_PROJECT", "test-project")
    monkeypatch.setenv("AUTO_STP_OVERPASS_URL", "overpass-api.de")
    with pytest.raises(ValueError, match="Not a URL"):
        with TestClient(main.app):
            pass


def _mock(monkeypatch, candidates, exclusion):
    monkeypatch.setattr(main.gee_pipeline, "get_suitable_pixels", lambda ring, end_date: candidates)
    monkeypatch.setattr(main.osm_pipeline, "get_exclusion_zones", lambda ring: exclusion)


def test_min_area_is_plant_footprint():
    assert main.MIN_AREA_SQM == pytest.approx(12.7 * 0.044 * 1e4)


def test_filters_exclusion_and_small_sites(client, monkeypatch):
    exclusion = gpd.GeoDataFrame(geometry=[NEAR_SCHOOL.buffer(0.001)], crs="EPSG:4326")
    _mock(monkeypatch, _fc(BIG, SMALL, NEAR_SCHOOL), exclusion)
    r = client.post("/analyze_region", json={"type": "Polygon", "coordinates": [RING]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["total_sites"] == 1
    site = body["sites"][0]
    assert 10_000 < site["area_sqm"] < 12_500
    assert site["centroid_lng"] == pytest.approx(77.4005, abs=1e-4)
    feature = body["geojson"]["features"][0]
    assert feature["properties"]["area_sqm"] == site["area_sqm"]


def test_no_candidates(client, monkeypatch):
    _mock(monkeypatch, {"type": "FeatureCollection", "features": []}, None)
    r = client.post("/analyze_region", json={"coordinates": [RING]})
    assert r.json()["status"] == "no_candidates"


def test_below_min_area(client, monkeypatch):
    _mock(monkeypatch, _fc(SMALL), gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    r = client.post("/analyze_region", json={"coordinates": [RING]})
    assert r.json()["status"] == "below_min_area"


def test_osm_failure_is_an_error_not_an_empty_exclusion(client, monkeypatch):
    def boom(ring):
        raise ConnectionError("Overpass unreachable")

    monkeypatch.setattr(main.gee_pipeline, "get_suitable_pixels", lambda ring, end_date: _fc(BIG))
    monkeypatch.setattr(main.osm_pipeline, "get_exclusion_zones", boom)
    r = client.post("/analyze_region", json={"coordinates": [RING]})
    assert r.status_code == 502
    assert "Overpass unreachable" in r.json()["detail"]


def test_too_many_polygons_is_422(client, monkeypatch):
    def too_many(ring, end_date):
        raise ValueError("6000 candidate polygons (> 5000); draw a smaller area")

    monkeypatch.setattr(main.gee_pipeline, "get_suitable_pixels", too_many)
    r = client.post("/analyze_region", json={"coordinates": [RING]})
    assert r.status_code == 422


def test_startup_requires_project(monkeypatch):
    monkeypatch.delenv("STP_EE_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="STP_EE_PROJECT"):
        with TestClient(main.app):
            pass


def test_osm_bbox_order_and_empty_tags(monkeypatch):
    calls = []

    def fake(bbox, tags):
        calls.append(bbox)
        if "natural" in tags:
            raise osm_pipeline.InsufficientResponseError("none")
        return gpd.GeoDataFrame(geometry=[BIG.centroid], crs="EPSG:4326")

    monkeypatch.setattr(osm_pipeline.ox, "features_from_bbox", fake)
    zones = osm_pipeline.get_exclusion_zones(RING)
    assert calls[0] == (77.39, 23.24, 77.42, 23.27)  # left, bottom, right, top
    assert len(zones) == 1
    assert zones.contains(BIG.centroid).all()


def test_osm_other_errors_propagate(monkeypatch):
    def fake(bbox, tags):
        raise ConnectionError("down")

    monkeypatch.setattr(osm_pipeline.ox, "features_from_bbox", fake)
    with pytest.raises(ConnectionError):
        osm_pipeline.get_exclusion_zones(RING)


def test_frontend_renders_without_backend():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(Path(__file__).parents[1] / "apps/auto_stp/frontend/app.py"), default_timeout=60
    ).run()
    assert not at.exception
    assert at.button[0].disabled  # nothing drawn yet
