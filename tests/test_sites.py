import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Point, box

from stp import io, sites

UTM = CRS.from_epsg(32643)


def _sites():
    return gpd.GeoDataFrame(
        {"area_sqm": [400.0, 400.0], "persistence": [0.9, 0.4]},
        geometry=[box(0, 20, 20, 40), box(20, 0, 40, 20)],
        crs=UTM,
    )


def test_raster_stats_inside_sites(tmp_path):
    arr = np.array([[1, 1, 5, 5], [1, 1, 5, 5], [9, 9, 3, 3], [9, 9, 3, 7]], dtype="float32")
    path = io.write_cog(tmp_path / "r.tif", arr, Affine(10, 0, 0, 0, -10, 40), UTM, -9999.0)
    out = sites.raster_stats(_sites(), path, ["mean", "max"], "v")
    assert out["v_mean"].tolist() == [1.0, 4.0]
    assert out["v_max"].tolist() == [1.0, 7.0]


def test_raster_stats_raises_when_site_has_no_valid_pixels(tmp_path):
    arr = np.full((4, 4), -9999.0, dtype="float32")
    path = io.write_cog(tmp_path / "r.tif", arr, Affine(10, 0, 0, 0, -10, 40), UTM, -9999.0)
    with pytest.raises(ValueError, match="no valid pixels"):
        sites.raster_stats(_sites(), path, ["mean"], "v")


def test_nearest_distance_from_site_edge():
    feats = gpd.GeoDataFrame(geometry=[Point(25, 30), Point(100, 100)], crs=UTM)
    out = sites.nearest_distance(_sites(), feats, "d")
    assert out["d"].tolist() == [pytest.approx(5.0), pytest.approx(10.0)]


def test_nearest_distance_crs_mismatch():
    feats = gpd.GeoDataFrame(geometry=[Point(77.4, 23.2)], crs=4326)
    with pytest.raises(ValueError, match="CRS"):
        sites.nearest_distance(_sites(), feats, "d")


def test_rank_sites_orders_and_flags():
    s = _sites()
    s["suit_mean"] = [8.1, 8.3]
    ranked = sites.rank_sites(s, robust_threshold=0.75)
    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["suit_mean"].tolist() == [8.3, 8.1]
    assert ranked["robust"].tolist() == [False, True]
