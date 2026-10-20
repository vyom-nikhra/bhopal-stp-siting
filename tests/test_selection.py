import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Point, box

from stp import selection

UTM = CRS.from_epsg(32643)
TF = Affine(10, 0, 740000, 0, -10, 2570000)  # inside Bhopal's UTM zone


def test_queries_chain_and_nodata():
    s = np.array([8.5, 8.5, 8.5, 7.9, np.nan])
    lulc = np.array([10, 10, 3, 10, 10])
    roads = np.array([7, 6, 9, 9, 9])
    q = selection.queries(s, lulc, roads, 8.0, 10, 7)
    assert q["query_01"].tolist() == [True, True, True, False, False]
    assert q["query_02"].tolist() == [True, True, False, False, False]
    assert q["query_03"].tolist() == [True, False, False, False, False]
    assert q["valid"].tolist() == [True, True, True, True, False]


def test_polygonize_connectivity():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)  # diagonal neighbours
    assert len(selection.polygonize(mask, TF, UTM, connectivity=4)) == 2
    assert len(selection.polygonize(mask, TF, UTM, connectivity=8)) == 1
    with pytest.raises(ValueError):
        selection.polygonize(mask, TF, UTM, connectivity=6)


def test_ellipsoidal_area_close_to_planar():
    gdf = gpd.GeoDataFrame(geometry=[box(740000, 2569900, 740100, 2570000)], crs=UTM)
    ell = selection.ellipsoidal_area(gdf)[0]
    # UTM scale distortion here is ~0.1%, so the true area is a little below 10,000 m2
    assert 9980 < ell < 10000


def test_exclusion_zones_buffer_points():
    pts = gpd.GeoDataFrame(geometry=[Point(740000, 2570000)], crs=UTM)
    z = selection.exclusion_zones([pts, pts], 500, 30)
    assert len(z) == 2
    assert z.geometry.area.iloc[0] == pytest.approx(np.pi * 500**2, rel=1e-3)


def test_exclusion_zones_requires_projected_crs():
    pts = gpd.GeoDataFrame(geometry=[Point(77.4, 23.2)], crs=4326)
    with pytest.raises(ValueError, match="projected"):
        selection.exclusion_zones([pts], 500, 30)


def test_select_sites_order_matters():
    # 100 m x 60 m site (~6000 m2); a buffer removes 30 m of it -> ~3000 m2 left
    cand = gpd.GeoDataFrame(geometry=[box(740000, 2570000, 740100, 2570060)], crs=UTM)
    excl = gpd.GeoDataFrame(geometry=[box(740070, 2569900, 740300, 2570200)], crs=UTM)
    qgis = selection.select_sites(cand, excl, 5000, "filter_then_difference")
    fixed = selection.select_sites(cand, excl, 5000, "difference_then_filter")
    assert len(qgis) == 1 and qgis["area_sqm"].iloc[0] < 5000  # kept despite being too small
    assert len(fixed) == 0
    with pytest.raises(ValueError):
        selection.select_sites(cand, excl, 5000, "sideways")
