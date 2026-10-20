import geopandas as gpd
import numpy as np
import pytest
from rasterio.crs import CRS
from shapely.geometry import LineString, box

from stp import proximity
from stp.config import Grid

UTM = CRS.from_epsg(32643)
GRID = Grid(origin=(0.0, 50.0), res=10.0, shape=(5, 5), crs=UTM)


def test_distance_to_single_pixel_is_euclidean_in_metres():
    target = np.zeros((5, 5), "uint8")
    target[2, 2] = 1
    d = proximity.distance_to(target, 10.0)
    assert d[2, 2] == 0
    assert d[2, 4] == pytest.approx(20.0)
    assert d[0, 0] == pytest.approx(np.hypot(20, 20))


def test_distance_to_without_targets_raises():
    with pytest.raises(ValueError, match="No target"):
        proximity.distance_to(np.zeros((3, 3), "uint8"), 10.0)


def test_burn_vertical_line_hits_one_column():
    gdf = gpd.GeoDataFrame(geometry=[LineString([(25, 0), (25, 50)])], crs=UTM)
    burned = proximity.burn(gdf, GRID)
    assert burned[:, 2].all()
    assert burned.sum() == 5


def test_burn_polygon_uses_pixel_centres():
    # covers centres x in {5, 15}, y in {45, 35}; not the pixel centred at x=25
    gdf = gpd.GeoDataFrame(geometry=[box(0, 30, 21, 50)], crs=UTM)
    burned = proximity.burn(gdf, GRID)
    assert burned[:2, :2].all() and burned.sum() == 4


def test_distance_raster_sets_nodata_outside_boundary():
    gdf = gpd.GeoDataFrame(geometry=[LineString([(5, 0), (5, 50)])], crs=UTM)
    boundary = box(0, 0, 30, 50)  # centres at x=35, 45 are outside
    dist, target = proximity.distance_raster(gdf, GRID, boundary)
    assert (dist[:, 3:] == proximity.DIST_NODATA).all()
    assert dist[0, 2] == pytest.approx(20.0)
    assert target[:, 0].all()
