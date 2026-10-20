import math

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from stp import io, preprocess


def test_horn_slope_of_tilted_plane():
    # z rises 1 m per 1 m east -> 45 degrees everywhere in the interior
    res = 10.0
    cols = np.arange(6) * res
    dem = np.tile(cols, (5, 1)) + 100.0
    slope = preprocess.horn_slope(dem, res, nodata=0)
    assert np.allclose(slope[1:-1, 1:-1], 45.0)
    # outer ring is NoData, as gdaldem without -compute_edges
    assert (slope[0] == preprocess.SLOPE_NODATA).all()
    assert (slope[:, -1] == preprocess.SLOPE_NODATA).all()


def test_horn_slope_flat_is_zero():
    slope = preprocess.horn_slope(np.full((4, 4), 500.0), 12.5, nodata=0)
    assert np.allclose(slope[1:-1, 1:-1], 0.0)


def test_horn_slope_nodata_neighbour_invalidates_window():
    dem = np.full((5, 5), 100.0)
    dem[2, 2] = 0  # NoData
    slope = preprocess.horn_slope(dem, 10.0, nodata=0)
    # every interior cell has (2, 2) in its 3x3 window
    assert (slope[1:4, 1:4] == preprocess.SLOPE_NODATA).all()


def test_normalized_difference_handles_zero_and_invalid():
    a = np.array([[3, 0, 5]], dtype="uint16")
    b = np.array([[1, 0, 5]], dtype="uint16")
    valid = np.array([[True, True, False]])
    nd = preprocess.normalized_difference(a, b, valid)
    assert nd[0, 0] == pytest.approx(0.5)
    assert np.isnan(nd[0, 1])  # a + b == 0
    assert np.isnan(nd[0, 2])  # outside valid


def test_normalized_difference_offset_changes_result():
    a = np.array([[3000]], dtype="uint16")
    b = np.array([[2000]], dtype="uint16")
    valid = np.ones((1, 1), bool)
    raw = preprocess.normalized_difference(a, b, valid)
    corrected = preprocess.normalized_difference(a, b, valid, offset=-1000)
    assert raw[0, 0] == pytest.approx(0.2)
    assert corrected[0, 0] == pytest.approx(1 / 3)


def test_crop_window_inward_keeps_whole_pixels_inside_bounds():
    tf = Affine(10, 0, 1000, 0, -10, 5000)
    win = preprocess.crop_window((1003.0, 4901.0, 1047.0, 4995.0), tf)  # default inward
    assert (win.col_off, win.row_off) == (1, 1)
    assert (win.width, win.height) == (3, 8)
    left, top = tf @ (win.col_off, win.row_off)
    right, bottom = tf @ (win.col_off + win.width, win.row_off + win.height)
    assert left >= 1003 and right <= 1047 and bottom >= 4901 and top <= 4995


def test_crop_window_rejects_unknown_snap():
    with pytest.raises(ValueError, match="snap"):
        preprocess.crop_window((0, 0, 1, 1), Affine(1, 0, 0, 0, -1, 1), snap="nearest")


def test_crop_window_outward_covers_bounds():
    tf = Affine(10, 0, 1000, 0, -10, 5000)
    win = preprocess.crop_window((1003.0, 4901.0, 1047.0, 4995.0), tf, snap="outward")
    assert (win.col_off, win.row_off) == (0, 0)
    assert (win.width, win.height) == (5, 10)
    left, top = tf @ (win.col_off, win.row_off)
    right, bottom = tf @ (win.col_off + win.width, win.row_off + win.height)
    assert left <= 1003 and right >= 1047 and bottom <= 4901 and top >= 4995
    assert math.isclose((right - left) % 10, 0)


def test_outside_mask_uses_pixel_centres():
    tf = Affine(10, 0, 0, 0, -10, 30)  # 3x3 grid over (0,0)-(30,30)
    geom = box(0, 0, 16, 30)  # covers centres at x=5 and x=15, not x=25
    out = preprocess.outside_mask(geom, (3, 3), tf)
    assert not out[:, :2].any()
    assert out[:, 2].all()


def test_read_on_finer_grid_duplicates_pixels(tmp_path):
    # 20 m source, 3x3, origin (0, 60)
    src_arr = np.arange(9, dtype="uint16").reshape(3, 3)
    path = io.write_cog(
        tmp_path / "b11.tif", src_arr, Affine(20, 0, 0, 0, -20, 60), CRS.from_epsg(32643), 0
    )
    # 10 m output starting half a source pixel in (x=10, y=50), 4x4
    out_tf = Affine(10, 0, 10, 0, -10, 50)
    with rasterio.open(path) as src:
        out = preprocess.read_on_finer_grid(src, out_tf, (4, 4))
    expected = np.repeat(np.repeat(src_arr, 2, 0), 2, 1)[1:5, 1:5]
    assert (out == expected).all()


def test_read_on_finer_grid_rejects_non_integer_factor(tmp_path):
    import rasterio

    path = io.write_cog(
        tmp_path / "a.tif",
        np.zeros((4, 4), "uint16"),
        Affine(15, 0, 0, 0, -15, 60),
        CRS.from_epsg(32643),
        0,
    )
    with rasterio.open(path) as src, pytest.raises(ValueError, match="integer multiple"):
        preprocess.read_on_finer_grid(src, Affine(10, 0, 0, 0, -10, 60), (2, 2))
