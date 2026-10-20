import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from stp import io, parity

CRS_UTM = CRS.from_epsg(32643)


@pytest.fixture
def ref_raster(tmp_path):
    tf = Affine(10, 0, 0, 0, -10, 40)
    arr = np.array([[1, 2, 3, 4]] * 4, dtype="float32")
    arr[0, 0] = -9999
    path = io.write_cog(tmp_path / "ref.tif", arr, tf, CRS_UTM, -9999)
    return path, arr, tf


def test_compare_identical(ref_raster):
    path, arr, tf = ref_raster
    r = parity.compare("x", arr, tf, -9999, path)
    assert r.same_grid and r.max_abs_diff == 0 and r.exact_share == 1 and r.mask_agreement == 1


def test_compare_detects_difference_and_mask_mismatch(ref_raster):
    path, arr, tf = ref_raster
    ours = arr.copy()
    ours[1, 1] += 0.5
    ours[0, 0] = 1  # valid where the reference is NoData
    r = parity.compare("x", ours, tf, -9999, path)
    assert r.max_abs_diff == pytest.approx(0.5)
    assert r.mask_agreement == pytest.approx(15 / 16)


def test_compare_grid_mismatch_raises(ref_raster):
    path, arr, _ = ref_raster
    with pytest.raises(ValueError, match="grid mismatch"):
        parity.compare("x", arr, Affine(10, 0, 5, 0, -10, 40), -9999, path)


def test_write_cog_roundtrip(tmp_path):
    arr = np.arange(12, dtype="uint16").reshape(3, 4)
    tf = Affine(10, 0, 100, 0, -10, 200)
    path = io.write_cog(tmp_path / "a.tif", arr, tf, CRS_UTM, 0)
    with rasterio.open(path) as src:
        assert src.nodata == 0 and src.crs == CRS_UTM and src.transform == tf
        assert (src.read(1) == arr).all()
