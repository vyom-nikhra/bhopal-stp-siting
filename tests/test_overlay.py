import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from stp import overlay, standardize
from stp.config import Grid

UTM = CRS.from_epsg(32643)


def test_rescale_score_linear_and_inverted():
    x = np.array([0.0, 50.0, 100.0])
    valid = np.ones(3, bool)
    assert standardize.rescale_score(x, valid, invert=False).tolist() == [1, 6, 10]
    assert standardize.rescale_score(x, valid, invert=True).tolist() == [10, 5, 1]


def test_rescale_score_rounds_input_first():
    # Slope-like input: with integer rounding first, 2.4 deg -> 2 and 2.6 deg -> 3,
    # which land in different score bins over a 0..46 range.
    x = np.array([0.0, 2.4, 2.6, 46.0])
    s = standardize.rescale_score(x, np.ones(4, bool), invert=True)
    assert s.tolist() == [10, 10, 9, 1]


def test_rescale_score_half_up_and_nodata():
    x = np.array([0.0, 1.0, 2.0, -9999.0])
    valid = np.array([True, True, True, False])
    # 1 + 9 * 1/2 = 5.5 -> 6 (half up; numpy's round would give 6 too, 4.5 would differ)
    s = standardize.rescale_score(x, valid, invert=False)
    assert s.tolist() == [1, 6, 10, standardize.SCORE_NODATA]
    assert standardize.round_half_up(np.array([4.5]))[0] == 5


def test_rescale_score_constant_layer_raises():
    with pytest.raises(ValueError, match="constant"):
        standardize.rescale_score(np.ones(3), np.ones(3, bool), invert=False)


def test_align_nearest_from_coarser_grid():
    # 20 m source 2x2 -> 10 m 4x4 template: each source pixel becomes a 2x2 block
    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    da = overlay.to_dataarray(src, Affine(20, 0, 0, 0, -20, 40), UTM, -9999.0)
    tmpl = overlay.grid_template(Grid((0.0, 40.0), 10.0, (4, 4), UTM))
    out = overlay.align(da, tmpl)
    assert (out == np.repeat(np.repeat(src, 2, 0), 2, 1)).all()


def test_to_dataarray_masks_nodata_and_qgis_float_min():
    a = np.array([[1.0, -9999.0], [-3.4e38, 2.0]])
    da = overlay.to_dataarray(a, Affine(10, 0, 0, 0, -10, 20), UTM, -9999.0)
    assert np.isnan(da.values).tolist() == [[False, True], [True, False]]


def test_weighted_sum_checks_weights_and_propagates_nan():
    layers = {"a": np.array([10.0, np.nan]), "b": np.array([0.0, 5.0])}
    s = overlay.weighted_sum(layers, {"a": 0.25, "b": 0.75})
    assert s[0] == pytest.approx(2.5) and np.isnan(s[1])
    with pytest.raises(ValueError, match="sum to"):
        overlay.weighted_sum(layers, {"a": 0.5, "b": 0.6})
    with pytest.raises(ValueError, match="do not match"):
        overlay.weighted_sum(layers, {"a": 1.0})


def test_rescale_score_with_cap_is_flat_beyond_cap():
    x = np.array([0.0, 50.0, 100.0, 500.0, 9000.0])
    s = standardize.rescale_score(x, np.ones(5, bool), invert=False, cap=100)
    assert s.tolist() == [1, 6, 10, 10, 10]
    with pytest.raises(ValueError, match="cap"):
        standardize.rescale_score(x, np.ones(5, bool), invert=False, cap=0)
