import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from stp import sensitivity

BASE = {"a": 0.5, "b": 0.3, "c": 0.2}
TF = Affine(10, 0, 0, 0, -10, 40)
UTM = CRS.from_epsg(32643)


def test_perturb_weights_sum_to_one_and_stay_in_band():
    rng = np.random.default_rng(0)
    for _ in range(100):
        w = sensitivity.perturb_weights(BASE, 0.2, rng)
        assert sum(w.values()) == pytest.approx(1.0)
        for k in BASE:
            # renormalisation keeps each weight within a (1.2/0.8)x band of its base value
            assert BASE[k] * 0.8 / 1.2 <= w[k] <= BASE[k] * 1.2 / 0.8


def test_shift_one_keeps_ratios_of_the_others():
    w = sensitivity.shift_one(BASE, "a", 1.2)
    assert w["a"] == pytest.approx(0.6)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["b"] / w["c"] == pytest.approx(BASE["b"] / BASE["c"])
    with pytest.raises(KeyError):
        sensitivity.shift_one(BASE, "z", 1.2)
    with pytest.raises(ValueError):
        sensitivity.shift_one(BASE, "a", 2.5)


def test_site_rasters_and_coverage():
    sites = gpd.GeoDataFrame(geometry=[box(0, 20, 20, 40), box(20, 0, 40, 20)], crs=UTM)
    ids = sensitivity.site_ids(sites, (4, 4), TF)
    assert (ids[:2, :2] == 1).all() and (ids[2:, 2:] == 2).all() and ids.sum() == 4 + 8
    selected = np.zeros((4, 4), bool)
    selected[0, :2] = True  # half of site 1
    cover = sensitivity.coverage_by_site(ids, selected, 2)
    assert cover.tolist() == [0.5, 0.0]
    mask = sensitivity.site_mask(sites, (4, 4), TF)
    assert mask.sum() == 8
    assert not sensitivity.site_mask(sites.iloc[:0], (4, 4), TF).any()


def test_iou():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert sensitivity.iou(a, b) == pytest.approx(1 / 3)
    assert sensitivity.iou(np.zeros(3, bool), np.zeros(3, bool)) == 1.0
