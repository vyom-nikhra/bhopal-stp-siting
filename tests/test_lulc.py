import numpy as np
import pytest

from stp import lulc


def test_pixel_table_masks_void_only_when_asked():
    stack = np.zeros((2, 2, 2), dtype="uint16")
    stack[:, 0, 0] = [5, 7]
    x_all, v_all = lulc.pixel_table(stack, mask_void=False)
    x_fix, v_fix = lulc.pixel_table(stack, mask_void=True)
    assert len(x_all) == 4 and v_all.all()
    assert len(x_fix) == 1 and x_fix[0].tolist() == [5, 7]


def test_fit_kmeans_separates_obvious_clusters():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(100, 1, (200, 2))])
    labels = lulc.fit_kmeans(x, k=2, batch_size=1024).labels_
    assert len(set(labels[:200])) == 1 and len(set(labels[200:])) == 1
    assert labels[0] != labels[-1]


def test_match_classes_recovers_permutation():
    ref = np.array([[1, 1, 2], [2, 3, 3]], dtype="int16")
    ours = np.array([[3, 3, 1], [1, 2, 2]], dtype="int16")  # relabelled ref
    m = lulc.match_classes(ours, ref)
    assert m["mapping"] == {3: 1, 1: 2, 2: 3}
    assert m["agreement"] == pytest.approx(1.0)
    assert (lulc.relabel(ours, m["mapping"]) == ref).all()


def test_match_classes_ignores_nodata():
    ref = np.array([1, 2, lulc.CLASS_NODATA], dtype="int16")
    ours = np.array([2, 1, 1], dtype="int16")
    assert lulc.match_classes(ours, ref)["agreement"] == pytest.approx(1.0)


def test_reclassify_none_becomes_nodata():
    classes = np.array([1, 2, 3], dtype="int16")
    out = lulc.reclassify(classes, {1: 2, 2: None, 3: 10})
    assert out.tolist() == [2.0, lulc.SCORE_NODATA, 10.0]


def test_k_selection_prefers_true_k_on_separated_blobs():
    rng = np.random.default_rng(1)
    centres = [(0, 0), (50, 0), (0, 50)]
    x = np.vstack([rng.normal(c, 1.0, (300, 2)) for c in centres])
    rows = lulc.k_selection(
        x, ks=[2, 3, 4], seeds=[0, 1], fit_size=900, eval_size=600, batch_size=1024
    )
    best = max(rows, key=lambda r: r["silhouette"])
    assert best["k"] == 3
    assert next(r for r in rows if r["k"] == 3)["stability_ari"] == pytest.approx(1.0)


def test_cluster_profiles_computes_indices_from_reflectance():
    class FakeModel:
        # DN with +1000 offset: reflectance B03=.05 B04=.05 B08=.45 B11=.15 (vegetation)
        cluster_centers_ = np.array([[1500.0, 1500.0, 1500.0, 5500.0, 2500.0]])

    rows = lulc.cluster_profiles(
        FakeModel(), ["B02", "B03", "B04", "B08", "B11"], np.zeros(4, dtype=int)
    )
    assert rows[0]["ndvi"] == pytest.approx((0.45 - 0.05) / 0.5)
    assert rows[0]["ndbi"] == pytest.approx((0.15 - 0.45) / 0.6)
    assert rows[0]["share"] == pytest.approx(1.0)
