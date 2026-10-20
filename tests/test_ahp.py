from itertools import combinations

import numpy as np
import pytest

from stp import ahp, threshold

CRIT = ["a", "b", "c"]


def _judge(a, b, prefer, s):
    return {"a": a, "b": b, "prefer": prefer, "strength": s}


def test_consistent_matrix_recovers_weights_with_zero_cr():
    # a = 2b, b = 2c  ->  a = 4c ; weights 4:2:1
    js = [_judge("a", "b", "a", 2), _judge("b", "c", "b", 2), _judge("a", "c", "a", 4)]
    a_mat = ahp.pairwise_matrix(CRIT, js)
    w, lam = ahp.priority_vector(a_mat)
    ci, cr = ahp.consistency(a_mat, lam)
    assert w == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    assert lam == pytest.approx(3.0)
    assert cr == pytest.approx(0.0, abs=1e-9)
    assert ahp.geometric_mean_weights(a_mat) == pytest.approx(w)


def test_contradictory_judgements_give_high_cr():
    # a > b, b > c, but c > a: circular
    js = [_judge("a", "b", "a", 5), _judge("b", "c", "b", 5), _judge("a", "c", "c", 5)]
    a_mat = ahp.pairwise_matrix(CRIT, js)
    _, lam = ahp.priority_vector(a_mat)
    assert ahp.consistency(a_mat, lam)[1] > 0.10
    worst = ahp.most_inconsistent(a_mat, ahp.priority_vector(a_mat)[0], CRIT, top=1)
    assert len(worst) == 1


def test_matrix_is_reciprocal():
    names = list("abcd")
    js = [_judge(x, y, x, 3) for x, y in combinations(names, 2)]
    a_mat = ahp.pairwise_matrix(names, js)
    assert np.allclose(a_mat * a_mat.T, 1.0)


def test_pairwise_matrix_validation():
    ok = [_judge("a", "b", "a", 2), _judge("b", "c", "b", 2)]
    with pytest.raises(ValueError, match="Missing"):
        ahp.pairwise_matrix(CRIT, ok)
    with pytest.raises(ValueError, match="twice"):
        ahp.pairwise_matrix(CRIT, ok + [_judge("b", "a", "a", 3), _judge("a", "c", "a", 4)])
    with pytest.raises(ValueError, match="prefer"):
        ahp.pairwise_matrix(CRIT, [_judge("a", "b", "c", 2)])
    with pytest.raises(ValueError, match="strength"):
        ahp.pairwise_matrix(CRIT, [_judge("a", "b", "a", 10)])


def test_top_pct_threshold_and_rules():
    surface = np.arange(1000, dtype=float)
    surface[0] = np.nan
    thr = threshold.top_pct_threshold(surface, 10)
    assert np.mean(surface[np.isfinite(surface)] >= thr) == pytest.approx(0.10, abs=0.002)
    assert threshold.rule_threshold(surface, {"kind": "fixed", "value": 8}) == 8.0
    with pytest.raises(ValueError):
        threshold.top_pct_threshold(surface, 0)
    with pytest.raises(ValueError):
        threshold.rule_threshold(surface, {"kind": "median"})
