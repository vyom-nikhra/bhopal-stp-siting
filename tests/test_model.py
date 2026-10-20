import pytest

from stp import model
from stp.config import load_config


def test_specs_from_repo_config():
    cfg = load_config()
    orig = model.get_spec(cfg, "original")
    assert orig.weights == cfg["weights"] and orig.residential_cap_m is None
    assert orig.rule == {"kind": "fixed", "value": 8.0} and orig.min_area_sqm == 5000

    final = model.get_spec(cfg, "final")
    assert sum(final.weights.values()) == pytest.approx(1.0)
    assert final.weights["water"] == pytest.approx(0.333, abs=1e-3)
    assert final.min_area_sqm == pytest.approx(12.7 * 0.044 * 1e4)
    assert final.residential_cap_m == 100
    assert final.rule == {"kind": "top_area", "pct_of_valid": 0.3395}
    assert final.constraints["eligible_mode_classes"] == [2, 5, 7]
    assert orig.constraints is None


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        model.get_spec(load_config(), "best")


def test_constraint_masks():
    import numpy as np

    mode = np.array([[7, 4, 2], [5, 6, 7], [0, 7, 7]])
    built = np.zeros((3, 3))
    built[1, 1] = 80
    water = np.zeros((3, 3))
    water[2, 2] = 30
    c = {
        "eligible_mode_classes": [2, 5, 7],
        "settlement_built_pct": 50,
        "settlement_buffer_m": 10,
        "flood_water_pct": 10,
    }
    m = model.constraint_masks({"mode": mode, "built_pct": built, "water_pct": water}, c, 10.0)
    # settlement zone = built pixel + its 4 edge neighbours (distance 10 m)
    assert m["settlement_zone"].sum() == 5
    assert m["flood"][2, 2]
    expected = np.array([[True, False, True], [False, False, False], [False, False, False]])
    assert (m["eligible"] == expected).all()
