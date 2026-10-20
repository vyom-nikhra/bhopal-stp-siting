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
    assert final.rule["kind"] == "top_pct"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        model.get_spec(load_config(), "best")
