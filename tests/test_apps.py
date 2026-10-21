"""App smoke tests. The dashboard test needs the local pipeline outputs."""

from pathlib import Path

import pytest

from stp.config import REPO_ROOT, load_config


def _final_outputs_exist() -> bool:
    out = Path(load_config()["paths"]["output"]) / "final"
    return (out / "sites_ranked.parquet").exists() and (out / "summary.json").exists()


@pytest.mark.needs_data
@pytest.mark.skipif(not _final_outputs_exist(), reason="final-model outputs not present")
def test_dashboard_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "apps" / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 3
    # Robust-only filter keeps a subset.
    at.sidebar.checkbox[0].check().run()
    assert not at.exception, at.exception
