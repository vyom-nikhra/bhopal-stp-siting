"""Model specifications: the original QGIS setup and the final (decided) setup.

The reproduction phases (1-5) always use the original setup so their parity checks
stay meaningful. The final setup was decided with the author on 2026-10-20:
- weights from the approved AHP judgements (ahp_judgements.yaml)
- threshold: the top 0.3395% of the surface instead of a fixed score of 8
- residential distance capped at 100 m (CPHEEO 2013 odour buffer, as quoted by CPCB)
- minimum site area = mean Bhopal plant capacity x land per MLD
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS

from stp import ahp, io, overlay, proximity, selection, standardize, threshold
from stp.config import REPO_ROOT, Grid


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: dict[str, float]
    rule: dict
    residential_cap_m: float | None
    min_area_sqm: float
    out_dir: str  # relative to paths.output


def ahp_weights(cfg: dict) -> dict[str, float]:
    with open(REPO_ROOT / cfg["ahp"]["judgements_file"], encoding="utf-8") as f:
        data = yaml.safe_load(f)
    criteria = list(cfg["weights"])
    w, _ = ahp.priority_vector(ahp.pairwise_matrix(criteria, data["judgements"]))
    return {c: float(x) for c, x in zip(criteria, w, strict=True)}


def get_spec(cfg: dict, name: str) -> ModelSpec:
    if name == "original":
        return ModelSpec(
            name="original",
            weights=dict(cfg["weights"]),
            rule={"kind": "fixed", "value": cfg["selection"]["suitability_min"]},
            residential_cap_m=None,
            min_area_sqm=float(cfg["selection"]["min_area_sqm"]),
            out_dir="phase6",
        )
    if name == "final":
        m = cfg["model"]
        if m["weights"] != "ahp":
            raise ValueError(f"model.weights must be 'ahp', got {m['weights']!r}")
        return ModelSpec(
            name="final",
            weights=ahp_weights(cfg),
            rule=dict(m["threshold"]),
            residential_cap_m=float(m["residential_cap_m"]),
            min_area_sqm=m["plant_capacity_mld"] * m["land_ha_per_mld"] * 1e4,
            out_dir="final",
        )
    raise ValueError(f"Unknown model {name!r}")


def build_inputs(cfg: dict, crs: CRS, grid: Grid, spec: ModelSpec):
    """Aligned score layers for `spec` (corrected land cover) and the exclusion cut."""
    outputs = Path(cfg["paths"]["output"])
    template = overlay.grid_template(grid)

    def read_da(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run the earlier phases first")
        with rasterio.open(path) as src:
            arr, tf, nodata = src.read(1), src.transform, src.nodata
        return arr, tf, nodata

    def aligned(arr, tf, nodata) -> np.ndarray:
        da = overlay.to_dataarray(arr, tf, crs, nodata)
        return overlay.align(da, template).astype("float32")

    layers = {}
    for name in cfg["standardize"]:
        layers[name] = aligned(*read_da(outputs / "phase4" / f"score_{name}.tif"))
    if spec.residential_cap_m is not None:
        sc = cfg["standardize"]["residential"]
        arr, tf, nodata = read_da(outputs / sc["source"])
        valid = np.isfinite(arr) & (arr != nodata)
        score = standardize.rescale_score(arr, valid, sc["invert"], cap=spec.residential_cap_m)
        layers["residential"] = aligned(score, tf, standardize.SCORE_NODATA)
    layers["lulc"] = aligned(*read_da(outputs / "phase3" / "lulc_score.tif"))

    geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
    sel = cfg["selection"]
    points = [
        proximity.load_clipped(cfg["paths"]["vectors_raw"], name, geom, crs)
        for name in sel["exclusion_layers"]
    ]
    cut = selection.exclusion_zones(points, sel["buffer_m"], sel["buffer_segments"]).union_all()
    return layers, cut


def make_selector(layers, cut, cfg: dict, grid: Grid, crs, spec: ModelSpec, rule=None):
    """surface -> final sites under `spec` (threshold rule, minimum area)."""
    sel = dict(cfg["selection"])
    sel["min_area_sqm"] = spec.min_area_sqm
    rule = rule or spec.rule

    def select(surface: np.ndarray):
        sel["suitability_min"] = threshold.rule_threshold(surface, rule)
        return selection.sites_from_surface(
            surface, layers["lulc"], layers["roads"], cut, sel, grid.transform, crs
        )[2]

    return select
