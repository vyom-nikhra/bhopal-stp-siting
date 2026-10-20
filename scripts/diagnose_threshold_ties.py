"""How many pixels sit exactly on the suitability threshold (score == 8.0)?

All criterion scores are integers and the weights are multiples of 0.01, so
100 x suitability is an exact integer. Pixels where it equals 800 are exact ties:
whether they pass `>= 8` depends only on floating-point rounding.
Run after `python -m stp.cli phase4`.
"""

from pathlib import Path

import numpy as np
from rasterio.crs import CRS

from stp import overlay
from stp.cli import _read_da
from stp.config import Grid, load_config

cfg = load_config()
crs = CRS.from_user_input(cfg["crs"])
out = Path(cfg["paths"]["output"])
grid = Grid.from_config(cfg, "sentinel_10m")
tmpl = overlay.grid_template(grid)
w100 = {k: round(v * 100) for k, v in cfg["weights"].items()}
assert sum(w100.values()) == 100, w100

layers = {
    n: overlay.align(_read_da(out / "phase4" / f"score_{n}.tif", crs)[0], tmpl)
    for n in cfg["standardize"]
}
lulc = {
    "original": overlay.align(
        _read_da(Path(cfg["paths"]["reference_rasters"]) / "Reclass_ML_LULC.tif", crs)[0], tmpl
    ),
    "corrected": overlay.align(_read_da(out / "phase3" / "lulc_score.tif", crs)[0], tmpl),
}
for kind, lc in lulc.items():
    stack = {**layers, "lulc": lc}
    valid = np.all([np.isfinite(a) for a in stack.values()], axis=0)
    exact = sum(w100[k] * np.where(valid, stack[k], 0).astype(np.int64) for k in stack)
    f64 = overlay.weighted_sum(stack, cfg["weights"])
    f32_from64 = f64.astype("float32")
    stack32 = {k: v.astype("float32") for k, v in stack.items()}
    f32_native = overlay.weighted_sum(stack32, cfg["weights"]).astype("float32")
    uncast = overlay.weighted_sum(stack32, cfg["weights"])  # float32 products, float64 sum
    ties = valid & (exact == 800)
    print(
        f"{kind}: pixels >= 8 exact {int((valid & (exact >= 800)).sum())}, "
        f"exact ties (==8.0) {int(ties.sum())}"
    )
    print(
        f"   float64->float32 passes {int((f32_from64[valid] >= 8).sum())} "
        f"(ties passing {int((f32_from64[ties] >= 8).sum())}); "
        f"float32-layer sum passes {int((f32_native[valid] >= 8).sum())} "
        f"(ties passing {int((f32_native[ties] >= 8).sum())})"
    )
    print(
        f"   float32 layers, float64 sum, no cast: passes {int((uncast[valid] >= 8).sum())} "
        f"(ties passing {int((uncast[ties] >= 8).sum())})"
    )
