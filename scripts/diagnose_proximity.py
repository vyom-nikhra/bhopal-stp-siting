"""Check the sign of the distance differences vs QGIS (GDAL Proximity).

If GDAL's two-pass scan is an approximation of the exact Euclidean distance
transform, it can only overestimate: every differing pixel should have QGIS > ours.
Run after `python -m stp.cli phase2`.
"""

from pathlib import Path

import numpy as np
import rasterio

from stp.config import load_config

cfg = load_config()
ref_dir = Path(cfg["paths"]["reference_rasters"])
out_dir = Path(cfg["paths"]["output"]) / "phase2"
for name, layer in cfg["proximity"]["layers"].items():
    with rasterio.open(out_dir / f"dist_{name}.tif") as s:
        ours = s.read(1).astype("float64")
    with rasterio.open(ref_dir / layer["reference"]) as s:
        ref = s.read(1).astype("float64")
    valid = (ours != -9999) & (ref != -9999)
    d = ref[valid] - ours[valid]
    diff = np.abs(d) > 1e-2
    print(
        f"{name:12s} differing px={int(diff.sum()):7d} "
        f"QGIS>ours={int((d > 1e-2).sum()):7d} QGIS<ours={int((d < -1e-2).sum()):4d} "
        f"max over={d.max():.3f} m  rel max={np.max(d[diff] / ref[valid][diff]):.4%}"
    )
