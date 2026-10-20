"""Download the 12-month Dynamic World statistics and describe them, to choose constraint
thresholds from evidence. Run before changing the model constraints.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from stp import dynamic_world as dw
from stp.config import Grid, load_config
from stp.sensitivity import site_ids

cfg = load_config()
grid = Grid.from_config(cfg, "sentinel_10m")
out = Path(cfg["paths"]["output"]) / "dynamic_world"
paths = dw.download_stats(grid, "2024-06-10", "2025-06-10", cfg["crosscheck"]["ee_project"], out)
s = dw.read_stats(paths)
with rasterio.open(Path(cfg["paths"]["output"]) / "final" / "suitability.tif") as src:
    valid = np.isfinite(src.read(1))
print((out / "dw_meta_2024-06-10_2025-06-10.txt").read_text())

mode = s["mode"][valid]
print("Mode class share of valid pixels:")
for k, name in dw.CLASSES.items():
    share = np.mean(mode == k)
    if share > 0:
        print(f"  {name:20s} {share:6.1%}")
for key in ("built_pct", "water_pct"):
    v = s[key][valid]
    qs = {t: np.mean(v >= t) for t in (10, 25, 50, 75)}
    print(f"{key}: " + ", ".join(f">={t}%: {q:.1%}" for t, q in qs.items()))

sites = gpd.read_parquet(Path(cfg["paths"]["output"]) / "final" / "sites_ranked.parquet")
ids = site_ids(sites, grid.shape, grid.transform)
inside = ids > 0
print(f"\nFinal sites ({len(sites)}), mode class share of their pixels:")
m = s["mode"][inside]
for k, name in dw.CLASSES.items():
    share = np.mean(m == k)
    if share > 0:
        print(f"  {name:20s} {share:6.1%}")
for key in ("built_pct", "water_pct"):
    v = s[key][inside]
    print(f"{key} in sites: >=10%: {np.mean(v >= 10):.1%}, >=50%: {np.mean(v >= 50):.1%}")
