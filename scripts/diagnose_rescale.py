"""Recover the exact 1-10 scoring rules behind the QGIS Reclass_* rasters.

1. For each distance/DEM layer, test candidate integer rescale rules (round, floor,
   ceil of the linear min-max map onto 1..10), with and without `11 - x` inversion.
2. For slope, tabulate the score assigned to each slope value to find the breakpoints.
"""

from pathlib import Path

import numpy as np
import rasterio

from stp.config import load_config

cfg = load_config()
R = Path(cfg["paths"]["reference_rasters"])


def read(name):
    with rasterio.open(R / name) as s:
        a = s.read(1).astype("float64")
        nd = s.nodata
    valid = np.isfinite(a) & (a > -1e30)
    if nd is not None:
        valid &= a != nd
    return a, valid


layers = {
    "roads": ("Raster_05_Dist_Roads_Clipped.tif", "Reclass_05_Dist_Roads.tif"),
    "drainage": ("Raster_06_Dist_Drainage_Clipped.tif", "Reclass_06_Dist_Drainage.tif"),
    "residential": ("Raster_07_Dist_Residential_Clipped.tif", "Reclass_07_Dist_Residential.tif"),
    "water": ("Raster_08_Dist_Water_Clipped.tif", "Reclass_08_Dist_Water.tif"),
    "dem": ("Raster_09_DEM_Bhopal_Clipped.tif", "Reclass_09_DEM.tif"),
}
for name, (raw_name, rec_name) in layers.items():
    raw, v_raw = read(raw_name)
    rec, v_rec = read(rec_name)
    if raw.shape != rec.shape:
        print(f"{name}: shape mismatch {raw.shape} vs {rec.shape}")
        continue
    v = v_raw & v_rec
    lo, hi = raw[v_raw].min(), raw[v_raw].max()
    lin = 1 + (raw - lo) * 9 / (hi - lo)
    print(
        f"{name}: raw range [{lo:.4f}, {hi:.4f}]  valid raw {v_raw.sum()} rec {v_rec.sum()} "
        f"both {v.sum()}  rec values {np.unique(rec[v_rec]).tolist()}"
    )
    for rule, f in (
        ("round", np.round),
        ("floor", np.floor),
        ("ceil", np.ceil),
        ("trunc+0.5", lambda a: np.floor(a + 0.5)),
    ):
        cand = f(lin)
        plain = np.mean(cand[v] == rec[v])
        inv = np.mean((11 - cand[v]) == rec[v])
        print(f"   {rule:9s} exact plain={plain:.6f} inverted={inv:.6f}")

# Slope: slope raster and its reclass share the 12.5 m grid.
slope, v_s = read("Raster_10_Slope.tif")
rec, v_r = read("Reclass_10_Slope.tif")
v = v_s & v_r
print(
    f"slope: range [{slope[v_s].min():.3f}, {slope[v_s].max():.3f}] valid slope {v_s.sum()} "
    f"rec {v_r.sum()} both {v.sum()}"
)
s, r = slope[v], rec[v]
for score in np.unique(r):
    sel = r == score
    print(
        f"   score {score:4.1f}: n={sel.sum():8d} slope in [{s[sel].min():.4f}, {s[sel].max():.4f}]"
    )
# Does a linear rule over a different max fit? Try 11 - round(1 + 9*s/smax) for smax candidates.
for smax in (15, 20, 25, 30, 37.6, s.max()):
    cand = 11 - np.round(1 + 9 * np.clip(s, 0, smax) / smax)
    print(f"   11-round(lin, max={smax:.2f}): exact={np.mean(cand == r):.4f}")

# Hypothesis: GRASS rounded the input to integers before r.rescale.
print("Integer-input hypothesis: score = round(1 + 9*(round(x)-lo)/(hi-lo)), lo/hi of round(x)")
checks = {name: pair for name, pair in layers.items()}
checks["slope"] = ("Raster_10_Slope.tif", "Reclass_10_Slope.tif")
for name, (raw_name, rec_name) in checks.items():
    raw, v_raw = read(raw_name)
    rec, v_rec = read(rec_name)
    v = v_raw & v_rec
    xi = np.round(raw)
    lo, hi = xi[v_raw].min(), xi[v_raw].max()
    cand = np.round(1 + 9 * (xi - lo) / (hi - lo))
    plain = np.mean(cand[v] == rec[v])
    inv = np.mean((11 - cand[v]) == rec[v])
    best = max(plain, inv)
    n_bad = int(round((1 - best) * v.sum()))
    print(
        f"   {name:11s} int range [{lo:.0f}, {hi:.0f}] exact plain={plain:.6f} "
        f"inverted={inv:.6f} mismatches={n_bad}"
    )
