"""Candidate fix for 3b: full (Lloyd) K-Means with restarts on a sample, k=4.

1. Seed stability of KMeans(n_init=10) vs the original MiniBatchKMeans setup.
2. Cross-tabulation of the k=4 classes with the original 5-class map, to carry the
   original scoring meaning over to the new classes.
Run after `python -m stp.cli phase1`.
"""

import time
from pathlib import Path

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from stp import lulc
from stp.config import load_config

cfg = load_config()
with rasterio.open(Path(cfg["paths"]["output"]) / "phase1" / "sentinel_stack.tif") as s:
    stack = s.read()
with rasterio.open(Path(cfg["paths"]["reference_rasters"]) / "ML_KMeans_LULC_Bhopal.tif") as s:
    original = s.read(1).reshape(-1)

x, valid = lulc.pixel_table(stack, mask_void=True)
rng = np.random.default_rng(0)
x_fit = x[rng.choice(len(x), size=1_000_000, replace=False)].astype("float64")
x_eval = x[rng.choice(len(x), size=200_000, replace=False)].astype("float64")

labelings, models = [], []
for seed in (42, 0, 1):
    t0 = time.perf_counter()
    m = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(x_fit)
    models.append(m)
    labelings.append(m.predict(x_eval))
    print(
        f"KMeans k=4 n_init=10 seed={seed}: inertia/px={m.inertia_ / len(x_fit):.4e} "
        f"iters={m.n_iter_} ({time.perf_counter() - t0:.0f}s)"
    )
aris = [adjusted_rand_score(labelings[i], labelings[j]) for i in range(3) for j in range(i + 1, 3)]
print(f"seed stability ARI: mean {np.mean(aris):.4f} min {np.min(aris):.4f}")

model = models[0]
labels = model.predict(x.astype("float64"))
prof = {p["cluster"]: p for p in lulc.cluster_profiles(model, cfg["sentinel"]["bands"], labels)}
orig = original[valid]
print("rows: new k=4 cluster; cols: original class 1..5 (share of the new cluster's pixels)")
for c in sorted(prof, key=lambda c: prof[c]["ndvi"]):
    sel = labels == c - 1
    shares = [(orig[sel] == o).mean() for o in range(1, 6)]
    p = prof[c]
    print(
        f"  c{c} ({p['share']:.3f}, ndvi {p['ndvi']:+.2f}, ndbi {p['ndbi']:+.2f}, "
        f"mndwi {p['mndwi']:+.2f}): " + " ".join(f"o{o + 1}={s:.3f}" for o, s in enumerate(shares))
    )
print(
    "original classes inside the boundary:",
    {o: round(float((orig == o).mean()), 3) for o in range(1, 6)},
)

# Same land-use check as diagnose_cluster_landuse.py, for this model.
from rasterio.crs import CRS  # noqa: E402

from stp import io, proximity  # noqa: E402
from stp.config import Grid  # noqa: E402

crs = CRS.from_user_input(cfg["crs"])
grid = Grid.from_config(cfg, "sentinel_10m")
geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
for name in ("residential", "water"):
    gdf = proximity.load_clipped(
        cfg["paths"]["vectors_raw"], cfg["proximity"]["layers"][name]["raw"], geom, crs
    )
    inside = (proximity.burn(gdf, grid).reshape(-1) == 1)[valid]
    shares = {}
    for c in sorted(prof, key=lambda c: prof[c]["ndvi"]):
        s = (labels[inside] == c - 1).mean()
        shares[f"c{c}"] = f"{s:.3f} (x{s / prof[c]['share']:.2f})"
    print(f"{name:11s} share by cluster: {shares}")
