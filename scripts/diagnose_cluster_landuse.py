"""Which K-Means cluster is built-up and which is barren? Check against OSM land use.

For k in 3..5 (void masked, fitted on a 1M-pixel sample, predicted on all pixels),
reports each cluster's share of all pixels, of pixels inside OSM residential areas,
and of pixels inside OSM water bodies. A cluster over-represented in residential
areas is built-up; one under-represented there and in water is the barren/bare-soil
candidate. Run after `python -m stp.cli phase1`.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from stp import io, lulc, proximity
from stp.config import Grid, load_config

cfg = load_config()
crs = CRS.from_user_input(cfg["crs"])
with rasterio.open(Path(cfg["paths"]["output"]) / "phase1" / "sentinel_stack.tif") as s:
    stack = s.read()
grid = Grid.from_config(cfg, "sentinel_10m")
geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
layers = cfg["proximity"]["layers"]
masks = {}
for name in ("residential", "water"):
    gdf = proximity.load_clipped(cfg["paths"]["vectors_raw"], layers[name]["raw"], geom, crs)
    masks[name] = proximity.burn(gdf, grid).reshape(-1) == 1

x, valid = lulc.pixel_table(stack, mask_void=True)
inside = {name: m[valid] for name, m in masks.items()}
rng = np.random.default_rng(0)
fit_idx = rng.choice(len(x), size=1_000_000, replace=False)
for k in (3, 4, 5):
    model = lulc.fit_kmeans(x[fit_idx].astype("float64"), k, random_state=42)
    labels = model.predict(x.astype("float64"))
    prof = {p["cluster"]: p for p in lulc.cluster_profiles(model, cfg["sentinel"]["bands"], labels)}
    print(
        f"k={k}  (residential px {inside['residential'].sum()}, water px {inside['water'].sum()})"
    )
    for c in sorted(prof, key=lambda c: prof[c]["ndvi"]):
        sel = labels == c - 1
        p = prof[c]
        res = sel[inside["residential"]].mean()
        wat = sel[inside["water"]].mean()
        print(
            f"  c{c}: all {p['share']:.3f}  residential {res:.3f} (x{res / p['share']:.2f})  "
            f"water {wat:.3f}  ndvi {p['ndvi']:+.2f} ndbi {p['ndbi']:+.2f} "
            f"mndwi {p['mndwi']:+.2f} B04 {p['reflectance']['B04']:.3f}"
        )
