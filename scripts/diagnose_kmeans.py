"""Is the 3a gap caused by the B11 artefact or by K-Means instability?

Fits the original configuration (k=5, void included) with several random seeds on
both stacks and reports agreement with the original map after label matching.
If changing only the seed moves agreement as much as changing the stack, the
gap is instability, not B11. Run after `python -m stp.cli phase1`.
"""

import time
from pathlib import Path

import rasterio

from stp import lulc
from stp.config import load_config

cfg = load_config()
ref_dir = Path(cfg["paths"]["reference_rasters"])
with rasterio.open(ref_dir / "ML_KMeans_LULC_Bhopal.tif") as s:
    ref = s.read(1)
stacks = {
    "qgis_stack": ref_dir / "Raster_01_Sentinel_Bhopal_Clipped.tif",
    "our_stack": Path(cfg["paths"]["output"]) / "phase1" / "sentinel_stack.tif",
}
for name, path in stacks.items():
    with rasterio.open(path) as s:
        stack = s.read()
    x, valid = lulc.pixel_table(stack, mask_void=False)
    for seed in (42, 0, 1, 2, 3):
        t0 = time.perf_counter()
        model = lulc.fit_kmeans(x, 5, random_state=seed)
        classes = lulc.class_raster(model.labels_, valid, stack.shape[1:])
        m = lulc.match_classes(classes, ref)
        secs = time.perf_counter() - t0
        print(
            f"{name:10s} seed={seed:2d} agreement={m['agreement']:.4f} "
            f"inertia={model.inertia_:.4e} iters={model.n_iter_:4d} ({secs:.0f}s)"
        )
