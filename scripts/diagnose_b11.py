"""Diagnose how QGIS upsampled B11 (20 m -> 10 m) in Raster_01_Sentinel_Bhopal_Clipped.tif.

Compares the reference B11 with candidate reconstructions from the raw 20 m JP2.
Kept in the repo as a record of how the parity gap was investigated.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from stp import preprocess
from stp.config import load_config

cfg = load_config()
ref_path = f"{cfg['paths']['reference_rasters']}/Raster_01_Sentinel_Bhopal_Clipped.tif"
with rasterio.open(ref_path) as r:
    ref = r.read(5).astype("int64")
    tf, shape, crs = r.transform, (r.height, r.width), r.crs
valid = ref > 0

b11_path = preprocess.find_band(cfg["paths"]["sentinel_granule"], "B11")
with rasterio.open(b11_path) as src:
    print("B11 grid:", src.transform, src.res)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            shifted = tf * tf.translation(dc, dr)
            cand = preprocess.read_on_finer_grid(src, shifted, shape).astype("int64")
            exact = np.mean(cand[valid] == ref[valid])
            print(f"duplication shift dr={dr:+d} dc={dc:+d}: exact={exact:.4f}")
    for alg in (Resampling.bilinear, Resampling.cubic, Resampling.average):
        dst = np.zeros(shape, "float64")
        reproject(rasterio.band(src, 1), dst, dst_transform=tf, dst_crs=crs, resampling=alg)
        d = np.abs(np.round(dst[valid]) - ref[valid])
        print(f"{alg.name:9s}: exact={np.mean(d == 0):.4f} mean|d|={d.mean():.2f}")

# Where do the mismatches sit? Pattern by row/col parity.
with rasterio.open(b11_path) as src:
    dup = preprocess.read_on_finer_grid(src, tf, shape).astype("int64")
mis = (dup != ref) & valid
rows, cols = np.nonzero(valid)
for pr in (0, 1):
    for pc in (0, 1):
        sel = (rows % 2 == pr) & (cols % 2 == pc)
        share = mis[rows[sel], cols[sel]].mean()
        print(f"row%2={pr} col%2={pc}: mismatch share={share:.4f}")

# Even rows only: compare with neighbouring rows of the duplicated reconstruction.
ev = np.arange(2, shape[0] - 1, 2)
v = valid[ev]
same = dup[ev] == ref[ev]
above = dup[ev - 1] == ref[ev]
below = dup[ev + 1] == ref[ev]
mean2 = np.round((dup[ev - 1] + dup[ev + 1]) / 2) == ref[ev]
print(
    f"even rows: =own {same[v].mean():.4f} =row-1 {above[v].mean():.4f} "
    f"=row+1 {below[v].mean():.4f} =mean(row-1,row+1) {mean2[v].mean():.4f} "
    f"=own or row-1 {(same | above)[v].mean():.4f}"
)
# Is the choice constant along a row (whole-row switch) or per pixel?
per_row = [above[i][v[i]].mean() for i in range(len(ev)) if v[i].any()]
per_row = np.array(per_row)
print(
    f"per-row share equal to row-1: min {per_row.min():.3f} median {np.median(per_row):.3f} "
    f"max {per_row.max():.3f}; rows fully row-1: {(per_row == 1).mean():.3f}, "
    f"fully own: {(per_row == 0).mean():.3f}"
)
first = [i for i in range(len(ev)) if v[i].any()][:1]
blocks = np.array([above[i][v[i]].mean() > 0.99 for i in range(len(ev)) if v[i].any()])
switch = np.flatnonzero(np.diff(blocks.astype(int))) if len(blocks) > 1 else []
print(
    "row-1 vs own switches at even-row indices (first 20):",
    (2 * (np.array(switch[:20]) + 1)).tolist(),
)

# Hypothesis: the gap comes from GDAL's block-wise VRT upsampling. Re-run the exact
# QGIS steps (gdal:buildvirtualraster -> gdal:cliprasterbymasklayer) through GDAL.
from osgeo import gdal  # noqa: E402

gdal.UseExceptions()
paths = [
    str(preprocess.find_band(cfg["paths"]["sentinel_granule"], b)) for b in cfg["sentinel"]["bands"]
]
vrt = gdal.BuildVRT(
    "/vsimem/stack.vrt", paths, separate=True, resolution="highest", resampleAlg="nearest"
)
warped = gdal.Warp(
    "/vsimem/clip.tif",
    vrt,
    cutlineDSName=cfg["paths"]["boundary"],
    cropToCutline=True,
    dstSRS=cfg["crs"],
)
gb11 = warped.GetRasterBand(5).ReadAsArray().astype("int64")
gt = warped.GetGeoTransform()
print(f"GDAL replay: shape {gb11.shape} origin {gt[0]},{gt[3]} (ref {tf.c},{tf.f} {shape})")
if gb11.shape == shape:
    print(f"GDAL replay B11 vs QGIS ref: exact={np.mean(gb11[valid] == ref[valid]):.4f}")
    print(f"GDAL replay B11 vs clean duplication: exact={np.mean(gb11[valid] == dup[valid]):.4f}")
