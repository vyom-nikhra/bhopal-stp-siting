"""Phase 1: band stack, DEM clip, slope and spectral indices.

Reproduces the QGIS steps documented in docs/qgis_workflow.md (sections 2 and 3):
- Sentinel-2 bands are read only inside the boundary's bounding box (windowed reads),
  B11 is upsampled 20 m -> 10 m with nearest neighbour on read, and pixels whose
  centre falls outside the boundary are set to NoData, like GDAL's crop-to-cutline.
- Unlike the QGIS output, NoData is declared in the file metadata (0 for the stack).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from shapely.geometry.base import BaseGeometry

STACK_NODATA = 0
FLOAT_NODATA = np.nan
SLOPE_NODATA = -9999.0


def find_band(granule_img_dir: str | Path, band: str) -> Path:
    """Locate a band's highest-resolution JP2 (B02/B03/B04/B08 at 10 m, B11 at 20 m)."""
    for res in ("10m", "20m", "60m"):
        hits = sorted(Path(granule_img_dir, f"R{res}").glob(f"*_{band}_{res}.jp2"))
        if len(hits) > 1:
            raise ValueError(f"Ambiguous files for {band} at {res}: {hits}")
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No JP2 found for band {band} under {granule_img_dir}")


def crop_window(
    bounds: tuple[float, float, float, float], transform: Affine, snap: str = "inward"
) -> Window:
    """Pixel-aligned window of `transform`'s grid for `bounds`.

    snap="inward" keeps only whole pixels inside the bounds. This is what GDAL's
    crop-to-cutline produced on every edge of the QGIS outputs (e.g. boundary left
    edge 727971.55 -> stack origin 727980, DEM origin 727972.3125).
    snap="outward" returns the smallest window that covers the bounds.
    """
    if snap not in ("inward", "outward"):
        raise ValueError(f"snap must be 'inward' or 'outward', got {snap!r}")
    lo, hi = (math.ceil, math.floor) if snap == "inward" else (math.floor, math.ceil)
    w = from_bounds(*bounds, transform=transform)
    col0, row0 = lo(w.col_off), lo(w.row_off)
    col1, row1 = hi(w.col_off + w.width), hi(w.row_off + w.height)
    return Window(col0, row0, col1 - col0, row1 - row0)


def outside_mask(geom: BaseGeometry, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """True where the pixel centre is outside `geom` (GDAL cutline semantics)."""
    return geometry_mask(
        [geom], out_shape=shape, transform=transform, invert=False, all_touched=False
    )


def build_stack(
    granule_img_dir: str | Path, bands: list[str], geom: BaseGeometry, crs
) -> tuple[np.ndarray, Affine]:
    """Read and clip the band stack on the 10 m grid of the first band.

    Returns a (bands, rows, cols) uint16 array and its transform.
    """
    ref_path = find_band(granule_img_dir, bands[0])
    with rasterio.open(ref_path) as ref:
        if ref.crs != crs:
            raise ValueError(f"{ref_path.name} is in {ref.crs}, expected {crs}")
        win = crop_window(geom.bounds, ref.transform)
        out_transform = ref.window_transform(win)
        shape = (int(win.height), int(win.width))

    layers = []
    for band in bands:
        path = find_band(granule_img_dir, band)
        with rasterio.open(path) as src:
            if src.crs != crs:
                raise ValueError(f"{path.name} is in {src.crs}, expected {crs}")
            layers.append(read_on_finer_grid(src, out_transform, shape))

    stack = np.stack(layers)
    stack[:, outside_mask(geom, shape, out_transform)] = STACK_NODATA
    return stack, out_transform


def read_on_finer_grid(src, out_transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """Read band 1 of `src` onto a grid whose pixel size divides `src`'s by an integer.

    Nearest-neighbour upsampling by an integer factor is exact pixel duplication, which
    is what the QGIS VRT did for B11 (20 m -> 10 m). Doing it explicitly avoids GDAL's
    decimated-read rounding, which picked different source pixels for ~28% of the area.
    """
    factor = src.res[0] / out_transform.a
    if not math.isclose(factor, round(factor)) or factor < 1:
        raise ValueError(f"Source res {src.res[0]} is not an integer multiple of {out_transform.a}")
    factor = round(factor)
    # Position of the output origin in source pixels: whole part + remainder in output pixels.
    col_f, row_f = ~src.transform @ (out_transform.c, out_transform.f)
    col0, row0 = math.floor(col_f + 1e-9), math.floor(row_f + 1e-9)
    dc = round((col_f - col0) * factor)
    dr = round((row_f - row0) * factor)
    n_cols = math.ceil((shape[1] + dc) / factor)
    n_rows = math.ceil((shape[0] + dr) / factor)
    data = src.read(1, window=Window(col0, row0, n_cols, n_rows))
    if factor > 1:
        data = np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)
    out = data[dr : dr + shape[0], dc : dc + shape[1]]
    if out.shape != tuple(shape):
        raise ValueError(f"Read {out.shape}, expected {shape}: window runs off the source raster")
    return out


def clip_dem(
    dem_path: str | Path, geom: BaseGeometry, crs, resampling: Resampling = Resampling.bilinear
) -> tuple[np.ndarray, Affine, float]:
    """Clip the DEM to the boundary, reprojecting first only if its CRS differs.

    Returns (array, transform, nodata). The array keeps the source dtype.
    """
    with rasterio.open(dem_path) as src:
        if src.nodata is None:
            raise ValueError(f"DEM has no NoData value set: {dem_path}")
        nodata = src.nodata
        if src.crs == crs:
            win = crop_window(geom.bounds, src.transform)
            data = src.read(1, window=win)
            transform = src.window_transform(win)
        else:
            # Same pixel size, target grid snapped to the boundary bounds.
            res = src.res[0]
            left, bottom, right, top = geom.bounds
            left, top = math.floor(left / res) * res, math.ceil(top / res) * res
            cols = math.ceil((right - left) / res)
            rows = math.ceil((top - bottom) / res)
            transform = Affine(res, 0, left, 0, -res, top)
            data = np.full((rows, cols), nodata, dtype=src.dtypes[0])
            reproject(
                rasterio.band(src, 1),
                data,
                dst_transform=transform,
                dst_crs=crs,
                resampling=resampling,
                src_nodata=nodata,
                dst_nodata=nodata,
            )

    data[outside_mask(geom, data.shape, transform)] = nodata
    return data, transform, nodata


def horn_slope(dem: np.ndarray, res: float, nodata: float) -> np.ndarray:
    """Slope in degrees with Horn's 3x3 method (as `gdaldem slope`, no edge computation).

    Output is NoData on the outer ring of pixels and wherever any cell of the 3x3
    window is NoData.
    """
    z = dem.astype("float64")
    invalid = dem == nodata
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)
    inner = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    # A window is invalid if any of its 9 cells is NoData.
    bad = np.zeros_like(inner, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            bad |= invalid[dr : dr + inner.shape[0], dc : dc + inner.shape[1]]

    out = np.full(dem.shape, SLOPE_NODATA, dtype="float32")
    out[1:-1, 1:-1] = np.where(bad, SLOPE_NODATA, inner)
    return out


def normalized_difference(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, offset: float = 0.0
) -> np.ndarray:
    """(a - b) / (a + b) as float32, NaN outside `valid` or where a + b == 0.

    `offset` is added to both inputs first (Sentinel-2 BOA_ADD_OFFSET, e.g. -1000).
    """
    a = a.astype("float64") + offset
    b = b.astype("float64") + offset
    denom = a + b
    ok = valid & (denom != 0)
    out = np.full(a.shape, FLOAT_NODATA, dtype="float32")
    out[ok] = ((a[ok] - b[ok]) / denom[ok]).astype("float32")
    return out


def spectral_indices(
    stack: np.ndarray, bands: list[str], offset: float = 0.0
) -> dict[str, np.ndarray]:
    """NDVI, NDBI and MNDWI from a clipped stack (NoData = 0)."""
    idx = {name: stack[bands.index(name)] for name in ("B03", "B04", "B08", "B11")}
    valid = np.all(stack != STACK_NODATA, axis=0)
    return {
        "ndvi": normalized_difference(idx["B08"], idx["B04"], valid, offset),
        "ndbi": normalized_difference(idx["B11"], idx["B08"], valid, offset),
        "mndwi": normalized_difference(idx["B03"], idx["B11"], valid, offset),
    }
