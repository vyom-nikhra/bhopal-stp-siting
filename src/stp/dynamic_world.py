"""Multi-temporal Dynamic World statistics on the Sentinel 10 m grid (Earth Engine).

A single June scene can't tell fallow farmland from genuinely open land: both look bare.
Over the 12 months up to the analysis scene, though, a fallow field is labelled `crops`
in other seasons. Per pixel, over all Dynamic World images in the period:
- mode:      most frequent class (0 water, 1 trees, 2 grass, 3 flooded vegetation,
             4 crops, 5 shrub and scrub, 6 built, 7 bare, 8 snow and ice)
- built_pct: % of observations labelled built (settlement mask)
- water_pct: % of observations labelled water (flood / inundation proxy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import requests

from stp.config import Grid

CLASSES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}


def _download(img, grid: Grid, path: Path) -> None:
    """Download one uint8 band exactly on `grid` (same CRS, transform and shape).

    Earth Engine rescales the transform when `dimensions` is given, so the grid is
    pinned with an explicit region (the grid's bounds) plus `crs_transform`.
    """
    import ee

    left, bottom, right, top = grid.bounds
    region = ee.Geometry.Rectangle([left, bottom, right, top], str(grid.crs), False)
    url = img.toUint8().getDownloadURL(
        {
            "format": "GEO_TIFF",
            "crs": str(grid.crs),
            "crs_transform": [grid.res, 0, grid.origin[0], 0, -grid.res, grid.origin[1]],
            "region": region,
        }
    )
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    path.write_bytes(r.content)
    with rasterio.open(path) as src:
        if src.shape != grid.shape or not src.transform.almost_equals(grid.transform):
            raise ValueError(f"{path.name} is not on the target grid: {src.shape} {src.transform}")


def download_stats(grid: Grid, start: str, end: str, project: str, out_dir: Path) -> dict:
    """Download mode / built_pct / water_pct for [start, end) to `out_dir` (cached)."""
    import ee

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"dw_{k}_{start}_{end}.tif" for k in ("mode", "built_pct", "water_pct")}
    if all(p.exists() for p in paths.values()):
        return {k: str(v) for k, v in paths.items()}

    ee.Initialize(project=project)
    left, bottom, right, top = grid.bounds
    region = ee.Geometry.Rectangle([left, bottom, right, top], str(grid.crs), False)
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(start, end)
        .select("label")
    )
    n_images = col.size().getInfo()
    if n_images == 0:
        raise RuntimeError(f"No Dynamic World images between {start} and {end}")
    count = col.count()
    images = {
        "mode": col.reduce(ee.Reducer.mode()),
        "built_pct": col.map(lambda i: i.eq(6)).sum().divide(count).multiply(100).round(),
        "water_pct": col.map(lambda i: i.eq(0)).sum().divide(count).multiply(100).round(),
    }
    for key, img in images.items():
        _download(img, grid, paths[key])
        print(f"[dynamic_world] {key} -> {paths[key].name}")
    (out_dir / f"dw_meta_{start}_{end}.txt").write_text(
        f"images={n_images}\nperiod={start}..{end}\n", encoding="utf-8"
    )
    return {k: str(v) for k, v in paths.items()}


def read_stats(paths: dict) -> dict[str, np.ndarray]:
    out = {}
    for key, p in paths.items():
        with rasterio.open(p) as src:
            out[key] = src.read(1)
    return out
