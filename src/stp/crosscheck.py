"""Independent cross-check of the final sites (an automated stand-in for a field check).

For each site, the share of its pixels in each class of three independent land-cover
products, harmonised to common classes, plus overlap with OSM land-use polygons:
- ESA WorldCover 10 m, 2021 (v200)
- Esri / Impact Observatory 10 m annual land cover, 2023 (Planetary Computer, io-lulc-annual-v02)
- Google Dynamic World, the classification of the same Sentinel-2 scene (2025-06-09)
These are evidence, not ground truth: each product has its own errors, and two of them
predate the scene. A site is flagged when the products agree it is cropland or built-up.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from stp import io, model
from stp.config import Grid
from stp.sensitivity import site_ids

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N21E075_Map.tif"
)
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/io-lulc-annual-v02"

HARMONISED = ["bare", "crops", "built", "trees", "low_veg", "water", "other"]

WORLDCOVER = {
    10: "trees",
    20: "low_veg",
    30: "low_veg",
    40: "crops",
    50: "built",
    60: "bare",
    80: "water",
    90: "water",
    95: "trees",
    100: "low_veg",
    70: "other",
}
ESRI_IO = {
    1: "water",
    2: "trees",
    4: "water",
    5: "crops",
    7: "built",
    8: "bare",
    9: "other",
    10: "other",
    11: "low_veg",
}
DYNAMIC_WORLD = {
    0: "water",
    1: "trees",
    2: "low_veg",
    3: "water",
    4: "crops",
    5: "low_veg",
    6: "built",
    7: "bare",
    8: "other",
}


def read_on_grid(url: str, grid: Grid) -> np.ndarray:
    """Read a (remote) categorical raster onto `grid` with nearest neighbour.

    Only the blocks covering the grid are fetched (GDAL range requests on COGs).
    """
    path = url if not url.startswith("http") else f"/vsicurl/{url}"
    with (
        rasterio.open(path) as src,
        WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.shape[1],
            height=grid.shape[0],
            resampling=Resampling.nearest,
        ) as vrt,
    ):
        return vrt.read(1)


def class_shares(ids: np.ndarray, classes: np.ndarray, mapping: dict, n_sites: int) -> list[dict]:
    """Per site (1..n): share of its pixels in each harmonised class."""
    harm = np.full(classes.shape, HARMONISED.index("other"), dtype=np.int16)
    for code, name in mapping.items():
        harm[classes == code] = HARMONISED.index(name)
    inside = ids > 0
    counts = np.zeros((n_sites + 1, len(HARMONISED)), dtype=np.int64)
    np.add.at(counts, (ids[inside], harm[inside]), 1)
    out = []
    for i in range(1, n_sites + 1):
        total = counts[i].sum()
        if total == 0:
            raise ValueError(f"Site {i} has no pixels on the grid")
        out.append({c: float(counts[i, k] / total) for k, c in enumerate(HARMONISED)})
    return out


def histogram_shares(hist: dict, mapping: dict) -> dict:
    """Harmonised shares from an Earth Engine frequency histogram {class_code: count}."""
    total = sum(hist.values())
    if total == 0:
        raise ValueError("Empty histogram")
    shares = dict.fromkeys(HARMONISED, 0.0)
    for code, count in hist.items():
        shares[mapping.get(int(float(code)), "other")] += count / total
    return shares


def dominant(shares: dict) -> tuple[str, float]:
    name = max(shares, key=shares.get)
    return name, shares[name]


def verdict(dominants: list[str]) -> str:
    """Flag from the products' dominant classes (majority of 3)."""
    for cls, flag in (
        ("crops", "likely_farmland"),
        ("built", "likely_built"),
        ("water", "likely_water"),
        ("trees", "likely_trees"),
    ):
        if dominants.count(cls) >= 2:
            return flag
    if sum(d in ("bare", "low_veg") for d in dominants) >= 2:
        return "consistent_open_land"
    return "mixed"


def read_osm_landuse(pbf: Path, area) -> gpd.GeoDataFrame:
    """OSM `landuse=*` areas intersecting `area` (EPSG:4326), from an .osm.pbf extract.

    GDAL's OSM driver puts closed ways and multipolygon relations tagged as areas in its
    `multipolygons` layer, with `landuse` as a column (default osmconf.ini).
    """
    import pyogrio

    if not pbf.exists():
        raise FileNotFoundError(f"{pbf} missing: download the extract named in config.yaml")
    lu = pyogrio.read_dataframe(
        pbf,
        layer="multipolygons",
        columns=["landuse"],
        bbox=tuple(area.bounds),
        where="landuse IS NOT NULL",
    )
    lu = lu.set_crs(4326, allow_override=True) if lu.crs is None else lu.to_crs(4326)
    return lu[lu.intersects(area)].reset_index(drop=True)


def osm_landuse_overlap(sites: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame) -> list[dict]:
    """Per site: share of its area under each OSM `landuse` value (planar, site CRS)."""
    lu = landuse[["landuse", "geometry"]].to_crs(sites.crs)
    lu = lu[lu.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    inter = gpd.overlay(sites[["site_id", "geometry"]], lu, how="intersection", keep_geom_type=True)
    inter["a"] = inter.area
    area = sites.set_index("site_id").area
    out = []
    for sid in sites["site_id"]:
        part = inter[inter["site_id"] == sid].groupby("landuse")["a"].sum() / area[sid]
        out.append({str(k): float(v) for k, v in part.sort_values(ascending=False).items()})
    return out


def site_id_raster(sites: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    return site_ids(sites, grid.shape, grid.transform)


def esri_io_url(bbox_lonlat, year: int) -> str:
    """Signed URL of the Esri/IO annual land-cover tile for `year` covering the bbox."""
    r = requests.post(
        PC_STAC,
        json={"collections": ["io-lulc-annual-v02"], "bbox": list(bbox_lonlat), "limit": 50},
        timeout=60,
    )
    r.raise_for_status()
    items = [
        f for f in r.json()["features"] if f["properties"]["start_datetime"].startswith(str(year))
    ]
    if len(items) != 1:
        raise RuntimeError(f"Expected one {year} item over {bbox_lonlat}, got {len(items)}")
    tok = requests.get(PC_TOKEN, timeout=60)
    tok.raise_for_status()
    return f"{items[0]['assets']['data']['href']}?{tok.json()['token']}"


def dynamic_world_hist(sites_lonlat: gpd.GeoDataFrame, date: str, project: str) -> list[dict]:
    """Dynamic World `label` histogram per site for the scene on `date` (Earth Engine)."""
    import ee

    ee.Initialize(project=project)
    fc = ee.FeatureCollection(json.loads(sites_lonlat[["site_id", "geometry"]].to_json()))
    start = ee.Date(date)
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(fc.geometry())
        .filterDate(start, start.advance(1, "day"))
    )
    if col.size().getInfo() == 0:
        raise RuntimeError(f"No Dynamic World image on {date}")
    res = (
        col.select("label")
        .mosaic()
        .reduceRegions(
            collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=10, crs="EPSG:32643"
        )
        .getInfo()
    )
    by_id = {f["properties"]["site_id"]: f["properties"]["histogram"] for f in res["features"]}
    return [by_id[sid] for sid in sites_lonlat["site_id"]]


def render_chips(
    sites: gpd.GeoDataFrame, stack_path: Path, out: Path, pad_m: float, offset: float
) -> None:
    """True- and false-colour Sentinel-2 chips per site, with its outline drawn on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stretch(b):
        ok = b > 0
        lo, hi = np.percentile(b[ok], [2, 98]) if ok.any() else (0.0, 1.0)
        return np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)

    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack_path) as src:
        for _, s in sites.iterrows():
            minx, miny, maxx, maxy = s.geometry.bounds
            win = (
                from_bounds(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, src.transform)
                .round_offsets()
                .round_lengths()
            )
            data = src.read(window=win, boundless=True, fill_value=0).astype("float64")
            tf = src.window_transform(win)
            refl = np.clip((data + offset) / 10000.0, 0, 1)
            extent = (tf.c, tf.c + tf.a * data.shape[2], tf.f + tf.e * data.shape[1], tf.f)
            true_c = np.dstack([stretch(refl[i]) for i in (2, 1, 0)])  # B04 B03 B02
            false_c = np.dstack([stretch(refl[i]) for i in (3, 2, 1)])  # B08 B04 B03
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
            panels = ((true_c, "true colour"), (false_c, "false colour (NIR, red, green)"))
            for ax, (img, title) in zip(axes, panels, strict=True):
                ax.imshow(img, extent=extent)
                gpd.GeoSeries([s.geometry]).boundary.plot(ax=ax, color="red", linewidth=1.2)
                ax.set_title(title, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(
                f"{s['site_id']} ({s['area_sqm'] / 1e4:.2f} ha); Sentinel-2 2025-06-09, 10 m; "
                f"frame = site + {pad_m:g} m",
                fontsize=9,
            )
            fig.tight_layout()
            fig.savefig(out / f"{s['site_id']}.png", dpi=110)
            plt.close(fig)


def run_crosscheck(cfg: dict, model_name: str = "final") -> gpd.GeoDataFrame:
    """Independent land-cover evidence per site, plus visual chips (`stp.cli crosscheck`)."""
    outputs = Path(cfg["paths"]["output"])
    spec = model.get_spec(cfg, model_name)
    out = outputs / spec.out_dir
    grid = Grid.from_config(cfg, "sentinel_10m")
    xc = cfg["crosscheck"]
    sites = gpd.read_parquet(out / "sites_ranked.parquet")
    sites["site_id"] = [f"S{r:02d}" for r in sites["rank"]]
    ids = site_id_raster(sites, grid)
    n = len(sites)

    # 1. Land-cover products, cached (the Esri read alone takes ~1.5 min). The cache is
    #    keyed by the site ids and areas, so it is ignored if the sites change.
    t0 = time.perf_counter()
    cache = out / "crosscheck_products.json"
    key = [[sid, round(a, 1)] for sid, a in zip(sites["site_id"], sites["area_sqm"], strict=True)]
    products = None
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["key"] == key:
            products = saved["products"]
            print("[crosscheck] land-cover products loaded from cache")
    if products is None:
        products = {
            "worldcover_2021": class_shares(ids, read_on_grid(WORLDCOVER_URL, grid), WORLDCOVER, n)
        }
        print(f"[crosscheck] WorldCover ({time.perf_counter() - t0:.0f}s)")
        bounds = io.read_boundary(cfg["paths"]["boundary"], 4326).total_bounds
        esri = read_on_grid(esri_io_url(bounds, xc["esri_io_year"]), grid)
        products[f"esri_io_{xc['esri_io_year']}"] = class_shares(ids, esri, ESRI_IO, n)
        print(f"[crosscheck] Esri IO ({time.perf_counter() - t0:.0f}s)")
        hists = dynamic_world_hist(sites.to_crs(4326), xc["dynamic_world_date"], xc["ee_project"])
        products["dynamic_world"] = [histogram_shares(h, DYNAMIC_WORLD) for h in hists]
        print(f"[crosscheck] Dynamic World ({time.perf_counter() - t0:.0f}s)")
        cache.write_text(json.dumps({"key": key, "products": products}), encoding="utf-8")

    for p, shares in products.items():
        sites[f"{p}_dominant"] = [dominant(s)[0] for s in shares]
        sites[f"{p}_dominant_share"] = [round(dominant(s)[1], 3) for s in shares]
        sites[f"{p}_crops"] = [round(s["crops"], 3) for s in shares]
        sites[f"{p}_built"] = [round(s["built"], 3) for s in shares]
    sites["product_verdict"] = [
        verdict([sites.loc[i, f"{p}_dominant"] for p in products]) for i in sites.index
    ]

    # 2. Visual chips (local data only).
    render_chips(
        sites,
        outputs / "phase1" / "sentinel_stack.tif",
        out / "chips",
        xc["chip_pad_m"],
        cfg["sentinel"]["boa_offset"],
    )
    print(f"[crosscheck] chips -> {out / 'chips'}")

    sites.to_parquet(out / "sites_crosscheck.parquet")
    detail = {"products": products, "site_ids": list(sites["site_id"])}
    (out / "crosscheck.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print(sites["product_verdict"].value_counts().to_string())
    print(f"[crosscheck] -> {out / 'sites_crosscheck.parquet'}; run `crosscheck-osm` for OSM")
    return sites


def run_crosscheck_osm(cfg: dict, model_name: str = "final") -> gpd.GeoDataFrame:
    """Add OSM land-use overlap to the cross-check, from a local OSM extract (.osm.pbf).

    Separate from `crosscheck` so the other evidence doesn't depend on OSM. A local
    extract replaced the Overpass API: the public server was at capacity ("all query
    workers are busy") and asks automated users not to use it.
    """
    outputs = Path(cfg["paths"]["output"])
    out = outputs / model.get_spec(cfg, model_name).out_dir
    src = out / "sites_crosscheck.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run `python -m stp.cli crosscheck` first")
    sites = gpd.read_parquet(src)
    pbf = Path(cfg["crosscheck"]["osm_pbf"])
    boundary = io.read_boundary(cfg["paths"]["boundary"], 4326)
    landuse = read_osm_landuse(pbf, boundary.union_all())
    osm = osm_landuse_overlap(sites, landuse)
    sites["osm_landuse"] = [
        "; ".join(f"{k} {v:.0%}" for k, v in list(d.items())[:3]) or "none" for d in osm
    ]
    sites.to_parquet(src)
    print(
        f"[crosscheck-osm] {pbf.name}: {len(landuse)} OSM landuse polygons; "
        f"{int((sites['osm_landuse'] != 'none').sum())}/{len(sites)} sites overlap one -> {src}"
    )
    return sites
