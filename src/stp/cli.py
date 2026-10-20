"""Command-line entry point: `python -m stp.cli <command> [--config config.yaml]`.

Commands: phase1, phase2, phase3, phase3k, phase4, phase5, sensitivity, sites, network,
threshold, ahp, review
(see `--help`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS

from stp import (
    io,
    lulc,
    network,
    overlay,
    parity,
    preprocess,
    proximity,
    selection,
    sensitivity,
    standardize,
)
from stp import sites as site_stats
from stp.config import DEFAULT_CONFIG, Grid, load_config

# See scripts/diagnose_b11.py and docs/parity.md.
B11_NOTE = "QGIS run took the adjacent 20 m row for 28% of pixels; GDAL 3.13 replay = ours"


def run_phase1(cfg: dict) -> list[parity.ParityResult]:
    crs = CRS.from_user_input(cfg["crs"])
    out = Path(cfg["paths"]["output"]) / "phase1"
    ref = Path(cfg["paths"]["reference_rasters"])
    bands = cfg["sentinel"]["bands"]
    offset = cfg["sentinel"]["boa_offset"] if cfg["indices"]["apply_boa_offset"] else 0.0

    t0 = time.perf_counter()
    boundary = io.read_boundary(cfg["paths"]["boundary"], crs)
    geom = boundary.union_all()
    bounds = tuple(round(v, 2) for v in geom.bounds)
    print(f"[phase1] boundary: {len(boundary)} feature(s), bounds {bounds}")

    stack, s_tf = preprocess.build_stack(cfg["paths"]["sentinel_granule"], bands, geom, crs)
    io.write_cog(out / "sentinel_stack.tif", stack, s_tf, crs, preprocess.STACK_NODATA)
    print(f"[phase1] stack {stack.shape} origin {s_tf.c:.4f},{s_tf.f:.4f}")

    indices = preprocess.spectral_indices(stack, bands, offset=offset)
    for name, arr in indices.items():
        io.write_cog(out / f"{name}.tif", arr, s_tf, crs, preprocess.FLOAT_NODATA)

    dem, d_tf, d_nodata = preprocess.clip_dem(cfg["paths"]["dem"], geom, crs)
    io.write_cog(out / "dem.tif", dem, d_tf, crs, d_nodata)
    slope = preprocess.horn_slope(dem, d_tf.a, d_nodata)
    io.write_cog(out / "slope.tif", slope, d_tf, crs, preprocess.SLOPE_NODATA)
    elapsed = time.perf_counter() - t0
    print(f"[phase1] dem {dem.shape} origin {d_tf.c:.4f},{d_tf.f:.4f}  ({elapsed:.1f}s)")

    # Parity against QGIS. QGIS computed indices on raw DN (no BOA offset), so the
    # check recomputes them without the offset; the files above keep the correction.
    raw = preprocess.spectral_indices(stack, bands, offset=0.0)
    results = [
        parity.compare(
            f"stack_{b}",
            stack[i],
            s_tf,
            preprocess.STACK_NODATA,
            ref / "Raster_01_Sentinel_Bhopal_Clipped.tif",
            band=i + 1,
            tol=0,
            ref_nodata_override=0,
            note=B11_NOTE if b == "B11" else "",
        )
        for i, b in enumerate(bands)
    ]
    results += [
        parity.compare(
            "ndbi_raw_dn",
            raw["ndbi"],
            s_tf,
            preprocess.FLOAT_NODATA,
            ref / "Raster_02_NDBI.tif",
            tol=1e-6,
            note="differs only where QGIS B11 differs",
        ),
        parity.compare(
            "ndvi_raw_dn",
            raw["ndvi"],
            s_tf,
            preprocess.FLOAT_NODATA,
            ref / "Raster_03_NDVI.tif",
            tol=1e-6,
            resample_to_ref=True,
            note="ref on 12.5 m grid; QGIS resampling rule unknown",
        ),
        parity.compare(
            "mndwi_raw_dn",
            raw["mndwi"],
            s_tf,
            preprocess.FLOAT_NODATA,
            ref / "Raster_04_MNDWI.tif",
            tol=1e-6,
            resample_to_ref=True,
            note="ref on 12.5 m grid; QGIS resampling rule unknown",
        ),
        parity.compare("dem", dem, d_tf, d_nodata, ref / "Raster_09_DEM_Bhopal_Clipped.tif", tol=0),
        parity.compare(
            "slope", slope, d_tf, preprocess.SLOPE_NODATA, ref / "Raster_10_Slope.tif", tol=1e-4
        ),
    ]
    # How much the missing BOA offset shifted the indices QGIS used.
    for name in ("ndvi", "ndbi", "mndwi"):
        d = np.abs(indices[name] - raw[name])
        mean_d, max_d = np.nanmean(d), np.nanmax(d)
        print(f"[phase1] BOA offset effect on {name}: mean|d|={mean_d:.4f} max|d|={max_d:.4f}")

    report = Path(cfg["paths"]["output"]) / "parity" / "phase1.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8")
    print(parity.format_table(results))
    print(f"[phase1] parity report -> {report}")
    return results


def run_phase2(cfg: dict) -> list[parity.ParityResult]:
    crs = CRS.from_user_input(cfg["crs"])
    out = Path(cfg["paths"]["output"]) / "phase2"
    ref = Path(cfg["paths"]["reference_rasters"])
    grid = Grid.from_config(cfg, "proximity_10m")
    geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
    inside = ~preprocess.outside_mask(geom, grid.shape, grid.transform)

    (out / "vectors").mkdir(parents=True, exist_ok=True)

    results, vector_checks = [], []
    for name, layer in cfg["proximity"]["layers"].items():
        t0 = time.perf_counter()
        gdf = proximity.load_clipped(cfg["paths"]["vectors_raw"], layer["raw"], geom, crs)
        gdf.to_parquet(out / "vectors" / f"{name}.parquet")
        dist, target = proximity.distance_raster(gdf, grid, geom)
        io.write_cog(out / f"dist_{name}.tif", dist, grid.transform, crs, proximity.DIST_NODATA)
        elapsed = time.perf_counter() - t0
        print(
            f"[phase2] {name}: {len(gdf)} features, {int(target.sum())} target px ({elapsed:.1f}s)"
        )

        vector_checks.append(
            parity.compare_vectors(
                name, gdf, Path(cfg["paths"]["reference_vectors"]) / f"{layer['qgis_clip']}.shp"
            )
        )
        # Burned pixels vs QGIS: distance == 0 inside the boundary.
        with rasterio.open(ref / layer["reference"]) as src:
            ref_zero = (src.read(1) == 0) & inside
        burned = (target == 1) & inside
        agree = float(np.mean(burned[inside] == ref_zero[inside]))
        print(
            f"[phase2] {name}: burned-pixel agreement {agree:.6f} "
            f"(ours {int(burned.sum())}, QGIS {int(ref_zero.sum())})"
        )
        results.append(
            parity.compare(
                f"dist_{name}",
                dist,
                grid.transform,
                proximity.DIST_NODATA,
                ref / layer["reference"],
                tol=1e-2,
            )
        )

    print(f"{'vector':12s} {'n_ours':>7s} {'n_qgis':>7s} {'measure':>7s} {'rel_diff':>10s}")
    for v in vector_checks:
        extra = f"  sym_diff_area={v['sym_diff_area']:.1f} m2" if "sym_diff_area" in v else ""
        print(
            f"{v['name']:12s} {v['n_ours']:7d} {v['n_ref']:7d} {v['measure']:>7s} "
            f"{v['rel_diff']:10.2e}{extra}"
        )
    report = Path(cfg["paths"]["output"]) / "parity" / "phase2.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rasters": [r.as_dict() for r in results], "vectors": vector_checks}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(parity.format_table(results))
    print(f"[phase2] parity report -> {report}")
    return results


def run_phase3(cfg: dict) -> dict:
    """3a: reproduce the original K-Means (void included) on both stacks and compare."""
    crs = CRS.from_user_input(cfg["crs"])
    out = Path(cfg["paths"]["output"]) / "phase3"
    ref = Path(cfg["paths"]["reference_rasters"])
    lc = cfg["lulc"]
    table = {int(k): v for k, v in lc["reclass"].items()}
    ours_stack = Path(cfg["paths"]["output"]) / "phase1" / "sentinel_stack.tif"
    if not ours_stack.exists():
        raise FileNotFoundError(f"{ours_stack} missing: run `python -m stp.cli phase1` first")
    inputs = {"qgis_stack": ref / "Raster_01_Sentinel_Bhopal_Clipped.tif", "our_stack": ours_stack}

    with rasterio.open(ref / "ML_KMeans_LULC_Bhopal.tif") as src:
        ref_classes = src.read(1)

    summary, results = {}, []
    for name, path in inputs.items():
        with rasterio.open(path) as src:
            stack, transform = src.read(), src.transform
        t0 = time.perf_counter()
        x, valid = lulc.pixel_table(stack, mask_void=False)
        model = lulc.fit_kmeans(x, lc["n_clusters"], lc["random_state"], lc["batch_size"])
        classes = lulc.class_raster(model.labels_, valid, stack.shape[1:])
        match = lulc.match_classes(classes, ref_classes)
        matched = lulc.relabel(classes, match["mapping"])
        scores = lulc.reclassify(matched, table)
        io.write_cog(
            out / f"kmeans_original_{name}.tif", matched, transform, crs, lulc.CLASS_NODATA
        )
        io.write_cog(
            out / f"lulc_score_original_{name}.tif", scores, transform, crs, lulc.SCORE_NODATA
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[phase3] {name}: k={lc['n_clusters']} void included, {elapsed:.1f}s; "
            f"mapping ours->ref {match['mapping']}; agreement {match['agreement']:.6f}"
        )
        summary[name] = match
        results.append(
            parity.compare(
                f"lulc_score_{name}",
                scores,
                transform,
                lulc.SCORE_NODATA,
                ref / "Reclass_ML_LULC.tif",
                tol=0,
            )
        )

    print(parity.format_table(results))
    corrected = _phase3_corrected(cfg, ours_stack, crs, out, ref / "Reclass_ML_LULC.tif")

    report = Path(cfg["paths"]["output"]) / "parity" / "phase3.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kmeans": summary,
        "rasters": [r.as_dict() for r in results],
        "corrected": corrected,
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[phase3] report -> {report}")
    return payload


def _phase3_corrected(
    cfg: dict, stack_path: Path, crs: CRS, out: Path, original_scores: Path
) -> dict:
    """3b: void-masked, k=4 full K-Means; clusters named with OSM evidence; scored."""
    cc = cfg["lulc"]["corrected"]
    bands = cfg["sentinel"]["bands"]
    with rasterio.open(stack_path) as src:
        stack, transform = src.read(), src.transform
    grid = Grid.from_config(cfg, "sentinel_10m")
    if stack.shape[1:] != grid.shape or not transform.almost_equals(grid.transform):
        raise ValueError("Phase 1 stack is not on the configured sentinel_10m grid")

    t0 = time.perf_counter()
    x, valid = lulc.pixel_table(stack, mask_void=True)
    model = lulc.fit_full_kmeans(
        x, cc["k"], cc["n_init"], cc["random_state"], cc["fit_size"], cc["sample_seed"]
    )
    labels = lulc.predict_chunked(model, x)
    profiles = lulc.cluster_profiles(model, bands, labels, offset=cfg["sentinel"]["boa_offset"])

    geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
    ratios = {}
    for name in ("residential", "water"):
        layer = cfg["proximity"]["layers"][name]["raw"]
        gdf = proximity.load_clipped(cfg["paths"]["vectors_raw"], layer, geom, crs)
        inside = (proximity.burn(gdf, grid).reshape(-1) == 1)[valid]
        ratios[name] = lulc.overlap_ratio(labels, inside, cc["k"])
    names = lulc.name_clusters(profiles, ratios["residential"], ratios["water"])

    to_code = {cluster: lulc.CLASS_CODES[name] for name, cluster in names.items()}
    codes = lulc.relabel(lulc.class_raster(labels, valid, grid.shape), to_code)
    score_table = {lulc.CLASS_CODES[name]: s for name, s in cc["scores"].items()}
    scores = lulc.reclassify(codes, score_table)
    io.write_cog(out / "lulc_classes.tif", codes, transform, crs, lulc.CLASS_NODATA)
    io.write_cog(out / "lulc_score.tif", scores, transform, crs, lulc.SCORE_NODATA)
    print(f"[phase3] corrected k={cc['k']} full K-Means: {time.perf_counter() - t0:.0f}s")

    for name, cluster in names.items():
        p = next(p for p in profiles if p["cluster"] == cluster)
        print(
            f"[phase3]   {name:10s} cluster {cluster}: {p['share']:.3f} of area, "
            f"ndvi {p['ndvi']:+.2f} mndwi {p['mndwi']:+.2f}, "
            f"residential x{ratios['residential'][cluster]:.2f} "
            f"water x{ratios['water'][cluster]:.2f} -> score {cc['scores'][name]}"
        )

    # How the corrected scores differ from the original Reclass_ML_LULC.
    with rasterio.open(original_scores) as src:
        orig = src.read(1)
    both = (orig != src.nodata) & (scores != lulc.SCORE_NODATA)
    change = {}
    for o in sorted(np.unique(orig[both]).tolist()):
        sel = both & (orig == o)
        vals, counts = np.unique(scores[sel], return_counts=True)
        change[str(o)] = {
            str(v): round(float(c / sel.sum()), 4) for v, c in zip(vals, counts, strict=True)
        }
    same = float(np.mean(orig[both] == scores[both]))
    print(f"[phase3]   score identical to original on {same:.1%} of pixels")
    for o, row in change.items():
        print(f"[phase3]     original score {o} -> corrected {row}")
    return {
        "names": names,
        "profiles": profiles,
        "residential_ratio": ratios["residential"],
        "water_ratio": ratios["water"],
        "score_same_as_original": same,
        "original_to_corrected": change,
    }


def run_phase3k(cfg: dict) -> list[dict]:
    """3b evidence: k-selection metrics and cluster profiles on the void-masked stack."""
    out = Path(cfg["paths"]["output"]) / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    stack_path = Path(cfg["paths"]["output"]) / "phase1" / "sentinel_stack.tif"
    if not stack_path.exists():
        raise FileNotFoundError(f"{stack_path} missing: run `python -m stp.cli phase1` first")
    sel = cfg["lulc"]["k_selection"]
    bands = cfg["sentinel"]["bands"]

    with rasterio.open(stack_path) as src:
        stack = src.read()
    x, valid = lulc.pixel_table(stack, mask_void=True)
    print(f"[phase3k] {len(x)} valid pixels ({valid.mean():.1%} of the grid)")

    t0 = time.perf_counter()
    rows = lulc.k_selection(x, sel["ks"], sel["seeds"], sel["fit_size"], sel["eval_size"])
    print(f"[phase3k] metrics in {time.perf_counter() - t0:.0f}s")
    print(f"{'k':>3s} {'inertia':>12s} {'silhouette':>13s} {'DB':>6s} {'ARI':>7s} {'ARImin':>7s}")
    for r in rows:
        print(
            f"{r['k']:3d} {r['inertia']:12.4e} {r['silhouette']:7.4f}+-{r['silhouette_sd']:.3f} "
            f"{r['davies_bouldin']:6.3f} {r['stability_ari']:7.4f} {r['stability_ari_min']:7.4f}"
        )

    # Cluster profiles per k (same fit sample, first seed) to help name the clusters.
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(len(x), size=min(sel["fit_size"], len(x)), replace=False)
    x_fit = x[fit_idx].astype("float64")
    profiles = {}
    for k in sel["ks"]:
        model = lulc.fit_kmeans(x_fit, k, random_state=sel["seeds"][0])
        profiles[k] = lulc.cluster_profiles(
            model, bands, model.labels_, offset=cfg["sentinel"]["boa_offset"]
        )

    _plot_k_selection(rows, out / "k_selection.png")
    payload = {"metrics": rows, "profiles": profiles, "n_pixels": int(len(x)), "config": sel}
    (out / "k_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[phase3k] -> {out / 'k_selection.json'}, {out / 'k_selection.png'}")
    return rows


def _plot_k_selection(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    panels = [
        ("inertia", "Inertia (elbow)", "lower = tighter clusters"),
        ("silhouette", "Silhouette", "higher is better"),
        ("davies_bouldin", "Davies-Bouldin", "lower is better"),
        ("stability_ari", "Seed stability (ARI)", "1 = same partition every seed"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (key, title, sub) in zip(axes, panels, strict=True):
        ax.plot(ks, [r[key] for r in rows], marker="o")
        ax.set_title(f"{title}\n({sub})", fontsize=10)
        ax.set_xlabel("k")
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _read_da(path: Path, crs: CRS):
    """Read band 1 as a georeferenced DataArray (NaN = NoData) plus array/transform."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the earlier phases first")
    with rasterio.open(path) as src:
        arr, transform, nodata = src.read(1), src.transform, src.nodata
    return overlay.to_dataarray(arr, transform, crs, nodata), arr, transform, nodata


def run_phase4(cfg: dict) -> dict:
    """Scores (4a), explicit grid alignment and weighted overlay (4b), with parity."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase4"
    ref = Path(cfg["paths"]["reference_rasters"])
    grid = Grid.from_config(cfg, "sentinel_10m")
    template = overlay.grid_template(grid)
    weights = cfg["weights"]

    # 4a: scores on each layer's native grid, compared with the QGIS Reclass_* rasters.
    results, ours, refs = [], {}, {}
    for name, sc in cfg["standardize"].items():
        _, arr, transform, nodata = _read_da(outputs / sc["source"], crs)
        valid = np.isfinite(arr) & (arr != nodata) if nodata is not None else np.isfinite(arr)
        score = standardize.rescale_score(arr, valid, sc["invert"])
        io.write_cog(out / f"score_{name}.tif", score, transform, crs, standardize.SCORE_NODATA)
        results.append(
            parity.compare(
                f"score_{name}",
                score,
                transform,
                standardize.SCORE_NODATA,
                ref / sc["reference"],
                tol=0,
            )
        )
        ours[name] = overlay.align(
            overlay.to_dataarray(score, transform, crs, standardize.SCORE_NODATA), template
        )
        refs[name] = overlay.align(_read_da(ref / sc["reference"], crs)[0], template)

    lulc_original = overlay.align(_read_da(ref / "Reclass_ML_LULC.tif", crs)[0], template)
    lulc_corrected = overlay.align(
        _read_da(outputs / "phase3" / "lulc_score.tif", crs)[0], template
    )

    # 4b: three overlays, so each difference can be traced to its cause.
    final_ref = ref / "Final_STP_Suitability_Map.tif"
    variants = {
        "qgis_scores": {**refs, "lulc": lulc_original},
        "pipeline_original_lulc": {**ours, "lulc": lulc_original},
        "pipeline_corrected_lulc": {**ours, "lulc": lulc_corrected},
    }
    summary = {}
    for vname, layers in variants.items():
        s = overlay.weighted_sum(layers, weights).astype("float32")
        io.write_cog(out / f"suitability_{vname}.tif", s, grid.transform, crs, np.nan)
        valid = np.isfinite(s)
        summary[vname] = {
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "mean": float(np.nanmean(s)),
            "share_ge_8": float(np.mean(s[valid] >= cfg["selection"]["suitability_min"])),
            "n_valid": int(valid.sum()),
        }
        if vname != "pipeline_corrected_lulc":
            results.append(
                parity.compare(f"suit_{vname}", s, grid.transform, np.nan, final_ref, tol=1e-4)
            )

    with rasterio.open(final_ref) as src:
        r = src.read(1).astype("float64")
    r = r[(r > -1e30) & np.isfinite(r)]
    summary["qgis_reference"] = {
        "min": float(r.min()),
        "max": float(r.max()),
        "mean": float(r.mean()),
        "share_ge_8": float(np.mean(r >= cfg["selection"]["suitability_min"])),
        "n_valid": int(r.size),
    }
    print(f"{'surface':26s} {'min':>6s} {'max':>6s} {'mean':>6s} {'>=8':>8s} {'valid px':>9s}")
    for k, v in summary.items():
        print(
            f"{k:26s} {v['min']:6.2f} {v['max']:6.2f} {v['mean']:6.2f} "
            f"{v['share_ge_8']:8.4%} {v['n_valid']:9d}"
        )

    report = outputs / "parity" / "phase4.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rasters": [x.as_dict() for x in results], "surfaces": summary}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(parity.format_table(results))
    print(f"[phase4] report -> {report}")
    return payload


def run_phase5(cfg: dict) -> dict:
    """Queries, polygonize, exclusion buffers, area filter; three variants with parity."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase5"
    ref = Path(cfg["paths"]["reference_rasters"])
    sel = cfg["selection"]
    grid = Grid.from_config(cfg, "sentinel_10m")
    template = overlay.grid_template(grid)

    def aligned(path: Path) -> np.ndarray:
        return overlay.align(_read_da(path, crs)[0], template)

    roads = aligned(outputs / "phase4" / "score_roads.tif")
    lulc_scores = {
        "original": aligned(ref / "Reclass_ML_LULC.tif"),
        "corrected": aligned(outputs / "phase3" / "lulc_score.tif"),
    }
    surfaces = {
        "original": aligned(outputs / "phase4" / "suitability_pipeline_original_lulc.tif"),
        "corrected": aligned(outputs / "phase4" / "suitability_pipeline_corrected_lulc.tif"),
    }

    geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
    points = [
        proximity.load_clipped(cfg["paths"]["vectors_raw"], name, geom, crs)
        for name in sel["exclusion_layers"]
    ]
    exclusions = selection.exclusion_zones(points, sel["buffer_m"], sel["buffer_segments"])
    print(f"[phase5] exclusion buffers: {[len(p) for p in points]} points -> {len(exclusions)}")
    cut = exclusions.union_all()

    variants = {
        "A_qgis_order_original_lulc": ("original", "filter_then_difference"),
        "B_corrected_order_original_lulc": ("original", "difference_then_filter"),
        "C_corrected_order_corrected_lulc": ("corrected", "difference_then_filter"),
    }
    query_files = {
        "query_01": "Query_01_High_Potential_Sites.tif",
        "query_02": "Query_02_Environmental_Check.tif",
        "query_03": "Query_03_Final_Recommendations.tif",
    }
    results, vectors, summary = [], [], {}
    for vname, (lulc_kind, order) in variants.items():
        vout = out / vname
        q, cand, sites = selection.sites_from_surface(
            surfaces[lulc_kind],
            lulc_scores[lulc_kind],
            roads,
            cut,
            sel,
            grid.transform,
            crs,
            order,
        )
        candidates = cand[np.round(cand["area_sqm"]) >= sel["min_area_sqm"]].copy()

        for qname in query_files:
            arr = np.where(q["valid"], q[qname].astype("float32"), np.nan).astype("float32")
            io.write_cog(vout / f"{qname}.tif", arr, grid.transform, crs, np.nan)
        cand.to_parquet(vout / "all_polygons.parquet")
        sites.to_parquet(vout / "sites.parquet")
        sites.to_crs(4326).to_file(vout / "sites.geojson", driver="GeoJSON")

        counts = {k: int(q[k].sum()) for k in query_files}
        summary[vname] = {
            "lulc": lulc_kind,
            "order": order,
            "query_pixels": counts,
            "polygons": len(cand),
            "candidates_ge_min_area": len(candidates),
            "sites": len(sites),
            "site_area_sqm": float(sites["area_sqm"].sum()),
            "smallest_site_sqm": float(sites["area_sqm"].min()) if len(sites) else None,
        }
        print(
            f"[phase5] {vname}: query px {counts}, polygons {len(cand)}, "
            f"candidates>=min {len(candidates)}, sites {len(sites)}, "
            f"area {sites['area_sqm'].sum():,.0f} m2"
        )

        if vname.startswith("A_"):
            q_ref = Path(cfg["paths"]["reference_queries"])
            for qname, fname in query_files.items():
                arr = np.where(q["valid"], q[qname].astype("float32"), np.nan)
                results.append(
                    parity.compare(qname, arr, grid.transform, np.nan, q_ref / fname, tol=0)
                )
            dash = Path(cfg["paths"]["reference_dashboard"]) / "Candidate_Sites_Polygons.geojson"
            vectors.append(parity.compare_vectors("candidates", candidates, dash))
            final_ref = Path(cfg["paths"]["reference_vectors"]) / "Final_Safe_Sites.shp"
            vectors.append(parity.compare_vectors("final_sites", sites, final_ref))

    print(parity.format_table(results))
    for v in vectors:
        print(
            f"[phase5] {v['name']:12s} ours {v['n_ours']} vs QGIS {v['n_ref']}; "
            f"{v['measure']} rel diff {v['rel_diff']:.2e}; "
            f"sym diff {v.get('sym_diff_area', float('nan')):.1f} m2"
        )
    report = outputs / "parity" / "phase5.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rasters": [r.as_dict() for r in results], "vectors": vectors, "variants": summary}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[phase5] report -> {report}")
    return payload


def _corrected_inputs(cfg: dict, crs: CRS, grid: Grid):
    """Aligned score layers (corrected land cover), roads score, and the exclusion cut."""
    outputs = Path(cfg["paths"]["output"])
    template = overlay.grid_template(grid)

    def aligned(path: Path) -> np.ndarray:
        return overlay.align(_read_da(path, crs)[0], template).astype("float32")

    layers = {
        name: aligned(outputs / "phase4" / f"score_{name}.tif") for name in cfg["standardize"]
    }
    layers["lulc"] = aligned(outputs / "phase3" / "lulc_score.tif")
    geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
    sel = cfg["selection"]
    points = [
        proximity.load_clipped(cfg["paths"]["vectors_raw"], name, geom, crs)
        for name in sel["exclusion_layers"]
    ]
    cut = selection.exclusion_zones(points, sel["buffer_m"], sel["buffer_segments"]).union_all()
    return layers, cut


def run_sensitivity(cfg: dict) -> dict:
    """Phase 6a: Monte Carlo and one-at-a-time weight sensitivity of the corrected result."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid.from_config(cfg, "sentinel_10m")
    sc = cfg["sensitivity"]
    sel = cfg["selection"]
    base = cfg["weights"]
    layers, cut = _corrected_inputs(cfg, crs, grid)

    def select(surface: np.ndarray):
        return selection.sites_from_surface(
            surface, layers["lulc"], layers["roads"], cut, sel, grid.transform, crs
        )[2]

    baseline = sensitivity.run_weights(layers, base, select)
    phase5 = gpd.read_parquet(
        outputs / "phase5" / "C_corrected_order_corrected_lulc" / "sites.parquet"
    )
    if len(baseline) != len(phase5):
        raise RuntimeError(f"Baseline has {len(baseline)} sites but Phase 5 C has {len(phase5)}")
    base_mask = sensitivity.site_mask(baseline, grid.shape, grid.transform)
    ids = sensitivity.site_ids(baseline, grid.shape, grid.transform)
    n = len(baseline)
    print(f"[sensitivity] baseline: {n} sites, {base_mask.sum()} px (matches Phase 5 C)")

    # Monte Carlo
    rng = np.random.default_rng(sc["seed"])
    freq = np.zeros(grid.shape, dtype="float32")
    persistent = np.zeros(n)
    draws = []
    t0 = time.perf_counter()
    for d in range(sc["draws"]):
        w = sensitivity.perturb_weights(base, sc["delta"], rng)
        sites = sensitivity.run_weights(layers, w, select)
        mask = sensitivity.site_mask(sites, grid.shape, grid.transform)
        freq += mask
        cover = sensitivity.coverage_by_site(ids, mask, n)
        persistent += cover >= sc["persistence_cover"]
        draws.append(
            {
                "weights": w,
                "sites": len(sites),
                "area_sqm": float(sites["area_sqm"].sum()),
                "iou_vs_baseline": sensitivity.iou(mask, base_mask),
            }
        )
        if (d + 1) % 50 == 0:
            rate = (time.perf_counter() - t0) / (d + 1)
            print(f"[sensitivity] {d + 1}/{sc['draws']} draws ({rate:.1f}s/draw)")
    freq /= sc["draws"]
    persistence = persistent / sc["draws"]
    io.write_cog(out / "selection_frequency.tif", freq, grid.transform, crs, None)

    baseline["persistence"] = persistence
    baseline["mean_selection_freq"] = [
        float(freq[ids == i + 1].mean()) if (ids == i + 1).any() else 0.0 for i in range(n)
    ]
    baseline.to_parquet(out / "sites_sensitivity.parquet")

    n_sites = np.array([x["sites"] for x in draws])
    area = np.array([x["area_sqm"] for x in draws]) / 1e4
    ious = np.array([x["iou_vs_baseline"] for x in draws])
    print(
        f"[sensitivity] sites per draw: median {np.median(n_sites):.0f} "
        f"(5-95%: {np.percentile(n_sites, 5):.0f}-{np.percentile(n_sites, 95):.0f}); "
        f"area median {np.median(area):.1f} ha (5-95%: {np.percentile(area, 5):.1f}-"
        f"{np.percentile(area, 95):.1f}); IoU vs baseline median {np.median(ious):.3f}"
    )
    for thr in (0.9, 0.75, 0.5):
        print(
            f"[sensitivity] sites persistent in >= {thr:.0%} of draws: "
            f"{int((persistence >= thr).sum())}/{n}"
        )

    # One-at-a-time
    oat = []
    for name in base:
        for sign in (-1, 1):
            w = sensitivity.shift_one(base, name, 1 + sign * sc["oat_delta"])
            sites = sensitivity.run_weights(layers, w, select)
            mask = sensitivity.site_mask(sites, grid.shape, grid.transform)
            oat.append(
                {
                    "criterion": name,
                    "change": sign * sc["oat_delta"],
                    "sites": len(sites),
                    "area_ha": float(sites["area_sqm"].sum() / 1e4),
                    "iou_vs_baseline": sensitivity.iou(mask, base_mask),
                }
            )
    print(
        f"{'criterion':12s} {'-20%: sites':>12s} {'ha':>7s} {'IoU':>6s} "
        f"{'+20%: sites':>12s} {'ha':>7s} {'IoU':>6s}"
    )
    for name in base:
        lo, hi = [r for r in oat if r["criterion"] == name]
        print(
            f"{name:12s} {lo['sites']:12d} {lo['area_ha']:7.1f} {lo['iou_vs_baseline']:6.3f} "
            f"{hi['sites']:12d} {hi['area_ha']:7.1f} {hi['iou_vs_baseline']:6.3f}"
        )

    payload = {
        "config": sc,
        "baseline_sites": n,
        "baseline_area_ha": float(baseline["area_sqm"].sum() / 1e4),
        "monte_carlo": draws,
        "persistence": persistence.tolist(),
        "one_at_a_time": oat,
    }
    (out / "sensitivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[sensitivity] -> {out}")
    return payload


def run_sites(cfg: dict) -> gpd.GeoDataFrame:
    """Phase 6b: per-site statistics, nearest-feature distances and ranking."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase6"
    src = out / "sites_sensitivity.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run `python -m stp.cli sensitivity` first")
    s = gpd.read_parquet(src)

    rasters = [
        (
            "suit",
            outputs / "phase4" / "suitability_pipeline_corrected_lulc.tif",
            ["mean", "min", "max"],
        ),
        ("slope_deg", outputs / "phase1" / "slope.tif", ["mean", "max"]),
        ("elev_m", outputs / "phase1" / "dem.tif", ["mean", "min", "max"]),
        ("road_dist_m", outputs / "phase2" / "dist_roads.tif", ["min", "mean"]),
        ("drain_dist_m", outputs / "phase2" / "dist_drainage.tif", ["min"]),
    ]
    for prefix, path, stats in rasters:
        s = site_stats.raster_stats(s, path, stats, prefix)
    s["elev_range_m"] = s["elev_m_max"] - s["elev_m_min"]

    geom = io.read_boundary(cfg["paths"]["boundary"], crs).union_all()
    near = {
        "school": "Schools_Bhopal",
        "hospital": "Hospital_Bhopal",
        "lake": cfg["proximity"]["layers"]["water"]["raw"],
        "residential": cfg["proximity"]["layers"]["residential"]["raw"],
    }
    for label, layer in near.items():
        feats = proximity.load_clipped(cfg["paths"]["vectors_raw"], layer, geom, crs)
        s = site_stats.nearest_distance(s, feats, f"nearest_{label}_m")
    # The 500 m buffers are 120-gon approximations of circles, so allow ~1 m below 500.
    floor = cfg["selection"]["buffer_m"] - 1.0
    for label in ("school", "hospital"):
        bad = s[f"nearest_{label}_m"] < floor
        if bad.any():
            raise RuntimeError(f"{int(bad.sum())} sites closer than {floor} m to a {label}")

    cent = s.geometry.centroid.to_crs(4326)
    s["lon"], s["lat"] = cent.x, cent.y
    ranked = site_stats.rank_sites(s, cfg["sensitivity"]["robust_threshold"])

    _write_site_table(ranked, out)
    table = ranked.drop(columns="geometry").copy()
    table["area_ha"] = table["area_sqm"] / 1e4

    cols = [
        "rank",
        "area_ha",
        "suit_mean",
        "persistence",
        "slope_deg_mean",
        "elev_range_m",
        "road_dist_m_min",
        "nearest_school_m",
        "nearest_hospital_m",
        "nearest_lake_m",
    ]
    print(table[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    robust = int(ranked["robust"].sum())
    thr = cfg["sensitivity"]["robust_threshold"]
    print(f"[sites] {len(ranked)} sites, {robust} robust (persistence >= {thr:.0%}) -> {out}")
    return ranked


def _write_site_table(ranked: gpd.GeoDataFrame, out: Path) -> None:
    ranked.to_parquet(out / "sites_ranked.parquet")
    ranked.to_crs(4326).to_file(out / "sites_ranked.geojson", driver="GeoJSON")
    table = ranked.drop(columns="geometry").copy()
    table["area_ha"] = table["area_sqm"] / 1e4
    num = table.select_dtypes("number").columns
    table[num] = table[num].round(4)
    table.to_csv(out / "sites_ranked.csv", index=False)


def run_network(cfg: dict) -> gpd.GeoDataFrame:
    """Phase 6d: drivable-network access from each site to the nearest major road."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase6"
    nc = cfg["network"]
    src = out / "sites_ranked.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run `python -m stp.cli sites` first")
    ranked = gpd.read_parquet(src)

    boundary = io.read_boundary(cfg["paths"]["boundary"], crs)
    area = boundary.buffer(nc["boundary_buffer_m"]).to_crs(4326).union_all()
    t0 = time.perf_counter()
    g = network.load_drive_graph(area, out / "drive_network.graphml", outputs / "cache", crs)
    print(
        f"[network] graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
        f"({time.perf_counter() - t0:.0f}s)"
    )
    majors = network.major_nodes(g)
    to_major = network.distance_to_nodes(g, majors)
    print(f"[network] {len(majors)} major-road nodes; {len(to_major)} nodes can reach one")

    ranked = network.site_access(ranked, g, to_major)
    ranked["access_total_m"] = ranked["last_mile_m"] + ranked["network_to_major_m"]
    _write_site_table(ranked, out)

    cols = [
        "rank",
        "area_sqm",
        "road_dist_m_min",
        "last_mile_m",
        "network_to_major_m",
        "access_total_m",
        "robust",
    ]
    print(ranked[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    fin = ranked.loc[ranked["reaches_major"], "access_total_m"]
    print(
        f"[network] {int(ranked['reaches_major'].sum())}/{len(ranked)} sites reach a major "
        f"road; access distance median {fin.median():.0f} m, max {fin.max():.0f} m"
    )
    return ranked


def main() -> None:
    p = argparse.ArgumentParser(prog="stp")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("phase1", help="band stack, DEM, slope, indices + parity")
    sub.add_parser("phase2", help="distance rasters + parity")
    sub.add_parser("phase3", help="original K-Means land cover (3a) + parity")
    sub.add_parser("phase3k", help="k-selection metrics for the corrected K-Means (3b)")
    sub.add_parser("phase4", help="1-10 scores, grid alignment, weighted overlay + parity")
    sub.add_parser("phase5", help="queries, polygons, exclusion buffers, final sites + parity")
    sub.add_parser("sensitivity", help="Phase 6a: weight sensitivity of the corrected result")
    sub.add_parser("sites", help="Phase 6b: per-site statistics and ranked site table")
    sub.add_parser("network", help="Phase 6d: road-network access per site (osmnx)")
    sub.add_parser("threshold", help="Phase 6e: compare suitability threshold rules")
    sub.add_parser("ahp", help="Phase 6c: AHP weights, consistency, questionnaire, effect")
    sub.add_parser("review", help="Site-check sheet (CSV with map links + KML outlines)")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "phase1":
        run_phase1(cfg)
    elif args.cmd == "phase2":
        run_phase2(cfg)
    elif args.cmd == "phase3":
        run_phase3(cfg)
    elif args.cmd == "phase3k":
        run_phase3k(cfg)
    elif args.cmd == "phase4":
        run_phase4(cfg)
    elif args.cmd == "phase5":
        run_phase5(cfg)
    elif args.cmd == "sensitivity":
        run_sensitivity(cfg)
    elif args.cmd == "sites":
        run_sites(cfg)
    elif args.cmd == "network":
        run_network(cfg)
    elif args.cmd in ("threshold", "ahp", "review"):
        from stp import analysis  # imports cli helpers, so import lazily

        {
            "threshold": analysis.run_threshold,
            "ahp": analysis.run_ahp,
            "review": analysis.run_review,
        }[args.cmd](cfg)


if __name__ == "__main__":
    main()
