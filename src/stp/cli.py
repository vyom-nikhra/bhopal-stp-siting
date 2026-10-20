"""Command-line entry point: `python -m stp.cli <phase1|phase2> [--config config.yaml]`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from stp import io, parity, preprocess, proximity
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


def main() -> None:
    p = argparse.ArgumentParser(prog="stp")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("phase1", help="band stack, DEM, slope, indices + parity")
    sub.add_parser("phase2", help="distance rasters + parity")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "phase1":
        run_phase1(cfg)
    elif args.cmd == "phase2":
        run_phase2(cfg)


if __name__ == "__main__":
    main()
