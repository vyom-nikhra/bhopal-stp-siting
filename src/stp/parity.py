"""Compare pipeline rasters with the QGIS reference rasters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

# QGIS writes float NoData as the most negative float32.
_FLOAT_MIN_SENTINEL = -1e30


@dataclass
class ParityResult:
    name: str
    reference: str
    same_grid: bool
    n_valid_ref: int
    n_valid_ours: int
    mask_agreement: float  # share of pixels where both agree on valid/NoData
    max_abs_diff: float  # over pixels valid in both
    mean_abs_diff: float
    exact_share: float  # share of jointly valid pixels with |diff| <= tol
    tol: float
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _valid(a: np.ndarray, nodata) -> np.ndarray:
    ok = np.isfinite(a) if np.issubdtype(a.dtype, np.floating) else np.ones(a.shape, bool)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        ok &= a != nodata
    if np.issubdtype(a.dtype, np.floating):
        ok &= a > _FLOAT_MIN_SENTINEL
    return ok


def compare(
    name: str,
    ours: np.ndarray,
    ours_transform: Affine,
    ours_nodata,
    ref_path: str | Path,
    band: int = 1,
    tol: float = 1e-6,
    ours_valid: np.ndarray | None = None,
    resample_to_ref: bool = False,
    ref_nodata_override=None,
    note: str = "",
) -> ParityResult:
    """Compare `ours` with band `band` of `ref_path`.

    By default both rasters must share a grid, otherwise ValueError. With
    `resample_to_ref=True`, `ours` is resampled (nearest) onto the reference grid
    first, and the result is a statistical comparison, not a pixel-exact one.
    `ref_nodata_override` replaces the reference's NoData value, for QGIS outputs that
    never declared one (the clipped band stack uses 0 outside the boundary).
    """
    with rasterio.open(ref_path) as src:
        ref = src.read(band)
        ref_nodata = src.nodata if ref_nodata_override is None else ref_nodata_override
        ref_transform = src.transform
        crs = src.crs

    same_grid = ref.shape == ours.shape and ref_transform.almost_equals(ours_transform)
    if not same_grid:
        if not resample_to_ref:
            raise ValueError(
                f"{name}: grid mismatch vs {Path(ref_path).name}: ours {ours.shape} "
                f"{tuple(ours_transform)[:6]} vs ref {ref.shape} {tuple(ref_transform)[:6]}"
            )
        src_valid = ours_valid if ours_valid is not None else _valid(ours, ours_nodata)
        src = np.where(src_valid, ours, np.nan).astype("float64")
        dst = np.full(ref.shape, np.nan)
        reproject(
            src,
            dst,
            src_transform=ours_transform,
            src_crs=crs,
            dst_transform=ref_transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        ours, ours_nodata, ours_valid = dst, np.nan, None

    v_ref = _valid(ref, ref_nodata)
    v_ours = ours_valid if ours_valid is not None else _valid(ours, ours_nodata)
    both = v_ref & v_ours
    if not both.any():
        raise ValueError(f"{name}: no pixels valid in both rasters")
    diff = np.abs(ours[both].astype("float64") - ref[both].astype("float64"))
    return ParityResult(
        name=name,
        reference=Path(ref_path).name,
        same_grid=same_grid,
        n_valid_ref=int(v_ref.sum()),
        n_valid_ours=int(v_ours.sum()),
        mask_agreement=float(np.mean(v_ref == v_ours)),
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        exact_share=float(np.mean(diff <= tol)),
        tol=tol,
        note=note,
    )


def compare_vectors(name: str, ours: gpd.GeoDataFrame, ref_path: str | Path) -> dict:
    """Feature count and total length/area of `ours` vs a reference vector file.

    For polygons, also the area of the symmetric difference (where coverage differs).
    """
    ref = gpd.read_file(ref_path, engine="pyogrio", columns=[])
    if ref.crs != ours.crs:
        ref = ref.to_crs(ours.crs)
    polygonal = bool(ours.geom_type.isin(["Polygon", "MultiPolygon"]).all())
    measure = "area" if polygonal else "length"
    m_ours = float(getattr(ours.geometry, measure).sum())
    m_ref = float(getattr(ref.geometry, measure).sum())
    out = {
        "name": name,
        "reference": Path(ref_path).name,
        "n_ours": len(ours),
        "n_ref": len(ref),
        "measure": measure,
        "total_ours": m_ours,
        "total_ref": m_ref,
        "rel_diff": (m_ours - m_ref) / m_ref,
    }
    if polygonal:
        out["sym_diff_area"] = float(ours.union_all().symmetric_difference(ref.union_all()).area)
    return out


def format_table(results: list[ParityResult]) -> str:
    head = (
        f"{'layer':16s} {'grid':5s} {'mask_agree':>10s} {'exact':>8s} "
        f"{'mean|d|':>10s} {'max|d|':>10s}  note"
    )
    rows = [head, "-" * len(head)]
    for r in results:
        rows.append(
            f"{r.name:16s} {'same' if r.same_grid else 'resm':5s} {r.mask_agreement:10.4f} "
            f"{r.exact_share:8.4f} {r.mean_abs_diff:10.3e} {r.max_abs_diff:10.3e}  {r.note}"
        )
    return "\n".join(rows)
