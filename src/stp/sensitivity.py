"""Phase 6a: how much do the final sites depend on the criterion weights?

The weights (residential 0.25, water 0.20, ...) are expert judgements, so the result
should be checked for robustness:
- Monte Carlo: in each draw every weight is multiplied by a random factor in
  [1 - delta, 1 + delta], the weights are renormalised to sum to 1, and the whole
  site selection is rerun. Pixels and sites that keep being selected are robust.
- One-at-a-time: each weight is raised or lowered by `delta` (the others rescaled
  proportionally so the sum stays 1), to see which criterion the result is most
  sensitive to.
"""

from __future__ import annotations

from collections.abc import Callable

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize

from stp import overlay


def perturb_weights(base: dict[str, float], delta: float, rng: np.random.Generator) -> dict:
    """Multiply each weight by U(1 - delta, 1 + delta), then renormalise to sum to 1."""
    raw = {k: w * rng.uniform(1 - delta, 1 + delta) for k, w in base.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def shift_one(base: dict[str, float], name: str, factor: float) -> dict:
    """Scale one weight by `factor`; rescale the others proportionally so the sum stays 1."""
    if name not in base:
        raise KeyError(name)
    new = base[name] * factor
    if not 0 < new < 1:
        raise ValueError(f"{name} weight {new} out of (0, 1)")
    rest = 1 - base[name]
    return {k: (new if k == name else w * (1 - new) / rest) for k, w in base.items()}


def site_mask(sites: gpd.GeoDataFrame, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """Pixels (by centre) inside any final site."""
    if len(sites) == 0:
        return np.zeros(shape, dtype=bool)
    burned = rasterize(
        ((g, 1) for g in sites.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)


def site_ids(sites: gpd.GeoDataFrame, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    """Raster of site ids 1..n (0 = no site), by pixel centre."""
    return rasterize(
        ((g, i + 1) for i, g in enumerate(sites.geometry)),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )


def run_weights(
    layers: dict[str, np.ndarray],
    weights: dict[str, float],
    select: Callable[[np.ndarray], gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Weighted overlay with `weights`, then `select(surface)` -> final sites."""
    return select(overlay.weighted_sum(layers, weights))


def coverage_by_site(ids: np.ndarray, selected: np.ndarray, n_sites: int) -> np.ndarray:
    """Share of each baseline site's pixels that are selected (index 0 = site 1)."""
    total = np.bincount(ids.ravel(), minlength=n_sites + 1)[1:]
    hit = np.bincount(ids[selected], minlength=n_sites + 1)[1:]
    return np.divide(hit, total, out=np.zeros(n_sites), where=total > 0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
