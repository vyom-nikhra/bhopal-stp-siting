"""Phase 3: unsupervised land cover with K-Means.

The original `kmeans_lulc.py` (MiniBatchKMeans, k=5, random_state=42) meant to skip
NoData pixels with `band1 != -9999`, but the stack is uint16 with 0 outside the
boundary, so nothing was skipped and one cluster became the out-of-boundary void.
`mask_void=False` reproduces that; `mask_void=True` is the fix.

K-Means on raw DN is unaffected by the Sentinel-2 BOA offset: subtracting the same
constant from every band translates all points and centroids equally.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

CLASS_NODATA = -9999
SCORE_NODATA = -9999.0


def pixel_table(stack: np.ndarray, mask_void: bool) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a (bands, rows, cols) stack to (n_pixels, bands) plus the valid mask.

    With mask_void=True, pixels that are 0 in every band (outside the boundary) are
    dropped. With False, every pixel is kept, as in the original script.
    """
    bands = stack.shape[0]
    flat = stack.reshape(bands, -1).T
    valid = np.any(flat != 0, axis=1) if mask_void else np.ones(len(flat), dtype=bool)
    return flat[valid], valid


def fit_kmeans(
    x: np.ndarray, k: int, random_state: int = 42, batch_size: int = 2048
) -> MiniBatchKMeans:
    """MiniBatchKMeans with the original script's settings."""
    model = MiniBatchKMeans(
        n_clusters=k, random_state=random_state, batch_size=batch_size, n_init="auto"
    )
    return model.fit(x)


def class_raster(labels: np.ndarray, valid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Labels (0..k-1) back onto the grid as classes 1..k, NoData elsewhere (int16)."""
    out = np.full(valid.shape, CLASS_NODATA, dtype="int16")
    out[valid] = labels + 1
    return out.reshape(shape)


def match_classes(ours: np.ndarray, ref: np.ndarray, nodata: int = CLASS_NODATA) -> dict:
    """Pair our cluster ids with reference ids by maximum pixel overlap (Hungarian method).

    Cluster numbers from K-Means are arbitrary, so agreement is only meaningful after
    relabelling. Returns the mapping {our_class: ref_class}, the agreement after
    relabelling, and the confusion matrix (rows: ours, cols: ref).
    """
    both = (ours != nodata) & (ref != nodata)
    a, b = ours[both], ref[both]
    ca, cb = np.unique(a), np.unique(b)
    conf = np.zeros((len(ca), len(cb)), dtype=np.int64)
    ia = np.searchsorted(ca, a)
    ib = np.searchsorted(cb, b)
    np.add.at(conf, (ia, ib), 1)
    rows, cols = linear_sum_assignment(conf, maximize=True)
    mapping = {int(ca[r]): int(cb[c]) for r, c in zip(rows, cols, strict=True)}
    return {
        "mapping": mapping,
        "agreement": float(conf[rows, cols].sum() / both.sum()),
        "confusion": conf.tolist(),
        "ours_classes": ca.tolist(),
        "ref_classes": cb.tolist(),
    }


def relabel(classes: np.ndarray, mapping: dict[int, int], nodata: int = CLASS_NODATA) -> np.ndarray:
    out = np.full_like(classes, nodata)
    for src, dst in mapping.items():
        out[classes == src] = dst
    return out


def k_selection(
    x: np.ndarray,
    ks: list[int],
    seeds: list[int],
    fit_size: int = 1_000_000,
    eval_size: int = 20_000,
    sample_seed: int = 0,
    batch_size: int = 2048,
) -> list[dict]:
    """Evidence for choosing k: elbow, silhouette, Davies-Bouldin and seed stability.

    Models are fitted on a fixed random sample of `fit_size` pixels (for speed) and all
    metrics are computed on a separate fixed sample of `eval_size` pixels:
    - inertia: mean squared distance to the nearest centroid (the "elbow" curve)
    - silhouette (higher is better, -1..1): how much closer pixels are to their own
      cluster than to the next one
    - Davies-Bouldin (lower is better): average similarity of each cluster to its
      most similar neighbour
    - stability: mean adjusted Rand index between the labelings from different seeds
      (1 = the same partition every time, ~0 = unrelated)
    """
    rng = np.random.default_rng(sample_seed)
    fit_idx = rng.choice(len(x), size=min(fit_size, len(x)), replace=False)
    eval_idx = rng.choice(len(x), size=min(eval_size, len(x)), replace=False)
    x_fit, x_eval = x[fit_idx].astype("float64"), x[eval_idx].astype("float64")

    rows = []
    for k in ks:
        labelings, inertias, sils, dbs = [], [], [], []
        for seed in seeds:
            model = fit_kmeans(x_fit, k, random_state=seed, batch_size=batch_size)
            lab = model.predict(x_eval)
            labelings.append(lab)
            inertias.append(float(-model.score(x_eval) / len(x_eval)))
            sils.append(float(silhouette_score(x_eval, lab)))
            dbs.append(float(davies_bouldin_score(x_eval, lab)))
        aris = [
            adjusted_rand_score(labelings[i], labelings[j])
            for i in range(len(seeds))
            for j in range(i + 1, len(seeds))
        ]
        rows.append(
            {
                "k": k,
                "inertia": float(np.mean(inertias)),
                "silhouette": float(np.mean(sils)),
                "silhouette_sd": float(np.std(sils)),
                "davies_bouldin": float(np.mean(dbs)),
                "stability_ari": float(np.mean(aris)),
                "stability_ari_min": float(np.min(aris)),
            }
        )
    return rows


def cluster_profiles(
    model: MiniBatchKMeans, bands: list[str], labels: np.ndarray, offset: float = -1000.0
) -> list[dict]:
    """Per-cluster pixel share, centroid reflectance and indices, sorted by NDVI.

    Reflectance = (DN + offset) / 10000 (Sentinel-2 L2A, baseline >= 04.00).
    """
    refl = (model.cluster_centers_ + offset) / 10000.0
    b = {name: refl[:, i] for i, name in enumerate(bands)}

    def nd(p, q):
        return (p - q) / (p + q)

    ndvi, ndbi, mndwi = nd(b["B08"], b["B04"]), nd(b["B11"], b["B08"]), nd(b["B03"], b["B11"])
    counts = np.bincount(labels, minlength=len(refl))
    rows = [
        {
            "cluster": c + 1,
            "share": float(counts[c] / counts.sum()),
            "reflectance": {name: round(float(b[name][c]), 4) for name in bands},
            "ndvi": float(ndvi[c]),
            "ndbi": float(ndbi[c]),
            "mndwi": float(mndwi[c]),
        }
        for c in range(len(refl))
    ]
    return sorted(rows, key=lambda r: r["ndvi"])


def reclassify(classes: np.ndarray, table: dict[int, float | None]) -> np.ndarray:
    """Class -> suitability score; a None score (or an unlisted class) becomes NoData."""
    out = np.full(classes.shape, SCORE_NODATA, dtype="float32")
    for cls, score in table.items():
        if score is not None:
            out[classes == int(cls)] = score
    return out
