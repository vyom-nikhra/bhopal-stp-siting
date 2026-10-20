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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

CLASS_NODATA = -9999
SCORE_NODATA = -9999.0

# Fixed class codes for the corrected (k=4) land-cover map.
CLASS_CODES = {"water": 1, "built_up": 2, "vegetation": 3, "barren": 4}


def fit_full_kmeans(
    x: np.ndarray, k: int, n_init: int, random_state: int, fit_size: int, sample_seed: int
) -> KMeans:
    """Lloyd K-Means with restarts, fitted on a fixed random sample of pixels.

    MiniBatchKMeans on all 13.8 M pixels stopped after one pass (n_iter_ == 1) and gave
    a different partition for every seed; full K-Means on a 1 M-pixel sample converges
    and is stable across seeds (see docs/parity.md, Phase 3).
    """
    rng = np.random.default_rng(sample_seed)
    idx = rng.choice(len(x), size=min(fit_size, len(x)), replace=False)
    return KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(
        x[idx].astype("float64")
    )


def predict_chunked(model, x: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Assign every pixel to its nearest centroid, in chunks to bound memory."""
    return np.concatenate(
        [model.predict(x[i : i + chunk].astype("float64")) for i in range(0, len(x), chunk)]
    )


def name_clusters(
    profiles: list[dict], residential_ratio: dict[int, float], water_ratio: dict[int, float]
) -> dict[str, int]:
    """Name the 4 clusters (1-based ids) as water, vegetation, built_up and barren.

    - water: highest MNDWI, and it must be the cluster most over-represented in OSM water
    - vegetation: highest NDVI among the rest
    - built_up / barren: of the remaining two, the one more over-represented in OSM
      residential areas is built-up
    `*_ratio` = (cluster's share of pixels inside the OSM layer) / (its share overall).
    Raises if the evidence is inconsistent instead of guessing.
    """
    if len(profiles) != 4:
        raise ValueError(f"Expected 4 clusters, got {len(profiles)}")
    by_id = {p["cluster"]: p for p in profiles}
    water = max(by_id, key=lambda c: by_id[c]["mndwi"])
    if water != max(water_ratio, key=water_ratio.get):
        raise ValueError(
            f"Highest-MNDWI cluster {water} is not the one most over-represented in OSM "
            f"water ({water_ratio}); cannot name the water class"
        )
    rest = [c for c in by_id if c != water]
    vegetation = max(rest, key=lambda c: by_id[c]["ndvi"])
    a, b = [c for c in rest if c != vegetation]
    built_up, barren = (a, b) if residential_ratio[a] > residential_ratio[b] else (b, a)
    if not residential_ratio[built_up] > 1.0 > residential_ratio[barren]:
        raise ValueError(
            f"Built-up/barren split is ambiguous: residential ratios {residential_ratio}"
        )
    return {"water": water, "vegetation": vegetation, "built_up": built_up, "barren": barren}


def overlap_ratio(labels: np.ndarray, inside: np.ndarray, k: int) -> dict[int, float]:
    """Per cluster (1-based): share of pixels inside a mask divided by share overall."""
    overall = np.bincount(labels, minlength=k) / len(labels)
    within = np.bincount(labels[inside], minlength=k) / max(int(inside.sum()), 1)
    return {c + 1: float(within[c] / overall[c]) for c in range(k)}


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
