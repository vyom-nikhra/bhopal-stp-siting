"""Phase 6c: Analytic Hierarchy Process (AHP) weights from pairwise judgements.

Instead of asserting weights directly, AHP asks for each pair of criteria "which
matters more, and how much?" on Saaty's 1-9 scale (1 equal, 3 moderate, 5 strong,
7 very strong, 9 extreme; even numbers in between). The answers form a reciprocal
matrix A (A[i, j] = how much i outweighs j, A[j, i] = 1 / A[i, j]). The weights are
A's principal eigenvector, and the consistency ratio CR checks that the judgements
don't contradict each other (e.g. a > b, b > c, but c > a); CR < 0.10 is the usual
acceptance threshold.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

# Saaty's random consistency index: the average CI of random reciprocal matrices.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def pairwise_matrix(criteria: list[str], judgements: list[dict]) -> np.ndarray:
    """Build the reciprocal matrix from judgements {a, b, prefer, strength}.

    Every unordered pair of criteria must be judged exactly once; `prefer` is a or b;
    `strength` is 1..9 (1 = equal, in which case `prefer` is irrelevant).
    """
    idx = {c: i for i, c in enumerate(criteria)}
    a_mat = np.ones((len(criteria), len(criteria)))
    seen = set()
    for j in judgements:
        a, b, prefer, s = j["a"], j["b"], j["prefer"], j["strength"]
        if a not in idx or b not in idx or a == b:
            raise ValueError(f"Bad pair {a!r} vs {b!r}")
        pair = frozenset((a, b))
        if pair in seen:
            raise ValueError(f"Pair {a} vs {b} judged twice")
        seen.add(pair)
        if prefer not in (a, b):
            raise ValueError(f"{a} vs {b}: prefer must be one of them, got {prefer!r}")
        if not (isinstance(s, int) and 1 <= s <= 9):
            raise ValueError(f"{a} vs {b}: strength must be an integer 1..9, got {s!r}")
        hi, lo = (a, b) if prefer == a else (b, a)
        a_mat[idx[hi], idx[lo]] = s
        a_mat[idx[lo], idx[hi]] = 1 / s
    missing = [f"{x} vs {y}" for x, y in combinations(criteria, 2) if frozenset((x, y)) not in seen]
    if missing:
        raise ValueError(f"Missing judgements: {missing}")
    return a_mat


def priority_vector(a_mat: np.ndarray) -> tuple[np.ndarray, float]:
    """Principal-eigenvector weights (sum 1) and the principal eigenvalue lambda_max."""
    vals, vecs = np.linalg.eig(a_mat)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def geometric_mean_weights(a_mat: np.ndarray) -> np.ndarray:
    """Row geometric-mean weights: a simpler approximation, used as a cross-check."""
    g = np.exp(np.log(a_mat).mean(axis=1))
    return g / g.sum()


def consistency(a_mat: np.ndarray, lambda_max: float) -> tuple[float, float]:
    """Consistency index CI = (lambda_max - n) / (n - 1) and ratio CR = CI / RI(n)."""
    n = a_mat.shape[0]
    ci = (lambda_max - n) / (n - 1)
    ri = RANDOM_INDEX[n]
    return ci, (ci / ri if ri else 0.0)


def most_inconsistent(a_mat: np.ndarray, w: np.ndarray, criteria: list[str], top: int = 3):
    """Judgements furthest from what the final weights imply (|log(a_ij / (w_i / w_j))|)."""
    n = len(criteria)
    errs = [
        (abs(np.log(a_mat[i, j] * w[j] / w[i])), criteria[i], criteria[j], a_mat[i, j], w[i] / w[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return sorted(errs, reverse=True)[:top]
