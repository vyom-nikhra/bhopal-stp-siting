"""Phase 4a: standardise each criterion to a 1-10 suitability score.

Recovered rule (docs/parity.md, Phase 4): the QGIS run used GRASS `r.rescale`, which
rounded the input to whole numbers, mapped the layer's own [min, max] linearly onto
[1, 10], and rounded the result. Roads, drainage and DEM were then inverted with
`11 - x` (closer to roads/drainage and lower ground score higher). The integer input
is why the slope scores step at 2.5, 7.5, 12.5 ... degrees.
"""

from __future__ import annotations

import numpy as np

SCORE_NODATA = -9999.0


def round_half_up(a: np.ndarray) -> np.ndarray:
    """Round .5 away from zero for non-negative values (C-style), unlike numpy's
    round-half-to-even."""
    return np.floor(a + 0.5)


def rescale_score(x: np.ndarray, valid: np.ndarray, invert: bool) -> np.ndarray:
    """Integer-input linear rescale of `x` onto 1..10 over its valid range.

    Returns float32 scores with SCORE_NODATA outside `valid`.
    """
    if not valid.any():
        raise ValueError("No valid pixels to rescale")
    xi = round_half_up(x[valid].astype("float64"))
    lo, hi = xi.min(), xi.max()
    if hi == lo:
        raise ValueError(f"Layer is constant ({lo}); cannot rescale")
    score = round_half_up(1 + 9 * (xi - lo) / (hi - lo))
    if invert:
        score = 11 - score
    out = np.full(x.shape, SCORE_NODATA, dtype="float32")
    out[valid] = score
    return out
