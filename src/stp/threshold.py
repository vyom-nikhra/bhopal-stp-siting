"""Phase 6e: alternatives to the fixed "suitability >= 8" cut.

The surface tops out at 8.48, so a fixed cut at 8 sits at the very top of the score
range, and small weight changes move thousands of pixels across it (docs/parity.md,
Phase 6b). Alternatives compared here:
- fixed:    suitability >= a fixed value (the original rule)
- top_pct:  the best X% of valid pixels, whatever their absolute score; the threshold
            moves with the weights, so the selected area stays roughly constant
- robust:   pixels selected in at least a given share of the weight-perturbation draws
            (`selection_frequency.tif` from the sensitivity run)
"""

from __future__ import annotations

import numpy as np


def top_pct_threshold(surface: np.ndarray, pct: float) -> float:
    """Score above which the best `pct` percent of valid pixels lie."""
    if not 0 < pct < 100:
        raise ValueError(f"pct must be in (0, 100), got {pct}")
    valid = surface[np.isfinite(surface)]
    if valid.size == 0:
        raise ValueError("Surface has no valid pixels")
    return float(np.quantile(valid, 1 - pct / 100))


def rule_threshold(surface: np.ndarray, rule: dict) -> float:
    """Suitability threshold for a rule.

    {"kind": "fixed", "value": 8} or {"kind": "top_pct", "pct": 0.34}.
    """
    if rule["kind"] == "fixed":
        return float(rule["value"])
    if rule["kind"] == "top_pct":
        return top_pct_threshold(surface, rule["pct"])
    raise ValueError(f"Unknown rule kind {rule['kind']!r}")
