"""How much do the seven criterion scores overlap? (Input to the AHP judgements.)

Spearman rank correlation between the aligned 1-10 score layers (corrected land
cover), on a fixed random sample of valid pixels, plus each score's distribution.
Run after `python -m stp.cli phase4`.
"""

import numpy as np
from rasterio.crs import CRS
from scipy.stats import spearmanr

from stp.cli import _corrected_inputs
from stp.config import Grid, load_config

cfg = load_config()
crs = CRS.from_user_input(cfg["crs"])
grid = Grid.from_config(cfg, "sentinel_10m")
layers, _ = _corrected_inputs(cfg, crs, grid)
names = list(cfg["weights"])
stack = np.stack([layers[n].ravel() for n in names], axis=1)
valid = np.all(np.isfinite(stack), axis=1)
rng = np.random.default_rng(0)
idx = rng.choice(np.flatnonzero(valid), size=200_000, replace=False)
rho = spearmanr(stack[idx]).statistic

print("Spearman correlation between criterion scores (200k valid pixels):")
print(" " * 12 + " ".join(f"{n[:7]:>8s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:12s}" + " ".join(f"{rho[i, j]:8.2f}" for j in range(len(names))))
print("\nScore distribution (share of valid pixels per score 1..10):")
for i, n in enumerate(names):
    vals = stack[valid, i]
    shares = [np.mean(vals == s) for s in range(1, 11)]
    print(
        f"{n:12s} mean {vals.mean():5.2f} sd {vals.std():4.2f}  "
        + " ".join(f"{s:4.2f}" for s in shares)
    )
