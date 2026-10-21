"""Render the README figures to docs/figures/ from the pipeline outputs.

Run after `python -m stp.cli final` (and phase5). Rasters are downsampled 4x for display.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stp import io  # noqa: E402
from stp.config import REPO_ROOT, load_config  # noqa: E402

cfg = load_config()
out = Path(cfg["paths"]["output"])
fig_dir = REPO_ROOT / "docs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
step = 4


def read(path):
    with rasterio.open(path) as src:
        a = src.read(1, out_shape=(src.height // step, src.width // step))
        b = src.bounds
    return a, (b.left, b.right, b.bottom, b.top)


boundary = io.read_boundary(cfg["paths"]["boundary"], cfg["crs"])
final = gpd.read_parquet(out / "final" / "sites_ranked.parquet")
qgis = gpd.read_parquet(out / "phase5" / "A_qgis_order_original_lulc" / "sites.parquet")

# 1. Overview: final suitability, final sites, original sites.
suit, extent = read(out / "final" / "suitability.tif")
fig, ax = plt.subplots(figsize=(9, 8.5))
im = ax.imshow(
    np.where(np.isfinite(suit), suit, np.nan), extent=extent, cmap="YlGn", vmin=3, vmax=10
)
boundary.boundary.plot(ax=ax, color="0.3", linewidth=0.8)
qgis.boundary.plot(ax=ax, color="#1f77b4", linewidth=1.6)
final.plot(ax=ax, color="#d62728", edgecolor="#d62728", linewidth=1.2)
ax.legend(
    handles=[
        Patch(
            color="#d62728",
            label=f"final sites ({len(final)}, {final['area_sqm'].sum() / 1e4:.0f} ha)",
        ),
        Patch(facecolor="none", edgecolor="#1f77b4", label=f"original QGIS sites ({len(qgis)})"),
    ],
    loc="upper right",
    fontsize=9,
)
fig.colorbar(im, ax=ax, shrink=0.7, label="suitability (final model, 1-10)")
ax.set_title("Bhopal STP siting: final model vs original QGIS result", fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(fig_dir / "overview.png", dpi=110)
plt.close(fig)

# 2. Constraints: eligible land, settlement zone, flood proxy.
elig, extent = read(out / "final" / "constraint_eligible.tif")
settle, _ = read(out / "final" / "constraint_settlement_zone.tif")
flood, _ = read(out / "final" / "constraint_flood.tif")
valid = np.isfinite(suit)
cat = np.full(elig.shape, np.nan)
cat[valid] = 0  # other ineligible land (cropland, trees)
cat[valid & (settle == 1)] = 1
cat[valid & (flood == 1)] = 2
cat[valid & (elig == 1)] = 3
cmap = ListedColormap(["#e8e2c8", "#9e9e9e", "#4a90d9", "#2ca02c"])
fig, ax = plt.subplots(figsize=(9, 8.5))
ax.imshow(cat, extent=extent, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
boundary.boundary.plot(ax=ax, color="0.3", linewidth=0.8)
final.boundary.plot(ax=ax, color="#d62728", linewidth=1.2)
# Exact full-resolution shares (the display raster is downsampled).
summary = json.loads((out / "final" / "summary.json").read_text(encoding="utf-8"))
cs = summary["constraint_shares"]
shares = {
    "eligible open land": cs["eligible"],
    "settlement + 100 m": cs["settlement_zone"],
    "water >= 10% of the year": cs["flood"],
}
ax.legend(
    handles=[
        Patch(color="#2ca02c", label=f"eligible open land ({shares['eligible open land']:.1%})"),
        Patch(color="#9e9e9e", label=f"settlement + 100 m ({shares['settlement + 100 m']:.1%})"),
        Patch(
            color="#4a90d9",
            label=f"water >= 10% of the year ({shares['water >= 10% of the year']:.1%})",
        ),
        Patch(color="#e8e2c8", label="cropland / trees (yearly mode)"),
        Patch(facecolor="none", edgecolor="#d62728", label="final sites"),
    ],
    loc="upper right",
    fontsize=9,
)
ax.set_title("Hard constraints from 12 months of Dynamic World (Jun 2024 - Jun 2025)", fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(fig_dir / "constraints.png", dpi=110)
plt.close(fig)
print("figures ->", fig_dir)
