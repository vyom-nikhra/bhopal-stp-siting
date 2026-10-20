"""How do the corrected sites (variant C) relate to the original 46 (variant A)?

Reports how much of the original site area is still selected, and how many of the
corrected sites are new (no overlap with any original site).
Run after `python -m stp.cli phase5`.
"""

from pathlib import Path

import geopandas as gpd

from stp.config import load_config

cfg = load_config()
base = Path(cfg["paths"]["output"]) / "phase5"
a = gpd.read_parquet(base / "A_qgis_order_original_lulc" / "sites.parquet")
c = gpd.read_parquet(base / "C_corrected_order_corrected_lulc" / "sites.parquet")

ua, uc = a.union_all(), c.union_all()
kept = ua.intersection(uc).area / ua.area
print(f"original sites: {len(a)} ({ua.area / 1e4:.1f} ha planar); corrected: {len(c)} "
      f"({uc.area / 1e4:.1f} ha planar)")
print(f"share of original site area still selected: {kept:.1%}")

a_hit = a.geometry.intersects(uc)
c_hit = c.geometry.intersects(ua)
print(f"original sites overlapping a corrected site: {int(a_hit.sum())}/{len(a)}")
print(f"corrected sites overlapping an original site: {int(c_hit.sum())}/{len(c)}; "
      f"new sites: {int((~c_hit).sum())} ({c.loc[~c_hit, 'area_sqm'].sum() / 1e4:.1f} ha)")
lost = a.loc[~a_hit]
if len(lost):
    print(f"original sites dropped entirely: {len(lost)} ({lost['area_sqm'].sum() / 1e4:.2f} ha)")
