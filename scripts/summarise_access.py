"""Compare straight-line road distance with drivable-network access per site.

Run after `python -m stp.cli network`.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np

from stp.config import load_config

cfg = load_config()
s = gpd.read_parquet(Path(cfg["paths"]["output"]) / "phase6" / "sites_ranked.parquet")
gap = s["last_mile_m"] - s["road_dist_m_min"]
print(f"sites: {len(s)}; reach a major road: {int(s['reaches_major'].sum())}")
print(
    f"nearest any-road distance (QuickOSM, all highway types): median "
    f"{s['road_dist_m_min'].median():.0f} m, max {s['road_dist_m_min'].max():.0f} m"
)
print(
    f"last mile to a drivable junction: median {s['last_mile_m'].median():.0f} m, "
    f"max {s['last_mile_m'].max():.0f} m"
)
for thr in (250, 500, 1000):
    print(
        f"  last mile > {thr} m: {int((s['last_mile_m'] > thr).sum())} sites "
        f"(any-road distance > {thr} m: {int((s['road_dist_m_min'] > thr).sum())})"
    )
print(
    f"sites where the drivable last mile exceeds the any-road distance by > 250 m: "
    f"{int((gap > 250).sum())}"
)
print(
    f"network distance to a major road: median {s['network_to_major_m'].median():.0f} m, "
    f"max {s['network_to_major_m'].max():.0f} m"
)
acc = s["access_total_m"]
print(
    f"total access (last mile + network): median {acc.median():.0f} m, "
    f"90th pct {np.percentile(acc, 90):.0f} m, max {acc.max():.0f} m"
)
r = s[s["robust"]]
print(f"robust sites ({len(r)}): total access median {r['access_total_m'].median():.0f} m")
