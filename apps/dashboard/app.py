"""STP site dashboard: explore the final model's ranked sites.

Run from the repo root:  streamlit run apps/dashboard/app.py
Needs the outputs of `python -m stp.cli final`, `sites`, `network` and `crosscheck`.
"""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from stp import io, proximity
from stp.config import load_config

st.set_page_config(page_title="Bhopal STP siting", layout="wide")


@st.cache_data
def load() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    cfg = load_config()
    out = Path(cfg["paths"]["output"]) / "final"
    path = out / "sites_crosscheck.parquet"
    if not path.exists():
        path = out / "sites_ranked.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the final-model commands first")
    sites = gpd.read_parquet(path)
    sites["site_id"] = [f"S{r:02d}" for r in sites["rank"]]
    sites["area_ha"] = sites["area_sqm"] / 1e4
    boundary = io.read_boundary(cfg["paths"]["boundary"], cfg["crs"])
    geom = boundary.union_all()
    lakes = proximity.load_clipped(
        cfg["paths"]["vectors_raw"], cfg["proximity"]["layers"]["water"]["raw"], geom, cfg["crs"]
    )
    summary = pd.read_json(out / "summary.json", typ="series").to_dict()
    return sites.to_crs(4326), boundary.to_crs(4326), lakes.to_crs(4326), summary


sites, boundary, lakes, summary = load()
has_verdict = "product_verdict" in sites.columns

# --- Sidebar filters ---
st.sidebar.title("Filters")
robust_only = st.sidebar.checkbox("Robust sites only (selected in >= 75% of weight draws)")
min_ha = st.sidebar.slider("Minimum area (ha)", 0.0, float(sites["area_ha"].max()), 0.0, step=0.5)
verdicts = sorted(sites["product_verdict"].unique()) if has_verdict else []
chosen = st.sidebar.multiselect("Land-cover verdict", verdicts, default=verdicts)
max_access = st.sidebar.slider(
    "Max access distance to a major road (km)",
    0.0,
    float(sites["access_total_m"].max() / 1000) + 0.5 if "access_total_m" in sites else 10.0,
    float(sites["access_total_m"].max() / 1000) + 0.5 if "access_total_m" in sites else 10.0,
    step=0.5,
)

view = sites[sites["area_ha"] >= min_ha]
if robust_only:
    view = view[view["robust"]]
if has_verdict:
    view = view[view["product_verdict"].isin(chosen)]
if "access_total_m" in view:
    view = view[view["access_total_m"] / 1000 <= max_access]

# --- Header and metrics ---
st.title("Decentralised STP site selection: Bhopal")
st.caption(
    "Final model: AHP weights, 100 m residential cap, 0.56 ha minimum, and constraints from "
    "12 months of Dynamic World. Method and evidence: README and docs/parity.md."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites shown", f"{len(view)} / {len(sites)}")
c2.metric("Area shown", f"{view['area_ha'].sum():.1f} ha")
c3.metric("Robust", int(view["robust"].sum()))
if has_verdict:
    c4.metric(
        "Open land by consensus", int((view["product_verdict"] == "consistent_open_land").sum())
    )

# --- Map ---
centre = boundary.union_all().centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=11, tiles="OpenStreetMap")
folium.GeoJson(
    boundary,
    name="Planning boundary 2031",
    style_function=lambda _: {"color": "#555", "weight": 1.5, "fillOpacity": 0},
).add_to(m)
folium.GeoJson(
    lakes,
    name="Lakes",
    style_function=lambda _: {"color": "#2b7bba", "fillColor": "#2b7bba", "fillOpacity": 0.4},
).add_to(m)
if len(view):
    tip_fields = ["site_id", "area_ha", "suit_mean", "persistence"]
    tip_alias = ["Site", "Area (ha)", "Mean suitability", "Robustness"]
    if has_verdict:
        tip_fields.append("product_verdict")
        tip_alias.append("Land-cover verdict")
    folium.GeoJson(
        view[[*tip_fields, "robust", "geometry"]].round(3),
        name="Sites",
        style_function=lambda f: {
            "color": "#1a9850" if f["properties"]["robust"] else "#fdae61",
            "fillColor": "#1a9850" if f["properties"]["robust"] else "#fdae61",
            "weight": 2,
            "fillOpacity": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=tip_fields, aliases=tip_alias),
    ).add_to(m)
folium.LayerControl().add_to(m)
st_folium(m, height=560, use_container_width=True, returned_objects=[])
st.caption("Green: robust (>= 75% of weight draws); orange: less robust.")

# --- Table and chart ---
cols = [
    c
    for c in [
        "site_id",
        "area_ha",
        "suit_mean",
        "persistence",
        "robust",
        "slope_deg_mean",
        "elev_range_m",
        "access_total_m",
        "nearest_school_m",
        "nearest_hospital_m",
        "nearest_lake_m",
        "product_verdict",
    ]
    if c in view
]
st.subheader("Sites")
st.dataframe(view[cols].round(2), hide_index=True, use_container_width=True)
if len(view):
    fig = px.bar(
        view.sort_values("area_ha", ascending=False),
        x="site_id",
        y="area_ha",
        color="robust",
        labels={"area_ha": "Area (ha)", "site_id": "Site"},
    )
    st.plotly_chart(fig, use_container_width=True)
