"""Auto-STP Streamlit frontend: draw an area, send it to the backend, show candidate sites.

Run from the repo root (backend must be running):
    AUTO_STP_BACKEND=http://127.0.0.1:8000 streamlit run apps/auto_stp/frontend/app.py

Changes from the original app: the backend URL comes from AUTO_STP_BACKEND, the basemap is
OpenStreetMap (CartoDB tiles now need an API key), polygons as well as rectangles can be
drawn, and the map tooltip shows each site's area (the original's empty tooltip errored).
"""

from __future__ import annotations

import json
import os

import folium
import requests
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

BACKEND_URL = os.environ.get("AUTO_STP_BACKEND", "http://127.0.0.1:8000")
SHAPE = {"color": "#0b7a75", "weight": 2, "fillOpacity": 0.1}

st.set_page_config(page_title="Auto-STP", layout="wide")
st.title("Auto-STP: candidate sewage treatment plant sites")
st.caption(
    "Draw an area. Candidate land is open in the 12-month Dynamic World mode (grass, scrub, "
    "bare), built in < 50% of observations, slope < 15 deg, at least the 0.56 ha plant "
    "footprint, and outside 500 m of OSM schools, hospitals, clinics and water."
)

if "results" not in st.session_state:
    st.session_state.results = None

col_map, col_panel = st.columns([3, 1], gap="medium")

with col_map:
    m = folium.Map(
        location=[23.25, 77.41], zoom_start=11, tiles="OpenStreetMap", control_scale=True
    )
    Draw(
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": {"shapeOptions": SHAPE},
            "polygon": {"shapeOptions": SHAPE},
        },
        edit_options={"edit": False},
    ).add_to(m)
    res = st.session_state.results
    if res and res["geojson"]["features"]:
        folium.GeoJson(
            res["geojson"],
            name="Candidate sites",
            style_function=lambda _: {
                "fillColor": "#d1495b",
                "color": "#d1495b",
                "weight": 2,
                "fillOpacity": 0.4,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["area_sqm"], aliases=["Area (m2)"], localize=True
            ),
        ).add_to(m)
    map_data = st_folium(m, height=620, use_container_width=True, key="main_map")

with col_panel:
    drawing = (map_data or {}).get("last_active_drawing")
    geometry = drawing["geometry"] if drawing else None
    if geometry and geometry["type"] == "Polygon":
        lngs = [c[0] for c in geometry["coordinates"][0]]
        lats = [c[1] for c in geometry["coordinates"][0]]
        st.success(
            f"Area drawn: {min(lngs):.4f} to {max(lngs):.4f} E, "
            f"{min(lats):.4f} to {max(lats):.4f} N"
        )
    else:
        geometry = None
        st.info("Draw a rectangle or polygon on the map.")

    if st.button("Analyse", use_container_width=True, disabled=geometry is None):
        with st.spinner("Querying Earth Engine and OpenStreetMap..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/analyze_region", json=geometry, timeout=1800)
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at {BACKEND_URL}.")
                st.stop()
            if not resp.ok:
                st.error(f"Backend error {resp.status_code}: {resp.text}")
                st.stop()
            st.session_state.results = resp.json()
            st.rerun()

    res = st.session_state.results
    if res:
        if res["status"] == "success":
            st.success("Candidate sites found.")
        else:
            st.warning(f"No sites: {res['status'].replace('_', ' ')}.")
        c1, c2 = st.columns(2)
        c1.metric("Sites", res["total_sites"])
        c2.metric("Area (ha)", f"{res['total_usable_area_sqm'] / 1e4:.2f}")
        if res["sites"]:
            st.dataframe(res["sites"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download GeoJSON",
                data=json.dumps(res["geojson"]),
                file_name="auto_stp_sites.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
