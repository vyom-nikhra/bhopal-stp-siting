# Auto-STP web app

Draw an area anywhere. The app returns land that could host a decentralised STP
(sewage treatment plant). It is a port of the original GEOTA `Auto_STP_Platform`
(FastAPI + Streamlit), rebuilt around the Bhopal study's final model.

| Step | Source | Rule |
|---|---|---|
| Open land | Dynamic World V1, 12 months to today | mode class is grass, shrub/scrub or bare |
| Not settled | Dynamic World V1 | built in < 50% of observations |
| Terrain | SRTM 30 m | slope < 15° |
| Exclusions | OpenStreetMap (Overpass) | 500 m from schools, hospitals, clinics, water |
| Size | `config.yaml` → `model` | ≥ 12.7 MLD × 0.044 ha/MLD = 0.56 ha (ellipsoidal area) |

## Run

```bash
earthengine authenticate          # once
STP_EE_PROJECT=<your-ee-project> uvicorn apps.auto_stp.backend.main:app
AUTO_STP_BACKEND=http://127.0.0.1:8000 streamlit run apps/auto_stp/frontend/app.py
```

Optional: set `AUTO_STP_OVERPASS_URL` to use another Overpass API endpoint, e.g. a mirror or
your own instance. The default, `https://overpass-api.de/api`, is often at capacity and asks
automated users to run their own. If the endpoint fails, the request still returns 502;
there is no automatic switch to another server.

Run both from the repo root, in the `stp` environment. On Windows PowerShell, set the
variables first with `$env:STP_EE_PROJECT = "..."`.

## What changed from the original, and why

| Original | Now | Why |
|---|---|---|
| NDVI < 0.2 and NDBI > 0 on a summer composite | Dynamic World 12-month mode + built share | Same fix as the main model: the spectral test picked fallow farmland (see `docs/parity.md`) |
| OSM errors caught, returned "no exclusions" | Errors raise → HTTP 502 | An Overpass outage silently approved sites next to schools |
| osmnx 1.x `bbox=(north, south, east, west)` | osmnx 2.x `(left, bottom, right, top)` | The old order is read as a different box |
| `ee.Authenticate()` inside requests | `ee.Initialize(project)` once at startup | Interactive auth can't run on a server |
| `bestEffort=True`, `limit(5000)` | Raise if > 5,000 polygons | Silent coarsening and truncation |
| `async def` endpoint | `def` | Blocking EE/Overpass calls froze the event loop |
| Planar UTM area, hand-written zone | Ellipsoidal area, `estimate_utm_crs()` | Matches QGIS `$area` and the pipeline |
| Hard-coded backend URL, CartoDB tiles, empty tooltip | `AUTO_STP_BACKEND`, OSM tiles, area tooltip | Config by environment; CartoDB now needs a key |

## Verified

- `tests/test_auto_stp.py`: 12 tests. Earth Engine and OSM are mocked. They cover exclusion
  and minimum-area filtering, per-site area in the GeoJSON, a 502 when OSM fails, a 422 when
  there are too many polygons, the bbox order, startup without a project, the Overpass URL
  setting, and a frontend
  render.
- Live Earth Engine run (2026-10-21) on an ~800 m box around one of the final model's
  sites: 46% open land, 24 candidate polygons.
- Live end-to-end request on the same day: the Earth Engine step succeeded. Overpass was
  unreachable from this network, so the API returned 502 "OSM error". The original app
  would have returned sites with no exclusions applied. A full live success still needs
  Overpass access.
