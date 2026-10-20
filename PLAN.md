# Plan

**Goal:** rebuild the QGIS STP site-suitability analysis for Bhopal as a reproducible Python pipeline, check it against the original QGIS outputs, and then extend it beyond what was done in QGIS. Audience: geospatial developer roles. QGIS stays the prototype, the parity reference and the tool for final map layouts.

**Ground rules**
- Original data folders are read-only references. The pipeline only writes to `outputs/`.
- Every phase ends with a **parity check** against the QGIS reference: measured differences, each one explained, not just "matches".
- Parameters live in `config.yaml`, not in code. Recovered parameters are documented in `docs/qgis_workflow.md`.
- Errors are raised, never swallowed.
- Script output stays ASCII (Windows cp1252 console).

## Phases

### Phase 0: Set-up and re-learning
- Repo, `environment.yml` (conda-forge), `pyproject.toml`, `config.yaml`, `docs/qgis_workflow.md`, `PROGRESS.md`.
- **Done when:** the environment builds and the core imports work.

### Phase 1: Preprocessing (rasterio, rasterio.warp, rasterio.mask, numpy)
- Stack B02/B03/B04/B08 plus B11 (nearest 20 m → 10 m), clip to the boundary with an explicit NoData value.
- Warp the DEM (bilinear) and clip it. Compute slope (Horn, degrees).
- Compute NDVI, NDBI and MNDWI with NoData handled properly.
- **Parity:** Raster_01–04, 09, 10 (max and mean absolute difference on the same grid).

### Phase 2: Distance rasters (geopandas + pyogrio, rasterio.features.rasterize, scipy.ndimage)
- Read vectors with pyogrio (geometry column only for the 630 MB roads file), rasterise on the `proximity_10m` grid, compute the Euclidean distance transform, clip.
- **Parity:** Raster_05–08.

### Phase 3: Land cover (scikit-learn)
- 3a: reproduce the original behaviour (void included, k=5) and match labels to the reference.
- 3b: fixed version: proper NoData mask, choose k deliberately (elbow/silhouette), document the change.

### Phase 4: Standardisation and weighted overlay (rioxarray, xarray)
- Explicitly align the 12.5 m and shifted 10 m grids onto the Sentinel 10 m grid (`reproject_match`).
- Linear 1–10 rescale, inverting roads/drainage/DEM. Reconstruct the slope rule from `Reclass_10_Slope`.
- Weighted sum. **Parity:** `Final_STP_Suitability_Map` (range 3.06–8.48).

### Phase 5: Site selection (rasterio.features.shapes, geopandas)
- Queries 1–3 → 4-connected polygonize → 500 m school/hospital buffers → difference → area ≥ 5000 m².
- **Parity:** 46 sites, 1,254,202 m². Report the effect of the corrected step order.

### Phase 6: Beyond QGIS
- AHP: pairwise comparison matrix → eigenvector weights → consistency ratio; compare with the asserted weights.
- Weight sensitivity analysis: Monte Carlo perturbation, per-site rank stability.
- Per-site attributes: `rasterstats.zonal_stats` (mean score, slope, elevation), `sjoin_nearest` (nearest school/hospital/lake) → ranked site table.
- Road-network access with `osmnx` (distance along the network, not straight-line).
- Outputs as Cloud-Optimized GeoTIFF and GeoParquet.
- Ground-truth check: ~200 stratified random points labelled on high-resolution imagery → confusion matrix for the land-cover map.

### Phase 7: Apps
- Dashboard reads the pipeline's GeoParquet (46 sites, not 47) plus the ranked table.
- Auto-STP fixes: OSM failures raise, `estimate_utm_crs()`, Earth Engine auth outside the request path, no blocking calls in `async def`.

### Phase 8: Polish
- README: QGIS-vs-Python parity table, figures, an honest "discrepancies found" section.
- `pytest` (synthetic-raster unit tests run anywhere; parity tests marked `needs_data`), `ruff`.
- Final map layouts in QGIS.

## Known discrepancies to document
- Indices computed on raw DN, without the Sentinel-2 BOA offset (−1000).
- B11 in the QGIS stack has a row-assignment artefact from that run (28% of pixels); see docs/parity.md.
- K-Means NoData mask never matched; one cluster is the out-of-boundary void.
- Report says 4 classes; code used k=5.
- Report orders Difference before the area filter; the real order was the reverse (no effect on the result here).
- Query 2 (barren filter) never removed a pixel.
- Dashboard showed 47 pre-buffer sites, including one inside an exclusion buffer.
- Mixed 10 m / shifted 10 m / 12.5 m grids resampled silently.
- Scoring rule not recorded; recovered from the data as GRASS's integer-input rescale (inputs rounded to whole numbers first, hence 5° slope steps).
