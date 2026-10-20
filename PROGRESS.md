# Progress log

Newest entries at the top. Each entry records what was done, what was verified, and what comes next, so a fresh session can resume from here.

## 2026-10-20: Phase 6 part 4 (network) blocked
- `stp.network` + `python -m stp.cli network` written and unit-tested (60 tests pass), committed.
- The real run failed: Overpass (overpass-api.de, overpass.private.coffee) connection timeout from this machine/sandbox, while other HTTPS (example.com) works. Nothing was written; `outputs/phase6/sites_ranked.*` is unchanged (no network columns yet).
- Retried outside the sandbox with the user's OK: still times out. DNS resolves and nominatim.openstreetmap.org works, but every public Overpass mirror (overpass-api.de, kumi.systems, mail.ru, private.coffee, osm.jp) fails to connect, so this network blocks Overpass. Not a sandbox issue.
- The download is needed only once: `python -m stp.cli network` saves `outputs/phase6/drive_network.graphml` and later runs are offline.
- **Waiting on the user:** run `network` once on another network (e.g. a phone hotspot), or build the graph from the local QuickOSM roads (no junction topology; bridges get joined), or skip part 4.

## 2026-10-20: Phase 6 parts 2-3 done (sensitivity, ranked table)
- Found and fixed: ~4,740 pixels score exactly 8.0; float rounding decided whether they pass (another arithmetic path drops 4,005). Threshold now `>= 8 - 1e-6` (`selection.threshold_tolerance`), matching QGIS; Phase 5 parity unchanged.
- `python -m stp.cli sensitivity` (~5 min): 300 draws -> sites 25-101 (5-95%), median IoU 0.555; 14/59 sites persistent in >= 90% of draws, 20 in >= 75%. Residential weight dominates (+-20% -> 23..105 sites). The fixed >= 8 threshold sits at the top of the 3-8.48 range, hence the fragility.
- `python -m stp.cli sites`: ranked table (zonal stats, nearest school/hospital/lake/residential, persistence, robust flag) in `outputs/phase6/sites_ranked.*`; all sites >= 499 m from schools/hospitals (checked).
- 55 unit tests pass; ruff clean.
- **Next:** Phase 6 part 4 (osmnx network access). Then ask the user about part 1 (AHP pairwise judgements) and part 5 (ground-truth labelling). Raise with the user: the threshold fragility (an option is a relative cut, e.g. top x% of the surface, or ranking by persistence).

## 2026-10-20: Phase 5 done
- `python -m stp.cli phase5`: queries, 4-connected polygonize (`rasterio.features.shapes`), ellipsoidal area (`pyproj.Geod`), 500 m school/hospital buffers, difference + area filter. Three variants in `outputs/phase5/<variant>/` (query COGs, `all_polygons.parquet`, `sites.parquet`, `sites.geojson`).
- **Exact parity:** query rasters identical; 47/47 candidates and 46/46 final sites with 0 m2 symmetric difference.
- Findings: Query 2 never removed a pixel; `Area_sqm` is ellipsoidal area; the QGIS step order did no harm here.
- Corrected result (variant C): 59 sites / 191.8 ha (vs 46 / 125.3 ha). 45/46 originals still overlap; 23 new sites (22.6 ha); 1 original (0.56 ha) dropped.
- 44 unit tests pass; ruff clean.
- **Next (Phase 6):** AHP pairwise matrix + consistency ratio, weight sensitivity (Monte Carlo), zonal stats + nearest-feature attributes -> ranked site table, osmnx network access, ground-truth check plan. Decide with the user which parts.

## 2026-10-20: Phase 4 done
- `python -m stp.cli phase4`: 1-10 scores (`stp.standardize`), explicit nearest-neighbour alignment onto the Sentinel 10 m grid (`rioxarray.reproject_match`, `stp.overlay`), weighted overlay. Writes `outputs/phase4/score_*.tif` and three suitability surfaces.
- **Exact parity:** all six scores identical to `Reclass_*`; the pipeline overlay is identical to `Final_STP_Suitability_Map` (0 difference, 8,105,749 px).
- Scoring rule recovered: round input to integer -> linear 1-10 over the integer range -> round half up -> `11 - x` for roads/drainage/DEM. Explains the slope steps; no unknown rule left.
- Corrected land cover raises the share of pixels >= 8 from 0.2208% to 0.3395%.
- Correction: the true suitability range is 3.06-8.48 (3.42 came from a 4x-subsampled read); fixed in config and docs.
- 38 unit tests pass; ruff clean.
- **Next (Phase 5):** Queries 1-3 -> 4-connected polygonize -> 500 m school/hospital buffers -> difference -> area >= 5000 m2. Parity vs 46 sites / 1,254,202 m2 (original order: area filter before difference; pipeline: difference first). Then the same for the corrected surface.

## 2026-10-20: Phase 3 done
- 3a: the original `kmeans_lulc.py` reproduced 100% in the user's `pytorch` env (sklearn 1.7.2). With sklearn 1.9.1: 98.9% label agreement on the QGIS stack, 73.5% on ours. The gap is K-Means instability (seed alone: 76-99% on the identical stack; `n_iter_ == 1`), not the B11 artefact.
- 3b: user chose k=4, full KMeans(n_init=10) on a 1M sample, original scores (barren 10, built-up 3, vegetation 2, water 1). Stable (ARI 0.957); clusters named automatically from MNDWI/NDVI + OSM residential/water overlap (`stp.lulc.name_clusters`, raises if ambiguous).
- Outputs: `outputs/phase3/lulc_classes.tif` (1 water, 2 built-up, 3 vegetation, 4 barren), `outputs/phase3/lulc_score.tif`. Score identical to the original on 79.7% of pixels; barren grows 44.7% -> 57.3% of the area (fallow farmland caveat).
- `python -m stp.cli phase3` runs both parts (~2.5 min, mostly the slow MiniBatch replay); `phase3k` regenerates the k-selection evidence. 31 unit tests pass; ruff clean.
- **Next (Phase 4):** align grids explicitly (rioxarray reproject_match onto sentinel_10m), linear 1-10 rescale with inversion for roads/drainage/DEM, reconstruct the slope rule from Reclass_10_Slope, weighted overlay. Parity vs Final_STP_Suitability_Map using the ORIGINAL LULC score; then a corrected version using lulc_score.tif.

## 2026-10-20: Phase 2 done
- `python -m stp.cli phase2`: geometry-only reads (pyogrio, `columns=[]`), `gpd.clip`, rasterise on the proximity grid, exact EDT distances, NoData outside the boundary. Writes COGs and GeoParquet (clipped vectors) to `outputs/phase2/`.
- Parity: clipped vectors identical (counts, length/area, 0 m² symmetric difference); burned pixels identical; distances 99.82–99.97% within 1 cm. The rest are GDAL Proximity overestimates (always QGIS > ours, max 5.72% relative), so the pipeline's values are exact. Details in `docs/parity.md`.
- 20 unit tests pass; ruff clean. Roads take ~43 s (42,798 lines); the others ~3 s each.
- **Next (Phase 3):** K-Means land cover. 3a: reproduce the original (k=5, void included), also run on the QGIS stack because of the B11 artefact; match labels to `ML_KMeans_LULC_Bhopal.tif`. 3b: proper mask, choose k.

## 2026-10-20: Phase 1 done
- `python -m stp.cli phase1` builds the stack, indices (BOA-offset corrected), clipped DEM and slope as COGs in `outputs/phase1/`, then checks parity against QGIS. Results in `docs/parity.md`.
- Identical to QGIS: B02/B03/B04/B08, DEM, slope, and both grids.
- Explained differences: B11 (a QGIS-run artefact; a GDAL 3.13 replay matches ours), NDBI (only where B11 differs), NDVI/MNDWI (QGIS put them on the 12.5 m grid with an unreproduced sampling rule; not scored criteria).
- Issues hit and fixed: missing JP2 driver (`libgdal-jp2openjpeg`, now in environment.yml); crop rule (GDAL snaps inward, not outward); rasterio decimated reads -> explicit integer-factor pixel duplication.
- 15 unit tests pass; ruff clean.
- **Next (Phase 2):** distance rasters on the `proximity_10m` grid (rasterize + EDT), parity vs Raster_05..08. Note: in Phase 3, run the K-Means parity on the QGIS stack as well, because of the B11 artefact.

## 2026-10-20: Phase 0 done
- Env `stp` built (Python 3.12.15, GDAL 3.13.3, rasterio 1.5.2, geopandas 1.2.0, osmnx 2.1.1); `scripts/check_env.py` passes.
- Issue fixed: packages in `%APPDATA%\Python\Python312\site-packages` (old geopandas/shapely built for NumPy 1) shadowed the env and crashed imports. Fixed with the env variable `PYTHONNOUSERSITE=1`. Run things via `conda run -n stp ...` or an activated env (direct `python.exe` calls skip GDAL_DATA/PROJ_DATA).
- `environment.lock.yml` pins the exact versions. `conda env export` mislabels scipy/pyarrow as pip packages; they were moved to the conda section by hand.
- Note for Phase 7: osmnx is 2.x here; the old web app used the 1.9 `bbox=(north, south, east, west)` order.

## 2026-10-20: Phase 0 started
- Created the repo with `PLAN.md`, `config.yaml`, `environment.yml`, `pyproject.toml` and `docs/qgis_workflow.md`.
- Recovered the QGIS parameters from `user-history.db` (63 runs) and from inspecting the rasters; see `docs/qgis_workflow.md`.
- Verified reference numbers: 46 final sites / 1,254,202 m²; suitability range 3.42–8.48 (later corrected to 3.06–8.48 at full resolution).
- **Next:** build the conda env (`conda env create -f environment.yml`), check imports, then start Phase 1.
- Repo published: https://github.com/vyom-nikhra/bhopal-stp-siting (public).
- Planning boundary source: DataMeet Municipal_Spatial_Data, CC BY 4.0 (attributed in the README).
- **Open question:** the slope scoring rule (Phase 4).
