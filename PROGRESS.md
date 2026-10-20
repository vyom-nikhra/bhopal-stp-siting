# Progress log

Newest entries at the top. Each entry records what was done, what was verified, and what comes next, so a fresh session can resume from here.

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
- Verified reference numbers: 46 final sites / 1,254,202 m²; suitability range 3.42–8.48.
- **Next:** build the conda env (`conda env create -f environment.yml`), check imports, then start Phase 1.
- Repo published: https://github.com/vyom-nikhra/bhopal-stp-siting (public).
- Planning boundary source: DataMeet Municipal_Spatial_Data, CC BY 4.0 (attributed in the README).
- **Open question:** the slope scoring rule (Phase 4).
