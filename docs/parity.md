# Parity with the QGIS outputs

Each phase compares the Python pipeline with the original QGIS rasters. "exact" is the share of pixels, valid in both rasters, whose values agree within the tolerance; "mask agree" is the share of pixels where both agree on valid vs NoData. Reproduce with `python -m stp.cli phase1` (the report is written to `outputs/parity/phase1.json`).

## Phase 1: preprocessing

| Layer | Grid | Mask agree | Exact | Mean \|d\| | Max \|d\| | Verdict |
|---|---|---|---|---|---|---|
| B02, B03, B04, B08 | same | 1.0000 | 1.0000 | 0 | 0 | **Identical** |
| B11 | same | 1.0000 | 0.7162 | 34.9 | 3598 | Explained: QGIS-run artefact (below) |
| NDBI (raw DN) | same | 1.0000 | 0.7162 | 4.9e-3 | 0.287 | Differs **only** where B11 differs (2,310,027 px; 0 elsewhere) |
| NDVI (raw DN) | resampled | 0.9994 | 0.4654 | 1.5e-2 | 0.404 | Statistical only: reference is on the 12.5 m grid |
| MNDWI (raw DN) | resampled | 0.9994 | 0.3355 | 1.3e-2 | 0.900 | Statistical only: reference is on the 12.5 m grid |
| DEM | same | 1.0000 | 1.0000 | 0 | 0 | **Identical** |
| Slope (Horn) | same | 1.0000 | 1.0000 | 0 | 0 | **Identical** (numpy implementation vs `gdaldem`) |

The stack grid (origin 727980, 2592280; 3769 × 3658) and the DEM grid (origin 727972.3125, 2592280.25; 3016 × 2926) match the references exactly. GDAL's crop-to-cutline snaps **inward** to the source pixel grid, and `preprocess.crop_window` reproduces this.

### Findings

**1. QGIS computed the indices on raw digital numbers.** Processing baseline N0511 stores surface reflectance as `(DN − 1000) / 10000` (`BOA_ADD_OFFSET = −1000` in `MTD_MSIL2A.xml`). NDBI matches raw DN to 1.5e-8 and differs by up to 0.30 once the offset is applied. The pipeline writes **offset-corrected** indices; the correction shifts NDVI by 0.088 on average (max 0.32), NDBI by 0.033 (max 0.30) and MNDWI by 0.126 (max 0.33). The parity check recomputes the raw-DN version only for comparison.

**2. B11 in the QGIS stack has a row-assignment artefact.** B11 is 20 m and was upsampled to 10 m with nearest neighbour inside the VRT, which should simply duplicate each 20 m pixel into a 2 × 2 block. In the QGIS output:
- every odd 10 m row matches clean duplication exactly;
- every even-row pixel equals either its own duplicated value or the row above (100% of pixels are one or the other), switching in blocks across the image.

Replaying the identical steps (`gdal.BuildVRT` → `gdal.Warp` with the cutline) under GDAL 3.13.3 gives clean duplication, 100% identical to the pipeline. The artefact came from the QGIS run (GDAL 3.12.1), most likely from block-wise fractional source windows in the VRT. The pipeline keeps the clean result. Investigation: `scripts/diagnose_b11.py`.

Impact: about 28% of B11 pixels in the QGIS stack hold the neighbouring 20 m row's value. B11 feeds NDBI and MNDWI (not scored) and **K-Means (scored)**, so the Phase 3 parity check also runs on the QGIS stack, to separate this effect from the clustering itself.

**3. NDVI and MNDWI were saved on the 12.5 m DEM grid.** They were computed in the Raster Calculator with a 12.5 m output grid. Neither GDAL nearest nor bilinear resampling, nor four pixel-centre/corner rounding rules, reproduce QGIS's sampling exactly (best: 46.5% exact). Since these layers are not among the seven scored criteria, they are compared statistically and the investigation stopped there.

**4. The DEM "warp" was a no-op.** The ALOS DEM is already in EPSG:32643 on the output grid, so the pipeline only clips it.

## Phase 2: distance rasters

Reproduce with `python -m stp.cli phase2` (report: `outputs/parity/phase2.json`).

**Clipped vectors** (`gpd.clip` vs QGIS `native:clip`):

| Layer | Features (ours / QGIS) | Total length or area, relative difference | Symmetric difference |
|---|---|---|---|
| Roads | 42,798 / 42,798 | −1.3e-16 (length) | |
| Drainage | 148 / 148 | −1.3e-16 (length) | |
| Residential | 386 / 386 | 0 (area) | 0 m² |
| Water | 79 / 79 | 7.5e-16 (area) | 0 m² |

**Rasterised features:** pixels with distance 0 inside the boundary are **identical** for all four layers (e.g. 668,913 road pixels in both).

**Distances:**

| Layer | Mask agree | Exact (≤ 1 cm) | Mean \|d\| | Max \|d\| |
|---|---|---|---|---|
| Roads | 1.0000 | 0.9982 | 2.1e-3 m | 2.72 m |
| Drainage | 1.0000 | 0.9993 | 3.4e-4 m | 2.09 m |
| Residential | 1.0000 | 0.9997 | 2.1e-4 m | 2.09 m |
| Water | 1.0000 | 0.9997 | 1.8e-4 m | 2.52 m |

### Finding

**5. GDAL Proximity slightly overestimates distances; the pipeline's are exact.** The pipeline uses `scipy.ndimage.distance_transform_edt`, an exact Euclidean distance transform. GDAL's Proximity uses a faster two-pass scan that approximates the nearest feature. Every one of the 25,844 differing pixels across the four layers has QGIS > ours (none the other way), and the largest relative error is exactly 5.72% in every layer: the fixed worst-case ratio of the approximation, not a data difference. Check: `scripts/diagnose_proximity.py`.
