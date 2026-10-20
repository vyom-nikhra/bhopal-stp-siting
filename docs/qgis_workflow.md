# The original QGIS workflow (reconstructed)

This is a record of what was actually done in QGIS 3.40.15 (Feb–Apr 2026), rebuilt from three sources:

1. **QGIS processing history**: `%APPDATA%/QGIS/QGIS3/profiles/default/user-history.db`, 63 logged runs.
2. **Inspection of the output rasters and vectors**: grids, NoData values, value distributions.
3. **The project report** (`GEOTA_Assignment_Final_S20230030424.pdf`): the source for steps QGIS doesn't log (Raster Calculator).

The source of each fact is marked: **[H]** history, **[D]** data inspection, **[R]** report.

## 1. Inputs

| Input | Details |
|---|---|
| Sentinel-2 L2A | `S2B_MSIL2A_20250609T051649_N0511_R062_T43QGF`, 9 June 2025, Copernicus Data Space |
| DEM | ALOS PALSAR RTC `AP_04940_FBS_F0450_RT1.dem.tif`, 12.5 m, UTM 43N, NoData 0 (ASF DAAC). A Copernicus 30 m DEM was tried and dropped **[H]** |
| Vectors | OpenStreetMap via QuickOSM: roads, natural drainage, residential land use, water, schools, hospitals, parks, power, railways |
| Study area | `Planning_Boundary_2031.geojson` (EPSG:4326) |

## 2. Preprocessing

| Step | Tool | Parameters |
|---|---|---|
| Band stack | `gdal:buildvirtualraster` | B02, B03, B04, B08 (10 m) + B11 (20 m), `SEPARATE=True`, `RESOLUTION=highest`, nearest resampling **[H]** |
| Clip stack | `gdal:cliprasterbymasklayer` | Mask = boundary, crop to cutline, target EPSG:32643, **no NoData set**, so outside pixels are 0 (UInt16) **[H][D]** |
| DEM | `gdal:warpreproject` → clip | Bilinear resampling to EPSG:32643, then clip to boundary **[H]** |
| Vector clips | `native:clip` | Each OSM layer clipped to the boundary **[H]** |
| Slope | `gdal:slope` | Horn, degrees, scale 1 **[H]**. Aspect was also computed but not used |

The ALOS DEM is already in EPSG:32643, so the warp step changed nothing **[D]**. In the QGIS stack, B11 has a row-assignment artefact from that run: see [parity.md](parity.md) **[D]**.

## 3. Spectral indices (Raster Calculator, not logged)

- NDVI = (B08 − B04) / (B08 + B04); NDBI = (B11 − B08) / (B11 + B08); MNDWI = (B03 − B11) / (B03 + B11) **[R]**
- Computed on **raw DN**, without the Sentinel-2 `BOA_ADD_OFFSET` of −1000 (NDBI matches raw DN to 1.5e-8) **[D]**.
- NDBI is on the Sentinel 10 m grid, but **NDVI and MNDWI are on the 12.5 m DEM grid** **[D]**. This was probably the calculator's reference layer at the time.
- An NDVI-based zone reclass (`Raster_10_LULC_Zones`: ≤0.2→1, ≤0.4→2, ≤1→3) was made but not used **[H]**.

## 4. Distance rasters

For roads, drainage, residential and water:

1. `gdal:rasterize`: burn 1, 10 m pixels, extent `727972.3125, 765672.3125, 2555705.25, 2592280.25` (EPSG:32643) **[H]**
2. `gdal:proximity`: distance in georeferenced units (metres), no maximum distance **[H]**
3. `gdal:cliprasterbymasklayer` to the boundary, NoData −9999 **[H]**

The water raster was first saved as `Raster_09_Dist_Water` and later re-clipped as `Raster_08_Dist_Water_Clipped` **[H]**.

## 5. Land cover with K-Means (`kmeans_lulc.py`)

- MiniBatchKMeans, k = 5, `random_state=42`, on all 5 bands of the clipped stack.
- **Bug:** the script masks `band1 != -9999`, but the stack is UInt16 with 0 outside the boundary. Nothing gets masked, and cluster 2 is exactly the out-of-boundary area (40.9% of pixels) **[D]**.
- Reclass table **[H]**: `{1→2, 2→NoData, 3→10 (barren), 4→3, 5→1}`. This is why the report mentions "4 classes".

## 6. Standardisation to a 1–10 score

- GRASS `r.rescale` with `from = [min, max]` of each layer and `to = [1, 10]` **[H]**.
- **Inverted afterwards (`11 − x`) for roads, drainage and DEM** (closer, or lower, is better). This was confirmed from the data: correlation with the raw layer is −0.96 / −0.99 / −0.98, and |reclass − (11 − linear)| ≤ 0.6 **[D]**. Residential and water distance are not inverted (corr +0.99) **[D]**.
- **Slope rule: unknown.** `Reclass_10_Slope` takes values 3–10 (8 distinct values). It matches neither a linear rescale nor the logged table `[0–5]→10, (5–15]→5, (15–90]→1` **[D]**.
- Output grids: roads and drainage Float32 on the 10 m proximity grid; residential and water Byte (NoData 255); DEM and slope on the 12.5 m grid **[D]**.

## 7. Weighted overlay (Raster Calculator, not logged)

```
S = 0.25·Residential + 0.20·Water + 0.15·Drainage + 0.12·Slope
  + 0.10·Roads + 0.10·LULC + 0.08·DEM
```

**[R]** The output is on the Sentinel 10 m grid. The other grids were resampled implicitly by the calculator. Values range from 3.42 to 8.48, and only 0.22% of pixels are ≥ 8 **[D]**.

## 8. Site selection

| Step | Logic | Source |
|---|---|---|
| Query 1 | suitability ≥ 8.0 | [R] |
| Query 2 | Query 1 AND LULC score == 10 | [R] |
| Query 3 | Query 2 AND roads score ≥ 7 | [R] |
| Polygonize | `gdal:polygonize`, **4-connected**, field `DN` → 984 polygons, including the DN=0 background | [H][D] |
| Area | `Area_sqm` field calculated (not logged) | [D] |
| Buffers | `native:buffer` 500 m, 30 segments, schools and hospitals separately → `native:mergevectorlayers` | [H] |
| Final | `native:difference` of (`Area_sqm >= 5000 AND DN = 1`) minus buffers → **46 sites, 1,254,202 m²** | [H][D] |

**Order:** the area filter ran *before* the buffer difference, which is the opposite of the order the report describes. It made no difference in practice: no final site is under 5000 m² **[D]**.

**Dashboard:** `STP_Dashboard/Candidate_Sites_Polygons.geojson` has **47** sites (1,258,324 m²). These are the candidates *before* the buffer difference, so the dashboard shows one site that falls inside an exclusion buffer **[D]**.
