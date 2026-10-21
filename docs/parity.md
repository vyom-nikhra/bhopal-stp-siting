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

## Phase 3: K-Means land cover

### 3a. Reproducing the original

- **Exact baseline.** The unmodified `kmeans_lulc.py`, run in the user's `pytorch` env (scikit-learn 1.7.2) on the QGIS stack, reproduces `ML_KMeans_LULC_Bhopal.tif` **100%** (all 13.8 M pixels). So that is how the original map was made.
- **Pipeline env (scikit-learn 1.9.1), same settings** (`python -m stp.cli phase3`), after matching cluster ids with the Hungarian method (`scipy.optimize.linear_sum_assignment`):

| Input | Label agreement | LULC score identical |
|---|---|---|
| QGIS stack | 0.9887 | 0.9808 |
| Pipeline stack (clean B11) | 0.7352 | 0.5515 |

### Finding

**6. The original K-Means map is one of many equally plausible partitions.** Changing only the random seed, on the identical QGIS stack, gives agreement from 0.76 to 0.99 with the original map; on the pipeline stack 0.74 to 0.93 (`scripts/diagnose_kmeans.py`). So the 0.735 above is K-Means instability, not the B11 artefact. In every run `n_iter_ == 1`: MiniBatchKMeans' early-stopping rule halts within its first pass over the 13.8 M pixels. Seed 42 landing at 98.9% on the QGIS stack is luck, not robustness.

### 3b. Evidence for the corrected version (void masked, 8,139,220 pixels)

`python -m stp.cli phase3k` (MiniBatchKMeans fitted on a 1 M-pixel sample, metrics on a separate 20 k sample, 3 seeds):

| k | Silhouette ↑ | Davies-Bouldin ↓ | Seed stability ARI ↑ (min) |
|---|---|---|---|
| 2 | 0.309 | 1.191 | 0.48 (0.28) |
| 3 | 0.327 | 1.058 | 0.82 (0.73) |
| **4** | **0.366** | **0.879** | **0.82 (0.74)** |
| 5 | 0.344 | 0.896 | 0.50 (0.35) |
| 6 | 0.332 | 0.942 | 0.54 (0.49) |
| 7 | 0.324 | 0.931 | 0.60 (0.50) |
| 8 | 0.314 | 0.965 | 0.53 (0.50) |

The elbow curve has no clear bend. k = 4 is best on silhouette, Davies-Bouldin and stability; k = 5 (the original) is unstable.

**Full K-Means** (Lloyd, `n_init=10`) at k = 4, fitted on the 1 M sample and applied to all pixels: seed-stability ARI **0.957** (min 0.937); converges in 12–41 iterations. Checked against OSM land use and the original classes (`scripts/diagnose_kmeans_fix.py`):

| Cluster | Area | In OSM residential | In OSM water | Original class overlap | Reading |
|---|---|---|---|---|---|
| c1 | 8.5% | 0.46× | 9.15× (78% of water px) | 58% o5, 42% o3 | Water / dark |
| c2 | 16.4% | 2.43× | 0.04× | 100% o4 | Built-up / bright |
| c4 | 17.8% | 1.43× | 0.70× | 97% o1 | Vegetation |
| c3 | 57.3% | 0.54× | 0.16× | 72% o3 | Barren / bare soil |

(The original's scores: o3 → 10 (barren), o4 → 3, o1 → 2, o5 → 1, void → NoData.)

Caveat: the scene is from 9 June (pre-monsoon), so the "barren / bare soil" class also contains fallow farmland; spectra from a single date cannot separate the two.

### 3b. The corrected land-cover layer (chosen setup)

Chosen with the user on 2026-10-20 from the evidence above: **void masked, k = 4, full K-Means (`n_init=10`) on a fixed 1 M-pixel sample, applied to all pixels**, scored with the original meanings (barren 10, built-up 3, vegetation 2, water 1). Clusters are named automatically by `stp.lulc.name_clusters`, which raises an error instead of guessing if the evidence is inconsistent:

| Class | Area | NDVI | MNDWI | In OSM residential | In OSM water | Score |
|---|---|---|---|---|---|---|
| Water / dark | 8.5% | +0.18 | −0.21 | 0.46× | 9.15× | 1 |
| Vegetation | 17.8% | +0.53 | −0.43 | 1.43× | 0.70× | 2 |
| Built-up | 16.4% | +0.15 | −0.25 | 2.43× | 0.04× | 3 |
| Barren / bare soil | 57.3% | +0.21 | −0.39 | 0.54× | 0.16× | 10 |

Compared with the original `Reclass_ML_LULC`, the score is identical on **79.7%** of pixels:

| Original score → | 1 | 2 | 3 | 10 |
|---|---|---|---|---|
| 1 (water/dark) | 99.8% | 0.1% | | 0.1% |
| 2 (vegetation) | 0.1% | 76.4% | 0.0% | 23.5% |
| 3 (built-up) | | 1.3% | 59.2% | 39.5% |
| 10 (barren) | 8.0% | 0.2% | 0.0% | 91.8% |

So the barren class grows (44.7% → 57.3% of the area), mostly taking from the original built-up and vegetation classes. Because the scene is pre-monsoon (9 June), some of this is fallow farmland, which a single date can't separate from true barren land. Phase 5's road-access and score thresholds narrow it down, and the Phase 6 ground-truth check will measure it.

## Phase 4: scoring and weighted overlay

Reproduce with `python -m stp.cli phase4` (report: `outputs/parity/phase4.json`).

| Check | Mask agree | Exact | Max \|d\| |
|---|---|---|---|
| Roads, drainage, residential, water, DEM, slope scores vs `Reclass_*` | 1.0000 | **1.0000** | 0 |
| Overlay of the QGIS `Reclass_*` rasters vs `Final_STP_Suitability_Map` | 1.0000 | **1.0000** | 0 |
| Full pipeline overlay (original land cover) vs `Final_STP_Suitability_Map` | 1.0000 | **1.0000** | 0 |

**The pipeline reproduces the QGIS suitability map exactly** (8,105,749 pixels, range 3.06–8.48, 0.2208% ≥ 8).

### Findings

**7. The scoring rule was GRASS's integer-input rescale.** `r.rescale` rounded each input to whole numbers, mapped the layer's integer [min, max] onto [1, 10], and rounded half up; roads, drainage and DEM were then inverted (`11 − x`). This reproduces all six score rasters with zero mismatches (`scripts/diagnose_rescale.py`). Rounding the input first is why the slope scores step at 2.5°, 7.5°, 12.5° …; it isn't a separate slope table. Round-half-up (not numpy's round-half-to-even) is needed for the last 4 water pixels.

**8. QGIS's implicit resampling was nearest neighbour.** The Raster Calculator combined three grids (Sentinel 10 m, the shifted 10 m proximity grid, and the 12.5 m DEM grid). Explicit nearest-neighbour alignment onto the Sentinel grid (`rioxarray.reproject_match`) gives an identical map, so the pipeline now does openly what QGIS did silently.

**9. Phase 2's small distance differences don't change any score.** GDAL Proximity's ≤ 2.7 m overestimates are far smaller than one score bin (≈ 234 m for roads, 1363 m for water) and never move a pixel across a bin boundary.

### Effect of the corrected land cover

| Surface | Min | Max | Mean | Pixels ≥ 8 |
|---|---|---|---|---|
| QGIS / pipeline with original land cover | 3.06 | 8.48 | 5.71 | 0.2208% |
| Pipeline with corrected land cover (Phase 3b) | 2.98 | 8.48 | 5.79 | 0.3395% |

The corrected land cover raises the area scoring ≥ 8 by about 54%, mostly because more land is now classed as barren (score 10).

## Phase 5: site selection

Reproduce with `python -m stp.cli phase5` (report: `outputs/parity/phase5.json`).

| Check | Result |
|---|---|
| Query 1, 2, 3 rasters vs QGIS | **Identical** (mask agree 1.0, exact 1.0) |
| Candidates ≥ 5000 m² vs `Candidate_Sites_Polygons.geojson` | 47 vs 47, 0 m² symmetric difference |
| Final sites vs `Final_Safe_Sites.shp` | 46 vs 46, 0 m² symmetric difference |

**The pipeline reproduces the QGIS site selection exactly.**

### Findings

**10. Query 2 (the "environmental check") never removed a pixel.** Query 1 and Query 2 select the same 17,896 pixels: every pixel scoring ≥ 8 was already barren (land-cover score 10). The same holds with the corrected land cover (27,519 = 27,519). With a 0.10 land-cover weight, a pixel can in principle reach 8 without being barren, but none does here.

**11. `Area_sqm` is QGIS's ellipsoidal area.** It equals the area on the WGS84 ellipsoid, rounded to whole m², for all 47 candidates; the planar UTM area is about 0.09% larger here (e.g. 5,100 vs 5,095.6 m²). The pipeline measures ellipsoidal area with `pyproj.Geod`, which is both faithful and the true ground area.

**12. Filtering before the buffer difference did no harm here.** Running the difference first and then re-measuring (the order the report describes) gives the same 46 sites: the buffers removed one 47th candidate entirely and trimmed none below 5000 m². The pipeline uses the safer order (difference, then filter), and a unit test shows a case where the two orders differ.

### Variants

| Variant | Land cover | Order | Query 3 px | Sites | Area (ellipsoidal) |
|---|---|---|---|---|---|
| A (QGIS reproduction) | original | filter → difference | 16,805 | 46 | 125.3 ha |
| B | original | difference → filter | 16,805 | 46 | 125.3 ha |
| **C (corrected)** | **corrected (Phase 3b)** | difference → filter | 25,694 | **59** | **191.8 ha** |

How C relates to A (`scripts/compare_site_sets.py`): 45 of the 46 original sites still overlap a corrected site and 83.3% of the original site area is still selected; 1 original site (0.56 ha) drops out; 23 corrected sites are entirely new (22.6 ha), and the rest of the gain comes from existing sites growing. So the land-cover correction extends the original result rather than replacing it. The barren class's fallow-farmland caveat (Phase 3b) applies to the added area.

## Phase 6: beyond QGIS

### 6a. Threshold ties (found while building the sensitivity analysis)

**13. Thousands of pixels score exactly 8.0, and floating-point rounding decided them.** All criterion scores are integers and the weights are multiples of 0.01, so 100 × suitability is an exact integer (`scripts/diagnose_threshold_ties.py`):

| Surface | Pixels ≥ 8 (exact) | Exact ties (= 8.0) | Ties passing: float64 → float32 (QGIS-like) | Ties passing: float32 layers, float64 sum |
|---|---|---|---|---|
| Original land cover | 17,896 | 2,417 | 2,417 | 200 |
| Corrected land cover | 27,519 | 4,740 | 4,740 | 735 |

In QGIS every tie happened to round up and pass. A different but equally reasonable arithmetic path drops most of them: 4,005 of 27,519 high-potential pixels (15%) would change. The pipeline now applies the threshold as `score ≥ 8 − 1e−6`, so exact ties pass deterministically, which matches what QGIS did. Phase 5 parity is unchanged. A consistency check caught this: the sensitivity baseline (60 sites) disagreed with Phase 5 (59) until the tolerance was added.

### 6b. Weight sensitivity

`python -m stp.cli sensitivity` (corrected result; 300 Monte Carlo draws, each weight × U(0.8, 1.2) then renormalised; plus one-at-a-time ±20%). Outputs: `outputs/phase6/selection_frequency.tif`, `sites_sensitivity.parquet`, `sensitivity.json`.

**Monte Carlo (300 draws):**

| Measure | Median | 5–95% range |
|---|---|---|
| Sites | 54 | 25–101 |
| Area | 188.0 ha | 83.8–454.6 ha |
| IoU with the baseline sites | 0.555 | |

Of the 59 baseline sites, 14 are selected (≥ 50% of their area) in ≥ 90% of draws, 20 in ≥ 75%, and 42 in ≥ 50%.

**One at a time (each weight × 0.8 or × 1.2, others rescaled):**

| Criterion | −20%: sites | ha | IoU | +20%: sites | ha | IoU |
|---|---|---|---|---|---|---|
| Residential distance | 105 | 568.0 | 0.333 | 23 | 85.7 | 0.407 |
| Water distance | 75 | 273.9 | 0.682 | 49 | 199.4 | 0.776 |
| Drainage distance | 60 | 212.0 | 0.618 | 64 | 201.4 | 0.595 |
| Slope | 43 | 117.8 | 0.614 | 71 | 263.7 | 0.727 |
| Road distance | 52 | 138.9 | 0.719 | 58 | 230.1 | 0.833 |
| Land cover | 47 | 124.6 | 0.650 | 64 | 240.5 | 0.797 |
| DEM | 59 | 162.4 | 0.818 | 45 | 196.8 | 0.897 |

**14. The site set is fragile to the weights, and the fixed "≥ 8" threshold is why.** The suitability surface tops out at 8.48, so the cut sits at the very top of the score range, where small weight changes move thousands of pixels (including the ~4,700 exact ties) across it. The residential-distance weight (0.25) dominates: ±20% swings the result between 23 and 105 sites. The weights were asserted rather than derived (see the AHP item in PLAN.md), so this is the main uncertainty in the final answer. Robustness should be reported alongside every site; the ranked table does this.

### 6c. Ranked site table

`python -m stp.cli sites` → `outputs/phase6/sites_ranked.{parquet,geojson,csv}`. For each of the 59 corrected sites:
- `rasterstats` zonal statistics: suitability (mean/min/max), slope, elevation range (relevant to gravity flow), distance to the nearest road and drainage line;
- `geopandas.sjoin_nearest`: distance from the site edge to the nearest school, hospital, lake and residential area. Every site is ≥ 499 m from any school or hospital, which the code checks (the 500 m buffers are 120-gon approximations);
- the persistence from 6b, a `robust` flag (≥ 75%), and centroid lon/lat.

Sites are ranked by mean suitability, then area. 20 of the 59 are robust.

### 6d. Road-network access

`python -m stp.cli network` adds network columns to `outputs/phase6/sites_ranked.*`. The drivable OSM network for the boundary plus 1 km (63,056 nodes, 162,786 edges, downloaded through osmnx on 2026-10-20 and cached as `drive_network.graphml`) is routed with one multi-source Dijkstra on the reversed graph, so one-way streets count. Major roads = motorway, trunk, primary, secondary (and their links). Summary from `scripts/summarise_access.py`:

| Measure | Median | Max |
|---|---|---|
| Straight-line distance to any QuickOSM road | 14 m | 626 m |
| Last mile to the nearest drivable junction | 258 m | 1,382 m |
| Network distance from there to a major road | 2,043 m | 6,123 m |
| Total access (last mile + network) | 2,620 m | 6,969 m (90th pct 5,501 m) |

All 59 sites reach a major road.

**15. Straight-line "distance to road" overstates truck access.** The QuickOSM roads layer includes footways, paths and tracks, which a vacuum truck can't use. Only 8 sites are more than 250 m from any road, but 31 are more than 250 m from a drivable junction (21 more than 500 m), and for 24 sites the drivable last mile is over 250 m longer than the any-road distance. The original road criterion (and Query 3) therefore rates some sites as accessible that aren't for trucks.

Caveat: the last mile is measured to the nearest junction (graph node), not the nearest point along a road edge, so it is an upper bound where a long road segment passes a site without a junction nearby. The OSM network was downloaded in October 2026, months after the February 2026 QuickOSM roads.

### 6e. Threshold rules

`python -m stp.cli threshold` compares rules for Query 1 under the same 100 random weight draws (each weight × U(0.8, 1.2), renormalised; corrected land cover). "Top X%" keeps the best X% of valid pixels, so its numeric cut moves with the weights.

| Rule | Cut | Baseline | Sites (5–95%) | Area (5–95%) | Location stability, median IoU (p5) |
|---|---|---|---|---|---|
| Fixed ≥ 8 (original) | 8.00 | 59 sites, 191.8 ha | 23–95 | 71–424 ha | 0.541 (0.283) |
| Top 0.25% | 8.02 | 55 sites, 154.6 ha | 38–58 | 130–157 ha | 0.679 (0.425) |
| **Top 0.3395%** (same share as ≥ 8) | 7.99 | 60 sites, 193.1 ha | **45–66** | **187–211 ha** | **0.653 (0.460)** |
| Top 0.5% | 7.93 | 74 sites, 310.3 ha | 62–87 | 301–335 ha | 0.731 (0.458) |

Robust core (pixels selected in ≥ t of the 300 sensitivity draws, then the same buffers and 0.5 ha filter): t = 75% → 33 sites, 89.4 ha; t = 90% → 24 sites, 58.4 ha.

**16. A relative ("top X%") cut removes most of the swing in how much land is selected, but only part of the swing in where it is.** At the same size as the original rule, the top-0.34% baseline is almost identical to the ≥ 8 result (IoU 0.993), while the selected area under weight changes narrows from 71–424 ha to 187–211 ha. Location stability improves from 0.54 to 0.65: the rest is genuine dependence on the weights, which the AHP review and per-site persistence address. Because scores take discrete values, top-X% cuts land slightly above their nominal share (0.25% selects 0.277%).

### 6f. AHP weights (draft judgements)

`python -m stp.cli ahp` reads `ahp_judgements.yaml` (21 pairwise judgements with reasons; a first draft by Claude for the author to review) and writes `docs/ahp_questionnaire.md` and `outputs/phase6/ahp/`. Inputs to the judgements came from `scripts/criteria_correlation.py`: criterion overlap is moderate at most (|ρ| ≤ 0.35), and slope and roads score 9–10 on 96% and 93% of the city, so they separate sites little.

| Criterion | Original (asserted) | AHP draft |
|---|---|---|
| Residential distance | 0.250 | 0.215 |
| Water distance | 0.200 | **0.333** |
| Drainage distance | 0.150 | 0.129 |
| Slope | 0.120 | 0.046 |
| Road distance | 0.100 | 0.074 |
| Land cover | 0.100 | 0.129 |
| DEM | 0.080 | 0.074 |

Consistency ratio CR = 0.0099 (acceptable below 0.10); the geometric-mean method agrees to 0.002. With the draft AHP weights: ≥ 8 rule → 38 sites, 219.2 ha (IoU 0.49 with the original-weight sites); top 0.3395% → 34 sites, 233.3 ha. 15 of the 20 robust sites are kept under both rules.

## Final model

Decided with the author on 2026-10-20/21. `python -m stp.cli final`; config section `model:`; code `stp.model`. Phases 1–5 keep the original setup so their parity checks still hold.

### Decisions and their sources

| Decision | Value | Source / evidence |
|---|---|---|
| Weights | AHP (CR 0.0099), lake distance 0.333 highest | `ahp_judgements.yaml`, reviewed and approved by the author |
| Residential distance | rescaled only up to 100 m, flat beyond | CPHEEO Manual on Sewerage and Sewage Treatment Systems (2013), Part A ch. 5, 100 m odour buffer without deodorisation, as quoted in a CPCB reply to the NGT. Before the cap, all 59 corrected sites were 2.1–7.6 km from mapped housing |
| Minimum site area | 5,588 m² = 12.7 MLD × 0.044 ha/MLD | Mean of Bhopal's six AMRUT 2.0 Package 3 plants; largest of three measured Indian plant footprints (Saxena, SuSanA) |
| Selection | best eligible pixels, as many as the top 0.3395% of all valid pixels | Phase 6e: a relative cut keeps the selected area stable under weight changes |
| Constraints | yearly Dynamic World mode ∈ {grass, scrub, bare}; built ≥ 50% + 100 m; water ≥ 10% | Below |

### Finding 17: the "barren" land was mostly fallow farmland

The first final-model run (AHP, cap, top 0.34%, no constraints) gave 44 sites (294 ha). An independent cross-check (`python -m stp.cli crosscheck`) put most of them on the wrong kind of land:

| Product | Crops | Built | Other |
|---|---|---|---|
| ESA WorldCover 2021 | 38 | 0 | 6 |
| Esri / Impact Observatory 2023 | 38 | 6 | 0 |
| Dynamic World (same scene, 2025-06-09) | 32 | 12 | 0 |

By majority: 36 likely farmland, 6 likely built-up, 0 open land. A visual check of Sentinel-2 chips found 8 sites inside settlements that OSM doesn't map as residential, and about 14 on riverbanks. Over the full year, the same 44 sites were 84.7% cropland and 13.7% built-up by Dynamic World's per-pixel mode (`scripts/explore_dynamic_world.py`).

**Cause.** A single pre-monsoon (June) scene can't tell fallow fields from open land, and OSM's residential polygons miss many villages.

**Fix.** Hard constraints from 12 months of Dynamic World (2024-06-10 to 2025-06-10, 141 images), applied **before** ranking:

| Constraint | Rule | Share of the city |
|---|---|---|
| Eligible land cover | yearly mode is grass (2), shrub and scrub (5) or bare (7) | 4.1% |
| Settlement zone | built in ≥ 50% of observations, + 100 m | 53.3% |
| Flood proxy | water in ≥ 10% of observations | 6.3% |
| **Eligible after all three** | | **3.3%** |

City-wide yearly mode: crops 47.4%, built 32.3%, trees 9.9%, water 5.0%, scrub 3.6%, flooded vegetation 1.2%, grass 0.3%, bare 0.2%.

### Result

66 sites, 233.8 ha. Cut-off 7.446 among eligible pixels; site sizes 0.57–21.3 ha (median 1.9 ha). Overlap with the original 46 QGIS sites: IoU 0.006.

| Check | Result |
|---|---|
| Weight sensitivity (300 draws, ±20%) | sites 59–68; IoU with baseline median 0.906 (p5 0.811); 58/66 sites persistent in ≥ 75% of draws, 51/66 in ≥ 90% |
| One-at-a-time ±20% | residential, roads and slope: no change (IoU ≥ 0.997); water: IoU 0.78–0.83; drainage: IoU 0.87–0.88 |
| Independent land cover | WorldCover 62/66 open land; Esri 35/66; single-scene Dynamic World 12/66. Majority: 35 open land, 31 likely farmland, 0 built |
| Visual check (Sentinel-2, 10 m) | 9 yes, 2 no (S35/S43: a facility with regular rows), 55 unsure |
| Road access | all 66 reach a major road; total access median 1.85 km, max 8.8 km; last mile median 276 m |
| Schools / hospitals | closest site 564 m from a school, 610 m from a hospital |

The stability is partly a **ceiling effect**: the top tier is saturated (score 10) on the capped residential criterion, lake distance and land cover, so changing those weights doesn't reorder it.

The single-scene Dynamic World "crops" calls drive most of the 31 "likely farmland" verdicts. That scene is the source shown above to confuse fallow fields with open land, so these 31 sites are best treated as uncertain rather than rejected.

**Remaining limitations.**
- The water-frequency proxy misses narrow riverbanks (only 0.9% of the first run's site pixels had water ≥ 10%), and no sourced riparian setback was available.
- The OSM land-use cross-check is pending because Overpass was unreachable from the author's network: one of its two IPv4 servers is unreachable from there, and the other timed out.
