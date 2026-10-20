# Bhopal STP Siting

Finding suitable land for decentralised **sewage treatment plants (STPs)** in Bhopal, the "City of Lakes", using multi-criteria GIS analysis of Sentinel-2 imagery, an ALOS PALSAR elevation model and OpenStreetMap data.

The analysis was first prototyped in QGIS. This repo rebuilds it as a reproducible Python pipeline (rasterio, rioxarray, geopandas, scikit-learn and others), checks every stage against the original QGIS outputs, and then extends it.

> **Status:** work in progress. See [PLAN.md](PLAN.md) for the roadmap and [PROGRESS.md](PROGRESS.md) for the current state.

## Method in brief
1. Preprocess Sentinel-2 L2A (10 m) and the ALOS PALSAR DEM (12.5 m) to UTM 43N and clip them to the 2031 planning boundary.
2. Build criteria layers: spectral indices, slope, distance to roads/drainage/residential areas/water, and K-Means land cover.
3. Score each criterion 1–10 and combine them with a weighted overlay.
4. Select sites: suitability threshold, barren land, road access, 500 m exclusion buffers around schools and hospitals, minimum area 0.5 ha.

The original QGIS workflow, reconstructed from the QGIS processing history, is documented in [docs/qgis_workflow.md](docs/qgis_workflow.md).

## Setup
```bash
conda env create -f environment.yml
conda activate stp
```
Then edit the data paths in `config.yaml`.

## Data sources
- Sentinel-2 MSI L2A, tile T43QGF, 9 June 2025: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- ALOS PALSAR RTC DEM (AP_04940_FBS_F0450_RT1): [ASF DAAC](https://search.asf.alaska.edu/)
- Roads, water, drainage, land use, schools, hospitals: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL)
- Planning boundary 2031: [Bhopal Planning Boundary 2031](https://github.com/datameet/Municipal_Spatial_Data/blob/master/Bhopal/Planning_%20Boundary_%202031.geojson) by [DataMeet India community](http://datameet.org/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

Data is not included in this repo.
