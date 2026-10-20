# Progress log

Newest entries at the top. Each entry records what was done, what was verified, and what comes next, so a fresh session can resume from here.

## 2026-10-20: Phase 0 started
- Created the repo with `PLAN.md`, `config.yaml`, `environment.yml`, `pyproject.toml` and `docs/qgis_workflow.md`.
- Recovered the QGIS parameters from `user-history.db` (63 runs) and from inspecting the rasters; see `docs/qgis_workflow.md`.
- Verified reference numbers: 46 final sites / 1,254,202 m²; suitability range 3.42–8.48.
- **Next:** build the conda env (`conda env create -f environment.yml`), check imports, then start Phase 1.
- Repo published: https://github.com/vyom-nikhra/bhopal-stp-siting (public).
- Planning boundary source: DataMeet Municipal_Spatial_Data, CC BY 4.0 (attributed in the README).
- **Open question:** the slope scoring rule (Phase 4).
