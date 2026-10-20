"""Phase 6 follow-ups: threshold alternatives, AHP weights, and the site-check sheet.

Kept out of cli.py (which wires the reproduction phases) to keep that file readable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.crs import CRS

from stp import ahp, overlay, selection, sensitivity, threshold
from stp.cli import _corrected_inputs
from stp.config import REPO_ROOT, Grid


def _selector(layers, cut, cfg, grid, crs, rule):
    """Final sites for a surface, with the suitability threshold set by `rule`."""
    sel = dict(cfg["selection"])

    def select(surface):
        sel["suitability_min"] = threshold.rule_threshold(surface, rule)
        return selection.sites_from_surface(
            surface, layers["lulc"], layers["roads"], cut, sel, grid.transform, crs
        )[2]

    return select


def _rule_name(rule: dict) -> str:
    return f"fixed_{rule['value']:g}" if rule["kind"] == "fixed" else f"top_{rule['pct']:g}pct"


def run_threshold(cfg: dict) -> dict:
    """Compare threshold rules on stability under the same random weight draws."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase6" / "threshold"
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid.from_config(cfg, "sentinel_10m")
    tc = cfg["threshold"]
    base = cfg["weights"]
    layers, cut = _corrected_inputs(cfg, crs, grid)
    valid_px = int(np.isfinite(overlay.weighted_sum(layers, base)).sum())

    rows, baselines = [], {}
    for rule in tc["rules"]:
        name = _rule_name(rule)
        select = _selector(layers, cut, cfg, grid, crs, rule)
        surface = overlay.weighted_sum(layers, base)
        thr = threshold.rule_threshold(surface, rule)
        base_sites = select(surface)
        base_mask = sensitivity.site_mask(base_sites, grid.shape, grid.transform)
        baselines[name] = base_mask
        base_sites.to_parquet(out / f"sites_{name}.parquet")

        rng = np.random.default_rng(tc["seed"])  # same draws for every rule
        n, area, ious = [], [], []
        t0 = time.perf_counter()
        for _ in range(tc["draws"]):
            w = sensitivity.perturb_weights(base, cfg["sensitivity"]["delta"], rng)
            s = sensitivity.run_weights(layers, w, select)
            m = sensitivity.site_mask(s, grid.shape, grid.transform)
            n.append(len(s))
            area.append(float(s["area_sqm"].sum()) / 1e4)
            ious.append(sensitivity.iou(m, base_mask))
        row = {
            "rule": name,
            "threshold": thr,
            "pixels_selected_q1_pct": float(np.mean(surface[np.isfinite(surface)] >= thr - 1e-6))
            * 100,
            "baseline_sites": len(base_sites),
            "baseline_area_ha": float(base_sites["area_sqm"].sum()) / 1e4,
            "sites_median": float(np.median(n)),
            "sites_p5": float(np.percentile(n, 5)),
            "sites_p95": float(np.percentile(n, 95)),
            "area_p5_ha": float(np.percentile(area, 5)),
            "area_p95_ha": float(np.percentile(area, 95)),
            "iou_median": float(np.median(ious)),
            "iou_p5": float(np.percentile(ious, 5)),
        }
        rows.append(row)
        print(
            f"[threshold] {name:12s} thr {thr:.3f}: {row['baseline_sites']} sites "
            f"{row['baseline_area_ha']:.1f} ha; draws sites {row['sites_p5']:.0f}-"
            f"{row['sites_p95']:.0f}, IoU median {row['iou_median']:.3f} "
            f"(p5 {row['iou_p5']:.3f}) [{time.perf_counter() - t0:.0f}s]"
        )

    # Robust core: pixels selected in >= t of the sensitivity draws (fixed >= 8 rule).
    freq_path = outputs / "phase6" / "selection_frequency.tif"
    if not freq_path.exists():
        raise FileNotFoundError(f"{freq_path} missing: run `python -m stp.cli sensitivity` first")
    with rasterio.open(freq_path) as src:
        freq = src.read(1)
    sel = cfg["selection"]
    robust_rows = []
    for t in tc["robust_core"]:
        mask = freq >= t
        polys = selection.polygonize(mask, grid.transform, crs, sel["polygonize_connectivity"])
        polys["area_sqm"] = selection.ellipsoidal_area(polys) if len(polys) else []
        sites = selection.select_sites(polys, cut, sel["min_area_sqm"], "difference_then_filter")
        sites.to_parquet(out / f"sites_robust_core_{t:g}.parquet")
        m = sensitivity.site_mask(sites, grid.shape, grid.transform)
        r = {
            "rule": f"robust_core_{t:g}",
            "baseline_sites": len(sites),
            "baseline_area_ha": float(sites["area_sqm"].sum()) / 1e4,
            "iou_vs_fixed_8": sensitivity.iou(m, baselines.get("fixed_8", m)),
        }
        robust_rows.append(r)
        print(
            f"[threshold] robust core >= {t:.0%} of draws: {r['baseline_sites']} sites, "
            f"{r['baseline_area_ha']:.1f} ha, IoU vs fixed 8 {r['iou_vs_fixed_8']:.3f}"
        )

    overlap = (
        {name: sensitivity.iou(m, baselines["fixed_8"]) for name, m in baselines.items()}
        if "fixed_8" in baselines
        else {}
    )
    payload = {
        "config": tc,
        "valid_pixels": valid_px,
        "rules": rows,
        "robust_core": robust_rows,
        "baseline_iou_vs_fixed_8": overlap,
    }
    (out / "threshold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[threshold] -> {out}")
    return payload


def _load_judgements(cfg: dict) -> dict:
    path = REPO_ROOT / cfg["ahp"]["judgements_file"]
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if set(data["criteria"]) != set(cfg["weights"]):
        raise ValueError(
            f"{path.name} criteria {sorted(data['criteria'])} != weights {sorted(cfg['weights'])}"
        )
    return data


def _coverage_of(sites_a: gpd.GeoDataFrame, mask_b: np.ndarray, grid: Grid) -> np.ndarray:
    """Share of each site in `sites_a` that is selected in `mask_b`."""
    ids = sensitivity.site_ids(sites_a, grid.shape, grid.transform)
    return sensitivity.coverage_by_site(ids, mask_b, len(sites_a))


def run_ahp(cfg: dict) -> dict:
    """AHP weights from ahp_judgements.yaml, consistency, effect on sites, questionnaire."""
    crs = CRS.from_user_input(cfg["crs"])
    outputs = Path(cfg["paths"]["output"])
    out = outputs / "phase6" / "ahp"
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid.from_config(cfg, "sentinel_10m")
    data = _load_judgements(cfg)
    criteria = list(cfg["weights"])
    a_mat = ahp.pairwise_matrix(criteria, data["judgements"])
    w, lam = ahp.priority_vector(a_mat)
    ci, cr = ahp.consistency(a_mat, lam)
    gm = ahp.geometric_mean_weights(a_mat)
    worst = ahp.most_inconsistent(a_mat, w, criteria)
    ahp_w = {c: float(x) for c, x in zip(criteria, w, strict=True)}
    print(
        f"[ahp] lambda_max {lam:.3f}, CI {ci:.4f}, CR {cr:.4f} "
        f"({'acceptable' if cr < 0.10 else 'TOO INCONSISTENT'}; threshold 0.10)"
    )
    print(f"{'criterion':12s} {'original':>9s} {'AHP':>7s} {'geo-mean':>9s}")
    for c in criteria:
        print(f"{c:12s} {cfg['weights'][c]:9.3f} {ahp_w[c]:7.3f} {gm[criteria.index(c)]:9.3f}")

    # Effect on the sites (corrected land cover), for the original rule and a top-% rule.
    layers, cut = _corrected_inputs(cfg, crs, grid)
    ranked = gpd.read_parquet(outputs / "phase6" / "sites_ranked.parquet")
    robust = ranked[ranked["robust"]].reset_index(drop=True)
    effect = []
    for rule in cfg["ahp"]["evaluate_rules"]:
        name = _rule_name(rule)
        select = _selector(layers, cut, cfg, grid, crs, rule)
        base_sites = sensitivity.run_weights(layers, cfg["weights"], select)
        ahp_sites = sensitivity.run_weights(layers, ahp_w, select)
        ahp_sites.to_parquet(out / f"sites_ahp_{name}.parquet")
        m_base = sensitivity.site_mask(base_sites, grid.shape, grid.transform)
        m_ahp = sensitivity.site_mask(ahp_sites, grid.shape, grid.transform)
        kept_robust = int((_coverage_of(robust, m_ahp, grid) >= 0.5).sum())
        row = {
            "rule": name,
            "original_sites": len(base_sites),
            "original_area_ha": float(base_sites["area_sqm"].sum()) / 1e4,
            "ahp_sites": len(ahp_sites),
            "ahp_area_ha": float(ahp_sites["area_sqm"].sum()) / 1e4,
            "iou_ahp_vs_original": sensitivity.iou(m_ahp, m_base),
            "robust_sites_kept": kept_robust,
            "robust_sites_total": len(robust),
        }
        effect.append(row)
        print(
            f"[ahp] {name}: original {row['original_sites']} sites "
            f"{row['original_area_ha']:.1f} ha -> AHP {row['ahp_sites']} sites "
            f"{row['ahp_area_ha']:.1f} ha; IoU {row['iou_ahp_vs_original']:.3f}; "
            f"robust sites kept {kept_robust}/{len(robust)}"
        )

    payload = {
        "criteria": criteria,
        "matrix": a_mat.tolist(),
        "weights_ahp": ahp_w,
        "weights_geometric_mean": {c: float(x) for c, x in zip(criteria, gm, strict=True)},
        "weights_original": cfg["weights"],
        "lambda_max": lam,
        "ci": ci,
        "cr": cr,
        "most_inconsistent": [
            {"a": a, "b": b, "judged": float(j), "implied": float(i)} for _, a, b, j, i in worst
        ],
        "effect": effect,
    }
    (out / "ahp.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    doc = REPO_ROOT / "docs" / "ahp_questionnaire.md"
    doc.write_text(_questionnaire_md(data, payload), encoding="utf-8", newline="\n")
    print(f"[ahp] -> {out}, {doc}")
    return payload


def _strength_word(s: int) -> str:
    return {
        1: "equal",
        2: "slightly",
        3: "moderately",
        4: "moderately to strongly",
        5: "strongly",
        6: "strongly to very strongly",
        7: "very strongly",
        8: "very to extremely",
        9: "extremely",
    }[s]


def _questionnaire_md(data: dict, p: dict) -> str:
    crit = data["criteria"]
    lines = [
        "# AHP questionnaire: weighting the siting criteria",
        "",
        "> Generated by `python -m stp.cli ahp` from"
        " [`ahp_judgements.yaml`](../ahp_judgements.yaml);"
        " don't edit this file. To change an answer, edit the YAML (or say which answer to"
        " change) and rerun.",
        "",
        f"**Status:** {data['status']}",
        "",
        "## How to answer",
        "",
        "For each pair, decide which criterion matters more for siting a **decentralised sewage"
        " treatment plant in Bhopal**, and how much more, on Saaty's scale:",
        "",
        "| Value | Meaning |",
        "|---|---|",
        "| 1 | Equally important |",
        "| 3 | Moderately more important |",
        "| 5 | Strongly more important |",
        "| 7 | Very strongly more important |",
        "| 9 | Extremely more important |",
        "| 2, 4, 6, 8 | In between |",
        "",
        "Judge importance **as the criteria actually vary in Bhopal**, not in general. For"
        " example, slope scores 9–10 on 96% of the city, so it hardly separates one site"
        " from another.",
        "",
        "## The criteria",
        "",
        "| Criterion | Meaning |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in crit.items()],
        "",
        "## Questions, with draft answers",
        "",
        "| # | Which matters more? | Your answer (criterion, 1–9) | Draft answer"
        " | Reason for the draft |",
        "|---|---|---|---|---|",
    ]
    for i, j in enumerate(data["judgements"], 1):
        if j["strength"] == 1:
            draft = "Equal (1)"
        else:
            draft = f"**{j['prefer']}**, {_strength_word(j['strength'])} ({j['strength']})"
        why = " ".join(str(j["why"]).split())
        lines.append(f"| {i} | `{j['a']}` or `{j['b']}` | | {draft} | {why} |")

    names = p["criteria"]
    lines += [
        "",
        "## Result of the draft answers",
        "",
        f"- Principal eigenvalue λmax = {p['lambda_max']:.3f} for n = {len(names)}",
        f"- Consistency index CI = {p['ci']:.4f}; consistency ratio **CR = {p['cr']:.4f}**"
        f" ({'below' if p['cr'] < 0.10 else 'ABOVE'} the 0.10 acceptance threshold)",
        "",
        "| Criterion | Original weight (asserted) | AHP weight (eigenvector)"
        " | Cross-check (geometric mean) |",
        "|---|---|---|---|",
        *[
            f"| `{c}` | {p['weights_original'][c]:.3f} | **{p['weights_ahp'][c]:.3f}** |"
            f" {p['weights_geometric_mean'][c]:.3f} |"
            for c in names
        ],
        "",
        "Judgements furthest from what the final weights imply (where a change would most"
        " improve consistency):",
        "",
        "| Pair | Judged (row ÷ column) | Implied by weights |",
        "|---|---|---|",
        *[
            f"| `{x['a']}` vs `{x['b']}` | {x['judged']:.2f} | {x['implied']:.2f} |"
            for x in p["most_inconsistent"]
        ],
        "",
        "### Pairwise matrix (row criterion vs column criterion)",
        "",
        "| | " + " | ".join(f"`{c}`" for c in names) + " |",
        "|---|" + "---|" * len(names),
        *[
            f"| `{r}` | " + " | ".join(_fmt_ratio(v) for v in row) + " |"
            for r, row in zip(names, p["matrix"], strict=True)
        ],
        "",
        "### Effect on the selected sites (corrected land cover)",
        "",
        "| Threshold rule | Original weights | AHP weights | Overlap (IoU) | Robust sites kept |",
        "|---|---|---|---|---|",
        *[
            f"| {e['rule']} | {e['original_sites']} sites, {e['original_area_ha']:.1f} ha |"
            f" {e['ahp_sites']} sites, {e['ahp_area_ha']:.1f} ha | {e['iou_ahp_vs_original']:.3f} |"
            f" {e['robust_sites_kept']}/{e['robust_sites_total']} |"
            for e in p["effect"]
        ],
        "",
        '"Robust sites" are the sites selected in ≥ 75% of the weight-perturbation draws;'
        ' "kept" means at least half of a robust site is still selected with the AHP weights.',
        "",
        "## Decisions on the open questions (2026-10-20)",
        "",
        "1. **Direction of the residential criterion: kept** (farther from homes = better)."
        " Caveat: all 59 corrected sites are 2.1–7.6 km (median 4.6 km) from the nearest"
        " mapped residential area, which suits a centralised plant more than a decentralised"
        " one, and OSM maps only 386 residential polygons, so true distances to housing are"
        " likely shorter. A capped version (score rising only up to a minimum separation) was"
        " suggested as a scenario.",
        "2. **Land cover stays in the weighted sum** as well as in Query 2. With the current"
        " weights Query 2 removes no pixels (every pixel scoring >= 8 is already barren), so"
        " it acts as a safety net if the weights change.",
        "",
    ]
    return "\n".join(lines)


def _fmt_ratio(v: float) -> str:
    if v >= 1:
        return f"{v:g}"
    return f"1/{round(1 / v):d}"


def _maps_links(lat: float, lon: float) -> tuple[str, str]:
    """Google Maps (Maps URLs API, satellite) and Bing Maps (aerial) links."""
    google = (
        "https://www.google.com/maps/@?api=1&map_action=map"
        f"&center={lat:.6f}%2C{lon:.6f}&zoom=18&basemap=satellite"
    )
    bing = f"https://www.bing.com/maps?cp={lat:.6f}~{lon:.6f}&lvl=18&style=a"
    return google, bing


def run_review(cfg: dict) -> None:
    """Site-check sheet: one row per site with map links and blank columns to fill in."""
    outputs = Path(cfg["paths"]["output"])
    grid = Grid.from_config(cfg, "sentinel_10m")
    ranked = gpd.read_parquet(outputs / "phase6" / "sites_ranked.parquet")
    original = gpd.read_parquet(outputs / "phase5" / "A_qgis_order_original_lulc" / "sites.parquet")
    m_orig = sensitivity.site_mask(original, grid.shape, grid.transform)
    ranked["in_original_46"] = _coverage_of(ranked, m_orig, grid) > 0
    ranked["site_id"] = [f"S{r:02d}" for r in ranked["rank"]]

    rows = []
    for _, s in ranked.iterrows():
        google, bing = _maps_links(s["lat"], s["lon"])
        rows.append(
            {
                "site_id": s["site_id"],
                "rank": int(s["rank"]),
                "area_ha": round(s["area_sqm"] / 1e4, 2),
                "robust": bool(s["robust"]),
                "persistence": round(float(s["persistence"]), 2),
                "in_original_46": bool(s["in_original_46"]),
                "suit_mean": round(float(s["suit_mean"]), 2),
                "drivable_last_mile_m": round(float(s["last_mile_m"])),
                "lat": round(float(s["lat"]), 6),
                "lon": round(float(s["lon"]), 6),
                "google_maps": google,
                "bing_aerial": bing,
                "observed_cover": "",
                "usable_for_stp": "",
                "confidence": "",
                "notes": "",
            }
        )
    review = REPO_ROOT / "review"
    review.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(review / "site_check.csv", index=False)
    kml = ranked[["site_id", "rank", "robust", "geometry"]].copy()
    kml["Name"] = [
        f"{sid} (rank {r}{', robust' if rb else ''})"
        for sid, r, rb in zip(kml["site_id"], kml["rank"], kml["robust"], strict=True)
    ]
    kml[["Name", "geometry"]].to_crs(4326).to_file(review / "sites.kml", driver="KML")
    print(f"[review] {len(rows)} sites -> {review / 'site_check.csv'}, {review / 'sites.kml'}")
