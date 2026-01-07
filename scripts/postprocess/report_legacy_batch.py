# scripts/report_legacy_batch.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------

MSA_TOTAL = "MSA_SQUARE_PIERRE"

PRESSURES = [
    "MSA_LU_PIERRE",
    "MSA_LU_ART_PIERRE",
    "MSA_LU_NON_ART_PIERRE",
    "MSA_ROAD_PIERRE",
    "MSA_CC_PIERRE",
    "MSA_N_PIERRE",
    "MSA_ENC_PIERRE",
]

DEFAULT_ID_COLS = [
    "siren",
    "siret",
    "plg_code_commune",
    "y_latitude",
    "x_longitude",
    "etatAdministratifEtablissement",
]

ACTION_SCOPE = {
    "MSA_LU_NON_ART_PIERRE": "Direct",
    "MSA_ENC_PIERRE": "Direct",
    "MSA_LU_ART_PIERRE": "Influence",
    "MSA_N_PIERRE": "Influence",
    "MSA_CC_PIERRE": "Influence",
    "MSA_ROAD_PIERRE": "Context",
    "MSA_LU_PIERRE": "Mixed",
}

RECO = {
    "MSA_LU_NON_ART_PIERRE": [
        "Éviter toute extension / nouvelle emprise dans les zones les plus exposées (signal 2100).",
        "Désimperméabiliser/renaturer les abords (parkings, friches), trame verte, continuités.",
        "Critères biodiversité dans l’arbitrage foncier + objectif ZAN (zéro artificialisation nette).",
    ],
    "MSA_ENC_PIERRE": [
        "Buffers végétalisés, haies, gestion écologique des abords, entretien différencié.",
        "Réduire nuisances (éclairage nocturne, bruit) et améliorer connectivité locale.",
        "Plan de gestion biodiversité site-level avec KPI simples.",
    ],
    "MSA_N_PIERRE": [
        "Exigences fournisseurs : réduire excédents N (plans de fertilisation, couverts, rotations).",
        "Accompagnement technique / contractualisation de pratiques agroécologiques.",
        "Prioriser les zones/filières où la contribution N est dominante (effet maximal).",
    ],
    "MSA_LU_ART_PIERRE": [
        "Politique 'zéro conversion' + traçabilité pour matières à risque (conversion d’habitats).",
        "Sourcing vers zones à moindre enjeu + restauration/agroforesterie.",
        "Programmes filières : incitations, audits, KPIs biodiversité amont.",
    ],
    "MSA_CC_PIERRE": [
        "Aligner avec plan climat (efficacité, ENR, électrification) – co-bénéfices biodiversité.",
        "Piloter par scénarios (sensibilité), prioriser actions robustes.",
        "Réduire postes à co-bénéfices (énergie/transport/sols) selon périmètre.",
    ],
    "MSA_ROAD_PIERRE": [
        "À traiter comme 'contexte' : éviter nouvelles implantations/expansions en zones très fragmentées.",
        "Optimiser flux si cela réduit pressions associées (à cadrer).",
        "Mesures locales de connectivité si projets fonciers (corridors/haies/buffers).",
    ],
    "MSA_LU_PIERRE": [
        "Diagnostiquer LU via LU_ART vs LU_NON_ART pour choisir levier (amont vs site).",
        "Éviter conversion + soutenir restauration/gestion durable dans zones prioritaires.",
        "Critères biodiversité dans achats/foncier selon le sous-driver dominant.",
    ],
}

FNAME_RE = re.compile(
    r"points_with_legacy_(?P<scen>\d+)_(?P<year>\d{4})(?P<checked>_checked)?\.(?P<ext>parquet|csv)$"
)


# -----------------------------
# Helpers
# -----------------------------

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[m] = a[m] / b[m]
    return out

def mk_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)
    (outdir / "fig").mkdir(parents=True, exist_ok=True)

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file: {path}")

def pick_reco(msa_col: str) -> str:
    items = RECO.get(msa_col, [])
    return " | ".join(items[:3]) if items else ""


# -----------------------------
# Enrichment
# -----------------------------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    ensure_cols(df, [MSA_TOTAL] + PRESSURES)
    out = df.copy()

    out["gap_total"] = 1.0 - out[MSA_TOTAL].astype("float64")

    # gaps for pressures
    for p in PRESSURES:
        key = p.replace("MSA_", "").replace("_PIERRE", "").lower()
        out[f"gap_{key}"] = 1.0 - out[p].astype("float64")

    gap_cols = [c for c in out.columns if c.startswith("gap_") and c != "gap_total"]
    out["gap_sum"] = out[gap_cols].sum(axis=1, skipna=True)

    # shares
    for p in PRESSURES:
        key = p.replace("MSA_", "").replace("_PIERRE", "").lower()
        out[f"share_{key}"] = safe_div(
            out[f"gap_{key}"].to_numpy(dtype="float64"),
            out["gap_sum"].to_numpy(dtype="float64"),
        )

    # drivers
    share_cols = [c for c in out.columns if c.startswith("share_")]
    mat = out[share_cols].to_numpy(dtype="float64")
    mat2 = np.where(np.isfinite(mat), mat, -np.inf)

    top1 = np.argmax(mat2, axis=1)
    mat2_bis = mat2.copy()
    rows = np.arange(mat2.shape[0])
    mat2_bis[rows, top1] = -np.inf
    top2 = np.argmax(mat2_bis, axis=1)

    out["driver_1"] = [share_cols[i].replace("share_", "").upper() for i in top1]
    out["driver_2"] = [share_cols[i].replace("share_", "").upper() for i in top2]

    out["driver_1_msa_col"] = "MSA_" + out["driver_1"] + "_PIERRE"
    out["driver_2_msa_col"] = "MSA_" + out["driver_2"] + "_PIERRE"

    out["scope_1"] = out["driver_1_msa_col"].map(ACTION_SCOPE).fillna("Unknown")
    out["scope_2"] = out["driver_2_msa_col"].map(ACTION_SCOPE).fillna("Unknown")

    # business-friendly indices (projection -> "signal d’exposition", pas un %)
    out["risk_artif_2100"] = 1.0 - out["MSA_LU_NON_ART_PIERRE"].astype("float64")
    out["risk_n_2100"] = 1.0 - out["MSA_N_PIERRE"].astype("float64")

    # priority flags
    out["priority_top5pct"] = out["gap_total"] >= out["gap_total"].quantile(0.95)
    out["priority_top10pct"] = out["gap_total"] >= out["gap_total"].quantile(0.90)

    out["recommendations_driver1"] = out["driver_1_msa_col"].apply(pick_reco)
    out["recommendations_driver2"] = out["driver_2_msa_col"].apply(pick_reco)

    return out


# -----------------------------
# Tables
# -----------------------------

def make_tables(df: pd.DataFrame, id_cols: List[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    keep = [c for c in id_cols if c in df.columns]
    # keep any "check" columns if present
    keep += [c for c in df.columns if "check" in c.lower() or "match" in c.lower() or "dist" in c.lower()]

    keep += [MSA_TOTAL] + [p for p in PRESSURES if p in df.columns]
    keep += [
        "gap_total",
        "gap_sum",
        "driver_1",
        "driver_2",
        "scope_1",
        "scope_2",
        "risk_artif_2100",
        "risk_n_2100",
        "priority_top5pct",
        "priority_top10pct",
        "recommendations_driver1",
        "recommendations_driver2",
    ]
    keep += [c for c in df.columns if c.startswith("share_")]
    keep = [c for c in keep if c in df.columns]

    tables["sites_enriched"] = df[keep].copy()

    tables["top_hotspots_200"] = (
        df.sort_values("gap_total", ascending=False)
          .loc[:, keep]
          .head(200)
          .copy()
    )

    tables["driver_distribution"] = (
        df["driver_1"]
          .value_counts(dropna=False)
          .rename_axis("driver_1")
          .reset_index(name="count")
          .assign(share=lambda x: x["count"] / x["count"].sum())
    )

    metrics = [MSA_TOTAL, "gap_total", "risk_artif_2100", "risk_n_2100"]
    tables["summary_overall"] = (
        df[metrics]
          .describe(percentiles=[0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95])
          .T
          .reset_index()
          .rename(columns={"index": "metric"})
    )

    if "plg_code_commune" in df.columns:
        agg_cols = [MSA_TOTAL, "gap_total", "risk_artif_2100"]
        g = df.groupby("plg_code_commune", dropna=False)[agg_cols].agg(["count", "mean", "median"])
        g.columns = ["_".join(c) for c in g.columns.to_flat_index()]
        tables["by_commune"] = g.reset_index().sort_values("gap_total_mean", ascending=False)

    if "etatAdministratifEtablissement" in df.columns:
        agg_cols = [MSA_TOTAL, "gap_total", "risk_artif_2100"]
        g = df.groupby("etatAdministratifEtablissement", dropna=False)[agg_cols].agg(["count", "mean", "median"])
        g.columns = ["_".join(c) for c in g.columns.to_flat_index()]
        tables["by_status"] = g.reset_index().sort_values("gap_total_mean", ascending=False)

    return tables


# -----------------------------
# Figures
# -----------------------------

def fig_hist(df: pd.DataFrame, outpath: Path, col: str, title: str, bins: int = 40) -> None:
    x = df[col].astype("float64").to_numpy()
    x = x[np.isfinite(x)]
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_bar(df_counts: pd.DataFrame, outpath: Path, xcol: str, ycol: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(df_counts[xcol].astype(str), df_counts[ycol].to_numpy())
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_pareto(df: pd.DataFrame, outpath: Path) -> None:
    x = df["gap_total"].astype("float64").to_numpy()
    x = x[np.isfinite(x)]
    x = np.sort(x)[::-1]
    cum = np.cumsum(x)
    cum_share = cum / (cum[-1] if cum[-1] != 0 else 1.0)
    idx = np.arange(1, len(x) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(idx, cum_share)
    plt.title("Pareto – part cumulée du gap_total")
    plt.xlabel("Nombre de sites (triés du plus exposé au moins exposé)")
    plt.ylabel("Part cumulée")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_scatter(df: pd.DataFrame, outpath: Path, x: str, y: str, title: str) -> None:
    xx = df[x].astype("float64").to_numpy()
    yy = df[y].astype("float64").to_numpy()
    m = np.isfinite(xx) & np.isfinite(yy)

    plt.figure(figsize=(10, 6))
    plt.scatter(xx[m], yy[m], s=10, alpha=0.5)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def fig_map_points(df: pd.DataFrame, outpath: Path, title: str = "") -> None:
    if "x_longitude" not in df.columns or "y_latitude" not in df.columns:
        return
    lon = df["x_longitude"].astype("float64").to_numpy()
    lat = df["y_latitude"].astype("float64").to_numpy()
    z = df["gap_total"].astype("float64").to_numpy()
    m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(z)

    plt.figure(figsize=(10, 10))
    sc = plt.scatter(lon[m], lat[m], c=z[m], s=10, alpha=0.7)
    plt.title(title or "Carte simple – hotspots (gap_total)")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    cb = plt.colorbar(sc)
    cb.set_label("gap_total = 1 - MSA_SQUARE")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------
# Cross-scenario sensitivity
# -----------------------------

def cross_scenario_report(dfs: Dict[Tuple[int, int], pd.DataFrame], outdir: Path, key_cols: List[str]) -> None:
    """
    dfs: {(scen, year): enriched_df}
    We build per-site scenario spread for MSA_SQUARE and risk_artif_2100.
    """
    # choose keys that exist
    keys = [c for c in key_cols if all(c in d.columns for d in dfs.values())]
    if not keys:
        return

    # build wide tables
    parts = []
    for (scen, year), d in dfs.items():
        small = d[keys + [MSA_TOTAL, "risk_artif_2100", "gap_total"]].copy()
        small = small.rename(columns={
            MSA_TOTAL: f"{MSA_TOTAL}_scen{scen}",
            "risk_artif_2100": f"risk_artif_2100_scen{scen}",
            "gap_total": f"gap_total_scen{scen}",
        })
        parts.append(small)

    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on=keys, how="inner")

    # compute spreads
    msa_cols = [c for c in wide.columns if c.startswith(f"{MSA_TOTAL}_scen")]
    art_cols = [c for c in wide.columns if c.startswith("risk_artif_2100_scen")]
    gap_cols = [c for c in wide.columns if c.startswith("gap_total_scen")]

    wide["msa_square_range"] = wide[msa_cols].max(axis=1) - wide[msa_cols].min(axis=1)
    wide["risk_artif_range"] = wide[art_cols].max(axis=1) - wide[art_cols].min(axis=1)
    wide["gap_total_range"] = wide[gap_cols].max(axis=1) - wide[gap_cols].min(axis=1)

    (outdir / "csv").mkdir(parents=True, exist_ok=True)
    (outdir / "fig").mkdir(parents=True, exist_ok=True)

    wide.to_csv(outdir / "csv" / "cross_scenario_sensitivity_wide.csv", index=False)

    # fig: hist ranges
    fig_hist(wide, outdir / "fig" / "hist_msa_square_range.png", "msa_square_range",
             "Sensibilité au scénario – range de MSA_SQUARE")
    fig_hist(wide, outdir / "fig" / "hist_risk_artif_range.png", "risk_artif_range",
             "Sensibilité au scénario – range du risque artificialisation (1 - MSA_LU_NON_ART)")

    # fig: boxplot of MSA across scenarios
    plt.figure(figsize=(10, 6))
    data = [wide[c].astype("float64").to_numpy() for c in msa_cols]
    plt.boxplot(data, labels=[c.replace(f"{MSA_TOTAL}_", "") for c in msa_cols], showfliers=False)
    plt.title("Distribution MSA_SQUARE par scénario (sites communs)")
    plt.ylabel("MSA_SQUARE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "fig" / "box_msa_square_by_scenario.png", dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="outputs/output_legacy", help="Directory containing points_with_legacy_*.parquet/csv")
    ap.add_argument("--out-root", default="outputs/report_legacy", help="Root output directory")
    ap.add_argument("--year", type=int, default=2100, help="Filter by year (default 2100)")
    ap.add_argument("--checked-only", type=int, default=1, help="1 => only *_checked.*, 0 => both")
    ap.add_argument("--id-cols", default=",".join(DEFAULT_ID_COLS))
    ap.add_argument("--glob", default="points_with_legacy_*", help="Glob pattern within in-dir")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_root)
    year_filter = int(args.year)
    checked_only = bool(int(args.checked_only))
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found in {in_dir} with glob={args.glob}")

    # collect enriched dfs per (scen, year)
    enriched: Dict[Tuple[int, int], pd.DataFrame] = {}

    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            continue

        scen = int(m.group("scen"))
        year = int(m.group("year"))
        is_checked = m.group("checked") is not None

        if year != year_filter:
            continue
        if checked_only and not is_checked:
            continue

        df = read_any(f)
        df2 = enrich(df)

        outdir = out_root / f"scen_{scen}_year_{year}" / ("checked" if is_checked else "raw")
        mk_outdir(outdir)

        # Write enriched full dataset
        df2.to_parquet(outdir / "csv" / "sites_enriched.parquet", index=False)
        df2.to_csv(outdir / "csv" / "sites_enriched.csv", index=False)

        # Tables
        tables = make_tables(df2, id_cols=id_cols)
        for name, t in tables.items():
            t.to_csv(outdir / "csv" / f"{name}.csv", index=False)

        # Figures
        fig_hist(df2, outdir / "fig" / "hist_msa_square.png", MSA_TOTAL, f"Distribution {MSA_TOTAL} (scen {scen}, {year})")
        fig_hist(df2, outdir / "fig" / "hist_risk_artif.png", "risk_artif_2100", f"Risque artificialisation (scen {scen}, {year})")
        fig_bar(tables["driver_distribution"], outdir / "fig" / "bar_driver1.png", "driver_1", "count",
                f"Driver principal (scen {scen}, {year})")
        fig_pareto(df2, outdir / "fig" / "pareto_gap_total.png")
        fig_scatter(df2, outdir / "fig" / "scatter_gap_vs_artif.png",
                    "risk_artif_2100", "gap_total",
                    f"Risque artificialisation vs gap_total (scen {scen}, {year})")
        fig_map_points(df2, outdir / "fig" / "map_hotspots.png", title=f"Hotspots gap_total – scen {scen}, {year}")

        # README (interprétation)
        (outdir / "README.txt").write_text(
            "Interprétation (projection 2100):\n"
            "- risk_artif_2100 = 1 - MSA_LU_NON_ART : index d’exposition / risque, PAS un % d’urbanisation.\n"
            "- À utiliser pour prioriser: (i) choix d’implantation/extension, (ii) actions site-level (renaturation),\n"
            "  et distinguer ce qui est pilotable (Direct/Influence) du contexte.\n",
            encoding="utf-8"
        )

        enriched[(scen, year)] = df2
        print(f"[OK] scen={scen} year={year} checked={is_checked} -> {outdir}")

    # Cross-scenario (if multiple scenarios exist)
    same_year = {(scen, year): df for (scen, year), df in enriched.items() if year == year_filter}
    if len(same_year) >= 2:
        cross_dir = out_root / f"cross_year_{year_filter}" / ("checked" if checked_only else "mixed")
        mk_outdir(cross_dir)
        cross_scenario_report(same_year, cross_dir, key_cols=["siret", "siren", "x_longitude", "y_latitude"])
        print(f"[OK] cross-scenario report -> {cross_dir}")
    else:
        print("[INFO] cross-scenario report skipped (need >=2 scenarios for the same year)")

    print("[DONE]")


if __name__ == "__main__":
    main()
