# carrefour/postprocess_points.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class PostprocessConfig:
    lat_col: str = "y_latitude"
    lon_col: str = "x_longitude"
    scenario_col: str = "scenario"

    # Which road column to fix
    road_col: str = "MSA_ROAD"
    road_fill_value: float = 1.0

    # MSA columns (auto-detected if None)
    msa_square_col: str = "MSA_SQUARE"
    msa_prefix: str = "MSA_"

    # outputs
    out_dir: Path = Path("outputs/points_postprocess")
    topk: int = 30


# =============================================================================
# IO
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def read_table(path: Path) -> pd.DataFrame:
    _require_exists(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input extension: {suf}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(path, index=False)
    elif suf == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output must end with .csv or .parquet")


# =============================================================================
# Utilities
# =============================================================================

def infer_scenario_from_filename(path: Path) -> str:
    """
    Examples:
      points_with_msa_585_2100.parquet -> 585
      points_with_msa_126_2100.parquet -> 126
    Falls back to 'unknown'.
    """
    name = path.stem
    for s in ("126", "370", "585"):
        if f"_{s}_" in f"_{name}_":
            return s
        if name.endswith(f"_{s}"):
            return s
    return "unknown"


def to_float(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    return s.astype(float).to_numpy()


def detect_msa_columns(df: pd.DataFrame, cfg: PostprocessConfig) -> List[str]:
    cols = [c for c in df.columns if c.startswith(cfg.msa_prefix)]
    # ensure MSA_SQUARE first if present
    if cfg.msa_square_col in cols:
        cols = [cfg.msa_square_col] + [c for c in cols if c != cfg.msa_square_col]
    return cols


def qc_summary(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        finite = np.isfinite(x.to_numpy())
        rows.append(
            {
                "col": c,
                "n": n,
                "nan_pct": 100.0 * (1.0 - float(finite.mean())) if n else np.nan,
                "min": float(np.nanmin(x)) if finite.any() else np.nan,
                "p05": float(np.nanpercentile(x, 5)) if finite.any() else np.nan,
                "median": float(np.nanmedian(x)) if finite.any() else np.nan,
                "p95": float(np.nanpercentile(x, 95)) if finite.any() else np.nan,
                "max": float(np.nanmax(x)) if finite.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fill_road_nan_as_one(df: pd.DataFrame, cfg: PostprocessConfig) -> Tuple[pd.DataFrame, dict]:
    """
    Replace NaN in ROAD column with 1.0 (neutral factor) if the column exists.
    Returns: (df_copy, info)
    """
    out = df.copy()
    info = {"road_col_found": cfg.road_col in out.columns, "road_nan_before": None, "road_nan_after": None}

    if cfg.road_col not in out.columns:
        return out, info

    road = pd.to_numeric(out[cfg.road_col], errors="coerce")
    info["road_nan_before"] = int(road.isna().sum())

    road = road.fillna(cfg.road_fill_value).astype(float)
    out[cfg.road_col] = road

    info["road_nan_after"] = int(pd.to_numeric(out[cfg.road_col], errors="coerce").isna().sum())
    return out, info


def top_k_worst(df: pd.DataFrame, cfg: PostprocessConfig, msa_cols: Sequence[str]) -> pd.DataFrame:
    """
    Return top-k worst points by MSA_SQUARE (lowest).
    """
    if cfg.msa_square_col not in df.columns:
        raise KeyError(f"Missing {cfg.msa_square_col} in columns")

    x = pd.to_numeric(df[cfg.msa_square_col], errors="coerce")
    tmp = df.copy()
    tmp["_MSA_SQUARE_NUM"] = x
    tmp = tmp.sort_values("_MSA_SQUARE_NUM", ascending=True)
    keep_cols = [c for c in [cfg.scenario_col, cfg.lat_col, cfg.lon_col] if c in tmp.columns] + list(msa_cols)
    keep_cols = [c for c in keep_cols if c in tmp.columns]
    return tmp.loc[:, keep_cols].head(cfg.topk).drop(columns=["_MSA_SQUARE_NUM"], errors="ignore")


def log_contributions(df: pd.DataFrame, cfg: PostprocessConfig, msa_cols: Sequence[str]) -> pd.DataFrame:
    """
    Decomposition “impact” in log space:
      impact_i = -log(MSA_i)
      impact_total = -log(MSA_SQUARE)
      share_i = impact_i / impact_total
    Returns a long table with one row per point and per component.
    """
    if cfg.msa_square_col not in msa_cols:
        msa_cols = [cfg.msa_square_col] + [c for c in msa_cols if c != cfg.msa_square_col]

    # components excluding square
    comps = [c for c in msa_cols if c != cfg.msa_square_col and c.startswith(cfg.msa_prefix)]
    if not comps:
        return pd.DataFrame()

    out_rows = []
    sq = pd.to_numeric(df[cfg.msa_square_col], errors="coerce").to_numpy(dtype=float)
    sq = np.clip(sq, 1e-12, 1.0)
    impact_total = -np.log(sq)

    for comp in comps:
        v = pd.to_numeric(df[comp], errors="coerce").to_numpy(dtype=float)
        v = np.clip(v, 1e-12, 1.0)
        impact = -np.log(v)

        share = np.full_like(impact, np.nan, dtype=float)
        mask = np.isfinite(impact_total) & (impact_total > 0)
        share[mask] = impact[mask] / impact_total[mask]

        out_rows.append(
            pd.DataFrame(
                {
                    cfg.scenario_col: df.get(cfg.scenario_col, "unknown"),
                    "point_id": np.arange(len(df)),
                    "component": comp,
                    "impact": impact,
                    "impact_total": impact_total,
                    "share": share,
                }
            )
        )

    return pd.concat(out_rows, axis=0, ignore_index=True)


# =============================================================================
# Plotting (matplotlib)
# =============================================================================

def make_plots(
    df: pd.DataFrame,
    cfg: PostprocessConfig,
    msa_cols: Sequence[str],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Histogram per scenario (MSA_SQUARE)
    if cfg.msa_square_col in df.columns:
        plt.figure(figsize=(10, 5))
        for scen, g in df.groupby(cfg.scenario_col):
            x = pd.to_numeric(g[cfg.msa_square_col], errors="coerce").dropna()
            if len(x) == 0:
                continue
            plt.hist(x, bins=40, alpha=0.4, density=True, label=str(scen))
        plt.xlabel("MSA_SQUARE")
        plt.ylabel("Density")
        plt.title("Distribution of MSA_SQUARE (per scenario)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "hist_msa_square_by_scenario.png", dpi=220)
        plt.close()

        # -------- Boxplot
        plt.figure(figsize=(9, 5))
        scen_order = sorted(df[cfg.scenario_col].astype(str).unique())
        data = []
        for s in scen_order:
            x = pd.to_numeric(df.loc[df[cfg.scenario_col].astype(str) == s, cfg.msa_square_col], errors="coerce").dropna()
            data.append(x.to_numpy())
        plt.boxplot(data, labels=scen_order, showfliers=False)
        plt.xlabel("Scenario")
        plt.ylabel("MSA_SQUARE")
        plt.title("MSA_SQUARE by scenario (boxplot)")
        plt.tight_layout()
        plt.savefig(out_dir / "boxplot_msa_square_by_scenario.png", dpi=220)
        plt.close()

    # -------- Scatter vs components (MSA_SQUARE vs each component)
    comps = [c for c in msa_cols if c != cfg.msa_square_col and c.startswith(cfg.msa_prefix)]
    if cfg.msa_square_col in df.columns and comps:
        for comp in comps:
            if comp not in df.columns:
                continue
            plt.figure(figsize=(6.5, 6))
            x = pd.to_numeric(df[comp], errors="coerce")
            y = pd.to_numeric(df[cfg.msa_square_col], errors="coerce")
            m = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
            plt.scatter(x[m], y[m], s=10, alpha=0.35)
            plt.xlabel(comp)
            plt.ylabel(cfg.msa_square_col)
            plt.title(f"{cfg.msa_square_col} vs {comp}")
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_{cfg.msa_square_col}_vs_{comp}.png", dpi=220)
            plt.close()

    # -------- Contributions stacked (topK worst)
    if cfg.msa_square_col in df.columns and comps:
        top = df.copy()
        top["_sq"] = pd.to_numeric(top[cfg.msa_square_col], errors="coerce")
        top = top.sort_values("_sq", ascending=True).head(cfg.topk).reset_index(drop=True)

        # compute shares on top
        sq = np.clip(pd.to_numeric(top[cfg.msa_square_col], errors="coerce").to_numpy(float), 1e-12, 1.0)
        impact_total = -np.log(sq)
        impact_total = np.where(impact_total <= 0, np.nan, impact_total)

        shares = []
        labels = []
        for comp in comps:
            v = np.clip(pd.to_numeric(top[comp], errors="coerce").to_numpy(float), 1e-12, 1.0)
            impact = -np.log(v)
            share = impact / impact_total
            shares.append(share)
            labels.append(comp)

        shares = np.vstack(shares)  # [n_comp, topk]

        plt.figure(figsize=(12, 6))
        bottom = np.zeros(top.shape[0], dtype=float)
        x = np.arange(top.shape[0])

        for i, lab in enumerate(labels):
            y = shares[i]
            y = np.where(np.isfinite(y), y, 0.0)
            plt.bar(x, y, bottom=bottom, label=lab)
            bottom += y

        plt.xticks(x, [f"{i}" for i in x], rotation=0)
        plt.xlabel(f"Top {cfg.topk} worst points (rank index)")
        plt.ylabel("Share of total impact (log-decomposition)")
        plt.title("Decomposition of impact by pressure (top worst points)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "topk_contributions_stacked.png", dpi=220)
        plt.close()


# =============================================================================
# Main pipeline
# =============================================================================

def postprocess_points_files(
    inputs: Sequence[Path],
    cfg: PostprocessConfig,
    *,
    make_plots_flag: bool = True,
) -> Path:
    """
    Reads one or many point tables, adds scenario, fills ROAD NaNs, writes outputs and plots.
    Returns path to merged parquet.
    """
    if not inputs:
        raise ValueError("No inputs provided")

    dfs = []
    for p in inputs:
        _log(f"[INFO] reading {p}")
        df = read_table(p)
        if cfg.scenario_col not in df.columns:
            df[cfg.scenario_col] = infer_scenario_from_filename(p)
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    # Ensure lat/lon are numeric for outputs/QC
    if cfg.lat_col in merged.columns:
        merged[cfg.lat_col] = pd.to_numeric(merged[cfg.lat_col], errors="coerce")
    if cfg.lon_col in merged.columns:
        merged[cfg.lon_col] = pd.to_numeric(merged[cfg.lon_col], errors="coerce")

    # Fill road NaNs with 1
    merged2, info = fill_road_nan_as_one(merged, cfg)
    _log(f"[QC] ROAD found={info['road_col_found']} | NaN before={info['road_nan_before']} | after={info['road_nan_after']}")

    msa_cols = detect_msa_columns(merged2, cfg)

    # Outputs
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_parquet = out_dir / "points_merged_clean.parquet"
    merged_csv = out_dir / "points_merged_clean.csv"

    write_table(merged2, merged_parquet)
    write_table(merged2, merged_csv)

    # Coordinates-only table (useful to inspect quickly)
    coord_cols = [c for c in [cfg.scenario_col, cfg.lat_col, cfg.lon_col] if c in merged2.columns]
    coord_cols += [c for c in msa_cols if c in merged2.columns]
    coords_df = merged2.loc[:, coord_cols]
    write_table(coords_df, out_dir / "points_coords_and_msa.csv")

    # Summary metrics
    summary = qc_summary(merged2, msa_cols)
    write_table(summary, out_dir / "summary_msa_stats.csv")

    # Top-K worst
    if cfg.msa_square_col in merged2.columns:
        top = top_k_worst(merged2, cfg, msa_cols)
        write_table(top, out_dir / f"top{cfg.topk}_worst_points.csv")

    # Contributions table (long)
    contrib = log_contributions(merged2, cfg, msa_cols)
    if len(contrib):
        write_table(contrib, out_dir / "log_contributions_long.csv")

    # Plots
    if make_plots_flag:
        make_plots(merged2, cfg, msa_cols, out_dir)

    _log(f"[OK] wrote outputs in {out_dir}")
    return merged_parquet
