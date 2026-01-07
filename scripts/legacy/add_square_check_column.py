# scripts/legacy/add_square_check_column.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported input: {p}")


def _write_any(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
        return
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
        return
    raise ValueError(f"Unsupported output: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Add Pierre MSA_SQUARE reconstruction + diff + OK flags.")
    ap.add_argument("--in", dest="inp", required=True, type=str, help="Input file (.parquet or .csv).")
    ap.add_argument("--out", required=True, type=str, help="Output file (.parquet or .csv).")
    ap.add_argument("--suffix", default="_PIERRE", type=str, help="Suffix used in attached columns.")
    ap.add_argument("--tol", default="1e-3", type=str, help="Tolerance for OK flag (e.g. 1e-3).")
    a = ap.parse_args()

    tol = float(a.tol)
    inp = Path(a.inp)
    out = Path(a.out)

    df = _read_any(inp)

    suf = a.suffix
    col_sq = f"MSA_SQUARE{suf}"
    col_lu_a = f"MSA_LU_ART{suf}"
    col_lu_n = f"MSA_LU_NON_ART{suf}"
    col_n = f"MSA_N{suf}"
    col_e = f"MSA_ENC{suf}"
    col_r = f"MSA_ROAD{suf}"
    col_c = f"MSA_CC{suf}"

    needed = [col_sq, col_lu_a, col_lu_n, col_n, col_e, col_r, col_c]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns for check: {missing}")

    sq = df[col_sq].astype(float)
    lu_a = df[col_lu_a].astype(float)
    lu_n = df[col_lu_n].astype(float)

    n = df[col_n].astype(float).fillna(1.0)
    e = df[col_e].astype(float).fillna(1.0)
    r = df[col_r].astype(float).fillna(1.0)
    c = df[col_c].astype(float).fillna(1.0)

    recon = lu_a + (lu_n * n * e * r * c)

    df["MSA_SQUARE_RECON_PIERRE"] = recon
    df["MSA_SQUARE_DIFF_PIERRE"] = sq - recon
    df["MSA_SQUARE_ABS_DIFF_PIERRE"] = (sq - recon).abs()

    # flags
    df["MSA_SQUARE_OK_1E3"] = df["MSA_SQUARE_ABS_DIFF_PIERRE"] <= 1e-3
    df["MSA_SQUARE_OK_1E2"] = df["MSA_SQUARE_ABS_DIFF_PIERRE"] <= 1e-2
    df["MSA_SQUARE_OK_TOL"] = df["MSA_SQUARE_ABS_DIFF_PIERRE"] <= tol

    # quick QC print
    ok = np.isfinite(df["MSA_SQUARE_ABS_DIFF_PIERRE"].to_numpy())
    if ok.any():
        mx = float(np.nanmax(df["MSA_SQUARE_ABS_DIFF_PIERRE"]))
        mae = float(np.nanmean(df["MSA_SQUARE_ABS_DIFF_PIERRE"]))
        pct = float(np.nanmean(df["MSA_SQUARE_OK_TOL"]) * 100.0)
        print(f"[CHECK] tol={tol:g} | mean_abs_diff={mae:.6g} | max_abs_diff={mx:.6g} | OK%={pct:.1f}")
    else:
        print("[CHECK] No finite diffs to report.")

    _write_any(df, out)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
