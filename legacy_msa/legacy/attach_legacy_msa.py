# carrefour/legacy/attach_legacy_msa.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# Added LU split files from Pierre (in legacy_root/MSA_LU/)
# - MSA_LU_<scen>_Arti.nc
# - MSA_LU_<scen>_non_Arti.nc
DEFAULT_VARS = [
    "MSA_SQUARE",
    "MSA_LU",
    "MSA_LU_ART",
    "MSA_LU_NON_ART",
    "MSA_ROAD",
    "MSA_CC",
    "MSA_N",
    "MSA_ENC",
]


@dataclass(frozen=True)
class AttachLegacyConfig:
    legacy_root: Path
    scen: str
    year: int
    lat_col: str
    lon_col: str
    suffix: str = "_PIERRE"
    vars: Tuple[str, ...] = tuple(DEFAULT_VARS)
    engine: Optional[str] = None
    max_full_load_mb: float = 400.0  # kept for compatibility / future use
    debug: bool = False
    # NOTE: Pierre legacy NetCDFs appear upside-down vs the 'lat' coordinate.
    # Use "flip" (default) to match real geography.
    lat_flip: str = "flip"  # "flip" | "none" | "auto"


# =============================================================================
# Logging / IO helpers
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _as_float_series(s: pd.Series) -> pd.Series:
    """Robust conversion for European decimals (comma) and stray spaces."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    ss = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(ss, errors="coerce")


def _wrap_lon_to_180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


# =============================================================================
# NetCDF open + variable selection
# =============================================================================

def _open_ds(nc_path: Path, engine: Optional[str]) -> xr.Dataset:
    try:
        return xr.open_dataset(nc_path, decode_cf=True, mask_and_scale=True, engine=engine)
    except Exception as e:
        _log(
            f"[WARN] time decode failed for {nc_path.name}. Reopening with decode_times=False. "
            f"({type(e).__name__}: {e})"
        )
        return xr.open_dataset(
            nc_path, decode_cf=False, decode_times=False, mask_and_scale=True, engine=engine
        )


def _pick_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


# =============================================================================
# Time selection
# =============================================================================

def _infer_base_year_from_units(units: str) -> Optional[int]:
    m = re.search(r"years\s+since\s+(\d{4})", units or "")
    if m:
        return int(m.group(1))
    return None


def _select_year_slice(da: xr.DataArray, year: int) -> xr.DataArray:
    if "year" in da.dims:
        y = da["year"].values.astype(int)
        if year not in y:
            nearest = int(y[np.argmin(np.abs(y - year))])
            _log(f"[WARN] year {year} not found; using nearest={nearest}")
            year = nearest
        return da.sel(year=year)

    if "time" in da.dims:
        t = da["time"]
        base_year = _infer_base_year_from_units(str(t.attrs.get("units", "")))

        # common legacy case: 86 layers => 2015..2100
        if base_year is None and t.size == 86 and (2015 <= year <= 2100):
            base_year = 2015

        if base_year is None:
            tv = np.asarray(t.values).astype(float)
            idx = int(np.nanargmin(np.abs(tv - float(year))))
            return da.isel(time=idx)

        idx = int(year - base_year)
        idx = max(0, min(idx, int(t.size) - 1))
        return da.isel(time=idx)

    return da


# =============================================================================
# Grid indexing (vectorized nearest, assumes increasing grid)
# =============================================================================

def _nearest_index_1d(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    v = np.asarray(values, dtype=float)

    if grid.size >= 2 and grid[0] > grid[-1]:
        raise ValueError("Grid axis is decreasing; _nearest_index_1d expects increasing grid.")

    idx = np.searchsorted(grid, v, side="left")
    idx = np.clip(idx, 1, len(grid) - 1)

    left = grid[idx - 1]
    right = grid[idx]
    choose_left = (v - left) <= (right - v)

    out = idx.copy()
    out[choose_left] = idx[choose_left] - 1
    return out.astype(np.int64)


def _clean_fillvalues(vals: np.ndarray) -> np.ndarray:
    out = vals.astype("float64", copy=False)
    out[out > 1e20] = np.nan  # typical legacy fill (e.g. 9.969e36)
    return out


def _qc(values: np.ndarray, label: str) -> None:
    v = np.asarray(values, dtype="float64")
    finite = np.isfinite(v)
    nan_pct = 100.0 * (1.0 - finite.mean()) if v.size else float("nan")
    if finite.any():
        mn = float(np.nanmin(v))
        mx = float(np.nanmax(v))
    else:
        mn = mx = float("nan")
    _log(f"[QC] {label}: min/max={mn:.6g}/{mx:.6g} | nan%={nan_pct:.2f}")


def _extract_at_indices(arr2d: np.ndarray, ilat: np.ndarray, ilon: np.ndarray, lat_flip: str) -> np.ndarray:
    """
    Extract arr2d at (ilat, ilon) with optional latitude flip.
    - lat_flip="none": arr2d[ilat, ilon]
    - lat_flip="flip": arr2d[nlat-1-ilat, ilon]
    - lat_flip="auto": choose mode giving more finite values
    """
    nlat = arr2d.shape[0]

    vals_none = _clean_fillvalues(arr2d[ilat, ilon])
    vals_flip = _clean_fillvalues(arr2d[(nlat - 1) - ilat, ilon])

    if lat_flip == "none":
        return vals_none
    if lat_flip == "flip":
        return vals_flip

    fin_none = int(np.isfinite(vals_none).sum())
    fin_flip = int(np.isfinite(vals_flip).sum())
    chosen = "flip" if fin_flip > fin_none else "none"
    _log(f"[INFO] lat_flip=auto -> choose '{chosen}' (finite none={fin_none} vs flip={fin_flip})")
    return vals_flip if chosen == "flip" else vals_none


# =============================================================================
# Legacy file discovery
# =============================================================================

_LU_SPLIT_VARS = {"MSA_LU_ART", "MSA_LU_NON_ART"}


def _legacy_nc_path(legacy_root: Path, var: str, scen: str) -> Path:
    """
    Resolve a legacy NetCDF path for a given var/scenario.

    Standard layout:
      legacy_root/<VAR>/<VAR>_<scen>.nc

    Special cases:
      - MSA_ROAD: sometimes 'msaroad.nc'
      - LU split: stored under legacy_root/MSA_LU with filenames:
          * MSA_LU_<scen>_Arti.nc
          * MSA_LU_<scen>_non_Arti.nc
    """
    # LU split files live inside MSA_LU folder (per your screenshot)
    if var in _LU_SPLIT_VARS:
        folder = legacy_root / "MSA_LU"
        if not folder.exists():
            raise FileNotFoundError(f"Missing legacy folder: {folder}")

        if var == "MSA_LU_ART":
            candidates = [
                folder / f"MSA_LU_{scen}_Arti.nc",
                folder / f"msalu_SSP{scen}_Arti.nc",
                folder / f"msalu_{scen}_Arti.nc",
            ]
        else:  # MSA_LU_NON_ART
            candidates = [
                folder / f"MSA_LU_{scen}_non_Arti.nc",
                folder / f"msalu_SSP{scen}_non_Arti.nc",
                folder / f"msalu_{scen}_non_Arti.nc",
            ]

        for c in candidates:
            if c.exists():
                return c

        # last resort: try any matching pattern
        pat = "*Arti.nc" if var == "MSA_LU_ART" else "*non_Arti.nc"
        hits = sorted(folder.glob(pat))
        if hits:
            _log(f"[WARN] Using fallback LU-split file for {var}: {hits[0].name}")
            return hits[0]

        raise FileNotFoundError(f"No LU-split .nc file found in {folder} for var={var} scen={scen}")

    # Standard layout
    folder = legacy_root / var
    if not folder.exists():
        raise FileNotFoundError(f"Missing legacy folder: {folder}")

    p1 = folder / f"{var}_{scen}.nc"
    if p1.exists():
        return p1

    if var.upper() == "MSA_ROAD":
        candidates = [
            folder / "msaroad.nc",
            folder / "MSA_ROAD.nc",
            folder / f"MSA_ROAD_{scen}.nc",
            folder / f"msaroad_{scen}.nc",
        ]
        for c in candidates:
            if c.exists():
                return c

    ncs = sorted(folder.glob("*.nc"))
    if ncs:
        _log(f"[WARN] Using fallback legacy file for {var}: {ncs[0].name}")
        return ncs[0]

    raise FileNotFoundError(f"No .nc file found in {folder}")


# =============================================================================
# MSA_SQUARE Pierre verification
# =============================================================================

def verify_square_pierre_points(df: pd.DataFrame, suffix: str) -> None:
    """
    Verify Pierre formula on points (when LU split is available):

      MSA_SQUARE = LU_ART + LU_NON_ART * N * ENC * ROAD * CC

    Notes:
    - We neutral-fill missing multipliers with 1.
    - We compare only rows where both square and both LU parts are finite.
    """
    col_sq = f"MSA_SQUARE{suffix}"
    col_lu_a = f"MSA_LU_ART{suffix}"
    col_lu_n = f"MSA_LU_NON_ART{suffix}"
    col_n = f"MSA_N{suffix}"
    col_e = f"MSA_ENC{suffix}"
    col_r = f"MSA_ROAD{suffix}"
    col_c = f"MSA_CC{suffix}"

    missing = [c for c in [col_sq, col_lu_a, col_lu_n, col_n, col_e, col_r, col_c] if c not in df.columns]
    if missing:
        _log(f"[VERIFY] skip: missing columns: {missing}")
        return

    sq = df[col_sq].astype(float)
    lu_a = df[col_lu_a].astype(float)
    lu_n = df[col_lu_n].astype(float)

    n = df[col_n].astype(float).fillna(1.0)
    e = df[col_e].astype(float).fillna(1.0)
    r = df[col_r].astype(float).fillna(1.0)
    c = df[col_c].astype(float).fillna(1.0)

    recon = lu_a + (lu_n * n * e * r * c)

    ok = np.isfinite(sq) & np.isfinite(recon) & np.isfinite(lu_a) & np.isfinite(lu_n)
    nn = int(ok.sum())
    if nn == 0:
        _log("[VERIFY] no finite rows to compare.")
        return

    d = (sq[ok] - recon[ok]).to_numpy()
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    mx = float(np.max(np.abs(d)))
    corr = float(pd.Series(sq[ok]).corr(pd.Series(recon[ok])))
    within_1e3 = float(np.mean(np.abs(d) <= 1e-3) * 100.0)
    within_1e2 = float(np.mean(np.abs(d) <= 1e-2) * 100.0)

    _log(f"[VERIFY] MSA_SQUARE Pierre vs reconstructed on n={nn} points")
    _log(f"         corr={corr:.6f} | MAE={mae:.6g} | RMSE={rmse:.6g} | max|diff|={mx:.6g}")
    _log(f"         |diff|<=1e-3: {within_1e3:.1f}% | |diff|<=1e-2: {within_1e2:.1f}%")


# =============================================================================
# Main attach
# =============================================================================

def attach_legacy_msa(df: pd.DataFrame, cfg: AttachLegacyConfig) -> pd.DataFrame:
    if cfg.lat_col not in df.columns or cfg.lon_col not in df.columns:
        raise ValueError(
            f"Missing lat/lon columns '{cfg.lat_col}'/'{cfg.lon_col}' in {list(df.columns)}"
        )

    out = df.copy()
    out[cfg.lat_col] = _as_float_series(out[cfg.lat_col])
    out[cfg.lon_col] = _as_float_series(out[cfg.lon_col])

    lats = out[cfg.lat_col].to_numpy(dtype=float)
    lons = _wrap_lon_to_180(out[cfg.lon_col].to_numpy(dtype=float))

    for var in cfg.vars:
        nc_path = _legacy_nc_path(cfg.legacy_root, var, cfg.scen)
        _log(f"[INFO] legacy {var}: {nc_path}")

        ds = _open_ds(nc_path, cfg.engine)
        # Important: LU split files usually have a single data_var -> _pick_var will pick it
        vname = _pick_var(ds, preferred=var)

        da = ds[vname]
        da2d = _select_year_slice(da, cfg.year)

        if set(da2d.dims) >= {"lat", "lon"}:
            da2d = da2d.transpose("lat", "lon")
        else:
            ds.close()
            raise ValueError(f"{var}: expected lat/lon dims, got dims={da2d.dims}")

        arr2d = da2d.values
        if arr2d.ndim != 2:
            ds.close()
            raise ValueError(f"{var}: expected 2D after year selection, got shape={arr2d.shape}")

        lat_grid = da2d["lat"].values
        lon_grid = da2d["lon"].values

        ilat = _nearest_index_1d(lat_grid, lats)
        ilon = _nearest_index_1d(lon_grid, lons)

        if cfg.debug:
            uniq_pairs = np.unique(np.stack([ilat, ilon], axis=1), axis=0).shape[0]
            _log(
                f"[DEBUG] {var}: unique ilat={np.unique(ilat).size}/{ilat.size}, "
                f"unique ilon={np.unique(ilon).size}/{ilon.size}, unique pairs={uniq_pairs}"
            )

        vals = _extract_at_indices(arr2d, ilat, ilon, cfg.lat_flip)

        col = f"{var}{cfg.suffix}"
        out[col] = vals
        _qc(out[col].to_numpy(), col)

        ds.close()

    return out


# =============================================================================
# Read / write points
# =============================================================================

def _read_points_any(path: Path) -> pd.DataFrame:
    _require_exists(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(path, sep=";", dtype=str)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", engine="python", dtype=str)
        return df

    raise ValueError(f"Unsupported input type: {path.suffix}")


def _write_any(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
        return

    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
        return

    raise ValueError(f"Unsupported output type: {out_path.suffix} (use .parquet or .csv)")


def run_attach_legacy_file(input_path: Path, output_path: Path, cfg: AttachLegacyConfig, verify_square: bool = False) -> None:
    _require_exists(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_points_any(input_path)
    df_out = attach_legacy_msa(df, cfg)

    if verify_square:
        verify_square_pierre_points(df_out, suffix=cfg.suffix)

    _write_any(df_out, output_path)
    _log(f"[OK] wrote {output_path}")
