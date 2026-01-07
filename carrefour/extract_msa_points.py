# carrefour/prepare/extract_msa_points.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import time


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class ExtractMSAConfig:
    scen: str
    year: int

    datamart_root: Path = Path("data/datamart")
    msasquare_path_tpl: str = "data/datamart/MSA_SQUARE/MSA_SQUARE_{scen}.nc"

    lat_col: str = "y_latitude"
    lon_col: str = "x_longitude"

    sheet: Optional[str] = None
    sep: Optional[str] = None

    include_components: bool = True

    # Performance knobs:
    # If the 2D raster is <= max_full_load_mb, we load it once then extract via NumPy (often fastest).
    max_full_load_mb: int = 300  # adjust if needed
    # Engine can matter depending on your stack; leave None unless you want to try "h5netcdf"
    engine: Optional[str] = None


# =============================================================================
# Logging / IO
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def read_points_table(path: Path, *, sheet: Optional[str] = None, sep: Optional[str] = None) -> pd.DataFrame:
    _require_exists(path)
    suf = path.suffix.lower()

    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=str)

    if suf == ".csv":
        if sep is None:
            return pd.read_csv(path, dtype=str, sep=None, engine="python")
        return pd.read_csv(path, dtype=str, sep=sep)

    raise ValueError(f"Unsupported input extension: {suf} (expected .csv or .xlsx/.xls)")


def _to_float_array(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    return s.astype(float).to_numpy()


# =============================================================================
# Xarray selection helpers
# =============================================================================

def _pick_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


def _normalize_latlon_names(da: xr.DataArray) -> xr.DataArray:
    rename: Dict[str, str] = {}
    if "lat" not in da.coords:
        for cand in ("latitude", "Latitude", "LAT", "y"):
            if cand in da.coords:
                rename[cand] = "lat"
                break
    if "lon" not in da.coords:
        for cand in ("longitude", "Longitude", "LON", "x"):
            if cand in da.coords:
                rename[cand] = "lon"
                break
    if rename:
        da = da.rename(rename)
    return da


def select_2d_slice(ds: xr.Dataset, varname: str, year: int) -> xr.DataArray:
    da = ds[varname]

    if "year" in da.dims:
        y = da["year"].values.astype(int)
        if year not in y:
            nearest = int(y[np.argmin(np.abs(y - year))])
            _log(f"[WARN] year {year} not found for {varname}, using nearest year={nearest}")
            year = nearest
        da = da.sel(year=year)

    elif "time" in da.dims:
        t = da["time"]
        if np.issubdtype(t.dtype, np.datetime64):
            target = np.datetime64(f"{year}-01-01")
            da = da.sel(time=target, method="nearest")
        elif t.dtype == object and len(t.values) and hasattr(t.values[0], "year"):
            years = np.array([getattr(x, "year", np.nan) for x in t.values], dtype=float)
            idx = int(np.nanargmin(np.abs(years - year)))
            da = da.isel(time=idx)
        else:
            tv = t.values.astype(float)
            idx = int(np.nanargmin(np.abs(tv - float(year))))
            da = da.isel(time=idx)
    else:
        raise ValueError(f"{varname}: expected 'year' or 'time' dim, got dims={da.dims}")

    da = _normalize_latlon_names(da)

    if set(da.dims) >= {"lat", "lon"}:
        da = da.transpose("lat", "lon")

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"{varname}: missing coords lat/lon (coords={list(da.coords)})")

    return da


def _maybe_convert_lon(points_lon: np.ndarray, ds_lon: np.ndarray) -> np.ndarray:
    ds_min = float(np.nanmin(ds_lon))
    ds_max = float(np.nanmax(ds_lon))
    has_negative_points = np.nanmin(points_lon) < 0.0
    if ds_min >= 0.0 and ds_max > 180.0 and has_negative_points:
        return (points_lon + 360.0) % 360.0
    return points_lon


# =============================================================================
# FAST nearest indices
# =============================================================================

def _nearest_index_1d(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    v = np.asarray(values)
    if g.ndim != 1:
        raise ValueError("grid must be 1D")

    # descending -> reverse
    if g[0] > g[-1]:
        gr = g[::-1]
        idx = np.searchsorted(gr, v, side="left")
        idx = np.clip(idx, 0, gr.size - 1)
        idx0 = np.maximum(idx - 1, 0)
        pick = np.where(np.abs(gr[idx] - v) < np.abs(gr[idx0] - v), idx, idx0)
        return (g.size - 1) - pick

    # ascending
    idx = np.searchsorted(g, v, side="left")
    idx = np.clip(idx, 0, g.size - 1)
    idx0 = np.maximum(idx - 1, 0)
    pick = np.where(np.abs(g[idx] - v) < np.abs(g[idx0] - v), idx, idx0)
    return pick


def _estimated_full_load_mb(da2d: xr.DataArray, dtype=np.float32) -> float:
    n = int(np.prod(da2d.shape))
    return (n * np.dtype(dtype).itemsize) / (1024.0 * 1024.0)


def extract_values_for_points_ultrafast(
    da2d: xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    max_full_load_mb: int,
) -> np.ndarray:
    """
    Strategy:
      - compute (lat_idx, lon_idx)
      - if raster not huge => load full 2D once and do numpy indexing (fast)
      - else => use xarray isel fancy indexing (still OK but can be slower)
    """
    da2d = _normalize_latlon_names(da2d)
    lat_grid = da2d["lat"].values
    lon_grid = da2d["lon"].values

    if lat_grid.ndim != 1 or lon_grid.ndim != 1:
        # fallback safe: xarray nearest (rare in your pipeline)
        lat_da = xr.DataArray(lats, dims="points")
        lon_da = xr.DataArray(lons, dims="points")
        out = da2d.sel(lat=lat_da, lon=lon_da, method="nearest")
        return out.values.astype(np.float32)

    lat_idx = _nearest_index_1d(lat_grid, lats)
    lon_idx = _nearest_index_1d(lon_grid, lons)

    need_mb = _estimated_full_load_mb(da2d, dtype=np.float32)

    if need_mb <= float(max_full_load_mb):
        # load full raster once, then numpy indexing
        arr = da2d.astype(np.float32).values  # this is where the I/O happens (1 big read)
        return arr[lat_idx, lon_idx].astype(np.float32)

    # fallback: vector isel (avoids loading full raster)
    out = da2d.isel(
        lat=xr.DataArray(lat_idx, dims="points"),
        lon=xr.DataArray(lon_idx, dims="points"),
    )
    return out.values.astype(np.float32)


# =============================================================================
# Discover components
# =============================================================================

def discover_component_files(datamart_root: Path, scen: str) -> List[Path]:
    out: List[Path] = []
    if not datamart_root.exists():
        return out

    for d in sorted(datamart_root.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith("MSA_"):
            continue
        if d.name == "MSA_SQUARE":
            continue
        cand = d / f"{d.name}_{scen}.nc"
        if cand.exists():
            out.append(cand)

    return out


# =============================================================================
# Main API
# =============================================================================

def attach_msa_to_points(df_points: pd.DataFrame, *, config: ExtractMSAConfig) -> pd.DataFrame:
    df = df_points.copy()

    if config.lat_col not in df.columns or config.lon_col not in df.columns:
        raise ValueError(
            f"Missing lat/lon columns: '{config.lat_col}'/'{config.lon_col}'. "
            f"Available columns={list(df.columns)}"
        )

    lats = _to_float_array(df[config.lat_col])
    lons = _to_float_array(df[config.lon_col])

    valid = np.isfinite(lats) & np.isfinite(lons)
    if not valid.all():
        _log(f"[WARN] {int((~valid).sum())} rows have invalid lat/lon -> outputs will be NaN for those rows.")

    # ---- MSA_SQUARE
    msasquare_path = Path(config.msasquare_path_tpl.format(scen=config.scen))
    _require_exists(msasquare_path)

    t0 = time.perf_counter()
    _log(f"[INFO] Reading MSA_SQUARE: {msasquare_path}")

    open_kwargs = dict(decode_times=False)
    if config.engine:
        open_kwargs["engine"] = config.engine

    with xr.open_dataset(msasquare_path, **open_kwargs) as ds_sq:
        var_sq = _pick_var(ds_sq, "MSA_SQUARE")
        da_sq = select_2d_slice(ds_sq, var_sq, year=config.year)

        lons_sq = _maybe_convert_lon(lons, da_sq["lon"].values)

        out_sq = np.full(len(df), np.nan, dtype=np.float32)
        if valid.any():
            out_sq[valid] = extract_values_for_points_ultrafast(
                da_sq,
                lats[valid],
                lons_sq[valid],
                max_full_load_mb=config.max_full_load_mb,
            )

        df["MSA_SQUARE"] = out_sq

    _log(f"[TIME] MSA_SQUARE done in {time.perf_counter() - t0:.2f}s")

    # ---- Components
    if config.include_components:
        comp_files = discover_component_files(config.datamart_root, scen=config.scen)
        if not comp_files:
            _log("[WARN] No component files found in data/datamart/MSA_*/MSA_*_{scen}.nc")
            return df

        _log(f"[INFO] Found {len(comp_files)} component file(s).")

        for p in comp_files:
            preferred = p.parent.name
            t1 = time.perf_counter()
            _log(f"[INFO] Component {preferred}: {p}")

            with xr.open_dataset(p, **open_kwargs) as ds:
                var = _pick_var(ds, preferred)
                da = select_2d_slice(ds, var, year=config.year)

                lons_use = _maybe_convert_lon(lons, da["lon"].values)

                out = np.full(len(df), np.nan, dtype=np.float32)
                if valid.any():
                    out[valid] = extract_values_for_points_ultrafast(
                        da,
                        lats[valid],
                        lons_use[valid],
                        max_full_load_mb=config.max_full_load_mb,
                    )

                df[var] = out

            _log(f"[TIME] {preferred} done in {time.perf_counter() - t1:.2f}s")

    return df


def run_extract_file(*, input_path: Path, output_path: Path, config: ExtractMSAConfig) -> Path:
    t0 = time.perf_counter()
    df = read_points_table(input_path, sheet=config.sheet, sep=config.sep)
    _log(f"[INFO] Loaded points: n={len(df)} | in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    df_out = attach_msa_to_points(df, config=config)
    _log(f"[INFO] Extraction total: {time.perf_counter() - t1:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suf = output_path.suffix.lower()
    if suf == ".parquet":
        df_out.to_parquet(output_path, index=False)
    elif suf == ".csv":
        df_out.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    _log(f"[OK] wrote {output_path} | n={len(df_out)} | cols={len(df_out.columns)}")
    return output_path
