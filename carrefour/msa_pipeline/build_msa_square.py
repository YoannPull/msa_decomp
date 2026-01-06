# scripts/build_msa_square.py
# -*- coding: utf-8 -*-

"""
Build MSA_SQUARE (combined MSA) datamart from individual pressures.

Optimized version:
- Avoid Dask chunk explosion during interp_like by:
  (1) doing all regridding with lat/lon in single chunks (lat=-1, lon=-1),
  (2) rechunking afterward for multiplication + writing.
- Skip regridding when grids already match.
- Optional "coverage mode": fill missing ROAD/CC on land with 1.0 (no pressure),
  so MSA_SQUARE is defined everywhere on land.

Usage:
  # Strict mode (propagate NaNs)
  poetry run python scripts/build_msa_square.py

  # Coverage mode (recommended for your ROAD sparsity)
  poetry run python scripts/build_msa_square.py --fill-road-na-as-one --fill-cc-na-as-one
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import xarray as xr


# =============================================================================
# 1) Paths
# =============================================================================

@dataclass(frozen=True)
class Paths:
    lu_dir: Path = Path("data/datamart/MSA_LU")
    road_dir: Path = Path("data/datamart/MSA_ROAD")
    cc_dir: Path = Path("data/datamart/MSA_CC")
    n_dir: Path = Path("data/datamart/MSA_N")
    enc_dir: Path = Path("data/datamart/MSA_ENC")
    oceans_nc: Path = Path("data/datamart/oceans.nc")
    out_dir: Path = Path("data/datamart/MSA_SQUARE")


SCENS = (126, 370, 585)

_UNITS_MONTHS_SINCE_RE = re.compile(r"^\s*months\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})", re.IGNORECASE)
_UNITS_YEARS_SINCE_RE = re.compile(r"^\s*years\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})", re.IGNORECASE)


# =============================================================================
# 2) Utilities
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _open_ds(p: Path, chunks: Optional[dict] = None) -> xr.Dataset:
    _require_exists(p)
    try:
        return xr.open_dataset(p, decode_times=True, use_cftime=True, chunks=chunks)
    except ValueError as e:
        msg = str(e)
        if "months since" in msg or "unable to decode time units" in msg:
            _log(f"[WARN] Time decode failed for {p.name}. Reopening with decode_times=False.")
            return xr.open_dataset(p, decode_times=False, chunks=chunks)
        raise


def _normalize_lon_to_180(da: xr.DataArray) -> xr.DataArray:
    if "lon" not in da.coords:
        return da
    lon = da["lon"].values
    if np.nanmin(lon) >= 0.0 and np.nanmax(lon) > 180.0:
        lon_new = ((lon + 180.0) % 360.0) - 180.0
        da = da.assign_coords(lon=lon_new).sortby("lon")
    return da


def _ensure_lat_ascending(da: xr.DataArray) -> xr.DataArray:
    if "lat" not in da.coords:
        return da
    lat = da["lat"].values
    if len(lat) >= 2 and lat[0] > lat[-1]:
        da = da.sortby("lat")
    return da


def _pick_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


def _year_from_months_since(time_vals: np.ndarray, units: str) -> np.ndarray:
    m = _UNITS_MONTHS_SINCE_RE.match(units or "")
    if m is None:
        raise ValueError(f"Unsupported 'months since' units: {units}")
    y0 = int(m.group(1))
    m0 = int(m.group(2))  # 1..12

    t = np.asarray(time_vals)
    if np.issubdtype(t.dtype, np.floating):
        t = np.round(t).astype(np.int64)
    else:
        t = t.astype(np.int64)

    abs_month = (y0 * 12 + (m0 - 1)) + t
    return (abs_month // 12).astype(np.int32)


def _year_from_years_since(time_vals: np.ndarray, units: str) -> np.ndarray:
    m = _UNITS_YEARS_SINCE_RE.match(units or "")
    if m is None:
        raise ValueError(f"Unsupported 'years since' units: {units}")
    y0 = int(m.group(1))

    t = np.asarray(time_vals)
    if np.issubdtype(t.dtype, np.floating):
        t = np.round(t).astype(np.int64)
    else:
        t = t.astype(np.int64)

    return (y0 + t).astype(np.int32)


def _to_yearly(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure output has a 'year' dimension.
    """
    if "year" in da.dims:
        return da.assign_coords(year=da["year"].astype(np.int32))

    if "time" not in da.dims:
        raise ValueError(f"Expected 'time' or 'year' dimension. Got dims={da.dims}")

    t = da["time"]

    # decoded datetime64
    if np.issubdtype(t.dtype, np.datetime64) and hasattr(t, "dt"):
        out = da.groupby(da["time"].dt.year).mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    # cftime-like objects
    if t.dtype == object and len(t.values) and hasattr(t.values[0], "year"):
        years = np.array([getattr(x, "year", np.nan) for x in t.values], dtype=np.int32)
        da2 = da.assign_coords(year=("time", years))
        out = da2.groupby("year").mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    # numeric time
    units = t.attrs.get("units", "")
    u = (units or "").lower()

    if "years since" in u:
        years = _year_from_years_since(t.values, units=units)
        da2 = da.assign_coords(year=("time", years))
        out = da2.groupby("year").mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    if "months since" in u:
        years = _year_from_months_since(t.values, units=units)
        da2 = da.assign_coords(year=("time", years))
        out = da2.groupby("year").mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    # last resort: if time already looks like years (2015..2100)
    try:
        tv = t.values.astype(np.int32)
        if np.all((tv >= 1800) & (tv <= 2500)):
            da2 = da.assign_coords(year=("time", tv))
            out = da2.groupby("year").mean("time", keep_attrs=True)
            return out.assign_coords(year=out["year"].astype(np.int32))
    except Exception:
        pass

    raise ValueError(f"Cannot convert time to year: units={units!r}")


def _infer_land_mask(oceans_nc: Path, template: xr.DataArray, fallback_from: xr.DataArray) -> xr.DataArray:
    """
    Return land mask aligned to template.
    Prefer oceans.nc if plausible; else fallback to non-NaN of fallback_from.
    """
    fallback = ~xr.ufuncs.isnan(fallback_from.interp_like(template, method="nearest"))

    if not oceans_nc.exists():
        return fallback

    try:
        ds = _open_ds(oceans_nc, chunks=None)
        var = "land" if "land" in ds.data_vars else list(ds.data_vars)[0]
        m = ds[var]
        m = _normalize_lon_to_180(_ensure_lat_ascending(m))
        m = m.interp_like(template, method="nearest")

        mv = m.values
        with np.errstate(invalid="ignore"):
            frac_ones = np.nanmean(mv == 1)
            frac_zeros = np.nanmean(mv == 0)

        def plausible(frac: float) -> bool:
            return 0.15 <= frac <= 0.60

        if plausible(frac_ones):
            _log(f"[INFO] oceans mask: land==1 (frac_ones={frac_ones:.3f})")
            return (m == 1) & (~xr.ufuncs.isnan(m))
        if plausible(frac_zeros):
            _log(f"[INFO] oceans mask: land==0 (frac_zeros={frac_zeros:.3f})")
            return (m == 0) & (~xr.ufuncs.isnan(m))

        _log("[WARN] oceans mask suspicious, fallback to raster-derived land mask.")
        return fallback
    except Exception as e:
        _log(f"[WARN] cannot use oceans mask ({oceans_nc}): {e}. Fallback to raster-derived land mask.")
        return fallback


def _safe_encoding_float32(da: xr.DataArray, max_chunks: Dict[str, int]) -> Dict[str, dict]:
    chunks = []
    for d in da.dims:
        size = int(da.sizes[d])
        target = int(max_chunks.get(d, size))
        chunks.append(min(target, size))
    return {
        da.name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": tuple(chunks),
            "_FillValue": np.float32(np.nan),
        }
    }


def _qc_summary(da: xr.DataArray, label: str) -> str:
    v = da.values
    finite = np.isfinite(v)
    nan_pct = 100.0 * (1.0 - finite.mean()) if v.size else float("nan")
    mn = float(np.nanmin(v)) if finite.any() else float("nan")
    mx = float(np.nanmax(v)) if finite.any() else float("nan")
    return f"[OK] wrote {label} | min/max={mn:.6g}/{mx:.6g} | nan%={nan_pct:.2f}% | dtype={da.dtype}"


def _load_msa_yearly(nc_path: Path, preferred_var: str, chunks: Optional[dict] = None) -> xr.DataArray:
    ds = _open_ds(nc_path, chunks=chunks)
    v = _pick_var(ds, preferred=preferred_var)
    da = ds[v]
    da = _normalize_lon_to_180(_ensure_lat_ascending(da))
    return _to_yearly(da).astype(np.float32)


def _grids_match(a: xr.DataArray, b: xr.DataArray, tol: float = 0.0) -> bool:
    if ("lat" not in a.coords) or ("lon" not in a.coords) or ("lat" not in b.coords) or ("lon" not in b.coords):
        return False
    alat, alon = a["lat"].values, a["lon"].values
    blat, blon = b["lat"].values, b["lon"].values
    if alat.shape != blat.shape or alon.shape != blon.shape:
        return False
    if tol <= 0.0:
        return np.array_equal(alat, blat) and np.array_equal(alon, blon)
    return np.allclose(alat, blat, atol=tol, rtol=0.0) and np.allclose(alon, blon, atol=tol, rtol=0.0)


def _chunk_for_interp(da: xr.DataArray, year_chunk: int = 1) -> xr.DataArray:
    # important: lat/lon in single chunks to avoid chunk explosion during interp_like
    chunks = {}
    if "year" in da.dims:
        chunks["year"] = min(year_chunk, int(da.sizes["year"]))
    if "lat" in da.dims:
        chunks["lat"] = -1
    if "lon" in da.dims:
        chunks["lon"] = -1
    return da.chunk(chunks) if chunks else da


def _chunk_for_mul(da: xr.DataArray, year_chunk: int, lat_chunk: int, lon_chunk: int) -> xr.DataArray:
    chunks = {}
    if "year" in da.dims:
        chunks["year"] = min(year_chunk, int(da.sizes["year"]))
    if "lat" in da.dims:
        chunks["lat"] = min(lat_chunk, int(da.sizes["lat"]))
    if "lon" in da.dims:
        chunks["lon"] = min(lon_chunk, int(da.sizes["lon"]))
    return da.chunk(chunks) if chunks else da


def _align_to_template(
    da: xr.DataArray,
    template: xr.DataArray,
    method: str = "nearest",
    tol: float = 0.0,
) -> xr.DataArray:
    if set(da.dims) >= {"year", "lat", "lon"}:
        da = da.transpose("year", "lat", "lon")
    if _grids_match(da, template, tol=tol):
        return da
    return da.interp_like(template, method=method)


def _fill_missing_on_land(da: xr.DataArray, land_mask: xr.DataArray, fill_value: float = 1.0) -> xr.DataArray:
    """
    Fill NaNs on land with fill_value, keep ocean as NaN.
    """
    da_land = da.where(land_mask)
    da_land = da_land.fillna(np.float32(fill_value))
    return da_land.where(land_mask)


# =============================================================================
# Build
# =============================================================================

def build_msa_square(
    paths: Paths,
    year_start: int = 2015,
    year_end: int = 2100,
    chunks_time: int = 60,
    year_chunk: int = 1,
    lat_chunk: int = 180,
    lon_chunk: int = 360,
    grid_tol: float = 0.0,
    fill_road_na_as_one: bool = False,
    fill_cc_na_as_one: bool = False,
) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    _log("== Build MSA_SQUARE (optimized) ==")
    _log(f"years      : {year_start}-{year_end}")
    _log(f"chunks mul : year={year_chunk}, lat={lat_chunk}, lon={lon_chunk}")
    _log(f"grid_tol   : {grid_tol}")
    _log(f"fill ROAD  : {fill_road_na_as_one}")
    _log(f"fill CC    : {fill_cc_na_as_one}")
    _log(f"oceans     : {paths.oceans_nc} (optional)")
    _log(f"out_dir    : {paths.out_dir}")
    _log("")

    scenario_das: List[xr.DataArray] = []

    for scen in SCENS:
        _log(f"\n-- Scenario {scen} --")

        p_lu = paths.lu_dir / f"MSA_LU_{scen}.nc"
        p_road = paths.road_dir / f"MSA_ROAD_{scen}.nc"
        p_cc = paths.cc_dir / f"MSA_CC_{scen}.nc"
        p_n = paths.n_dir / f"MSA_N_{scen}.nc"
        p_enc = paths.enc_dir / f"MSA_ENC_{scen}.nc"

        # Load yearly
        lu = _load_msa_yearly(p_lu, preferred_var="MSA_LU", chunks={"time": chunks_time})
        road = _load_msa_yearly(p_road, preferred_var="MSA_ROAD", chunks={"time": chunks_time})
        cc = _load_msa_yearly(p_cc, preferred_var="MSA_CC", chunks={"time": chunks_time})
        n = _load_msa_yearly(p_n, preferred_var="MSA_N", chunks=None)
        enc = _load_msa_yearly(p_enc, preferred_var="MSA_ENC", chunks={"time": chunks_time})

        # Year window
        lu = lu.sel(year=slice(year_start, year_end))
        road = road.sel(year=slice(year_start, year_end))
        cc = cc.sel(year=slice(year_start, year_end))
        n = n.sel(year=slice(year_start, year_end))
        enc = enc.sel(year=slice(year_start, year_end))

        # Common years
        y = np.intersect1d(lu["year"].values, road["year"].values)
        y = np.intersect1d(y, cc["year"].values)
        y = np.intersect1d(y, n["year"].values)
        y = np.intersect1d(y, enc["year"].values)
        if y.size == 0:
            raise ValueError(f"No overlapping years across pressures for scen {scen}.")

        lu = lu.sel(year=y)

        # Template for interpolation: single-chunk lat/lon
        template_interp = _chunk_for_interp(lu, year_chunk=year_chunk)

        # Land mask ONCE on template grid
        land_mask = _infer_land_mask(
            paths.oceans_nc,
            template=template_interp.isel(year=0),
            fallback_from=lu.isel(year=0),
        )

        # Prepare other pressures for interpolation: single-chunk lat/lon
        road_i = _chunk_for_interp(road.sel(year=y), year_chunk=year_chunk)
        cc_i = _chunk_for_interp(cc.sel(year=y), year_chunk=year_chunk)
        n_i = _chunk_for_interp(n.sel(year=y), year_chunk=year_chunk)
        enc_i = _chunk_for_interp(enc.sel(year=y), year_chunk=year_chunk)

        # Align (skip if same grid)
        road_i = _align_to_template(road_i, template_interp, method="nearest", tol=grid_tol)
        cc_i = _align_to_template(cc_i, template_interp, method="nearest", tol=grid_tol)
        n_i = _align_to_template(n_i, template_interp, method="nearest", tol=grid_tol)
        enc_i = _align_to_template(enc_i, template_interp, method="nearest", tol=grid_tol)

        # Coverage mode: fill missing on land as "no pressure"
        if fill_road_na_as_one:
            road_i = _fill_missing_on_land(road_i, land_mask, fill_value=1.0)
        if fill_cc_na_as_one:
            cc_i = _fill_missing_on_land(cc_i, land_mask, fill_value=1.0)

        # Rechunk for multiplication/writing
        template = _chunk_for_mul(lu, year_chunk, lat_chunk, lon_chunk)
        road_m = _chunk_for_mul(road_i, year_chunk, lat_chunk, lon_chunk)
        cc_m = _chunk_for_mul(cc_i, year_chunk, lat_chunk, lon_chunk)
        n_m = _chunk_for_mul(n_i, year_chunk, lat_chunk, lon_chunk)
        enc_m = _chunk_for_mul(enc_i, year_chunk, lat_chunk, lon_chunk)

        # Combine
        msa = (template * road_m * cc_m * n_m * enc_m).astype(np.float32)
        msa = msa.where(land_mask)
        msa = msa.where(np.isfinite(msa), np.nan).clip(min=0.0, max=1.0)
        msa.name = "MSA_SQUARE"

        # Write per scenario
        out_path = paths.out_dir / f"MSA_SQUARE_{scen}.nc"
        ds_out = msa.to_dataset()

        enc_nc = _safe_encoding_float32(msa, max_chunks={"year": 100, "lat": 360, "lon": 720})
        ds_out.to_netcdf(out_path, encoding=enc_nc)
        _log(_qc_summary(ds_out["MSA_SQUARE"], str(out_path)))

        scenario_das.append(msa.assign_coords(scenario=np.int32(scen)).expand_dims("scenario"))

    # Combined file
    msa_all = xr.concat(scenario_das, dim="scenario")
    out_all = paths.out_dir / "MSA_SQUARE.nc"
    ds_all = msa_all.to_dataset()

    enc_all = _safe_encoding_float32(msa_all, max_chunks={"scenario": 3, "year": 100, "lat": 360, "lon": 720})
    ds_all.to_netcdf(out_all, encoding=enc_all)
    _log(_qc_summary(ds_all["MSA_SQUARE"], str(out_all)))

    _log("\n[DONE] MSA_SQUARE datamart built.")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MSA_SQUARE datamart (combined MSA).")

    p.add_argument("--year-start", type=int, default=2015)
    p.add_argument("--year-end", type=int, default=2100)
    p.add_argument("--chunks-time", type=int, default=60)

    p.add_argument("--year-chunk", type=int, default=1)
    p.add_argument("--lat-chunk", type=int, default=180)
    p.add_argument("--lon-chunk", type=int, default=360)
    p.add_argument("--grid-tol", type=float, default=0.0)

    p.add_argument("--fill-road-na-as-one", action="store_true", help="Fill ROAD NaNs on land with 1.0.")
    p.add_argument("--fill-cc-na-as-one", action="store_true", help="Fill CC NaNs on land with 1.0.")

    p.add_argument("--lu-dir", type=str, default=str(Paths.lu_dir))
    p.add_argument("--road-dir", type=str, default=str(Paths.road_dir))
    p.add_argument("--cc-dir", type=str, default=str(Paths.cc_dir))
    p.add_argument("--n-dir", type=str, default=str(Paths.n_dir))
    p.add_argument("--enc-dir", type=str, default=str(Paths.enc_dir))

    p.add_argument("--oceans-nc", type=str, default=str(Paths.oceans_nc))
    p.add_argument("--out-dir", type=str, default=str(Paths.out_dir))

    return p.parse_args()


def main() -> None:
    a = _parse_args()
    paths = Paths(
        lu_dir=Path(a.lu_dir),
        road_dir=Path(a.road_dir),
        cc_dir=Path(a.cc_dir),
        n_dir=Path(a.n_dir),
        enc_dir=Path(a.enc_dir),
        oceans_nc=Path(a.oceans_nc),
        out_dir=Path(a.out_dir),
    )

    build_msa_square(
        paths=paths,
        year_start=a.year_start,
        year_end=a.year_end,
        chunks_time=a.chunks_time,
        year_chunk=a.year_chunk,
        lat_chunk=a.lat_chunk,
        lon_chunk=a.lon_chunk,
        grid_tol=a.grid_tol,
        fill_road_na_as_one=a.fill_road_na_as_one,
        fill_cc_na_as_one=a.fill_cc_na_as_one,
    )


if __name__ == "__main__":
    main()
