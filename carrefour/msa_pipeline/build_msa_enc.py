# scripts/build_msa_enc.py
# -*- coding: utf-8 -*-

"""
Build MSA_ENC (Encroachment / artificialisation pressure) datamart.

Robust pipeline version (xarray + rasterio) in the same style as build_msa_n.py.

Core logic (matches notebook intent)
------------------------------------
Inputs (per scenario):
- LUH2 land-use fractions from SSP*.nc
  Required variables (summed to build "artificialisation"):
    pastr + range + urban + c3per + c4ann + c3ann + c4per + c3nfx
- Protected areas rasters:
    PA_0.tif and PA_1.tif (same grid OR regridded to LUH grid using nearest)
- Optional oceans mask:
    data/datamart/oceans.nc  (land==1 or land==0; auto-detected)
    If unusable, fallback to "non-NaN" of LUH template.

Steps:
1) Open SSP*.nc with robust time decoding
2) Build arti = sum(required LU fractions)
3) Ensure a yearly coordinate "year"
   - If time is decoded -> use time.dt.year (and yearly mean if needed)
   - If numeric time with 'years since' or 'months since' -> compute years manually
   - If already "year" dim -> passthrough
4) Read PA_0 + PA_1 GeoTIFF -> protected_mask = (PA_0 + PA_1) > 0
   - Convert rasters to DataArray with lat/lon coords
   - Normalize lon to [-180, 180], ensure lat ascending
   - Regrid protected_mask to LUH grid via interp_like(nearest)
5) For each year (vectorized):
   protected -> MSA=1
   non-protected:
     arti<=0 -> 1
     0<arti<=thr -> linear from 1 to msa_min
     arti>thr -> msa_min
   Clip [0,1], set invalid->1
6) Apply land mask (oceans -> NaN)
7) Save per scenario + combined file (scenario stacked)

Outputs:
- data/datamart/MSA_ENC/MSA_ENC_126.nc
- data/datamart/MSA_ENC/MSA_ENC_370.nc
- data/datamart/MSA_ENC/MSA_ENC_585.nc
- data/datamart/MSA_ENC/MSA_ENC.nc

Usage:
poetry run python scripts/build_msa_enc.py
or with overrides:
poetry run python scripts/build_msa_enc.py --ssp126 data/legacy/MSA_ENC/SSP126.nc --pa0 data/legacy/MSA_ENC/PA_0.tif --pa1 data/legacy/MSA_ENC/PA_1.tif
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import xarray as xr

try:
    import rasterio
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "rasterio is required to read PA GeoTIFFs. Add it to your deps (poetry add rasterio)."
    ) from e


# =============================================================================
# 1) Required variables (artificialisation components)
# =============================================================================

REQUIRED_ARTI_VARS: Tuple[str, ...] = (
    "pastr",
    "range",
    "urban",
    "c3per",
    "c4ann",
    "c3ann",
    "c4per",
    "c3nfx",
)


# =============================================================================
# 2) Paths (defaults; override via CLI)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    # LUH2 land-use fractions
    ssp126: Path = Path("data/legacy/MSA_ENC/SSP126.nc")
    ssp370: Path = Path("data/legacy/MSA_ENC/SSP370.nc")
    ssp585: Path = Path("data/legacy/MSA_ENC/SSP585.nc")

    # Protected areas
    pa0_tif: Path = Path("data/legacy/MSA_ENC/PA_0.tif")
    pa1_tif: Path = Path("data/legacy/MSA_ENC/PA_1.tif")

    # Optional oceans mask
    oceans_nc: Path = Path("data/datamart/oceans.nc")

    # Output
    out_dir: Path = Path("data/datamart/MSA_ENC")


SCEN_TO_FILE = {
    126: "ssp126",
    370: "ssp370",
    585: "ssp585",
}


# =============================================================================
# 3) Utilities (same spirit as build_msa_n.py)
# =============================================================================

_UNITS_MONTHS_SINCE_RE = re.compile(r"^\s*months\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})", re.IGNORECASE)
_UNITS_YEARS_SINCE_RE = re.compile(r"^\s*years\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})", re.IGNORECASE)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _open_ds(p: Path, chunks: Optional[dict] = None) -> xr.Dataset:
    """
    Robust open for weird CF-time encodings.
    If time decoding fails (e.g., 'months since ...' with 365_day), reopen decode_times=False.
    """
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
    year = (abs_month // 12).astype(np.int32)
    return year


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
    Ensure the output has a 'year' dimension.
    - If already 'year' dim -> passthrough
    - If 'time' decoded -> groupby time.dt.year and mean over time
    - If numeric time with 'years since' or 'months since' -> manual and mean
    """
    if "year" in da.dims:
        return da.assign_coords(year=da["year"].astype(np.int32))

    if "time" not in da.dims:
        # some LUH files may already be (t, lat, lon) with a non-standard dim; we keep strict
        raise ValueError(f"Expected 'time' or 'year' dimension. Got dims={da.dims}")

    # decoded time
    if hasattr(da["time"], "dt"):
        out = da.groupby(da["time"].dt.year).mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    # numeric time
    units = da["time"].attrs.get("units", "")
    u = (units or "").lower()

    if "years since" in u:
        years = _year_from_years_since(da["time"].values, units=units)
        da2 = da.assign_coords(year=("time", years))
        out = da2.groupby("year").mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    if "months since" in u:
        years = _year_from_months_since(da["time"].values, units=units)
        da2 = da.assign_coords(year=("time", years))
        out = da2.groupby("year").mean("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    raise ValueError(f"Cannot convert time to year: units={units!r}")


def _read_tif_as_da(path: Path, name: str) -> xr.DataArray:
    """
    Read a GeoTIFF as a DataArray with lat/lon coordinates (cell centers).
    """
    _require_exists(path)
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        width, height = src.width, src.height

        xs = (np.arange(width) + 0.5) * transform.a + transform.c
        ys = (np.arange(height) + 0.5) * transform.e + transform.f

        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"x": xs.astype(np.float32), "y": ys.astype(np.float32)},
            name=name,
        )

    da = da.rename({"x": "lon", "y": "lat"})
    da = _normalize_lon_to_180(_ensure_lat_ascending(da))
    return da


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
    """
    NetCDF encoding with chunk sizes in da.dims order, capped by dim sizes (prevents netCDF4 crash).
    """
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


# =============================================================================
# 4) Build
# =============================================================================

def build_msa_enc(
    paths: Paths,
    year_start: int = 2015,
    year_end: int = 2100,
    threshold: float = 0.015,
    msa_min: float = 0.85,
    chunks_time: int = 50,
) -> None:
    """
    Build MSA_ENC per scenario + combined file.

    threshold: artificialisation threshold (e.g., 0.015)
    msa_min  : plateau MSA when arti > threshold (e.g., 0.85)
    """
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    _log("== Build MSA_ENC ==")
    _log(f"pa0      : {paths.pa0_tif}")
    _log(f"pa1      : {paths.pa1_tif}")
    _log(f"oceans   : {paths.oceans_nc} (optional)")
    _log(f"out_dir  : {paths.out_dir}")
    _log(f"years    : {year_start}-{year_end}")
    _log(f"thr      : {threshold}")
    _log(f"msa_min  : {msa_min}")
    _log("")

    # Read PA rasters once
    pa0 = _read_tif_as_da(paths.pa0_tif, name="PA_0")
    pa1 = _read_tif_as_da(paths.pa1_tif, name="PA_1")
    protected_raster = (pa0 + pa1) > 0
    protected_raster = protected_raster.astype(np.uint8)
    protected_raster.name = "protected"

    scenario_das: List[xr.DataArray] = []

    for scen, attr in SCEN_TO_FILE.items():
        ssp_path = getattr(paths, attr)

        _log(f"\n-- Scenario {scen} --")
        _log(f"SSP: {ssp_path}")

        ds = _open_ds(ssp_path, chunks={"time": chunks_time} if True else None)

        missing = [v for v in REQUIRED_ARTI_VARS if v not in ds.variables and v not in ds.data_vars]
        if missing:
            raise KeyError(
                f"{ssp_path} missing required vars: {missing}. "
                f"Available: {list(ds.variables)[:40]}..."
            )

        # Build artificialisation (keep as DataArray on LUH grid)
        arti = None
        for v in REQUIRED_ARTI_VARS:
            da = ds[v]
            da = _normalize_lon_to_180(_ensure_lat_ascending(da))
            arti = da if arti is None else (arti + da)

        arti.name = "arti"

        # Ensure yearly
        arti_y = _to_yearly(arti).sel(year=slice(year_start, year_end))

        # Regrid protected mask to LUH grid (nearest)
        protected_on_luh = protected_raster.interp_like(arti_y.isel(year=0), method="nearest")
        protected_on_luh = protected_on_luh.astype(bool)

        # Piecewise MSA
        # Start with 1 everywhere
        msa = xr.full_like(arti_y, 1.0, dtype=np.float32)

        # Only apply where NOT protected
        nonprot = ~protected_on_luh

        # linear zone: (0, threshold]
        m2 = nonprot & (arti_y > 0) & (arti_y <= threshold)
        msa = msa.where(~m2, 1.0 - (arti_y / threshold) * (1.0 - msa_min))

        # plateau zone: > threshold
        m1 = nonprot & (arti_y > threshold)
        msa = msa.where(~m1, msa_min)

        # safety: invalid -> 1, clip
        msa = msa.where(np.isfinite(msa), 1.0).clip(min=0.0, max=1.0).astype(np.float32)
        msa.name = "MSA_ENC"

        # Land mask (oceans -> NaN) on LUH grid
        land_mask = _infer_land_mask(
            paths.oceans_nc,
            template=msa.isel(year=0),
            fallback_from=arti_y.isel(year=0),
        )
        msa = msa.where(land_mask)

        # Write per scenario
        out_path = paths.out_dir / f"MSA_ENC_{scen}.nc"
        ds_out = msa.to_dataset()

        enc = _safe_encoding_float32(msa, max_chunks={"year": 100, "lat": 360, "lon": 720})
        ds_out.to_netcdf(out_path, encoding=enc)
        _log(_qc_summary(ds_out["MSA_ENC"], str(out_path)))

        scenario_das.append(msa.assign_coords(scenario=np.int32(scen)).expand_dims("scenario"))

    # Combined file
    msa_all = xr.concat(scenario_das, dim="scenario")
    out_all = paths.out_dir / "MSA_ENC.nc"
    ds_all = msa_all.to_dataset()

    enc_all = _safe_encoding_float32(msa_all, max_chunks={"scenario": 3, "year": 100, "lat": 360, "lon": 720})
    ds_all.to_netcdf(out_all, encoding=enc_all)
    _log(_qc_summary(ds_all["MSA_ENC"], str(out_all)))

    _log("\n[DONE] MSA_ENC datamart built.")


# =============================================================================
# 5) CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MSA_ENC datamart (Encroachment / artificialisation).")

    p.add_argument("--year-start", type=int, default=2015)
    p.add_argument("--year-end", type=int, default=2100)

    p.add_argument("--threshold", type=float, default=0.015)
    p.add_argument("--msa-min", type=float, default=0.85)

    p.add_argument("--chunks-time", type=int, default=50)

    # Path overrides (keep aligned with Paths defaults)
    p.add_argument("--ssp126", type=str, default=str(Paths.ssp126))
    p.add_argument("--ssp370", type=str, default=str(Paths.ssp370))
    p.add_argument("--ssp585", type=str, default=str(Paths.ssp585))

    p.add_argument("--pa0", type=str, default=str(Paths.pa0_tif))
    p.add_argument("--pa1", type=str, default=str(Paths.pa1_tif))

    p.add_argument("--oceans-nc", type=str, default=str(Paths.oceans_nc))
    p.add_argument("--out-dir", type=str, default=str(Paths.out_dir))

    return p.parse_args()


def main() -> None:
    a = _parse_args()
    paths = Paths(
        ssp126=Path(a.ssp126),
        ssp370=Path(a.ssp370),
        ssp585=Path(a.ssp585),
        pa0_tif=Path(a.pa0),
        pa1_tif=Path(a.pa1),
        oceans_nc=Path(a.oceans_nc),
        out_dir=Path(a.out_dir),
    )

    build_msa_enc(
        paths=paths,
        year_start=a.year_start,
        year_end=a.year_end,
        threshold=a.threshold,
        msa_min=a.msa_min,
        chunks_time=a.chunks_time,
    )


if __name__ == "__main__":
    main()
