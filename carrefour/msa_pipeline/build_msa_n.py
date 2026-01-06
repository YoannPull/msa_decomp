# scripts/build_msa_n.py
# -*- coding: utf-8 -*-

"""
Build MSA_N (Nitrogen deposition pressure) datamart.

Robust pipeline version (handles weird CF-time like 'months since ...' with 365_day).

Core logic (matches notebook intent)
------------------------------------
Inputs:
- Monthly nitrogen deposition components: NOy and NHx (per scenario)
- Biomes raster -> critical loads (mapping biome_code -> critical_load)
- Land-use fractions from SSP*.nc: primf, primn, secdf, secdn

Steps:
1) Open NOy/NHx with a robust time decoder:
   - try decode_times=True
   - if it fails (months since ... with 365_day), reopen with decode_times=False
2) Annualize deposition with SUM over months: annual_total = sum(months in year)
   - If time is decoded -> groupby(time.year)
   - Else if units 'months since' -> compute year manually from units + month index
3) Compute Ne = (NOy + NHx) * scale_factor  (notebook uses *10)
4) Compute critical load CL from biome raster via mapping
5) Exceedance Nexed = max(Ne - CL, 0)
6) Load LU fractions; convert to yearly if needed
7) Forest = primn + primf ; Grassland = secdf + secdn (same as notebook snippet)
8) Regrid Nexed to LU grid (nearest)
9) Compute:
      weight = Forest / (Forest + Grassland)  (safe when denom==0)
      MSA_N  = 1 - weight * log(Nexed + 1) * beta
   Replace invalid by 1, clip [0,1], mask oceans to NaN
10) Save per scenario + combined file (scenario stacked), with safe chunking

Outputs:
- data/datamart/MSA_N/MSA_N_126.nc
- data/datamart/MSA_N/MSA_N_370.nc
- data/datamart/MSA_N/MSA_N_585.nc
- data/datamart/MSA_N/MSA_N.nc

Usage:
poetry run python scripts/build_msa_n.py
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
        "rasterio is required to read the biomes GeoTIFF. Add it to your deps (poetry add rasterio)."
    ) from e


# =============================================================================
# 1) Mapping biome -> critical load  (keep consistent with your notebook)
# =============================================================================
BIOME_TO_CRITICAL_LOAD: Dict[int, float] = {
    1: 10.0,
    2: 10.0,
    3: 10.0,
    4: 10.0,
    5: 12.5,
    6: 7.5,
    7: 10.0,
    8: 20.0,
    9: 10.0,
    10: 10.0,
    11: 10.0,
    12: 6.75,
    13: 5.0,
    14: 10.0,
}


# =============================================================================
# 2) Paths (YOUR PROVIDED PATHS)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    # N deposition (monthly)
    noy_126: Path = Path("data/legacy/MSA_NDEP/ndep-noy_ssp126soc_monthly_2015_2100.nc")
    noy_370: Path = Path("data/legacy/MSA_NDEP/ndep-noy_ssp370soc_monthly_2015_2100.nc")
    noy_585: Path = Path("data/legacy/MSA_NDEP/ndep-noy_ssp585soc_monthly_2015_2100.nc")

    nhx_126: Path = Path("data/legacy/MSA_NDEP/ndep-nhx_ssp126soc_monthly_2015_2100.nc")
    nhx_370: Path = Path("data/legacy/MSA_NDEP/ndep-nhx_ssp370soc_monthly_2015_2100.nc")
    nhx_585: Path = Path("data/legacy/MSA_NDEP/ndep-nhx_ssp585soc_monthly_2015_2100.nc")

    # Biomes raster (biome codes)
    biomes_tif: Path = Path("data/legacy/MSA_NDEP/Biomes_Raster_Full.tif")

    # Land-use fractions (from LU inputs)
    ssp126_lu: Path = Path("data/legacy/MSA_LU/SSP126.nc")
    ssp370_lu: Path = Path("data/legacy/MSA_LU/SSP370.nc")
    ssp585_lu: Path = Path("data/legacy/MSA_LU/SSP585.nc")

    # Optional oceans mask
    oceans_nc: Path = Path("data/datamart/oceans.nc")

    # Output
    out_dir: Path = Path("data/datamart/MSA_N")


SCEN_TO_FILES = {
    126: ("noy_126", "nhx_126", "ssp126_lu"),
    370: ("noy_370", "nhx_370", "ssp370_lu"),
    585: ("noy_585", "nhx_585", "ssp585_lu"),
}


# =============================================================================
# 3) Utilities
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


def _annual_sum_monthly(da: xr.DataArray) -> xr.DataArray:
    """
    Annualize monthly data by SUM over months.
    Supports:
      - decoded time: groupby(time.dt.year)
      - numeric time + units 'months since ...': manual year computation
    """
    if "year" in da.dims:
        return da.assign_coords(year=da["year"].astype(np.int32))

    if "time" not in da.dims:
        raise ValueError(f"Expected dim 'time'. Got dims={da.dims}")

    # decoded time
    if hasattr(da["time"], "dt"):
        out = da.groupby(da["time"].dt.year).sum("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    # numeric time
    units = da["time"].attrs.get("units", "")
    if "months since" in (units or "").lower():
        years = _year_from_months_since(da["time"].values, units=units)
        da2 = da.assign_coords(year=("time", years))
        out = da2.groupby("year").sum("time", keep_attrs=True)
        return out.assign_coords(year=out["year"].astype(np.int32))

    raise ValueError(f"Cannot annualize: non-decoded time with unsupported units={units!r}")


def _to_yearly_fraction(da: xr.DataArray) -> xr.DataArray:
    """
    Convert LU fractions to yearly series.
    - decoded time: yearly mean
    - numeric time with 'years since' or 'months since': manual
    - already year dim: passthrough
    """
    if "year" in da.dims:
        return da.assign_coords(year=da["year"].astype(np.int32))

    if "time" not in da.dims:
        return da

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

    raise ValueError(f"Cannot convert LU time to year: units={units!r}")


def _read_biomes_as_da(path: Path) -> xr.DataArray:
    _require_exists(path)
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        width, height = src.width, src.height

        xs = (np.arange(width) + 0.5) * transform.a + transform.c
        ys = (np.arange(height) + 0.5) * transform.e + transform.f

        da = xr.DataArray(
            arr.astype(np.int16),
            dims=("y", "x"),
            coords={"x": xs.astype(np.float32), "y": ys.astype(np.float32)},
            name="biome_code",
        )

    da = da.rename({"x": "lon", "y": "lat"})
    da = _normalize_lon_to_180(_ensure_lat_ascending(da))
    return da


def _lut_from_mapping(mapping: Dict[int, float]) -> np.ndarray:
    if not mapping:
        raise ValueError("Empty mapping.")
    if any(k < 0 for k in mapping):
        raise ValueError(f"Negative biome code in mapping: {[k for k in mapping if k < 0]}")
    max_code = int(max(mapping.keys()))
    lut = np.full((max_code + 1,), np.nan, dtype=np.float32)
    for k, v in mapping.items():
        lut[int(k)] = np.float32(float(v))
    return lut


def _map_codes_to_values(codes: xr.DataArray, lut: np.ndarray, name: str) -> xr.DataArray:
    max_code = len(lut) - 1

    def _lookup(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.int32)
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = (arr >= 0) & (arr <= max_code)
        out[valid] = lut[arr[valid]]
        return out

    out = xr.apply_ufunc(
        _lookup,
        codes,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    out.name = name
    return out


def _infer_land_mask(oceans_nc: Path, template: xr.DataArray, fallback_from: xr.DataArray) -> xr.DataArray:
    """
    Return land mask aligned to template.
    Prefer oceans.nc if plausible; else fallback to non-NaN fallback_from.
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

def build_msa_n(
    paths: Paths,
    year_start: int = 2015,
    year_end: int = 2100,
    scale_factor: float = 10.0,  # notebook uses *10
    beta: float = 0.0630,        # notebook uses 0.0630
    noy_var: str = "noy",
    nhx_var: str = "nhx",
    chunks_time: int = 120,
) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    _log("== Build MSA_N ==")
    _log(f"biomes   : {paths.biomes_tif}")
    _log(f"oceans   : {paths.oceans_nc} (optional)")
    _log(f"out_dir  : {paths.out_dir}")
    _log(f"years    : {year_start}-{year_end}")
    _log(f"scale    : {scale_factor}")
    _log(f"beta     : {beta}")
    _log("")

    # Biomes & critical loads (CL)
    biomes = _read_biomes_as_da(paths.biomes_tif)
    cl_lut = _lut_from_mapping(BIOME_TO_CRITICAL_LOAD)
    cl = _map_codes_to_values(biomes, cl_lut, name="critical_load")

    scenario_das: List[xr.DataArray] = []

    for scen, (noy_attr, nhx_attr, lu_attr) in SCEN_TO_FILES.items():
        noy_path = getattr(paths, noy_attr)
        nhx_path = getattr(paths, nhx_attr)
        lu_path = getattr(paths, lu_attr)

        _log(f"\n-- Scenario {scen} --")
        _log(f"NOy: {noy_path}")
        _log(f"NHx: {nhx_path}")
        _log(f"LU : {lu_path}")

        # NOy + NHx monthly -> annual SUM
        ds_noy = _open_ds(noy_path, chunks={"time": chunks_time})
        ds_nhx = _open_ds(nhx_path, chunks={"time": chunks_time})

        v_noy = _pick_var(ds_noy, noy_var)
        v_nhx = _pick_var(ds_nhx, nhx_var)

        noy = _normalize_lon_to_180(_ensure_lat_ascending(ds_noy[v_noy]))
        nhx = _normalize_lon_to_180(_ensure_lat_ascending(ds_nhx[v_nhx]))

        noy_y = _annual_sum_monthly(noy)
        nhx_y = _annual_sum_monthly(nhx)

        # align years and select window
        years = np.intersect1d(noy_y["year"].values, nhx_y["year"].values)
        noy_y = noy_y.sel(year=years).sel(year=slice(year_start, year_end))
        nhx_y = nhx_y.sel(year=years).sel(year=slice(year_start, year_end))

        # Total deposition + scaling
        Ne = (noy_y + nhx_y) * scale_factor

        # CL aligned to deposition grid
        cl_on_dep = cl.interp_like(Ne.isel(year=0), method="nearest")

        # Exceedance
        Nexed = (Ne - cl_on_dep).clip(min=0.0)
        Nexed.name = "Nexed"

        # Land-use fractions
        ds_lu = _open_ds(lu_path, chunks=None)

        for req in ["primf", "primn", "secdf", "secdn"]:
            if req not in ds_lu.variables:
                raise KeyError(f"{lu_path} missing variable '{req}'. Available: {list(ds_lu.variables)[:30]}...")

        primf = _normalize_lon_to_180(_ensure_lat_ascending(ds_lu["primf"]))
        primn = _normalize_lon_to_180(_ensure_lat_ascending(ds_lu["primn"]))
        secdf = _normalize_lon_to_180(_ensure_lat_ascending(ds_lu["secdf"]))
        secdn = _normalize_lon_to_180(_ensure_lat_ascending(ds_lu["secdn"]))

        primf_y = _to_yearly_fraction(primf)
        primn_y = _to_yearly_fraction(primn)
        secdf_y = _to_yearly_fraction(secdf)
        secdn_y = _to_yearly_fraction(secdn)

        # Align years between Nexed and LU
        y_common = np.intersect1d(Nexed["year"].values, primf_y["year"].values)
        if y_common.size == 0:
            raise ValueError(
                f"No overlapping years between NDEP and LU for scen {scen}. "
                f"NDEP years: {Nexed['year'].values.min()}..{Nexed['year'].values.max()} ; "
                f"LU years: {primf_y['year'].values.min()}..{primf_y['year'].values.max()}"
            )

        Nexed = Nexed.sel(year=y_common)
        primf_y = primf_y.sel(year=y_common)
        primn_y = primn_y.sel(year=y_common)
        secdf_y = secdf_y.sel(year=y_common)
        secdn_y = secdn_y.sel(year=y_common)

        # Forest and "Grassland"
        Forest = primn_y + primf_y
        Grassland = secdf_y + secdn_y

        # Regrid Nexed to LU grid
        Nexed_on_lu = Nexed.interp_like(Forest, method="nearest")

        # Weight safe
        denom = Forest + Grassland
        weight = xr.where(denom > 0, Forest / denom, 0.0)

        # MSA_N formula
        msa = 1.0 - weight * xr.ufuncs.log(Nexed_on_lu + 1.0) * beta

        # Fill invalid -> 1, clip [0,1]
        msa = msa.where(np.isfinite(msa), 1.0)
        msa = msa.clip(min=0.0, max=1.0).astype(np.float32)
        msa.name = "MSA_N"

        # Land mask on LU grid
        land_mask = _infer_land_mask(paths.oceans_nc, template=msa.isel(year=0), fallback_from=Forest.isel(year=0))
        msa = msa.where(land_mask)

        # Write per scenario
        out_path = paths.out_dir / f"MSA_N_{scen}.nc"
        ds_out = msa.to_dataset()

        enc = _safe_encoding_float32(msa, max_chunks={"year": 100, "lat": 360, "lon": 720})
        ds_out.to_netcdf(out_path, encoding=enc)
        _log(_qc_summary(ds_out["MSA_N"], str(out_path)))

        scenario_das.append(msa.assign_coords(scenario=np.int32(scen)).expand_dims("scenario"))

    # Combined file
    msa_all = xr.concat(scenario_das, dim="scenario")
    out_all = paths.out_dir / "MSA_N.nc"
    ds_all = msa_all.to_dataset()

    enc_all = _safe_encoding_float32(msa_all, max_chunks={"scenario": 3, "year": 100, "lat": 360, "lon": 720})
    ds_all.to_netcdf(out_all, encoding=enc_all)
    _log(_qc_summary(ds_all["MSA_N"], str(out_all)))

    _log("\n[DONE] MSA_N datamart built.")


# =============================================================================
# 5) CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MSA_N datamart (Nitrogen deposition).")

    p.add_argument("--year-start", type=int, default=2015)
    p.add_argument("--year-end", type=int, default=2100)

    p.add_argument("--scale-factor", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=0.0630)

    p.add_argument("--noy-var", type=str, default="noy")
    p.add_argument("--nhx-var", type=str, default="nhx")
    p.add_argument("--chunks-time", type=int, default=120)

    # path overrides (keep aligned with Paths defaults)
    p.add_argument("--noy-126", type=str, default=str(Paths.noy_126))
    p.add_argument("--noy-370", type=str, default=str(Paths.noy_370))
    p.add_argument("--noy-585", type=str, default=str(Paths.noy_585))

    p.add_argument("--nhx-126", type=str, default=str(Paths.nhx_126))
    p.add_argument("--nhx-370", type=str, default=str(Paths.nhx_370))
    p.add_argument("--nhx-585", type=str, default=str(Paths.nhx_585))

    p.add_argument("--biomes-tif", type=str, default=str(Paths.biomes_tif))

    p.add_argument("--ssp126-lu", type=str, default=str(Paths.ssp126_lu))
    p.add_argument("--ssp370-lu", type=str, default=str(Paths.ssp370_lu))
    p.add_argument("--ssp585-lu", type=str, default=str(Paths.ssp585_lu))

    p.add_argument("--oceans-nc", type=str, default=str(Paths.oceans_nc))
    p.add_argument("--out-dir", type=str, default=str(Paths.out_dir))

    return p.parse_args()


def main() -> None:
    a = _parse_args()
    paths = Paths(
        noy_126=Path(a.noy_126),
        noy_370=Path(a.noy_370),
        noy_585=Path(a.noy_585),
        nhx_126=Path(a.nhx_126),
        nhx_370=Path(a.nhx_370),
        nhx_585=Path(a.nhx_585),
        biomes_tif=Path(a.biomes_tif),
        ssp126_lu=Path(a.ssp126_lu),
        ssp370_lu=Path(a.ssp370_lu),
        ssp585_lu=Path(a.ssp585_lu),
        oceans_nc=Path(a.oceans_nc),
        out_dir=Path(a.out_dir),
    )

    build_msa_n(
        paths=paths,
        year_start=a.year_start,
        year_end=a.year_end,
        scale_factor=a.scale_factor,
        beta=a.beta,
        noy_var=a.noy_var,
        nhx_var=a.nhx_var,
        chunks_time=a.chunks_time,
    )


if __name__ == "__main__":
    main()
