# carrefour/prepare/msa_lu.py
# -*- coding: utf-8 -*-

"""
Prepare MSA_LU (Land Use) from LUH2 landState input4MIPs.

Important LUH convention:
- Sum of landState fractions = land fraction of the grid cell (near 0 over oceans,
  near 1 over full land, intermediate near coasts/mixed cells).
- We therefore renormalize by landfrac before computing MSA.

Outputs:
- NetCDF: data/datamart/MSA_LU/MSA_LU_<scen>.nc
- Variable: msa_lu (time, lat, lon) float32 in [0,1], oceans as NaN

This module contains NO CLI parsing; use scripts/prepare/prepare_msa_lu.py as wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class MsaLuConfig:
    # IO roots (repo-root-relative)
    luh_dir: Path = Path("data/outputs/LUH")
    out_dir: Path = Path("data/datamart/MSA_LU")

    # Optional external land mask (variable expected land==1 for land)
    ocean_mask_candidates: Tuple[Path, ...] = (
        Path("data/datamart/oceans.nc"),
        Path("data/outputs/ocean.nc"),
        Path("data/outputs/oceans.nc"),
    )

    # Reference grid candidates (lat/lon)
    grid_ref_candidates: Tuple[Path, ...] = (
        Path("data/datamart/ROAD/road_on_msa_grid.nc"),
        Path("data/datamart/MSA_ROAD/MSA_ROAD_126.nc"),
    )

    # "GLOBIO-like" coefficients (starter)
    msa_coeff_primary: float = 1.00
    msa_coeff_secondary: float = 0.50
    msa_coeff_cropland: float = 0.10
    msa_coeff_grazing: float = 0.60
    msa_coeff_urban: float = 0.05

    # thresholds
    eps_landfrac: float = 1e-6
    valid_full_land: float = 0.95  # logs only

    # Dask safety (avoid chunk explosion on interp_like)
    pre_interp_single_chunk: bool = True
    post_interp_chunks: Optional[dict] = None  # e.g. {"lat": 180, "lon": 360}


# -----------------------------
# Small utilities
# -----------------------------

def _abs(p: Path, repo_root: Path) -> Path:
    return p if p.is_absolute() else (repo_root / p)


def _resolve_existing_path(candidates: Tuple[Path, ...], repo_root: Path, what: str) -> Path:
    tried = []
    for p in candidates:
        ap = _abs(p, repo_root)
        tried.append(ap)
        if ap.exists():
            return ap
    msg = "Could not find {what}. Tried:\n".format(what=what) + "\n".join(str(x) for x in tried)
    raise FileNotFoundError(msg)


def _select_year_index(year: int) -> int:
    # LUH file shown: 2015..2100 inclusive => 86
    if year < 2015 or year > 2100:
        raise ValueError("year must be in [2015, 2100] for LUH 2015-2100 files.")
    return year - 2015


def _luh_path_for_scen(repo_root: Path, cfg: MsaLuConfig, scen: int) -> Path:
    """
    Match your LUH filenames:
      126 -> UofMD-IMAGE-ssp126
      370 -> UofMD-AIM-ssp370
      585 -> UofMD-MAGPIE-ssp585
    """
    if scen == 126:
        ssp, model = "ssp126", "IMAGE"
    elif scen == 370:
        ssp, model = "ssp370", "AIM"
    elif scen == 585:
        ssp, model = "ssp585", "MAGPIE"
    else:
        raise ValueError("scen must be one of: 126, 370, 585")

    luh_dir = _abs(cfg.luh_dir, repo_root)
    pat = f"multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-{model}-{ssp}*2015-2100.nc"
    matches = sorted(luh_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No LUH file matching pattern: {luh_dir / pat}")
    if len(matches) > 1:
        print(f"[WARN] multiple matches for {pat}. Using: {matches[0].name}")
    return matches[0]


def _load_land_mask(repo_root: Path, cfg: MsaLuConfig) -> xr.DataArray:
    """
    Return int8 land mask aligned on its own grid: 1 land, 0 ocean.
    """
    p = _resolve_existing_path(cfg.ocean_mask_candidates, repo_root, what="ocean/land mask (oceans.nc/ocean.nc)")
    ds = xr.open_dataset(p, decode_times=False)

    for v in ("land", "Land", "mask", "lsmask"):
        if v in ds.data_vars:
            land = ds[v]
            break
    else:
        raise KeyError(f"Could not find land mask variable in {p}. Vars: {list(ds.data_vars)}")

    land01 = xr.where(land == 1, 1, 0).astype("int8")
    frac_ones = float(land01.mean(skipna=True))
    print(f"[INFO] oceans mask: land==1 (frac_ones={frac_ones:.3f}) | path={p}")
    return land01


def _load_grid_ref(repo_root: Path, cfg: MsaLuConfig, scen: int, grid_ref: Optional[Path]) -> xr.Dataset:
    if grid_ref is not None:
        p = _abs(grid_ref, repo_root)
        if not p.exists():
            raise FileNotFoundError(f"grid_ref not found: {p}")
        ds = xr.open_dataset(p, decode_times=False)
        return ds

    # try candidates + scenario-specific MSA_ROAD if it exists
    candidates = list(cfg.grid_ref_candidates) + [
        Path(f"data/datamart/MSA_ROAD/MSA_ROAD_{scen}.nc"),
    ]
    p = _resolve_existing_path(tuple(candidates), repo_root, what="reference grid dataset (lat/lon)")
    return xr.open_dataset(p, decode_times=False)


def _same_grid(a: xr.DataArray, target: xr.Dataset) -> bool:
    if "lat" not in a.coords or "lon" not in a.coords:
        return False
    if "lat" not in target.coords or "lon" not in target.coords:
        return False
    if a.sizes.get("lat") != target.sizes.get("lat"):
        return False
    if a.sizes.get("lon") != target.sizes.get("lon"):
        return False
    return (
        np.allclose(a["lat"].values, target["lat"].values)
        and np.allclose(a["lon"].values, target["lon"].values)
    )


# -----------------------------
# Core computation
# -----------------------------

def build_msa_lu_native(ds_luh: xr.Dataset, *, year: int, cfg: MsaLuConfig) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute MSA_LU on LUH native grid for a given year.

    Returns:
      msa_lu: (lat, lon) float32, NaN where landfrac<=eps
      landfrac: (lat, lon) float32 in [0,1]
    """
    required = [
        "primf","primn","secdf","secdn","urban",
        "c3ann","c4ann","c3per","c4per","c3nfx",
        "pastr","range",
    ]
    missing = [v for v in required if v not in ds_luh.data_vars]
    if missing:
        raise KeyError(f"Missing required LUH variables: {missing}")

    tidx = _select_year_index(year)
    ds = ds_luh.isel(time=tidx)

    # Aggregate states
    f_primary = (ds["primf"] + ds["primn"]).clip(0, 1)
    f_secondary = (ds["secdf"] + ds["secdn"]).clip(0, 1)
    f_cropland = (ds["c3ann"] + ds["c4ann"] + ds["c3per"] + ds["c4per"] + ds["c3nfx"]).clip(0, 1)
    f_grazing = (ds["pastr"] + ds["range"]).clip(0, 1)
    f_urban = ds["urban"].clip(0, 1)

    landfrac = (f_primary + f_secondary + f_cropland + f_grazing + f_urban).astype("float32")
    landfrac.name = "landfrac"
    landfrac.attrs.update(long_name="LUH land fraction (sum of landState states)", units="1", year=year)

    # Renormalize conditional on land
    inv = xr.where(landfrac > cfg.eps_landfrac, 1.0 / landfrac, np.nan)

    p = f_primary * inv
    s = f_secondary * inv
    c = f_cropland * inv
    g = f_grazing * inv
    u = f_urban * inv

    msa = (
        p * cfg.msa_coeff_primary
        + s * cfg.msa_coeff_secondary
        + c * cfg.msa_coeff_cropland
        + g * cfg.msa_coeff_grazing
        + u * cfg.msa_coeff_urban
    ).clip(0, 1).astype("float32")

    # Mask ocean/non-modeled based on LUH
    msa = msa.where(landfrac > cfg.eps_landfrac)

    msa.name = "msa_lu"
    msa.attrs.update(
        long_name="MSA from land use (LUH2 landState, renormalized on land fraction)",
        units="1",
        source="LUH2 landState input4MIPs",
        year=year,
        msa_coeff_primary=cfg.msa_coeff_primary,
        msa_coeff_secondary=cfg.msa_coeff_secondary,
        msa_coeff_cropland=cfg.msa_coeff_cropland,
        msa_coeff_grazing=cfg.msa_coeff_grazing,
        msa_coeff_urban=cfg.msa_coeff_urban,
    )

    # Logs: landfrac + normalization check on full-land-ish
    lf = landfrac
    print(
        "[INFO] LUH landfrac (global) min/max/mean: "
        f"{float(lf.min()):.6f}/{float(lf.max()):.6f}/{float(lf.mean()):.6f}"
    )

    sum_norm = (p + s + c + g + u).where(landfrac > cfg.valid_full_land)
    if np.isfinite(sum_norm.values).any():
        vmin = float(sum_norm.min(skipna=True))
        vmax = float(sum_norm.max(skipna=True))
        vmean = float(sum_norm.mean(skipna=True))
        max_dev = float(np.nanmax(np.abs((sum_norm - 1.0).values)))
        print(
            f"[INFO] sum(normalized states) on landfrac>{cfg.valid_full_land}: "
            f"min/max/mean={vmin:.6f}/{vmax:.6f}/{vmean:.6f} | max|sum-1|={max_dev:.6f}"
        )

    return msa, landfrac


def regrid_to_target(msa_native: xr.DataArray, target: xr.Dataset, cfg: MsaLuConfig) -> xr.DataArray:
    """
    Regrid msa_native (lat/lon) onto target lat/lon using interp_like if needed.
    Handles optional dask chunking guard.
    """
    if _same_grid(msa_native, target):
        print("[INFO] Grids already match; skip interp_like.")
        return msa_native

    msa_in = msa_native
    if cfg.pre_interp_single_chunk:
        # avoid dask chunk explosion during interpolation
        try:
            msa_in = msa_in.chunk({"lat": -1, "lon": -1})
        except ValueError:
            # if not dask-backed, ignore
            pass

    msa_out = msa_in.interp_like(target, method="linear").clip(0, 1).astype("float32")

    if cfg.post_interp_chunks:
        try:
            msa_out = msa_out.chunk(cfg.post_interp_chunks)
        except ValueError:
            pass

    return msa_out


def write_msa_lu(out_path: Path, msa: xr.DataArray, *, year: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_out = msa.to_dataset()
    ds_out = ds_out.expand_dims(time=[year])

    enc = {"msa_lu": {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
    ds_out.to_netcdf(out_path, encoding=enc)

    v = ds_out["msa_lu"].values
    nanpct = float(np.isnan(v).mean() * 100.0)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    print(f"[OK] wrote {out_path} | min/max={vmin}/{vmax} | nan%={nanpct:.2f}% | dtype=float32")


# -----------------------------
# Public entry point (called by CLI)
# -----------------------------

def prepare_msa_lu(
    *,
    repo_root: Path,
    scen: int,
    year: int,
    grid_ref: Optional[Path] = None,
    use_only_luh_mask: bool = False,
    cfg: Optional[MsaLuConfig] = None,
) -> Path:
    """
    Full pipeline: read LUH -> build native MSA -> mask -> regrid -> write.

    Returns output path.
    """
    cfg = cfg or MsaLuConfig()

    luh_path = _luh_path_for_scen(repo_root, cfg, scen)
    print(f"[INFO] LUH landState: {luh_path}")

    ds_luh = xr.open_dataset(luh_path, decode_times=False)

    msa_native, landfrac = build_msa_lu_native(ds_luh, year=year, cfg=cfg)

    # Optional external land mask for consistency across pressures
    if not use_only_luh_mask:
        land01 = _load_land_mask(repo_root, cfg)
        land01_aligned = land01.interp_like(msa_native, method="nearest")
        msa_native = msa_native.where(land01_aligned == 1)

    # Regrid to target grid
    ds_ref = _load_grid_ref(repo_root, cfg, scen, grid_ref)
    if "lat" not in ds_ref.coords or "lon" not in ds_ref.coords:
        raise KeyError(f"Reference grid must contain lat/lon coords. Got coords: {list(ds_ref.coords)}")

    target = xr.Dataset(coords={"lat": ds_ref["lat"], "lon": ds_ref["lon"]})
    msa_out = regrid_to_target(msa_native, target, cfg)

    # Write
    out_dir = _abs(cfg.out_dir, repo_root)
    out_path = out_dir / f"MSA_LU_{scen}.nc"
    write_msa_lu(out_path, msa_out, year=year)

    return out_path
