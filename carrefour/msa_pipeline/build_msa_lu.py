# carrefour/msa_pipeline/build_msa_lu.py
# -*- coding: utf-8 -*-

"""
Build MSA_LU datamart files from legacy LUH2 NetCDFs.

Inputs (legacy):  data/legacy/MSA_LU/*.nc  (files containing 126 / 370 / 585 in name)
Outputs:       data/datamart/MSA_LU/MSA_LU_126.nc (and 370/585)
               data/datamart/oceans.nc  (land=1, ocean=NA)

Notes:
- Step A: we assume legacy data is already on the target grid (720x1440) and 86 years (2015-2100).
- We keep masked arrays to avoid "9.969e36" fill-value issues.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Tuple

import numpy as np
import netCDF4 as nc


# -----------------------------
# Config (adaptable)
# -----------------------------
legacy_DIR = "data/legacy/MSA_LU"
OUT_ROOT = "data/datamart"
OUT_DIR = os.path.join(OUT_ROOT, "MSA_LU")
OCEANS_FILE = os.path.join(OUT_ROOT, "oceans.nc")

YEARS = np.arange(2015, 2101)  # 2015..2100 inclusive
TIME_VALUES = np.arange(len(YEARS), dtype=np.float64)  # 0..85
FILL_VALUE = np.float32(-9999.0)

# --- Land-use coefficients (GLOBIO-like; adjust if you want to match exactly their notebook)
# MSA_LU = sum( landuse_fraction * coeff )
COEFF = {
    "primf": 1.0,
    "primn": 1.0,
    "secdf": 0.5,
    "secdn": 1.0,
    "pastr": 0.6,
    "range": 0.3,
    "urban": 0.05,
    "c3ann": 0.1,
    "c4ann": 0.1,
    "c3per": 0.1,
    "c4per": 0.1,
    "c3nfx": 0.1,
}


# -----------------------------
# Helpers
# -----------------------------
def find_scenario_files(legacy_dir: str) -> Dict[str, str]:
    """
    Detect files for scenarios 126/370/585 based on filename containing these numbers.
    Returns dict like {"126": "/.../something_126.nc", ...}
    """
    if not os.path.isdir(legacy_dir):
        raise FileNotFoundError(f"legacy dir not found: {legacy_dir}")

    files = [os.path.join(legacy_dir, f) for f in os.listdir(legacy_dir) if f.lower().endswith(".nc")]
    scen_map: Dict[str, str] = {}

    for fp in sorted(files):
        m = re.search(r"(126|370|585)", os.path.basename(fp))
        if m:
            scen_map[m.group(1)] = fp

    return scen_map


def _require_vars(ds: nc.Dataset, required: Dict[str, float]) -> None:
    missing = [k for k in required.keys() if k not in ds.variables]
    if missing:
        raise KeyError(f"Missing variables in legacy LU file: {missing}\nAvailable: {list(ds.variables.keys())}")


def compute_msa_lu(ds: nc.Dataset) -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ndarray, np.ndarray]:
    """
    Compute MSA_LU (time, lat, lon) and a land mask (lat, lon) derived from total land fraction.

    Returns:
      msa_lu: masked array float32 shape (time,lat,lon) in [0,1] with ocean masked
      land_mask: masked array float32 shape (lat,lon) with land=1 and ocean masked
      lat, lon: coordinate arrays
    """
    _require_vars(ds, COEFF)

    if "lat" not in ds.variables or "lon" not in ds.variables:
        raise KeyError("legacy LU file must contain 'lat' and 'lon' variables.")

    lat = np.array(ds.variables["lat"][:])
    lon = np.array(ds.variables["lon"][:])

    msa = None
    total = None

    for var, w in COEFF.items():
        arr = ds.variables[var][:]          # can be masked array
        arr = np.ma.masked_invalid(arr)

        msa = (arr * w) if msa is None else (msa + arr * w)
        total = arr if total is None else (total + arr)

    # Build land mask (land where total fraction > tiny threshold)
    land_mask = np.ma.masked_where(total[0, :, :] < 1e-6, np.ones(total.shape[1:], dtype=np.float32))

    # Clip and mask ocean
    msa = np.ma.masked_invalid(msa)
    msa = np.ma.clip(msa, 0.0, 1.0)

    ocean_mask = np.ma.getmaskarray(land_mask)  # True on ocean
    msa = np.ma.masked_where(np.broadcast_to(ocean_mask, msa.shape), msa)

    # Basic expected shape check for Step A
    if msa.shape[0] != 86:
        print(f"[WARN] time length is {msa.shape[0]} (expected 86 for 2015-2100).")
    if msa.shape[1] != 720 or msa.shape[2] != 1440:
        raise ValueError(f"[Step A] Grid is {msa.shape[1:]} but expected (720,1440). Regridding comes later.")

    return msa.astype(np.float32), land_mask.astype(np.float32), lat, lon


def qc_report(name: str, arr: np.ma.MaskedArray) -> None:
    data = arr.compressed()
    mask_pct = 100.0 * np.ma.getmaskarray(arr).mean()
    print(f"\nQC — {name}")
    print("  shape:", arr.shape)
    print("  masked %:", f"{mask_pct:.2f}%")
    print("  min/max:", float(data.min()), float(data.max()))
    print("  p1 p5 p50 p95 p99:", np.quantile(data, [0.01, 0.05, 0.5, 0.95, 0.99]))


def write_msa_nc(out_path: str, lat: np.ndarray, lon: np.ndarray, msa: np.ma.MaskedArray, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with nc.Dataset(out_path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("time", msa.shape[0])
        ds.createDimension("lat", msa.shape[1])
        ds.createDimension("lon", msa.shape[2])

        ds.title = title
        ds.subtitle = "MSA Land Use (LUH2) — values in [0,1]"

        vlat = ds.createVariable("lat", np.float32, ("lat",))
        vlat.units = "degrees_north"
        vlat.long_name = "latitude"
        vlat[:] = lat.astype(np.float32)

        vlon = ds.createVariable("lon", np.float32, ("lon",))
        vlon.units = "degrees_east"
        vlon.long_name = "longitude"
        vlon[:] = lon.astype(np.float32)

        vtime = ds.createVariable("time", np.float64, ("time",))
        vtime.units = "years since 2015-01-01"
        vtime.long_name = "time"
        vtime[:] = TIME_VALUES

        v = ds.createVariable("MSA_LU", np.float32, ("time", "lat", "lon"), fill_value=FILL_VALUE)
        v.units = "MSA"
        v.standard_name = "MSA_LU"

        v[:, :, :] = np.ma.filled(msa, FILL_VALUE)


def write_oceans_nc(out_path: str, lat: np.ndarray, lon: np.ndarray, land_mask: np.ma.MaskedArray) -> None:
    """
    oceans.nc with variable 'Oceans' :
      land = 1
      ocean = NA (fill value)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with nc.Dataset(out_path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("lat", land_mask.shape[0])
        ds.createDimension("lon", land_mask.shape[1])

        ds.title = "oceans mask (land=1, ocean=NA)"
        ds.subtitle = "Derived from LUH2 total land fraction"

        vlat = ds.createVariable("lat", np.float32, ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = lat.astype(np.float32)

        vlon = ds.createVariable("lon", np.float32, ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = lon.astype(np.float32)

        v = ds.createVariable("Oceans", np.float32, ("lat", "lon"), fill_value=FILL_VALUE)
        v.units = "1"
        v.long_name = "land mask (1 on land, NA on ocean)"

        v[:, :] = np.ma.filled(land_mask, FILL_VALUE)


def main() -> None:
    scen_files = find_scenario_files(legacy_DIR)
    if not scen_files:
        raise FileNotFoundError(f"No .nc files containing 126/370/585 found in {legacy_DIR}")

    print("Detected legacy LU files:", scen_files)

    first_land_mask = None
    first_lat = None
    first_lon = None

    for scen in ["126", "370", "585"]:
        if scen not in scen_files:
            print(f"[WARN] Scenario {scen} not found, skipping.")
            continue

        legacy_fp = scen_files[scen]
        out_fp = os.path.join(OUT_DIR, f"MSA_LU_{scen}.nc")

        with nc.Dataset(legacy_fp) as ds:
            msa, land_mask, lat, lon = compute_msa_lu(ds)

        qc_report(f"MSA_LU_{scen}", msa)
        write_msa_nc(out_fp, lat, lon, msa, title=f"MSA_LU_{scen}")
        print("Wrote:", out_fp)

        if first_land_mask is None:
            first_land_mask, first_lat, first_lon = land_mask, lat, lon

    if first_land_mask is not None:
        write_oceans_nc(OCEANS_FILE, first_lat, first_lon, first_land_mask)
        print("Wrote:", OCEANS_FILE)
