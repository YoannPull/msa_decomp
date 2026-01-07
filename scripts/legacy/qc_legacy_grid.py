# scripts/legacy/qc_legacy_grid.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def open_ds(p: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(p, decode_times=True, mask_and_scale=True)
    except Exception:
        return xr.open_dataset(p, decode_times=False, mask_and_scale=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", type=str, required=True)
    ap.add_argument("--scen", type=str, required=True)
    ap.add_argument("--var", type=str, default="MSA_SQUARE")
    ap.add_argument("--year", type=int, default=2100)
    args = ap.parse_args()

    root = Path(args.legacy_root)
    if args.var == "MSA_ROAD":
        nc_path = root / "MSA_ROAD" / "msaroad.nc"
    else:
        nc_path = root / args.var / f"{args.var}_{args.scen}.nc"

    print("[FILE]", nc_path)
    ds = open_ds(nc_path)

    vname = args.var if args.var in ds.data_vars else list(ds.data_vars)[0]
    da = ds[vname]
    print("[VAR]", vname, "dims=", da.dims, "shape=", da.shape)

    # check coords
    for c in ["lat", "lon", "time", "year"]:
        if c in da.coords:
            cv = da[c].values
            print(f"[COORD] {c}: dtype={cv.dtype} shape={cv.shape} min={np.nanmin(cv):.3g} max={np.nanmax(cv):.3g}")

    # Excel grids
    lat_xlsx = root / "Latitudes.xlsx"
    lon_xlsx = root / "Longitudes.xlsx"
    if lat_xlsx.exists() and lon_xlsx.exists():
        lat_df = pd.read_excel(lat_xlsx)
        lon_df = pd.read_excel(lon_xlsx)
        lat = pd.to_numeric(lat_df["Latitude_degre"], errors="coerce").dropna().to_numpy()
        lon = pd.to_numeric(lon_df["Longitude_degre"], errors="coerce").dropna().to_numpy()
        print(f"[EXCEL] lat n={lat.size} first/last={lat[0]:.3f}/{lat[-1]:.3f}")
        print(f"[EXCEL] lon n={lon.size} first/last={lon[0]:.3f}/{lon[-1]:.3f}")

    # infer "layout" possibilities
    if da.ndim >= 2:
        a2 = da
        while a2.ndim > 2:
            a2 = a2.isel({a2.dims[0]: 0})
        arr = a2.values
        arr = np.where(np.isfinite(arr) & (arr > 1e20), np.nan, arr)
        print("[ARR] 2D shape=", arr.shape, "min/max=", np.nanmin(arr), np.nanmax(arr), "nan%=", 100*(~np.isfinite(arr)).mean())

        # compare with excel sizes
        if lat_xlsx.exists() and lon_xlsx.exists():
            lat_df = pd.read_excel(lat_xlsx)
            lon_df = pd.read_excel(lon_xlsx)
            lat = pd.to_numeric(lat_df["Latitude_degre"], errors="coerce").dropna().to_numpy()
            lon = pd.to_numeric(lon_df["Longitude_degre"], errors="coerce").dropna().to_numpy()
            print("[MATCH] shape vs (lat,lon) =>", (arr.shape[0] == lat.size and arr.shape[1] == lon.size))
            print("[MATCH] shape vs (lon,lat) =>", (arr.shape[0] == lon.size and arr.shape[1] == lat.size))


if __name__ == "__main__":
    main()
