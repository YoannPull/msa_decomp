# scripts/plot/plot_road_input.py
# -*- coding: utf-8 -*-

"""
Plot ROAD input produced by prepare-road:
  data/outputs/ROAD/road_on_msa_grid.nc

Colors ocean using ocean mask from:
  data/outputs/ocean.nc

Usage:
  poetry run python scripts/plot/plot_road_input.py
  poetry run python scripts/plot/plot_road_input.py --var road_density
  poetry run python scripts/plot/plot_road_input.py --var road_bin --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


DEFAULT_IN = Path("data/outputs/ROAD/road_on_msa_grid.nc")
DEFAULT_OCEANS = Path("data/outputs/oceans.nc")
DEFAULT_OUTDIR = Path("outputs/plots")


def _pick_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    if var in ds:
        return ds[var]
    raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")


def _infer_origin(lat: np.ndarray) -> str:
    return "lower" if lat[1] > lat[0] else "upper"


def _load_land_mask(oceans_nc: Path) -> xr.DataArray:
    ds = xr.open_dataset(oceans_nc)
    if "land" in ds:
        land = ds["land"]
    else:
        # fallback: take first data var
        land = next(iter(ds.data_vars.values()))
    # normalize coord names
    if "latitude" in land.coords and "lat" not in land.coords:
        land = land.rename({"latitude": "lat"})
    if "longitude" in land.coords and "lon" not in land.coords:
        land = land.rename({"longitude": "lon"})
    if "lat" not in land.coords or "lon" not in land.coords:
        raise ValueError(f"ocean mask must have lat/lon coords. Got coords: {list(land.coords)}")
    return land


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-nc", type=Path, default=DEFAULT_IN)
    p.add_argument("--oceans-nc", type=Path, default=DEFAULT_OCEANS)
    p.add_argument("--var", type=str, default="road_bin", choices=["road_bin", "road_density", "land"])
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--show", action="store_true")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--ocean-value", type=float, default=-1.0, help="Sentinel value used to color oceans.")
    p.add_argument("--land-threshold", type=float, default=0.5, help="Land if land_mask > threshold.")
    args = p.parse_args()

    ds = xr.open_dataset(args.in_nc)
    da = _pick_var(ds, args.var)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"{args.var} must have lat/lon coords. Got coords: {list(da.coords)}")

    lat = da["lat"].values
    lon = da["lon"].values
    origin = _infer_origin(lat)

    # Load land mask and align to ROAD grid
    land = _load_land_mask(args.oceans_nc)
    land_aligned = land.interp(lat=da["lat"], lon=da["lon"], method="nearest")

    is_land = np.isfinite(land_aligned.values) & (land_aligned.values > args.land_threshold)
    is_ocean = ~is_land

    # Build array to plot:
    # - keep data on land
    # - set ocean to sentinel value (so it gets a fixed color)
    arr = da.values.astype(float)
    arr_plot = arr.copy()
    arr_plot[is_ocean] = float(args.ocean_value)

    # Stats computed on land only (more meaningful)
    land_vals = arr_plot[is_land]
    vmin = float(np.nanmin(land_vals))
    vmax = float(np.nanmax(land_vals))
    nan_land_pct = float(np.isnan(land_vals).mean() * 100.0) if land_vals.size else float("nan")

    # Choose colormap + ocean color
    cmap = plt.get_cmap("viridis").copy()
    # If there are still NaNs (shouldn't on land), color them too
    cmap.set_bad("lightgray")
    # Ocean will be colored with "under" if we set vmin > ocean_value
    cmap.set_under("lightblue")

    # Define normalization so ocean_value is "under"
    # Ensure vmin is above ocean_value
    vmin_norm = max(vmin, args.ocean_value + 1e-6)

    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        arr_plot,
        origin=origin,
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        interpolation="nearest",
        vmin=vmin_norm,
        vmax=vmax,
        cmap=cmap,
    )
    plt.colorbar(im, label=args.var)

    title = args.title or f"{args.var} | land min={vmin:.3g} max={vmax:.3g} nan% land={nan_land_pct:.2f}"
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"road_input_{args.var}_with_ocean.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] wrote {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
