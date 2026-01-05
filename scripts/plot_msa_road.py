# scripts/plot_msa_road.py
# -*- coding: utf-8 -*-

"""
Plot MSA_ROAD for a given scenario/year and save the figure to outputs/plots/.

Settings aligned with the "brown gradient" example:
- origin forced to "upper"
- brown colormap (copper)
- default display range: [0.75, 1.02] (change with --vmin/--vmax)
- oceans masked using data/datamart/oceans.nc (transparent)

Usage:
  poetry run python scripts/plot_msa_road.py --scen 126 --year 2015
  poetry run python scripts/plot_msa_road.py --scen 126 --year 2100 --vmin 0.90 --vmax 1.0
  poetry run python scripts/plot_msa_road.py --scen 370 --year 2030 --show
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc


def load_ocean_mask(mask_file: str) -> np.ndarray:
    """Return ocean mask (True=ocean) from oceans.nc variable 'Oceans'."""
    if not os.path.exists(mask_file):
        raise FileNotFoundError(
            f"Ocean mask not found: {mask_file}\n"
            "Run LU build first (it creates data/datamart/oceans.nc)."
        )

    with nc.Dataset(mask_file) as ds:
        if "Oceans" not in ds.variables:
            raise KeyError(f"'Oceans' not found in {mask_file}. Available: {list(ds.variables.keys())}")
        M = ds.variables["Oceans"][:, :]  # may be masked array

    if np.ma.isMaskedArray(M):
        ocean = np.ma.getmaskarray(M) | np.isnan(np.ma.filled(M, np.nan))
    else:
        ocean = np.isnan(M)

    return ocean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2015, help="Year in [2015,2100]")
    parser.add_argument("--scen", type=str, default="126", choices=["126", "370", "585"], help="Scenario suffix")
    parser.add_argument("--data-root", type=str, default="data/datamart", help="Datamart root folder")
    parser.add_argument("--outdir", type=str, default=os.path.join("outputs", "plots"), help="Output plots directory")
    parser.add_argument("--show", action="store_true", help="Also display the plot interactively")

    # Brown-gradient style defaults (like your example)
    parser.add_argument("--vmin", type=float, default=0.75, help="Color scale minimum (default: 0.75)")
    parser.add_argument("--vmax", type=float, default=1.02, help="Color scale maximum (default: 1.02)")

    # Viz details
    parser.add_argument("--interp", type=str, default="nearest", choices=["nearest", "bilinear"])
    parser.add_argument("--no-ocean-mask", action="store_true", help="Disable oceans masking (debug)")
    args = parser.parse_args()

    year = args.year
    scen = args.scen

    if year < 2015 or year > 2100:
        raise ValueError("year must be between 2015 and 2100")

    t_idx = year - 2015  # 2015->0 ... 2100->85

    nc_file = os.path.join(args.data_root, "MSA_ROAD", f"MSA_ROAD_{scen}.nc")
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"NetCDF not found: {nc_file}\nDid you run: make road ?")

    # Load ROAD
    with nc.Dataset(nc_file) as ds:
        if "MSA_ROAD" not in ds.variables:
            raise KeyError(f"'MSA_ROAD' not found. Available: {list(ds.variables.keys())}")
        Z = ds.variables["MSA_ROAD"][t_idx, :, :]  # (lat, lon)

    # Apply ocean mask
    if not args.no_ocean_mask:
        mask_file = os.path.join(args.data_root, "oceans.nc")
        ocean = load_ocean_mask(mask_file)
        Z = np.ma.masked_where(ocean, Z)

    # Brown colormap like your example
    cmap = plt.cm.copper.copy()
    cmap.set_bad((1, 1, 1, 0))  # transparent oceans

    os.makedirs(args.outdir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.title(f"MSA Road {year}")

    im = plt.imshow(
        Z,
        origin="upper",                 # <- forced as requested
        cmap=cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        interpolation=args.interp,
    )

    cb = plt.colorbar(im, orientation="horizontal")
    cb.set_label("MSA_ROAD")

    plt.tight_layout()

    out_png = os.path.join(args.outdir, f"MSA_ROAD_brown_SSP{scen}_{year}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Saved:", out_png)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
