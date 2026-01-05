import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2100)
    parser.add_argument("--scen", type=str, default="126", choices=["126", "370", "585"])
    args = parser.parse_args()

    DATA_ROOT = "data/datamart"
    YEAR = args.year
    SCEN = args.scen

    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    nc_file = os.path.join(DATA_ROOT, "MSA_LU", f"MSA_LU_{SCEN}.nc")
    t_idx = YEAR - 2015

    with nc.Dataset(nc_file) as ds:
        Z = ds.variables["MSA_LU"][t_idx, :, :]
        lat = ds.variables["lat"][:]

    origin = "upper" if lat[0] > lat[-1] else "lower"

    cmap = plt.cm.Greens.copy()
    cmap.set_bad((1, 1, 1, 0))  # NA transparent

    plt.figure(figsize=(10, 6))
    plt.title(f"MSA_LU en {YEAR} (SSP{SCEN})")
    im = plt.imshow(Z, origin=origin, vmin=0, vmax=1, cmap=cmap, interpolation="bilinear")
    plt.colorbar(im, orientation="horizontal", label="MSA_LU")
    plt.tight_layout()

    out_png = os.path.join(plots_dir, f"MSA_LU_SSP{SCEN}_{YEAR}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Saved:", out_png)

if __name__ == "__main__":
    main()
