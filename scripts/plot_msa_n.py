# scripts/plot_msa_n.py
# -*- coding: utf-8 -*-

"""
Plot MSA_N for a given scenario/year and save to outputs/plots/.

Defaults tuned for readability:
- cmap: 'ndep_white' with OPTIONAL black top (values near vmax rendered black)
- norm: PowerNorm(gamma) to reveal subtle differences close to 1
- background land mask (oceans.nc) in light grey

Usage:
  poetry run python scripts/plot_msa_n.py --scen 585 --year 2100
  poetry run python scripts/plot_msa_n.py --scen 585 --year 2100 --black-top --black-top-frac 0.02
  poetry run python scripts/plot_msa_n.py --scen 585 --year 2100 --no-black-top
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from xarray.coding.times import CFDatetimeCoder


DEFAULT_DIR = Path("data/datamart/MSA_N")
DEFAULT_OCEANS = Path("data/datamart/oceans.nc")
DEFAULT_OUT = Path("outputs/plots")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _open_ds(p: Path) -> xr.Dataset:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    time_coder = CFDatetimeCoder(use_cftime=True)
    return xr.open_dataset(p, decode_times=time_coder)


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


def _pick_var(ds: xr.Dataset, preferred: str = "MSA_N") -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


def _load_msa_n(msa_dir: Path, scen: int) -> xr.DataArray:
    combined = msa_dir / "MSA_N.nc"
    if combined.exists():
        ds = _open_ds(combined)
        var = _pick_var(ds, "MSA_N")
        da = ds[var]
        da = _normalize_lon_to_180(_ensure_lat_ascending(da))
        if "scenario" not in da.dims:
            raise ValueError(f"{combined} has no 'scenario' dim.")
        return da.sel(scenario=scen)

    scen_file = msa_dir / f"MSA_N_{scen}.nc"
    ds = _open_ds(scen_file)
    var = _pick_var(ds, "MSA_N")
    da = ds[var]
    da = _normalize_lon_to_180(_ensure_lat_ascending(da))
    return da


def _load_land_mask(oceans_nc: Path, template: xr.DataArray) -> Optional[xr.DataArray]:
    if not oceans_nc.exists():
        return None
    try:
        ds = _open_ds(oceans_nc)
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

        _log("[WARN] oceans mask suspicious, skipping background land mask.")
        return None
    except Exception as e:
        _log(f"[WARN] cannot use oceans mask: {e}. skipping background land mask.")
        return None


def _base_ndep_white() -> LinearSegmentedColormap:
    # 1.0 is near-white by default (great for showing impacts), then we may override top to black.
    stops = [
        (0.00, "#081d58"),  # deep blue
        (0.15, "#225ea8"),  # blue
        (0.30, "#41b6c4"),  # cyan
        (0.45, "#7fcdbb"),  # blue-green
        (0.60, "#c7e9b4"),  # light green
        (0.72, "#ffffcc"),  # pale yellow
        (0.84, "#fed976"),  # yellow/orange
        (0.92, "#fd8d3c"),  # orange
        (0.97, "#f03b20"),  # red-orange
        (1.00, "#fff5f0"),  # near-white
    ]
    return LinearSegmentedColormap.from_list("ndep_white", stops, N=256)


def _with_black_top(cmap: LinearSegmentedColormap, top_frac: float = 0.02) -> LinearSegmentedColormap:
    """
    Replace the top end of the colormap with black.

    top_frac = 0.02 -> last 2% of the scale becomes black.
    This approximates "black when value is 1" while remaining robust numerically.
    """
    top_frac = float(top_frac)
    if not (0.0 < top_frac < 0.2):
        raise ValueError("top_frac should be in (0, 0.2)")

    n = 256
    arr = cmap(np.linspace(0, 1, n))
    k = max(1, int(round(top_frac * n)))
    arr[-k:, :3] = 0.0  # RGB -> black
    arr[-k:, 3] = 1.0   # alpha
    return LinearSegmentedColormap.from_list(f"{cmap.name}_blacktop", arr, N=n)


def _get_cmap(name: str, black_top: bool, black_top_frac: float) -> LinearSegmentedColormap | str:
    n = (name or "").strip().lower()

    if n in {"ndep_white", "best", "readable"}:
        cmap = _base_ndep_white()
        if black_top:
            cmap = _with_black_top(cmap, top_frac=black_top_frac)
        return cmap

    if n in {"turbo"}:
        return "turbo"

    return name


def plot_msa_n(
    scen: int,
    year: int,
    msa_dir: Path = DEFAULT_DIR,
    oceans_nc: Path = DEFAULT_OCEANS,
    out_dir: Path = DEFAULT_OUT,
    cmap: str = "ndep_white",
    vmin: float = 0.8,
    vmax: float = 1.0,
    gamma: float = 0.35,
    origin: str = "lower",
    alpha: float = 0.90,
    black_top: bool = True,
    black_top_frac: float = 0.02,
    show: bool = False,
    dpi: int = 200,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    da = _load_msa_n(msa_dir, scen)

    if "year" not in da.dims:
        raise ValueError(f"Expected dim 'year'. dims={da.dims}")
    if year not in da["year"].values:
        yrs = da["year"].values
        raise ValueError(f"Year {year} not available. Range {yrs.min()}..{yrs.max()}")

    da_y = _normalize_lon_to_180(_ensure_lat_ascending(da.sel(year=year)))

    lon = da_y["lon"].values
    lat = da_y["lat"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # Background land mask (grey)
    land = _load_land_mask(oceans_nc, template=da_y)
    if land is not None:
        da_y = da_y.where(land)  # ocean -> NaN

    # Boost contrast near 1
    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    cmap_obj = _get_cmap(cmap, black_top=black_top, black_top_frac=black_top_frac)
    if hasattr(cmap_obj, "set_bad"):
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_bad(color=(1, 1, 1, 0))  # transparent ocean

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # draw land background first
    if land is not None:
        ax.imshow(
            land.astype(np.float32).values,
            extent=extent,
            origin=origin,
            interpolation="nearest",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            alpha=0.22,
            aspect="auto",
        )

    # overlay msa layer
    im = ax.imshow(
        da_y.values,
        extent=extent,
        origin=origin,
        interpolation="nearest",
        cmap=cmap_obj,
        norm=norm,
        alpha=alpha,
        aspect="auto",
    )

    ax.set_title(f"MSA_N — SSP{scen} — {year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("MSA_N (0=impacted, 1=intact)")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    fig.tight_layout()

    out_path = out_dir / f"MSA_N_{scen}_{year}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    _log(f"[OK] wrote {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MSA_N for a given scenario/year.")
    p.add_argument("--scen", type=int, required=True, choices=[126, 370, 585])
    p.add_argument("--year", type=int, required=True)

    p.add_argument("--msa-dir", type=str, default=str(DEFAULT_DIR))
    p.add_argument("--oceans-nc", type=str, default=str(DEFAULT_OCEANS))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))

    p.add_argument("--cmap", type=str, default="ndep_white")
    p.add_argument("--vmin", type=float, default=0.8)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.35)
    p.add_argument("--alpha", type=float, default=0.90)
    p.add_argument("--black-top", action="store_true", default=True)
    p.add_argument("--no-black-top", dest="black_top", action="store_false")
    p.add_argument("--black-top-frac", type=float, default=0.02, help="fraction of top colormap set to black (e.g. 0.02)")
    p.add_argument("--origin", type=str, default="lower", choices=["lower", "upper"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    plot_msa_n(
        scen=a.scen,
        year=a.year,
        msa_dir=Path(a.msa_dir),
        oceans_nc=Path(a.oceans_nc),
        out_dir=Path(a.out_dir),
        cmap=a.cmap,
        vmin=a.vmin,
        vmax=a.vmax,
        gamma=a.gamma,
        alpha=a.alpha,
        black_top=a.black_top,
        black_top_frac=a.black_top_frac,
        origin=a.origin,
        show=a.show,
        dpi=a.dpi,
    )


if __name__ == "__main__":
    main()
