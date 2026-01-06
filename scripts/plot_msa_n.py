# scripts/plot_msa_n.py
# -*- coding: utf-8 -*-

"""
Plot MSA_N for a given scenario/year and save to outputs/plots/.

Highlights (best readability for near-1 fields):
- Default colormap: 'best' -> matplotlib 'turbo' (very readable / high contrast)
- Default normalization: PowerNorm(gamma=0.35) to reveal differences close to 1
- Default range: vmin=0.8, vmax=1.0
- Optional 'ndep' custom palette (blue->cyan->green->yellow->orange->red)

Usage:
  poetry run python scripts/plot_msa_n.py --scen 585 --year 2100
  poetry run python scripts/plot_msa_n.py --scen 585 --year 2100 --cmap ndep --vmin 0.8 --vmax 1.0 --gamma 0.35
  poetry run python scripts/plot_msa_n.py --scen 585 --year 2100 --cmap best --vmin 0.9 --vmax 1.0 --gamma 0.30 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_DIR = Path("data/datamart/MSA_N")
DEFAULT_OCEANS = Path("data/datamart/oceans.nc")
DEFAULT_OUT = Path("outputs/plots")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _open_ds(p: Path) -> xr.Dataset:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return xr.open_dataset(p, decode_times=True, use_cftime=True)


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


def _infer_land_mask(oceans_nc: Path, template: xr.DataArray) -> Optional[xr.DataArray]:
    """
    Optional: mask oceans if oceans.nc exists and looks plausible.
    """
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

        _log("[WARN] oceans mask suspicious, skipping ocean masking.")
        return None
    except Exception as e:
        _log(f"[WARN] cannot use oceans mask: {e}. skipping ocean masking.")
        return None


def _load_msa_n(msa_dir: Path, scen: int) -> xr.DataArray:
    """
    Load MSA_N either from combined file MSA_N.nc (with scenario dim)
    or from per-scenario file MSA_N_{scen}.nc.
    """
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


def _get_cmap(name: str) -> Union[str, LinearSegmentedColormap]:
    """
    Colormap choices:
    - 'best' or 'turbo' : matplotlib 'turbo' (high contrast, best readability)
    - 'ndep'            : custom blue->red palette (close to your example)
    - anything else     : passed through to matplotlib (e.g., 'viridis')
    """
    n = (name or "").strip().lower()

    if n in {"best", "turbo"}:
        return "turbo"

    if n in {"ndep", "msa_ndep", "msa_n"}:
        colors = [
            "#001a8f",  # deep blue
            "#005bff",  # blue
            "#35cfff",  # cyan
            "#66ff66",  # green
            "#ffff66",  # yellow
            "#ff9933",  # orange
            "#7a0c0c",  # dark red
        ]
        return LinearSegmentedColormap.from_list("ndep_like", colors, N=256)

    return name


def plot_msa_n(
    scen: int,
    year: int,
    msa_dir: Path = DEFAULT_DIR,
    oceans_nc: Path = DEFAULT_OCEANS,
    out_dir: Path = DEFAULT_OUT,
    cmap: str = "best",
    vmin: float = 0.8,
    vmax: float = 1.0,
    gamma: float = 0.35,
    origin: str = "lower",
    show: bool = False,
    dpi: int = 200,
) -> Path:
    """
    gamma:
      - < 1 : boosts contrast near vmax (recommended for MSA close to 1)
      - = 1 : linear mapping
      - > 1 : compresses near vmax
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    da = _load_msa_n(msa_dir, scen)

    if "year" not in da.dims:
        raise ValueError(f"Expected dim 'year'. dims={da.dims}")
    if year not in da["year"].values:
        yrs = da["year"].values
        raise ValueError(f"Year {year} not available. Range {yrs.min()}..{yrs.max()}")

    da_y = da.sel(year=year)

    land = _infer_land_mask(oceans_nc, da_y)
    if land is not None:
        da_y = da_y.where(land)

    da_y = _normalize_lon_to_180(_ensure_lat_ascending(da_y))

    lon = da_y["lon"].values
    lat = da_y["lat"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # Perceptual contrast for near-1 fields
    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(
        da_y.values,
        extent=extent,
        origin=origin,
        interpolation="nearest",
        cmap=_get_cmap(cmap),
        norm=norm,
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

    p.add_argument("--cmap", type=str, default="best", help="best/turbo (recommended) or ndep or any matplotlib cmap.")
    p.add_argument("--vmin", type=float, default=0.8)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.35, help="PowerNorm gamma (<1 boosts contrast near vmax).")
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
        origin=a.origin,
        show=a.show,
        dpi=a.dpi,
    )


if __name__ == "__main__":
    main()
