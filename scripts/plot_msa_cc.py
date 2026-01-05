# scripts/plot_msa_cc.py
# -*- coding: utf-8 -*-

"""
Plot MSA_CC for a given scenario/year and save the figure to outputs/plots/.

Features
--------
- Reads MSA_CC either from:
    - data/datamart/MSA_CC/MSA_CC.nc (preferred, with scenario dim)
    - or data/datamart/MSA_CC/MSA_CC_{scen}.nc
- Masks oceans using data/datamart/oceans.nc when available (robust heuristics),
  otherwise leaves as-is (or relies on NaNs already present).
- Handles lon in [0,360) by converting to [-180,180).
- Handles lat descending by sorting to ascending.
- imshow with explicit extent and configurable origin ('lower' or 'upper').

Usage
-----
poetry run python scripts/plot_msa_cc.py --scen 126 --year 2100
poetry run python scripts/plot_msa_cc.py --scen 370 --year 2030 --show
poetry run python scripts/plot_msa_cc.py --scen 585 --year 2050 --origin upper --cmap RdYlGn

Notes
-----
- Default cmap is "RdYlGn" (0=red impacted, 1=green intact).
- MSA_CC is expected in [0,1]. You can override vmin/vmax if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


DEFAULT_MSA_DIR = Path("data/datamart/MSA_CC")
DEFAULT_OCEANS = Path("data/datamart/oceans.nc")
DEFAULT_OUTDIR = Path("outputs/plots")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


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


def _open_nc(path: Path, chunks: Optional[dict] = None) -> xr.Dataset:
    _require_exists(path)
    return xr.open_dataset(path, decode_times=True, use_cftime=True, chunks=chunks)


def _pick_var(ds: xr.Dataset, preferred: str = "MSA_CC") -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


def _infer_land_mask(oceans_nc: Path, template: xr.DataArray) -> xr.DataArray | None:
    """
    Return land_mask(lat, lon) aligned to template, or None if not possible.

    Heuristics:
      - If mask values look like {0,1}, decide whether land is 1 or 0 based on
        plausible global land fraction.
    """
    if not oceans_nc.exists():
        return None

    try:
        ds = _open_nc(oceans_nc, chunks=None)
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
            land = (m == 1) & (~xr.ufuncs.isnan(m))
            _log(f"[INFO] Using oceans mask: land == 1 (frac_ones={frac_ones:.3f})")
            return land

        if plausible(frac_zeros):
            land = (m == 0) & (~xr.ufuncs.isnan(m))
            _log(f"[INFO] Using oceans mask: land == 0 (frac_zeros={frac_zeros:.3f})")
            return land

        _log(
            f"[WARN] oceans mask not plausible (frac_ones={frac_ones:.3f}, frac_zeros={frac_zeros:.3f}). "
            "Skipping ocean masking."
        )
        return None

    except Exception as e:
        _log(f"[WARN] Could not read/apply oceans mask ({oceans_nc}): {e}. Skipping ocean masking.")
        return None


def _load_msa_cc(msa_dir: Path, scen: int) -> xr.DataArray:
    """
    Load MSA_CC as DataArray with dims including year, lat, lon.
    Supports combined file with scenario dimension, or scenario-specific file.
    """
    combined = msa_dir / "MSA_CC.nc"
    if combined.exists():
        ds = _open_nc(combined, chunks=None)
        var = _pick_var(ds, preferred="MSA_CC")
        da = ds[var]
        da = _normalize_lon_to_180(_ensure_lat_ascending(da))
        if "scenario" not in da.dims:
            raise ValueError(f"{combined} exists but has no 'scenario' dim.")
        if scen not in da["scenario"].values:
            raise ValueError(f"Scenario {scen} not found in {combined}. Available={da['scenario'].values}")
        return da.sel(scenario=scen)

    scen_file = msa_dir / f"MSA_CC_{scen}.nc"
    _require_exists(scen_file)
    ds = _open_nc(scen_file, chunks=None)
    var = _pick_var(ds, preferred="MSA_CC")
    da = ds[var]
    da = _normalize_lon_to_180(_ensure_lat_ascending(da))
    return da


def plot_msa_cc(
    scen: int,
    year: int,
    msa_dir: Path = DEFAULT_MSA_DIR,
    oceans_nc: Path = DEFAULT_OCEANS,
    out_dir: Path = DEFAULT_OUTDIR,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    origin: str = "lower",
    show: bool = False,
    dpi: int = 200,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    da = _load_msa_cc(msa_dir=msa_dir, scen=scen)

    if "year" not in da.dims:
        raise ValueError(f"MSA_CC must have dim 'year'. Found dims={da.dims}")
    if year not in da["year"].values:
        years = da["year"].values
        raise ValueError(f"Year {year} not available. Available range: {years.min()}..{years.max()}")

    da_y = da.sel(year=year)

    # Ocean masking (optional)
    land_mask = _infer_land_mask(oceans_nc=oceans_nc, template=da_y)
    if land_mask is not None:
        da_y = da_y.where(land_mask)

    # Ensure coordinates sane
    da_y = _normalize_lon_to_180(_ensure_lat_ascending(da_y))

    # Extent for imshow: [xmin, xmax, ymin, ymax]
    lon = da_y["lon"].values
    lat = da_y["lat"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(
        da_y.values,
        extent=extent,
        origin=origin,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="auto",
    )

    ax.set_title(f"MSA_CC (Climate Change) — SSP{scen} — {year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("MSA_CC (0=impacted, 1=intact)")

    # Keep map-like framing
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    fig.tight_layout()

    out_path = out_dir / f"MSA_CC_{scen}_{year}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    _log(f"[OK] wrote {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MSA_CC for a given scenario/year.")

    p.add_argument("--scen", type=int, required=True, choices=[126, 370, 585], help="Scenario (126, 370, 585).")
    p.add_argument("--year", type=int, required=True, help="Year to plot (e.g., 2030, 2050, 2100).")

    p.add_argument("--msa-dir", type=str, default=str(DEFAULT_MSA_DIR), help="Directory containing MSA_CC NetCDF files.")
    p.add_argument("--oceans-nc", type=str, default=str(DEFAULT_OCEANS), help="Optional oceans mask NetCDF.")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTDIR), help="Output directory for plots.")

    p.add_argument("--cmap", type=str, default="RdYlGn", help="Matplotlib colormap name.")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)

    p.add_argument("--origin", type=str, default="lower", choices=["lower", "upper"], help="imshow origin.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show", action="store_true", help="Display interactively.")

    return p.parse_args()


def main() -> None:
    a = _parse_args()
    plot_msa_cc(
        scen=a.scen,
        year=a.year,
        msa_dir=Path(a.msa_dir),
        oceans_nc=Path(a.oceans_nc),
        out_dir=Path(a.out_dir),
        cmap=a.cmap,
        vmin=a.vmin,
        vmax=a.vmax,
        origin=a.origin,
        show=a.show,
        dpi=a.dpi,
    )


if __name__ == "__main__":
    main()
