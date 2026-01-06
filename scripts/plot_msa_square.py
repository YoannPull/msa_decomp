# scripts/plot_msa_square.py
# -*- coding: utf-8 -*-

"""
Plot MSA_SQUARE maps from the datamart and save figures to outputs/plots/.

Features
--------
- Reads MSA_SQUARE_{scen}.nc produced by scripts/build_msa_square.py
- Robust year selection:
  - If dataset has 'year' dim -> select directly
  - Else if has datetime 'time' -> select nearest to Jan 1st of requested year
  - Else if has numeric 'time' -> select nearest numeric value
- Correct orientation:
  - If lat is ascending (-90..90): origin="lower"
  - Else: origin="upper"
- Colormap: Matplotlib "Greens" (same spirit as your reference gradient)
- Optional vmin/vmax
- Optional "black at 1" (pixels with value==1 to black)
- Optional gamma correction (PowerNorm):
  - gamma < 1 pushes more pixels toward green
  - gamma > 1 pushes toward white
- Optional show

Outputs:
- outputs/plots/MSA_SQUARE_{scen}_{year}.png

Usage:
poetry run python scripts/plot_msa_square.py --scen 126 --year 2100
poetry run python scripts/plot_msa_square.py --scen 370 --year 2030 --vmin 0.5 --vmax 1.0
poetry run python scripts/plot_msa_square.py --scen 585 --year 2050 --black-at-1
poetry run python scripts/plot_msa_square.py --scen 585 --year 2050 --gamma 0.7
poetry run python scripts/plot_msa_square.py --scen 585 --year 2050 --gamma 0.7 --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm


# =============================================================================
# 1) Paths (override via CLI if needed)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    datamart_dir: Path = Path("data/datamart/MSA_SQUARE")
    out_dir: Path = Path("outputs/plots")


# =============================================================================
# 2) Colormap ("Greens" like your reference)
# =============================================================================

def _truncate_cmap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256) -> LinearSegmentedColormap:
    """
    Keep only a sub-range of an existing colormap.
    Useful to avoid pure white at the very start.
    """
    minval = float(minval)
    maxval = float(maxval)
    if not (0.0 <= minval < maxval <= 1.0):
        raise ValueError(f"Invalid minval/maxval: {minval}/{maxval} (must satisfy 0<=min<max<=1)")
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )

# Base colormap: matches the gradient in your screenshot
BASE_CMAP = cm.get_cmap("Greens")

# Option A (default): full Greens (0 is almost white)
MSA_CMAP = BASE_CMAP

# Option B: avoid the very-white beginning (uncomment if you want a greener low end)
# MSA_CMAP = _truncate_cmap(BASE_CMAP, minval=0.05, maxval=1.0)

DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 1.0
DEFAULT_GAMMA = 1.0


# =============================================================================
# 3) Utilities
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _pick_var(ds: xr.Dataset, preferred: str = "MSA_SQUARE") -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


def _infer_origin_from_lat(da2d: xr.DataArray) -> str:
    if "lat" not in da2d.coords:
        return "upper"
    lat = da2d["lat"].values
    if lat.size >= 2 and lat[0] < lat[-1]:
        return "lower"
    return "upper"


def _select_slice(ds: xr.Dataset, varname: str, year: int) -> xr.DataArray:
    """
    Return a 2D slice (lat, lon) for the requested year.
    Supports:
      - 'year' dim
      - 'time' datetime-like
      - 'time' numeric (nearest)
    """
    da = ds[varname]

    if "year" in da.dims:
        y = da["year"].values.astype(int)
        if year not in y:
            nearest = int(y[np.argmin(np.abs(y - year))])
            _log(f"[WARN] year {year} not found, using nearest year={nearest}")
            year = nearest
        return da.sel(year=year)

    if "time" in da.dims:
        t = da["time"]

        # datetime-like
        if np.issubdtype(t.dtype, np.datetime64):
            target = np.datetime64(f"{year}-01-01")
            return da.sel(time=target, method="nearest")

        # cftime objects (object dtype)
        if t.dtype == object and len(t.values) and hasattr(t.values[0], "year"):
            years = np.array([getattr(x, "year", np.nan) for x in t.values], dtype=float)
            idx = int(np.nanargmin(np.abs(years - year)))
            return da.isel(time=idx)

        # numeric time axis
        tv = t.values.astype(float)
        idx = int(np.nanargmin(np.abs(tv - float(year))))
        return da.isel(time=idx)

    raise ValueError(f"Could not find 'year' or 'time' dimension in variable dims={da.dims}")


def _qc_2d(arr: np.ndarray) -> Tuple[float, float, float]:
    finite = np.isfinite(arr)
    nan_pct = 100.0 * (1.0 - finite.mean()) if arr.size else float("nan")
    mn = float(np.nanmin(arr)) if finite.any() else float("nan")
    mx = float(np.nanmax(arr)) if finite.any() else float("nan")
    return mn, mx, nan_pct


def _make_norm(vmin: float, vmax: float, gamma: float) -> Normalize:
    """
    Create a normalization:
      - gamma == 1 -> linear Normalize
      - gamma != 1 -> PowerNorm (gamma correction)
    """
    if gamma is None or float(gamma) == 1.0:
        return Normalize(vmin=vmin, vmax=vmax, clip=True)
    return PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax, clip=True)


def _apply_black_at_1(arr: np.ndarray, norm: Normalize, eps: float = 1e-6) -> np.ndarray:
    """
    Return an RGBA image array:
      - apply norm + colormap
      - pixels ~1.0 -> black (alpha=1)
      - NaNs -> transparent (alpha=0)
    """
    rgba = MSA_CMAP(norm(arr))

    # NaNs -> fully transparent
    nanmask = ~np.isfinite(arr)
    rgba[nanmask, 3] = 0.0

    # ~1.0 -> black (keep alpha)
    ones = np.isfinite(arr) & (np.abs(arr - 1.0) <= eps)
    rgba[ones, 0:3] = 0.0
    rgba[ones, 3] = 1.0
    return rgba


# =============================================================================
# 4) Plot
# =============================================================================

def plot_msa_square(
    nc_path: Path,
    out_png: Path,
    year: int,
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    title: Optional[str] = None,
    black_at_1: bool = False,
    show: bool = False,
    gamma: float = DEFAULT_GAMMA,
) -> None:
    _require_exists(nc_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    var = _pick_var(ds, preferred="MSA_SQUARE")
    sl = _select_slice(ds, varname=var, year=year)

    # Ensure 2D in lat/lon order if possible
    if set(sl.dims) >= {"lat", "lon"}:
        sl = sl.transpose("lat", "lon")

    arr = sl.values.astype(np.float32)
    mn, mx, nan_pct = _qc_2d(arr)

    origin = _infer_origin_from_lat(sl)
    norm = _make_norm(vmin=vmin, vmax=vmax, gamma=gamma)

    plt.figure(figsize=(14, 6))

    if black_at_1:
        rgba = _apply_black_at_1(arr, norm=norm)
        plt.imshow(rgba, origin=origin, interpolation="nearest")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=MSA_CMAP)
        plt.colorbar(sm, label="MSA")
    else:
        im = plt.imshow(
            arr,
            origin=origin,
            cmap=MSA_CMAP,
            norm=norm,
            interpolation="nearest",
        )
        plt.colorbar(im, label="MSA")

    scen_label = nc_path.stem.replace("MSA_SQUARE_", "SSP")
    ttl = title or (
        f"MSA_SQUARE — {scen_label} — year≈{year} | "
        f"min/max={mn:.3g}/{mx:.3g} | nan%={nan_pct:.2f} | gamma={gamma:g}"
    )
    plt.title(ttl)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    if show:
        plt.show()
    plt.close()

    _log(f"[OK] saved {out_png}")


# =============================================================================
# 5) CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MSA_SQUARE for a given scenario/year.")
    p.add_argument("--scen", required=True, choices=["126", "370", "585"], help="Scenario (126/370/585).")
    p.add_argument("--year", type=int, required=True, help="Target year (nearest will be used if missing).")

    p.add_argument("--datamart-dir", type=str, default=str(Paths.datamart_dir))
    p.add_argument("--out-dir", type=str, default=str(Paths.out_dir))

    p.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    p.add_argument("--vmax", type=float, default=DEFAULT_VMAX)

    p.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Gamma correction for color normalization. <1 pushes more pixels to green, >1 pushes to white.",
    )

    p.add_argument("--black-at-1", action="store_true", help="Render pixels with value==1 in black.")
    p.add_argument("--show", action="store_true", help="Display the figure.")
    return p.parse_args()


def main() -> None:
    a = _parse_args()

    datamart_dir = Path(a.datamart_dir)
    out_dir = Path(a.out_dir)

    nc_path = datamart_dir / f"MSA_SQUARE_{a.scen}.nc"
    out_png = out_dir / f"MSA_SQUARE_{a.scen}_{a.year}.png"

    plot_msa_square(
        nc_path=nc_path,
        out_png=out_png,
        year=a.year,
        vmin=a.vmin,
        vmax=a.vmax,
        gamma=a.gamma,
        black_at_1=a.black_at_1,
        show=a.show,
    )


if __name__ == "__main__":
    main()
