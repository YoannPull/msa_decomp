# scripts/plot_msa_enc.py
# -*- coding: utf-8 -*-

"""
Plot MSA_ENC maps from the datamart and save figures to outputs/plots/.

Features (same spirit as your other plot scripts)
-------------------------------------------------
- Reads MSA_ENC_{scen}.nc produced by scripts/build_msa_enc.py
- Robust year selection:
  - If dataset has 'year' dim -> select directly
  - Else if has datetime 'time' -> select nearest to Jan 1st of requested year
  - Else if numeric 'time' -> select nearest numeric value
- Correct orientation:
  - If lat is ascending (-90..90): origin="lower"
  - Else: origin="upper"
- Consistent MSA colormap (red -> yellow -> green)
- Optional vmin/vmax
- Optional show

Outputs:
- outputs/plots/MSA_ENC_{scen}_{year}.png

Usage:
poetry run python scripts/plot_msa_enc.py --scen 126 --year 2100
poetry run python scripts/plot_msa_enc.py --scen 370 --year 2030 --vmin 0.85 --vmax 1.0
poetry run python scripts/plot_msa_enc.py --scen 585 --year 2050 --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 1) Paths (override via CLI if needed)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    datamart_dir: Path = Path("data/datamart/MSA_ENC")
    out_dir: Path = Path("outputs/plots")


# =============================================================================
# 2) Colormap (MSA-style gradient)
# =============================================================================

MSA_CMAP = LinearSegmentedColormap.from_list(
    "msa_grad",
    [
        "#7f0000",  # very low
        "#d7301f",
        "#fc8d59",
        "#fee08b",
        "#1a9850",  # high
    ],
)

DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 1.0


# =============================================================================
# 3) Utilities
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _pick_var(ds: xr.Dataset, preferred: str = "MSA_ENC") -> str:
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
        if year not in da["year"].values:
            # nearest year
            y = da["year"].values.astype(int)
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
        if t.dtype == object and hasattr(t.values[0], "year"):
            years = np.array([getattr(x, "year", np.nan) for x in t.values], dtype=float)
            idx = int(np.nanargmin(np.abs(years - year)))
            return da.isel(time=idx)

        # numeric
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


# =============================================================================
# 4) Plot
# =============================================================================

def plot_msa_enc(
    nc_path: Path,
    out_png: Path,
    year: int,
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    _require_exists(nc_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    var = _pick_var(ds, preferred="MSA_ENC")
    sl = _select_slice(ds, varname=var, year=year)

    # Ensure 2D in lat/lon order if possible
    # (xarray might keep dims as (lon, lat) in some weird files; we handle it)
    if set(sl.dims) >= {"lat", "lon"}:
        sl = sl.transpose("lat", "lon")

    arr = sl.values.astype(np.float32)
    mn, mx, nan_pct = _qc_2d(arr)

    origin = _infer_origin_from_lat(sl)

    plt.figure(figsize=(14, 6))
    im = plt.imshow(
        arr,
        origin=origin,
        cmap=MSA_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    scen_label = nc_path.stem.replace("MSA_ENC_", "SSP")
    plt.title(title or f"MSA_ENC — {scen_label} — year≈{year} | min/max={mn:.3g}/{mx:.3g} | nan%={nan_pct:.2f}")
    plt.colorbar(im, label="MSA")
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
    p = argparse.ArgumentParser(description="Plot MSA_ENC for a given scenario/year.")
    p.add_argument("--scen", required=True, choices=["126", "370", "585"], help="Scenario (126/370/585).")
    p.add_argument("--year", type=int, required=True, help="Target year (nearest will be used if missing).")

    p.add_argument("--datamart-dir", type=str, default=str(Paths.datamart_dir))
    p.add_argument("--out-dir", type=str, default=str(Paths.out_dir))

    p.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    p.add_argument("--vmax", type=float, default=DEFAULT_VMAX)

    p.add_argument("--show", action="store_true", help="Display the figure.")
    return p.parse_args()


def main() -> None:
    a = _parse_args()

    datamart_dir = Path(a.datamart_dir)
    out_dir = Path(a.out_dir)

    nc_path = datamart_dir / f"MSA_ENC_{a.scen}.nc"
    out_png = out_dir / f"MSA_ENC_{a.scen}_{a.year}.png"

    plot_msa_enc(
        nc_path=nc_path,
        out_png=out_png,
        year=a.year,
        vmin=a.vmin,
        vmax=a.vmax,
        show=a.show, 
    )


if __name__ == "__main__":
    main()
