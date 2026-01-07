# carrefour/legacy/plot_legacy_msa.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


@dataclass(frozen=True)
class PlotLegacyConfig:
    legacy_root: Path
    out_dir: Path
    year: int
    scens: List[str]
    vars: List[str]
    cmap: str = "msa"  # "msa" | "whitegreen" | "greens"
    gamma: float = 1.0
    base_year: int = 2015
    engine: Optional[str] = None
    flip_lat: bool = False
    origin: str = "upper"  # "upper" | "lower" | "auto"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _open_ds(nc_path: Path, engine: Optional[str]) -> xr.Dataset:
    return xr.open_dataset(nc_path, decode_times=False, mask_and_scale=True, engine=engine)


def _legacy_nc_path(root: Path, var: str, scen: str) -> Path:
    if var == "MSA_ROAD":
        return root / "MSA_ROAD" / "msaroad.nc"
    return root / var / f"{var}_{scen}.nc"


def _pick_var(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not find '{preferred}' in data_vars={list(ds.data_vars)}")


def _select_year_slice(da: xr.DataArray, year: int, base_year: int) -> xr.DataArray:
    if "year" in da.dims:
        years = da["year"].values.astype(int)
        if year in years:
            return da.sel(year=year)
        nearest = int(years[np.argmin(np.abs(years - year))])
        _log(f"[WARN] year {year} not found, using nearest year={nearest}")
        return da.sel(year=nearest)

    if "time" in da.dims:
        idx = int(year - base_year)
        idx = max(0, min(idx, da.sizes["time"] - 1))
        return da.isel(time=idx)

    return da


def _clean_fill(arr: np.ndarray, fill_thresh: float = 1e20) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    bad = ~np.isfinite(arr) | (np.abs(arr) > fill_thresh)
    if bad.any():
        arr = arr.copy()
        arr[bad] = np.nan
    return arr


def _apply_gamma(arr: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return arr
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    out = arr.copy()
    m = np.isfinite(out)
    out[m] = np.clip(out[m], 0.0, 1.0) ** gamma
    return out


def _resolve_origin(cfg_origin: str, lat: Optional[np.ndarray]) -> str:
    if cfg_origin in ("upper", "lower"):
        return cfg_origin
    if cfg_origin != "auto":
        raise ValueError("origin must be one of: upper, lower, auto")
    if lat is None or lat.size < 2:
        return "upper"
    return "lower" if lat[0] < lat[-1] else "upper"


def _make_cmap(name: str):
    """
    Supported:
      - 'greens'    -> Matplotlib Greens (classic)
      - 'whitegreen'-> white->green custom
      - 'msa'       -> red->yellow->green
    """
    if name == "greens":
        return cm.get_cmap("Greens").copy()

    if name == "whitegreen":
        return LinearSegmentedColormap.from_list(
            "white_to_green",
            [(0.0, "#ffffff"), (1.0, "#1a9850")],
        )

    if name == "msa":
        return LinearSegmentedColormap.from_list(
            "msa",
            ["#7f0000", "#d7301f", "#fc8d59", "#fee08b", "#1a9850"],
        )

    raise ValueError("cmap must be one of: msa, whitegreen, greens")


def plot_legacy(cfg: PlotLegacyConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    cmap = _make_cmap(cfg.cmap)
    cmap = cmap.copy()
    cmap.set_bad(alpha=0.0)  # NaN transparent

    for scen in cfg.scens:
        for var in cfg.vars:
            nc_path = _legacy_nc_path(cfg.legacy_root, var, scen)
            _require_exists(nc_path)

            ds = _open_ds(nc_path, cfg.engine)
            vname = _pick_var(ds, preferred=var)
            da = ds[vname]

            sl = _select_year_slice(da, cfg.year, cfg.base_year)

            if set(sl.dims) >= {"lat", "lon"}:
                sl = sl.transpose("lat", "lon")
            while sl.ndim > 2:
                sl = sl.isel({sl.dims[0]: 0})

            arr = _clean_fill(sl.values)

            if cfg.flip_lat:
                arr = np.flipud(arr)

            arr = _apply_gamma(arr, cfg.gamma)

            lat = np.asarray(sl["lat"].values, dtype=float) if "lat" in sl.coords else None
            lon = np.asarray(sl["lon"].values, dtype=float) if "lon" in sl.coords else None
            origin = _resolve_origin(cfg.origin, lat)

            finite = np.isfinite(arr)
            nanpct = 100.0 * (1.0 - finite.mean()) if arr.size else float("nan")
            mn = float(np.nanmin(arr)) if finite.any() else float("nan")
            mx = float(np.nanmax(arr)) if finite.any() else float("nan")

            plt.figure(figsize=(14, 6))

            if lat is not None and lon is not None:
                extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
                im = plt.imshow(
                    arr,
                    origin=origin,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    extent=extent,
                )
            else:
                im = plt.imshow(arr, origin=origin, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")

            plt.colorbar(im, label=var)
            plt.title(
                f"LEGACY {var} — scen={scen} — year≈{cfg.year} | min/max={mn:.3g}/{mx:.3g} | nan%={nanpct:.2f}"
                f" | origin={origin} flip={cfg.flip_lat} cmap={cfg.cmap}"
            )
            plt.axis("off")
            plt.tight_layout()

            out_png = cfg.out_dir / f"LEGACY_{var}_{scen}_{cfg.year}.png"
            plt.savefig(out_png, dpi=220)
            plt.close()

            _log(f"[OK] saved {out_png}")


__all__ = ["PlotLegacyConfig", "plot_legacy"]
