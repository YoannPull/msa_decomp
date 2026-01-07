# carrefour/legacy/plot_legacy_msa.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


# -----------------------------------------------------------------------------
# Pretty metadata (pressure-aware)
# -----------------------------------------------------------------------------

# You can adapt these names to your paper
VAR_META: Dict[str, Dict[str, str]] = {
    "MSA_SQUARE": {"label": "MSA (combined)", "pressure": "MSA"},
    "MSA_LU": {"label": "MSA (land use)", "pressure": "LU"},
    "MSA_ROAD": {"label": "MSA (roads)", "pressure": "ROAD"},
    "MSA_CC": {"label": "MSA (climate change)", "pressure": "CC"},
    "MSA_N": {"label": "MSA (nitrogen deposition)", "pressure": "N"},
    "MSA_ENC": {"label": "MSA (human encroachment)", "pressure": "ENC"},
}

# Pressure hues (colorblind-friendly-ish, distinct)
PRESSURE_HEX = {
    "LU": "#b35806",    # brown/orange (land use)
    "ROAD": "#6a51a3",  # purple (roads)
    "CC": "#2171b5",    # blue (climate)
    "N": "#238b45",     # green (nitrogen)
    "ENC": "#dd1c77",   # magenta (encroachment)
    "MSA": "#1a9850",   # green (overall)
}


def _scenario_label(scen: str) -> str:
    """
    Optional: prettier scenario names.
    If you don't want assumptions, keep only 'scen=...'.
    """
    mapping = {
        "126": "SSP1-2.6",
        "245": "SSP2-4.5",
        "370": "SSP3-7.0",
        "585": "SSP5-8.5",
    }
    return mapping.get(str(scen), f"scen {scen}")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotLegacyConfig:
    legacy_root: Path
    out_dir: Path
    year: int
    scens: List[str]
    vars: List[str]

    # cmap:
    # - "auto"      -> per-pressure palettes + Greens for MSA
    # - "greens"    -> classic green
    # - "whitegreen"-> white->green
    # - "msa"       -> red->yellow->green (legacy)
    # - or any Matplotlib cmap name (e.g. "viridis", "YlGn", "PuBu")
    cmap: str = "auto"

    gamma: float = 1.0
    base_year: int = 2015
    engine: Optional[str] = None
    flip_lat: bool = False
    origin: str = "upper"  # "upper" | "lower" | "auto"

    # style:
    # - "screen" -> debug-friendly (transparent NaNs, verbose title)
    # - "paper"  -> publication-ready (grey NaNs, clean titles, dpi 300)
    style: str = "paper"

    # plotting
    figsize: Tuple[float, float] = (12.5, 5.2)
    dpi: int = 300


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _open_ds(nc_path: Path, engine: Optional[str]) -> xr.Dataset:
    return xr.open_dataset(nc_path, decode_times=False, mask_and_scale=True, engine=engine)


def _legacy_nc_path(root: Path, var: str, scen: str) -> Path:
    # Pierre special-case
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


def _white_to_hex_cmap(name: str, hex_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [(0.0, "#ffffff"), (1.0, hex_color)])


def _make_base_cmap(name: str):
    """
    Supported explicit names:
      - 'greens'     -> Matplotlib Greens (classic)
      - 'whitegreen' -> white->green custom
      - 'msa'        -> red->yellow->green
    Otherwise, treat as Matplotlib cmap name.
    """
    if name == "greens":
        return cm.get_cmap("Greens").copy()

    if name == "whitegreen":
        return _white_to_hex_cmap("white_to_green", "#1a9850")

    if name == "msa":
        return LinearSegmentedColormap.from_list(
            "msa",
            ["#7f0000", "#d7301f", "#fc8d59", "#fee08b", "#1a9850"],
        )

    # any Matplotlib cmap name
    try:
        return cm.get_cmap(name).copy()
    except Exception as e:
        raise ValueError(f"Unknown cmap '{name}'. Use auto|msa|greens|whitegreen or a Matplotlib cmap name.") from e


def _cmap_for_var(cfg_cmap: str, var: str) -> LinearSegmentedColormap:
    """
    If cfg_cmap == 'auto': choose per-pressure hues + Greens for MSA.
    Otherwise: use cfg_cmap as requested.
    """
    if cfg_cmap != "auto":
        return _make_base_cmap(cfg_cmap)

    meta = VAR_META.get(var, {"label": var, "pressure": "MSA"})
    pressure = meta.get("pressure", "MSA")

    # MSA combined: classic Greens
    if var == "MSA_SQUARE" or pressure == "MSA":
        return _make_base_cmap("greens")

    # per-pressure hue (white -> hue)
    hex_color = PRESSURE_HEX.get(pressure, "#1a9850")
    return _white_to_hex_cmap(f"{pressure.lower()}_white_to_color", hex_color)


def _set_style(style: str) -> None:
    if style not in ("screen", "paper"):
        raise ValueError("style must be one of: screen, paper")

    if style == "paper":
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "figure.titlesize": 14,
                "legend.fontsize": 10,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
            }
        )
    else:
        # keep defaults (or set minimal)
        plt.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _pretty_label(var: str) -> str:
    return VAR_META.get(var, {}).get("label", var)


def _colorbar_label(var: str) -> str:
    # Classic academic label
    # If you want "BII" or something else, change here.
    return "Mean Species Abundance (MSA, 0–1)"


# -----------------------------------------------------------------------------
# Main plotting
# -----------------------------------------------------------------------------

def plot_legacy(cfg: PlotLegacyConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _set_style(cfg.style)

    for scen in cfg.scens:
        scen_str = str(scen)
        scen_lab = _scenario_label(scen_str)

        for var in cfg.vars:
            nc_path = _legacy_nc_path(cfg.legacy_root, var, scen_str)
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

            cmap = _cmap_for_var(cfg.cmap, var).copy()

            # NaN rendering: paper -> light grey, screen -> transparent
            if cfg.style == "paper":
                cmap.set_bad(color="#f0f0f0", alpha=1.0)
            else:
                cmap.set_bad(alpha=0.0)

            finite = np.isfinite(arr)
            nanpct = 100.0 * (1.0 - finite.mean()) if arr.size else float("nan")
            mn = float(np.nanmin(arr)) if finite.any() else float("nan")
            mx = float(np.nanmax(arr)) if finite.any() else float("nan")

            fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)

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

            # Colorbar: clean for paper
            cbar = plt.colorbar(im, fraction=0.035, pad=0.02)
            cbar.set_label(_colorbar_label(var))
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

            # Titles
            if cfg.style == "paper":
                title = f"{_pretty_label(var)} — {scen_lab}, {cfg.year}"
            else:
                title = (
                    f"LEGACY {var} — scen={scen_str} — year≈{cfg.year} | "
                    f"min/max={mn:.3g}/{mx:.3g} | nan%={nanpct:.2f} | origin={origin} flip={cfg.flip_lat} cmap={cfg.cmap}"
                )

            plt.title(title)
            plt.axis("off")
            plt.tight_layout()

            suffix = "paper" if cfg.style == "paper" else "screen"
            out_png = cfg.out_dir / f"LEGACY_{var}_{scen_str}_{cfg.year}_{suffix}.png"
            fig.savefig(out_png)
            plt.close(fig)

            _log(f"[OK] saved {out_png} (min={mn:.3g}, max={mx:.3g}, nan%={nanpct:.2f})")


__all__ = ["PlotLegacyConfig", "plot_legacy"]
