# carrefour/msa_pipeline/build_msa_road.py
# -*- coding: utf-8 -*-
"""
Build MSA_ROAD datamart NetCDF from road rasters.

Implements the "doc" logic (GRIP-like):
- Convert road rasters to a PRESENCE grid (cell is "non-null" if any road type > 0).
- If input is hi-res (2160x4320), compute n = number of non-null cells in each 3x3 block -> 720x1440.
- Compute MSA with:  MSA = 0.78*(n/9) + (9-n)/9
  -> MSA=1 when n=0, MSA=0.78 when n=9.

Rigorous choices:
- Road rasters nodata is treated as 0 (absence of road), NOT missing scientific data.
- Deterministic orientation normalization to datamart convention:
    row0=south (lat increasing), col0=west (lon increasing, -180..180)
- Ocean masking supports NaN/masked or binary 0/1 masks.

Usage:
  poetry run python carrefour/msa_pipeline/build_msa_road.py
  poetry run python carrefour/msa_pipeline/build_msa_road.py --debug
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import netCDF4 as nc

try:
    import rasterio
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'rasterio'. Install with:\n"
        "  poetry add rasterio\n"
        "Then retry."
    ) from e


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class RoadConfig:
    legacy_root: Path = Path("data/legacy/MSA_ROAD")
    out_root: Path = Path("data/datamart/MSA_ROAD")

    # adapt to your filenames
    tp1: Path = Path("Road/Type_1_Raster.tif")
    tp2: Path = Path("Road/Type_2_Raster.tif")
    tp3: Path = Path("Road/Type_3_Raster.tif")

    scenarios: Tuple[str, ...] = ("126", "370", "585")

    base_year: int = 2015
    n_time: int = 86  # 2015..2100 inclusive

    # grids
    target_shape: Tuple[int, int] = (720, 1440)
    hi_shape: Tuple[int, int] = (2160, 4320)
    downsample_factors: Tuple[int, int] = (3, 3)

    # NetCDF output
    out_dtype = np.float32
    netcdf_format: str = "NETCDF4_CLASSIC"
    compress: bool = True
    complevel: int = 4
    chunksizes: Tuple[int, int, int] = (1, 180, 360)  # (time, lat, lon)
    fill_value: float = np.float32(np.nan)

    # Ocean/land mask
    oceans_path: Path = Path("data/datamart/oceans.nc")
    oceans_var: str = "Oceans"
    oceans_is_land_if_binary: bool = True  # if mask is 0/1, interpret 1 as land (else 1=ocean)

    write_legacy: bool = True


# -----------------------------
# Small helpers
# -----------------------------
def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")


def _pct_nan(x: np.ndarray) -> float:
    return float(np.isnan(x).mean() * 100.0)


def _nodata_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.zeros(arr.shape, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        return np.isclose(arr, nodata, rtol=0.0, atol=0.0) | np.isnan(arr)
    return arr == nodata


def read_raster_2d(
    path: Path,
) -> tuple[np.ndarray, float | None, rasterio.Affine, rasterio.coords.BoundingBox, Optional[object]]:
    """Read band1 as float64 + nodata + transform + bounds + crs."""
    _assert_exists(path)
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64, copy=False)
        nodata = src.nodata
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    return arr, nodata, transform, bounds, crs


def nodata_to_zero(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """
    Roads rasters: nodata usually means "no road", so treat nodata as 0.
    This avoids NaN explosions (like 95% NaN).
    """
    if nodata is None:
        # still sanitize NaNs (just in case)
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
            out = arr.copy()
            out[np.isnan(out)] = 0.0
            return out
        return arr

    out = arr.copy()
    m = _nodata_mask(out, nodata)
    if m.any():
        out[m] = 0.0
    return out


# -----------------------------
# Orientation normalization
# -----------------------------
def normalize_orientation_to_datamart(a_2d: np.ndarray, transform, bounds) -> np.ndarray:
    """
    Enforce datamart orientation:
      - row 0 = south, row end = north
      - col 0 = west, col end = east
      - lon convention: -180..180

    Rasterio typical north-up rasters have transform.e negative => row0 is north.
    """
    out = a_2d

    xres = float(transform.a)
    yres = float(transform.e)
    left = float(bounds.left)
    right = float(bounds.right)

    # latitude: north-up => flip vertically to get south-up
    if yres < 0:
        out = np.flipud(out)

    # longitude direction: if xres < 0, columns reversed
    if xres < 0:
        out = np.fliplr(out)

    # lon convention: 0..360 -> shift to -180..180
    if left >= -1e-6 and right > 180.0 + 1e-6:
        out = np.roll(out, shift=out.shape[1] // 2, axis=1)

    return out


# -----------------------------
# Grid coords (centers)
# -----------------------------
def make_lat_lon_centers(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cell centers for a global regular grid:
      lat: (-90, 90), lon: (-180, 180)
    For (720,1440): 0.25° centers => -89.875..+89.875, -179.875..+179.875
    """
    dlat = 180.0 / ny
    dlon = 360.0 / nx
    lat = (-90.0 + dlat / 2.0) + np.arange(ny) * dlat
    lon = (-180.0 + dlon / 2.0) + np.arange(nx) * dlon
    return lat.astype(np.float32), lon.astype(np.float32)


# -----------------------------
# Ocean/Land mask (robust)
# -----------------------------
def load_land_mask(cfg: RoadConfig) -> np.ndarray | None:
    """
    Returns boolean mask: True=land, False=ocean.
    Supports:
      - NaN/masked oceans convention
      - binary 0/1 masks (configurable meaning)
      - fallback heuristic if weird (warns)
    """
    if not cfg.oceans_path.exists():
        print(f"[WARN] oceans.nc not found at {cfg.oceans_path}. No ocean masking.")
        return None

    with nc.Dataset(cfg.oceans_path) as ds:
        if cfg.oceans_var not in ds.variables:
            raise KeyError(
                f"'{cfg.oceans_var}' not found in {cfg.oceans_path}. "
                f"Available: {list(ds.variables.keys())}"
            )
        M = ds.variables[cfg.oceans_var][:, :]

    Mm = np.ma.filled(M, np.nan) if np.ma.isMaskedArray(M) else np.asarray(M)

    # Case A: NaN-based oceans
    if np.isnan(Mm).any():
        ocean = np.isnan(Mm)
        return ~ocean

    # Case B: binary 0/1
    u = np.unique(Mm)
    if u.size <= 3 and np.all(np.isfinite(u)) and np.nanmin(Mm) >= 0 and np.nanmax(Mm) <= 1:
        if cfg.oceans_is_land_if_binary:
            return Mm.astype(np.int8) == 1
        else:
            return Mm.astype(np.int8) == 0

    # Case C: weird numeric mask
    print(
        "[WARN] oceans mask is neither NaN-based nor binary 0/1. "
        "Applying heuristic thresholding (values > median => land)."
    )
    med = float(np.median(Mm))
    return Mm > med


# -----------------------------
# Core build (doc method)
# -----------------------------
def build_msa_road(cfg: RoadConfig, debug: bool = False) -> np.ndarray:
    """
    Build MSA_ROAD 2D on target grid (720,1440) using the doc method:
      1) presence = (any type > 0)
      2) if hi-res, count non-null cells per 3x3 block => n in [0..9]
      3) msa = 0.78*(n/9) + (9-n)/9
      4) normalize orientation to datamart, mask oceans to NaN
    """
    p1 = cfg.legacy_root / cfg.tp1
    p2 = cfg.legacy_root / cfg.tp2
    p3 = cfg.legacy_root / cfg.tp3

    print("[INFO] Reading road rasters:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)

    tp1, nd1, tr1, b1, crs1 = read_raster_2d(p1)
    tp2, nd2, tr2, b2, crs2 = read_raster_2d(p2)
    tp3, nd3, tr3, b3, crs3 = read_raster_2d(p3)

    if tp1.shape != tp2.shape or tp1.shape != tp3.shape:
        raise ValueError(f"Road rasters must have same shape, got: {tp1.shape}, {tp2.shape}, {tp3.shape}")

    if debug:
        print("[DEBUG] nodata:", nd1, nd2, nd3)
        print("[DEBUG] shapes:", tp1.shape)

    # Treat nodata as 0 (absence of road)
    tp1 = nodata_to_zero(tp1, nd1)
    tp2 = nodata_to_zero(tp2, nd2)
    tp3 = nodata_to_zero(tp3, nd3)

    # Presence of any road type
    presence = (tp1 > 0.0) | (tp2 > 0.0) | (tp3 > 0.0)
    presence = presence.astype(np.float64)

    # Normalize orientation BEFORE aggregation so blocks align with datamart convention
    presence = normalize_orientation_to_datamart(presence, tr1, b1)

    if debug:
        print("[DEBUG] presence%:", float(presence.mean() * 100.0))

    # Aggregate to target grid
    if presence.shape == cfg.hi_shape:
        fy, fx = cfg.downsample_factors
        ny, nx = presence.shape
        if ny % fy != 0 or nx % fx != 0:
            raise ValueError(f"hi_shape {presence.shape} not divisible by {(fy, fx)}")
        reshaped = presence.reshape(ny // fy, fy, nx // fx, fx)
        n_non_null = reshaped.sum(axis=(1, 3))  # 0..9
    elif presence.shape == cfg.target_shape:
        # If already 0.25°, doc's 3x3 neighbor count is not reconstructible.
        # Minimal consistent approximation: n=9 if presence, else 0.
        print("[WARN] Input rasters already at target resolution; approximating n_non_null as 9 if presence else 0.")
        n_non_null = presence * 9.0
    else:
        raise ValueError(
            f"Unexpected raster shape {presence.shape}. Expected {cfg.hi_shape} or {cfg.target_shape}."
        )

    # Doc formula
    msa_2d = 0.78 * (n_non_null / 9.0) + ((9.0 - n_non_null) / 9.0)
    msa_2d = np.clip(msa_2d, 0.0, 1.0)

    if debug:
        print("[DEBUG] msa_2d nan% pre-ocean:", _pct_nan(msa_2d))

    # Ocean masking
    land = load_land_mask(cfg)
    if land is not None:
        if land.shape != cfg.target_shape:
            raise ValueError(f"oceans mask shape={land.shape} expected={cfg.target_shape}")
        if debug:
            print("[DEBUG] land%:", float(land.mean() * 100.0))
        msa_2d = np.where(land, msa_2d, np.nan)

    # Final sanity
    if debug:
        print("[DEBUG] msa_2d nan% post-ocean:", _pct_nan(msa_2d))
        print("[DEBUG] msa_2d min/max:", float(np.nanmin(msa_2d)), float(np.nanmax(msa_2d)))

    return msa_2d.astype(cfg.out_dtype, copy=False)


# -----------------------------
# NetCDF writer
# -----------------------------
def write_netcdf(out_path: Path, msa_2d: np.ndarray, cfg: RoadConfig, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if msa_2d.shape != cfg.target_shape:
        raise ValueError(f"Expected {cfg.target_shape}, got {msa_2d.shape}")

    msa_3d = np.repeat(msa_2d[None, :, :], cfg.n_time, axis=0)

    lat, lon = make_lat_lon_centers(cfg.target_shape[0], cfg.target_shape[1])
    time_vals = np.arange(cfg.n_time, dtype=np.float64)

    with nc.Dataset(out_path, mode="w", format=cfg.netcdf_format) as ds:
        ds.createDimension("time", cfg.n_time)
        ds.createDimension("lat", cfg.target_shape[0])
        ds.createDimension("lon", cfg.target_shape[1])

        ds.title = title
        ds.subtitle = "MSA_ROAD computed from road presence (3x3 neighbor count) with deterministic orientation."
        ds.Conventions = "CF-1.8"

        vlat = ds.createVariable("lat", np.float32, ("lat",))
        vlat.units = "degrees_north"
        vlat.standard_name = "latitude"
        vlat.long_name = "latitude (cell center)"
        vlat[:] = lat

        vlon = ds.createVariable("lon", np.float32, ("lon",))
        vlon.units = "degrees_east"
        vlon.standard_name = "longitude"
        vlon.long_name = "longitude (cell center)"
        vlon[:] = lon

        vtime = ds.createVariable("time", np.float64, ("time",))
        vtime.units = f"years since {cfg.base_year}-01-01"
        vtime.long_name = "time"
        vtime.calendar = "standard"
        vtime[:] = time_vals

        var_kwargs = {}
        if cfg.compress:
            var_kwargs.update(
                zlib=True,
                complevel=int(cfg.complevel),
                chunksizes=tuple(cfg.chunksizes),
            )

        v = ds.createVariable(
            "MSA_ROAD",
            cfg.out_dtype,
            ("time", "lat", "lon"),
            fill_value=cfg.fill_value,
            **var_kwargs,
        )
        v.units = "1"
        v.long_name = "Mean Species Abundance under road impact"
        v.standard_name = "MSA_ROAD"
        v.missing_value = cfg.fill_value
        v[:, :, :] = msa_3d

    print(
        f"[OK] wrote {out_path} | min/max={np.nanmin(msa_3d):.6g}/{np.nanmax(msa_3d):.6g} "
        f"| nan%={np.isnan(msa_3d).mean()*100:.2f}% | dtype={msa_3d.dtype}"
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Print debug diagnostics (nan%, land%, etc.)")
    args = parser.parse_args()

    cfg = RoadConfig()
    msa_2d = build_msa_road(cfg, debug=bool(args.debug))

    for scen in cfg.scenarios:
        out = cfg.out_root / f"MSA_ROAD_{scen}.nc"
        write_netcdf(out, msa_2d, cfg, title=f"MSA_ROAD_SSP{scen}")

    if cfg.write_legacy:
        out_legacy = cfg.out_root / "MSA_ROAD.nc"
        write_netcdf(out_legacy, msa_2d, cfg, title="MSA_ROAD")

    print("[DONE] MSA_ROAD datamart built.")


if __name__ == "__main__":
    main()
