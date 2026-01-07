# carrefour/prepare/road_grip4.py
# -*- coding: utf-8 -*-
"""
Prepare ROAD input on the model grid from GRIP4 ASCII raster.

Inputs
------
- GRIP4 density raster as Arc/Info ASCII grid (.asc)
- Land/ocean mask (oceans.nc) with variable 'land' (1 on land, NaN on ocean) or equivalent

Outputs
-------
- NetCDF on target grid (same as oceans.nc):
    - road_density  float32  (GRIP4 density regridded)
    - road_bin      uint8    (1 if density > threshold else 0)
    - land          float32  (land mask)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class RoadPrepConfig:
    grip4_asc: Path
    oceans_nc: Path
    out_nc: Path
    # density threshold to define road presence
    bin_threshold: float = 0.0
    # resampling method for reproject_match
    # "average" is appropriate when aggregating densities to coarser grids
    resampling: str = "average"
    # treat land cells as land if land > land_threshold
    land_threshold: float = 0.5


def _pick_land_var(ds: xr.Dataset) -> xr.DataArray:
    """
    Return the land mask variable from oceans.nc.
    Prefers 'land', otherwise takes the only var if unique.
    """
    if "land" in ds.data_vars:
        return ds["land"]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    raise ValueError(
        f"Could not infer land variable. Found vars: {list(ds.data_vars)}. "
        "Expected 'land' or a single variable dataset."
    )


def _ensure_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure coordinates are named lat/lon.
    """
    # Many masks are already lat/lon. If not, try common aliases.
    rename = {}
    if "latitude" in da.coords and "lat" not in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords and "lon" not in da.coords:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(
            "Land mask must have lat/lon coords (or latitude/longitude). "
            f"Got coords: {list(da.coords)}"
        )
    return da


def _qa_density(dens: xr.DataArray, land: xr.DataArray) -> dict[str, Any]:
    """
    Compute basic QA stats on density and land/ocean masking.
    """
    a = dens.values
    land_ok = np.isfinite(land.values) & (land.values > 0.5)
    ocean_ok = ~land_ok

    def pct(x: np.ndarray) -> float:
        return float(100.0 * x.mean()) if x.size else float("nan")

    qa = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "nan_pct_global": pct(np.isnan(a)),
        "nan_pct_on_land": pct(np.isnan(a[land_ok])) if land_ok.any() else float("nan"),
        "nan_pct_on_ocean": pct(np.isnan(a[ocean_ok])) if ocean_ok.any() else float("nan"),
        "pct_land_cells_with_road": pct((a[land_ok] > 0)) if land_ok.any() else float("nan"),
    }
    return qa


def _print_qa(tag: str, qa: dict[str, Any]) -> None:
    print(f"== QA {tag} ==")
    for k, v in qa.items():
        print(f"{k:>22}: {v}")


def _build_template_from_latlon(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """
    Build a rioxarray template DataArray with correct affine transform for EPSG:4326.
    """
    import rasterio
    import rioxarray  # noqa: F401

    # resolution
    if lat.size < 2 or lon.size < 2:
        raise ValueError("lat/lon must have at least 2 points to infer resolution.")

    # assume regular grid
    res_x = float(np.mean(np.abs(np.diff(lon))))
    res_y = float(np.mean(np.abs(np.diff(lat))))

    # rasterio's from_origin uses top-left corner (west, north)
    west = float(lon.min()) - res_x / 2.0
    north = float(lat.max()) + res_y / 2.0
    transform = rasterio.transform.from_origin(west, north, res_x, res_y)

    tmpl = xr.DataArray(
        np.zeros((lat.size, lon.size), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": lat, "x": lon},
        name="template",
    )
    tmpl = tmpl.rio.write_crs("EPSG:4326").rio.write_transform(transform)
    return tmpl


def prepare_road_on_ocean_grid(cfg: RoadPrepConfig) -> xr.Dataset:
    """
    Main function: read GRIP4 ASCII density, regrid on oceans.nc grid,
    apply land mask and produce road_density + road_bin.
    """
    cfg = RoadPrepConfig(**{**cfg.__dict__})  # type: ignore

    cfg.out_nc.parent.mkdir(parents=True, exist_ok=True)

    # Load land mask and define target grid
    oceans = xr.open_dataset(cfg.oceans_nc)
    land = _ensure_lat_lon(_pick_land_var(oceans)).astype("float32")

    lat = land["lat"].values
    lon = land["lon"].values

    # Build template on target grid
    tmpl = _build_template_from_latlon(lat, lon)

    # Read GRIP4 ASCII
    import rioxarray  # noqa: F401
    import rasterio

    src = rioxarray.open_rasterio(cfg.grip4_asc, masked=True).squeeze(drop=True)

    # GRIP4 ASCII usually has no CRS; assume EPSG:4326
    if src.rio.crs is None:
        src = src.rio.write_crs("EPSG:4326")

    # Regrid to target grid
    resampling_map = {
        "nearest": rasterio.enums.Resampling.nearest,
        "bilinear": rasterio.enums.Resampling.bilinear,
        "average": rasterio.enums.Resampling.average,
        "mode": rasterio.enums.Resampling.mode,
    }
    if cfg.resampling not in resampling_map:
        raise ValueError(f"Unknown resampling='{cfg.resampling}'. Choose from {list(resampling_map)}")

    dens = src.rio.reproject_match(tmpl, resampling=resampling_map[cfg.resampling])

    # Rename dims to lat/lon
    dens = dens.rename({"y": "lat", "x": "lon"}).astype("float32")
    dens.name = "road_density"

    # Safety: negative densities -> 0
    dens = dens.where(dens >= 0, 0.0)

    # Apply land mask: NaN over ocean
    dens = dens.where(land > cfg.land_threshold)

    # Binary road presence
    road_bin = (dens > cfg.bin_threshold).astype("uint8").rename("road_bin")
    road_bin = road_bin.where(land > cfg.land_threshold)

    # QA
    qa = _qa_density(dens, land)
    _print_qa("GRIP4 -> target grid", qa)

    out = xr.Dataset(
        data_vars={
            "road_density": dens,
            "road_bin": road_bin,
            "land": land,
        },
        attrs={
            "title": "ROAD input on target grid from GRIP4 density",
            "source": "GRIP4 road density (Arc/Info ASCII grid)",
            "bin_threshold": str(cfg.bin_threshold),
            "resampling": cfg.resampling,
            "note": "NaN over ocean (masked by land). road_bin=1 if road_density>threshold on land.",
        },
    )
    return out


def write_netcdf(ds: xr.Dataset, out_nc: Path) -> None:
    encoding = {
        "road_density": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "road_bin": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "land": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    ds.to_netcdf(out_nc, encoding=encoding)
