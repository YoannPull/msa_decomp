# carrefour/prepare/attach_globio_msa.py
# -*- coding: utf-8 -*-

"""
Attach GLOBIO aggregated MSA (GeoTIFF) to a points table (lat/lon).

- Works with very large GeoTIFFs efficiently by sampling only the required points.
- Automatically reprojects points (EPSG:4326 lon/lat) to the raster CRS when needed.
- Keeps all original columns, appends:
    - MSA_GLOBIO

Typical filenames you showed:
- MSA_2015.tif
- MSA_2050_SSP5.tif

Usage is via scripts/prepare/attach_globio_msa.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps with friendly error
try:
    import rasterio
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'rasterio'. Install it (e.g. `poetry add rasterio` or `pip install rasterio`)."
    ) from e

try:
    from pyproj import Transformer
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'pyproj'. Install it (e.g. `poetry add pyproj` or `pip install pyproj`)."
    ) from e


@dataclass(frozen=True)
class AttachGlobioMSAConfig:
    tif_path: Optional[Path] = None  # if provided, use directly

    # If tif_path is not provided, build it from globio_dir + year (+ ssp)
    globio_dir: Path = Path("data/sources/GLOBIO")
    year: int = 2015
    ssp: Optional[str] = None  # e.g. "5" for SSP5, or "SSP5"

    # Input column names
    lat_col: str = "y_latitude"
    lon_col: str = "x_longitude"

    # Output column name
    out_col: str = "MSA_GLOBIO"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _to_float_array(series: pd.Series) -> np.ndarray:
    """
    Convert a column to float, handling French decimal commas.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    return s.astype(float).to_numpy()


def _resolve_tif_path(cfg: AttachGlobioMSAConfig) -> Path:
    if cfg.tif_path is not None:
        p = Path(cfg.tif_path)
        _require_exists(p)
        return p

    # Build filename from year (+ssp)
    # - year 2015 -> MSA_2015.tif
    # - year 2050 + ssp=5 -> MSA_2050_SSP5.tif
    if cfg.ssp is None:
        fname = f"MSA_{cfg.year}.tif"
    else:
        s = str(cfg.ssp).upper().replace("SSP", "")
        fname = f"MSA_{cfg.year}_SSP{s}.tif"

    p = cfg.globio_dir / fname
    _require_exists(p)
    return p


def _sample_raster_at_points(
    tif_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Sample raster values at given points (EPSG:4326 lon/lat).
    Returns float32 array with NaN for nodata/outside.
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        raster_crs = src.crs
        if raster_crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")

        # Transform points from EPSG:4326 -> raster CRS if needed
        if str(raster_crs).upper() in ("EPSG:4326", "WGS84") or raster_crs.to_epsg() == 4326:
            xs, ys = lons, lats
        else:
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)

        coords = list(zip(xs, ys))

        # rasterio.sample returns an iterator of arrays (bands,)
        out = np.full(len(coords), np.nan, dtype=np.float32)

        for i, val in enumerate(src.sample(coords)):
            # single-band expected for MSA, take band 1
            v = float(val[0]) if val is not None and len(val) else np.nan

            if np.isnan(v):
                out[i] = np.nan
                continue

            if nodata is not None and np.isfinite(nodata) and np.isclose(v, nodata):
                out[i] = np.nan
                continue

            # Some rasters use very negative nodata even if nodata isn't set
            if v <= -1e20:
                out[i] = np.nan
                continue

            out[i] = np.float32(v)

        return out


def attach_globio_msa(df_points: pd.DataFrame, *, cfg: AttachGlobioMSAConfig) -> pd.DataFrame:
    """
    Return df with a new column cfg.out_col containing sampled GLOBIO MSA.
    """
    if cfg.lat_col not in df_points.columns or cfg.lon_col not in df_points.columns:
        raise ValueError(
            f"Missing lat/lon columns '{cfg.lat_col}'/'{cfg.lon_col}'. "
            f"Available columns={list(df_points.columns)}"
        )

    df = df_points.copy()

    lats = _to_float_array(df[cfg.lat_col])
    lons = _to_float_array(df[cfg.lon_col])

    valid = np.isfinite(lats) & np.isfinite(lons)
    if not valid.all():
        _log(f"[WARN] {int((~valid).sum())} rows have invalid lat/lon -> MSA_GLOBIO will be NaN.")

    tif_path = _resolve_tif_path(cfg)
    _log(f"[INFO] Sampling GLOBIO MSA from: {tif_path}")

    out = np.full(len(df), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = _sample_raster_at_points(tif_path, lats[valid], lons[valid])

    df[cfg.out_col] = out

    # quick QC
    finite = np.isfinite(out)
    if finite.any():
        _log(f"[QC] {cfg.out_col}: min/max={out[finite].min():.3g}/{out[finite].max():.3g} | nan%={100*(~finite).mean():.2f}")
    else:
        _log(f"[QC] {cfg.out_col}: all NaN (check CRS / bounds / input coords)")

    return df


def run_attach_globio_file(
    *,
    input_path: Path,
    output_path: Path,
    cfg: AttachGlobioMSAConfig,
    sheet: Optional[str] = None,
    sep: Optional[str] = None,
) -> Path:
    """
    High-level helper:
    - read CSV or Excel
    - attach GLOBIO MSA
    - write CSV or Parquet
    """
    _require_exists(input_path)

    suf = input_path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, sheet_name=sheet or 0, dtype=str)
    elif suf == ".csv":
        if sep is None:
            df = pd.read_csv(input_path, dtype=str, sep=None, engine="python")
        else:
            df = pd.read_csv(input_path, dtype=str, sep=sep)
    else:
        raise ValueError(f"Unsupported input extension: {suf}")

    df_out = attach_globio_msa(df, cfg=cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suf = output_path.suffix.lower()
    if out_suf == ".parquet":
        df_out.to_parquet(output_path, index=False)
    elif out_suf == ".csv":
        df_out.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    _log(f"[OK] wrote {output_path}")
    return output_path
