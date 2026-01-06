# carrefour/msa_pipeline/build_msa_cc.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import xarray as xr

try:
    import rasterio
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "rasterio is required to read the biomes GeoTIFF. "
        "Add it to your env/deps (e.g., poetry add rasterio)."
    ) from e


# =============================================================================
# 1) Biome -> MSA loss per °C (YOUR PROVIDED MAPPING)
# =============================================================================

BIOME_MSA_LOSS_PER_DEGC: Dict[int, float] = {
    6: 0.0367,
    5: 0.1126,
    8: 0.1201,
    13: 0.1201,
    12: 0.0661,
    10: 0.0775,
    4: 0.0487,
    1: 0.1075,
    11: 0.0426,
    14: 0.0521,
    2: 0.1075,
    3: 0.1075,
    7: 0.1201,
    9: 0.1201,
}


# =============================================================================
# 2) Paths (YOUR REAL FILES)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    tas_hist: Path = Path(
        "data/legacy/MSA_CC/Data_Hist_Temp_TAS/"
        "tas_Amon_CNRM-CM6-1-HR_historical_r1i1p1f2_gr_185001-201412.nc"
    )
    tas_ssp126: Path = Path(
        "data/legacy/MSA_CC/Data_ssp_Temp_TAS/"
        "tas_Amon_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_201501-210012.nc"
    )
    tas_ssp370: Path = Path(
        "data/legacy/MSA_CC/Data_ssp_Temp_TAS/"
        "tas_Amon_CNRM-CM6-1-HR_ssp370_r1i1p1f2_gr_201501-210012.nc"
    )
    tas_ssp585: Path = Path(
        "data/legacy/MSA_CC/Data_ssp_Temp_TAS/"
        "tas_Amon_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gr_201501-210012.nc"
    )

    biomes_tif: Path = Path("data/legacy/MSA_CC/Data_Biomes/Biomes_Raster_Full.tif")
    oceans_nc: Path = Path("data/datamart/oceans.nc")

    out_dir: Path = Path("data/datamart/MSA_CC")


SCEN_TO_FILE = {126: "tas_ssp126", 370: "tas_ssp370", 585: "tas_ssp585"}


# =============================================================================
# 3) Utils
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def _open_netcdf(path: Path, chunks: Optional[dict] = None) -> xr.Dataset:
    _require_exists(path)
    return xr.open_dataset(path, decode_times=True, use_cftime=True, chunks=chunks)


def _infer_var(ds: xr.Dataset, preferred: str = "tas") -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(
        f"Could not find variable '{preferred}' in {list(ds.data_vars)}. "
        f"Pass --tas-var to override."
    )


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


def _tas_to_celsius(tas: xr.DataArray, units: str) -> xr.DataArray:
    u = units.strip().lower()
    if u in {"k", "kelvin"}:
        return tas - 273.15
    if u in {"c", "celsius", "degc", "°c"}:
        return tas
    raise ValueError(f"Unsupported --tas-units: {units} (expected kelvin or celsius)")


def _annual_mean(tas: xr.DataArray) -> xr.DataArray:
    if "year" in tas.dims:
        out = tas
    elif "time" in tas.dims:
        out = tas.groupby("time.year").mean("time", keep_attrs=True)
    else:
        raise ValueError(f"tas dims must include 'time' or 'year'. Got: {tas.dims}")
    return out.assign_coords(year=out["year"].astype(np.int32))


def _read_biomes_geotiff(path: Path) -> xr.DataArray:
    _require_exists(path)
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        width, height = src.width, src.height

        xs = (np.arange(width) + 0.5) * transform.a + transform.c
        ys = (np.arange(height) + 0.5) * transform.e + transform.f

        da = xr.DataArray(
            arr.astype(np.int16),
            dims=("y", "x"),
            coords={"x": xs.astype(np.float32), "y": ys.astype(np.float32)},
            name="biome_code",
        )

    da = da.rename({"x": "lon", "y": "lat"})
    da = _normalize_lon_to_180(da)
    da = _ensure_lat_ascending(da)
    return da


def _mapping_to_lut(mapping: Dict[int, float]) -> Tuple[np.ndarray, Dict[int, float]]:
    if not mapping:
        raise ValueError("Empty biome mapping.")
    if any(k < 0 for k in mapping):
        bad = [k for k in mapping if k < 0]
        raise ValueError(f"Negative biome_code in mapping: {bad}")

    max_code = int(max(mapping.keys()))
    lut = np.full((max_code + 1,), np.nan, dtype=np.float32)
    for k, v in mapping.items():
        lut[int(k)] = np.float32(float(v))
    return lut, {int(k): float(v) for k, v in mapping.items()}


def _infer_land_mask(
    biomes: xr.DataArray,
    oceans_nc: Path,
    template: xr.DataArray,
) -> xr.DataArray:
    biomes_on = biomes.interp_like(template, method="nearest")
    land_from_biomes = (biomes_on > 0) & (~xr.ufuncs.isnan(biomes_on))

    if not oceans_nc.exists():
        return land_from_biomes

    try:
        ds = _open_netcdf(oceans_nc)
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

        _log(
            f"[WARN] oceans mask suspicious (frac_ones={frac_ones:.3f}, frac_zeros={frac_zeros:.3f}). "
            "Falling back to biomes-derived land mask."
        )
        return land_from_biomes

    except Exception as e:
        _log(f"[WARN] cannot use oceans mask ({oceans_nc}): {e}. Falling back to biomes-derived land mask.")
        return land_from_biomes


def _biome_coeff_array(biomes_on: xr.DataArray, lut: np.ndarray) -> xr.DataArray:
    if biomes_on.dtype.kind not in {"i", "u"}:
        biomes_on = biomes_on.astype(np.int32)

    max_code = len(lut) - 1

    def _lookup(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.int32)
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = (arr >= 0) & (arr <= max_code)
        out[valid] = lut[arr[valid]]
        return out

    coeff = xr.apply_ufunc(
        _lookup,
        biomes_on,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    coeff.name = "msa_loss_per_degC"
    return coeff


def _check_mapping_coverage_on_land(
    biomes_on: xr.DataArray,
    land_mask: xr.DataArray,
    mapping: Dict[int, float],
) -> None:
    vals = biomes_on.where(land_mask).values
    vals = vals[np.isfinite(vals)].astype(np.int32)
    unique_codes = np.unique(vals)
    missing = [int(c) for c in unique_codes.tolist() if int(c) not in mapping and int(c) != 0]
    if missing:
        raise ValueError(
            "Biome mapping incomplete: found biome codes on land that are not in BIOME_MSA_LOSS_PER_DEGC. "
            f"Missing codes: {missing}."
        )


def _encoding_float32_chunked(da: xr.DataArray, max_chunks: Dict[str, int]) -> Dict[str, dict]:
    """
    Build safe netCDF encoding for da:
      - chunksizes follow da.dims order
      - each chunk <= corresponding dim size
    """
    chunks = []
    for d in da.dims:
        size = int(da.sizes[d])
        target = int(max_chunks.get(d, size))
        chunks.append(min(target, size))

    return {
        da.name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": tuple(chunks),
            "_FillValue": np.float32(np.nan),
        }
    }


def _qc_summary(da: xr.DataArray, label: str) -> str:
    v = da.values
    finite = np.isfinite(v)
    nan_pct = 100.0 * (1.0 - finite.mean()) if v.size else float("nan")
    if finite.any():
        mn = float(np.nanmin(v))
        mx = float(np.nanmax(v))
    else:
        mn, mx = float("nan"), float("nan")
    return f"[OK] wrote {label} | min/max={mn:.6g}/{mx:.6g} | nan%={nan_pct:.2f}% | dtype={da.dtype}"


# =============================================================================
# 4) Build
# =============================================================================

def build_msa_cc(
    paths: Paths,
    out_year_start: int = 2015,
    out_year_end: int = 2100,
    baseline_start: int = 1850,
    baseline_end: int = 1899,
    tas_units: str = "kelvin",
    tas_var: Optional[str] = None,
    chunks_time: int = 120,
) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    _log("== Build MSA_CC ==")
    _log(f"tas_hist : {paths.tas_hist}")
    _log(f"tas_126  : {paths.tas_ssp126}")
    _log(f"tas_370  : {paths.tas_ssp370}")
    _log(f"tas_585  : {paths.tas_ssp585}")
    _log(f"biomes   : {paths.biomes_tif}")
    _log(f"oceans   : {paths.oceans_nc} (optional)")
    _log(f"out_dir  : {paths.out_dir}")
    _log(f"baseline : {baseline_start}-{baseline_end}")
    _log(f"years    : {out_year_start}-{out_year_end}")
    _log("")

    biomes = _read_biomes_geotiff(paths.biomes_tif)

    lut, mapping = _mapping_to_lut(BIOME_MSA_LOSS_PER_DEGC)
    _log(f"Loaded mapping: {len(mapping)} biome coeffs (max_code={len(lut)-1}).")

    ds_hist = _open_netcdf(paths.tas_hist, chunks={"time": chunks_time})
    var_hist = tas_var or _infer_var(ds_hist, preferred="tas")
    tas_hist = _normalize_lon_to_180(_ensure_lat_ascending(ds_hist[var_hist]))
    tas_hist_c = _tas_to_celsius(tas_hist, units=tas_units)
    tas_hist_ann = _annual_mean(tas_hist_c)

    baseline = tas_hist_ann.sel(year=slice(baseline_start, baseline_end)).mean("year")
    baseline.name = "tas_baseline_degC"

    land_mask = _infer_land_mask(biomes=biomes, oceans_nc=paths.oceans_nc, template=baseline)
    biomes_on = biomes.interp_like(baseline, method="nearest")

    _check_mapping_coverage_on_land(biomes_on=biomes_on, land_mask=land_mask, mapping=mapping)

    coeff = _biome_coeff_array(biomes_on, lut=lut)

    scenario_das: List[xr.DataArray] = []

    for scen, attr in SCEN_TO_FILE.items():
        scen_path: Path = getattr(paths, attr)
        _log(f"\n-- Scenario {scen} --")
        _log(f"Reading: {scen_path}")

        ds = _open_netcdf(scen_path, chunks={"time": chunks_time})
        var = tas_var or _infer_var(ds, preferred="tas")
        tas = _normalize_lon_to_180(_ensure_lat_ascending(ds[var]))
        tas_c = _tas_to_celsius(tas, units=tas_units)

        tas_ann = _annual_mean(tas_c).sel(year=slice(out_year_start, out_year_end))
        dT = tas_ann - baseline

        msa_cc = (1.0 - coeff * dT).clip(min=0.0, max=1.0).astype(np.float32)
        msa_cc = msa_cc.where(land_mask)
        msa_cc.name = "MSA_CC"

        # ---- per-scenario write (SAFE chunking) ----
        out_path = paths.out_dir / f"MSA_CC_{scen}.nc"
        ds_out = msa_cc.to_dataset()

        enc = _encoding_float32_chunked(
            msa_cc,
            max_chunks={"year": 100, "lat": 360, "lon": 720},
        )
        ds_out.to_netcdf(out_path, encoding=enc)
        _log(_qc_summary(ds_out["MSA_CC"], label=str(out_path)))

        scenario_das.append(msa_cc.assign_coords(scenario=np.int32(scen)).expand_dims("scenario"))

    # ---- combined write (SAFE chunking) ----
    msa_all = xr.concat(scenario_das, dim="scenario")
    out_all = paths.out_dir / "MSA_CC.nc"
    ds_all = msa_all.to_dataset()

    enc_all = _encoding_float32_chunked(
        msa_all,
        max_chunks={"scenario": 3, "year": 100, "lat": 360, "lon": 720},
    )
    ds_all.to_netcdf(out_all, encoding=enc_all)
    _log(_qc_summary(ds_all["MSA_CC"], label=str(out_all)))

    _log("\n[DONE] MSA_CC datamart built.")


# =============================================================================
# 5) CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MSA_CC datamart (Climate Change).")

    p.add_argument("--out-year-start", type=int, default=2015)
    p.add_argument("--out-year-end", type=int, default=2100)
    p.add_argument("--baseline-start", type=int, default=1850)
    p.add_argument("--baseline-end", type=int, default=1899)

    p.add_argument("--tas-units", default="kelvin", choices=["kelvin", "celsius"])
    p.add_argument("--tas-var", default=None, help="Override variable name if not 'tas'.")
    p.add_argument("--chunks-time", type=int, default=120)

    p.add_argument("--tas-hist", default=str(Paths.tas_hist))
    p.add_argument("--tas-ssp126", default=str(Paths.tas_ssp126))
    p.add_argument("--tas-ssp370", default=str(Paths.tas_ssp370))
    p.add_argument("--tas-ssp585", default=str(Paths.tas_ssp585))
    p.add_argument("--biomes-tif", default=str(Paths.biomes_tif))
    p.add_argument("--oceans-nc", default=str(Paths.oceans_nc))
    p.add_argument("--out-dir", default=str(Paths.out_dir))

    return p.parse_args()


def main() -> None:
    a = _parse_args()
    paths = Paths(
        tas_hist=Path(a.tas_hist),
        tas_ssp126=Path(a.tas_ssp126),
        tas_ssp370=Path(a.tas_ssp370),
        tas_ssp585=Path(a.tas_ssp585),
        biomes_tif=Path(a.biomes_tif),
        oceans_nc=Path(a.oceans_nc),
        out_dir=Path(a.out_dir),
    )

    build_msa_cc(
        paths=paths,
        out_year_start=a.out_year_start,
        out_year_end=a.out_year_end,
        baseline_start=a.baseline_start,
        baseline_end=a.baseline_end,
        tas_units=a.tas_units,
        tas_var=a.tas_var,
        chunks_time=a.chunks_time,
    )


if __name__ == "__main__":
    main()
