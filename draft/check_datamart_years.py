# scripts/check_datamart_years.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import xarray as xr


@dataclass
class YearCheckResult:
    path: Path
    dims: Dict[str, int]
    mode: str
    years: List[int]
    units: Optional[str] = None
    note: Optional[str] = None


YEAR_UNITS_RE = re.compile(r"years since (\d{4})-\d{2}-\d{2}")


def _safe_int_list(arr) -> List[int]:
    vals = np.atleast_1d(arr)
    out = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            pass
    return sorted(set(out))


def extract_years_from_dataset(ds: xr.Dataset) -> Tuple[str, List[int], Optional[str], Optional[str]]:
    """
    Returns: (mode, years, units, note)
    mode explains how years were obtained.
    """
    # 1) explicit year coord
    if "year" in ds.coords:
        years = _safe_int_list(ds["year"].values)
        return ("coord:year", years, None, None)

    # 2) time coord
    if "time" in ds.coords:
        t = np.asarray(ds["time"].values)
        units = ds["time"].attrs.get("units", "")
        n_time = ds.sizes.get("time", None)

        # Parse base year from units
        m = YEAR_UNITS_RE.match(units or "")
        base_year = int(m.group(1)) if m else None

        # Determine if time values are usable (not all fill-like)
        # Many broken files have time filled with ~9.969e36 without missing_value/_FillValue
        finite = np.isfinite(t)
        not_huge = np.abs(t) < 1e20
        valid = finite & not_huge

        # If *any* valid values, derive years from them (rounded)
        if valid.any() and base_year is not None:
            years = np.unique((base_year + np.rint(t[valid])).astype(int)).tolist()
            years = sorted(set(years))
            return ("coord:time(values)", years, units, None)

        # Otherwise fallback to index-based mapping if we have base_year and length
        if base_year is not None and n_time is not None:
            years = (base_year + np.arange(n_time)).astype(int).tolist()
            note = None
            if valid.any() is False:
                note = "time values look missing/filled; used index-based years"
            elif base_year is None:
                note = "could not parse base_year from units; cannot infer"
            return ("coord:time(index)", years, units, note)

        return ("coord:time(unusable)", [], units, "time present but cannot infer years")

    # 3) global attrs fallback
    for k in ["year", "target_year", "YEAR", "reference_year"]:
        if k in ds.attrs:
            try:
                return (f"attr:{k}", [int(ds.attrs[k])], None, None)
            except Exception:
                return (f"attr:{k}", [], None, f"attr {k} not int: {ds.attrs[k]!r}")

    return ("NO_TIME_OR_YEAR", [], None, None)


def check_file(nc_path: Path) -> YearCheckResult:
    ds = xr.open_dataset(nc_path, decode_times=False)
    try:
        mode, years, units, note = extract_years_from_dataset(ds)
        dims = dict(ds.sizes)
        return YearCheckResult(path=nc_path, dims=dims, mode=mode, years=years, units=units or None, note=note)
    finally:
        ds.close()


def main(root: str = "data/datamart") -> None:
    root_path = Path(root)
    files = sorted(root_path.rglob("*.nc"))
    if not files:
        raise SystemExit(f"No .nc files found under {root_path}")

    results: List[YearCheckResult] = []
    for f in files:
        try:
            results.append(check_file(f))
        except Exception as e:
            results.append(YearCheckResult(path=f, dims={}, mode="ERROR", years=[], note=str(e)))

    # Print per-file report
    print(f"Scanned {len(results)} NetCDF files under {root_path}\n")
    for r in results:
        years_str = (
            f"{min(r.years)}..{max(r.years)} (n={len(r.years)})"
            if r.years else
            "—"
        )
        rel = r.path.relative_to(root_path)
        print(f"{str(rel):55s}  mode={r.mode:20s}  years={years_str:18s}  dims={r.dims}")
        if r.units:
            print(f"{'':55s}  units={r.units}")
        if r.note:
            print(f"{'':55s}  note={r.note}")

    # Global summary: intersection of years across files that have years
    year_sets = [set(r.years) for r in results if r.years]
    if year_sets:
        inter = set.intersection(*year_sets)
        union = set.union(*year_sets)
        print("\n--- SUMMARY ---")
        print(f"Files with inferred years: {len(year_sets)}/{len(results)}")
        print(f"Union years: {min(union)}..{max(union)} (n={len(union)})")
        if inter:
            print(f"Intersection years (common to all those files): {min(inter)}..{max(inter)} (n={len(inter)})")
        else:
            print("Intersection years: EMPTY (not all files share the same year set)")
    else:
        print("\n--- SUMMARY ---")
        print("No years could be inferred from any file.")


if __name__ == "__main__":
    main()
