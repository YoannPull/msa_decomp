# scripts/legacy/attach_legacy_msa.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from legacy_msa.legacy.attach_legacy_msa import (
    AttachLegacyConfig,
    DEFAULT_VARS,
    run_attach_legacy_file,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach Pierre legacy MSA_* variables to point coordinates.")

    p.add_argument("--input", required=True, type=str, help="Input points file (.csv with ; or .parquet).")
    p.add_argument("--out", required=True, type=str, help="Output file (.parquet or .csv).")

    p.add_argument("--legacy-root", required=True, type=str, help="Legacy datamart root (e.g. data/datamart_legacy).")
    p.add_argument("--scen", required=True, choices=["126", "370", "585"], help="Scenario.")
    p.add_argument("--year", required=True, type=int, help="Target year (e.g. 2100).")

    p.add_argument("--lat-col", default="y_latitude", help="Latitude column name in input points.")
    p.add_argument("--lon-col", default="x_longitude", help="Longitude column name in input points.")

    p.add_argument("--suffix", default="", help="Suffix appended to attached columns.")
    p.add_argument(
        "--vars",
        default=",".join(DEFAULT_VARS),
        help=f"Comma-separated vars to attach (default: {','.join(DEFAULT_VARS)}).",
    )

    p.add_argument("--engine", default=None, help="xarray engine (optional), e.g. h5netcdf")
    p.add_argument("--max-full-load-mb", type=float, default=400.0)
    p.add_argument("--debug", action="store_true", help="Verbose logging / QC.")

    p.add_argument(
        "--lat-flip",
        default="flip",
        choices=["flip", "none", "auto"],
        help="Latitude index handling. 'flip' recommended for Pierre legacy (default).",
    )

    p.add_argument(
        "--verify-square",
        action="store_true",
        help="After attaching, verify Pierre formula for MSA_SQUARE using LU split (if present).",
    )

    return p.parse_args()


def main() -> None:
    a = _parse_args()

    vars_list = tuple(v.strip() for v in a.vars.split(",") if v.strip())
    if not vars_list:
        raise SystemExit("Empty --vars list after parsing.")

    cfg = AttachLegacyConfig(
        legacy_root=Path(a.legacy_root),
        scen=str(a.scen),
        year=int(a.year),
        lat_col=str(a.lat_col),
        lon_col=str(a.lon_col),
        suffix=str(a.suffix),
        vars=vars_list,
        engine=a.engine,
        max_full_load_mb=float(a.max_full_load_mb),
        debug=bool(a.debug),
        lat_flip=str(a.lat_flip),
    )

    run_attach_legacy_file(
        input_path=Path(a.input),
        output_path=Path(a.out),
        cfg=cfg,
        verify_square=bool(a.verify_square),
    )


if __name__ == "__main__":
    main()
