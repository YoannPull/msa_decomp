# scripts/prepare/extract_msa_points.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from carrefour.extract_msa_points import ExtractMSAConfig, run_extract_file


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach MSA_SQUARE + MSA_* components to point coordinates.")
    p.add_argument("--input", required=True, type=str, help="Input file (.csv or .xlsx/.xls).")
    p.add_argument("--out", required=True, type=str, help="Output file (.csv or .parquet).")

    p.add_argument("--scen", required=True, choices=["126", "370", "585"], help="Scenario.")
    p.add_argument("--year", required=True, type=int, help="Target year (nearest will be used).")

    p.add_argument("--lat-col", default="y_latitude", type=str, help="Latitude column name.")
    p.add_argument("--lon-col", default="x_longitude", type=str, help="Longitude column name.")

    p.add_argument("--datamart-root", default="data/datamart", type=str, help="Root datamart directory.")
    p.add_argument("--msasquare-path-tpl", default="data/datamart/MSA_SQUARE/MSA_SQUARE_{scen}.nc", type=str)

    p.add_argument("--no-components", action="store_true", help="Only attach MSA_SQUARE (skip MSA_* components).")

    p.add_argument("--sheet", default=None, type=str, help="Excel sheet name (if input is xlsx/xls).")
    p.add_argument("--sep", default=None, type=str, help="CSV separator (default: auto-detect).")

    p.add_argument(
        "--max-full-load-mb",
        type=int,
        default=300,
        help="If raster <= this size (MB), load full raster once then numpy-index (often faster).",
    )

    p.add_argument(
        "--engine",
        type=str,
        default="",
        help="Optional xarray engine (e.g. 'h5netcdf'). Empty means default.",
    )

    return p.parse_args()


def main() -> None:
    a = _parse_args()

    cfg = ExtractMSAConfig(
        scen=a.scen,
        year=a.year,
        datamart_root=Path(a.datamart_root),
        msasquare_path_tpl=a.msasquare_path_tpl,
        lat_col=a.lat_col,
        lon_col=a.lon_col,
        sheet=a.sheet,
        sep=a.sep,
        include_components=(not a.no_components),
        max_full_load_mb=a.max_full_load_mb,
        engine=(a.engine.strip() or None),
    )

    run_extract_file(
        input_path=Path(a.input),
        output_path=Path(a.out),
        config=cfg,
    )


if __name__ == "__main__":
    main()
