# scripts/prepare/attach_globio_msa.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from carrefour.attach_globio_msa import (
    AttachGlobioMSAConfig,
    run_attach_globio_file,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach GLOBIO aggregated MSA (GeoTIFF) to a points table.")
    p.add_argument("--input", required=True, type=str, help="Input file (.csv or .xlsx/.xls).")
    p.add_argument("--out", required=True, type=str, help="Output file (.csv or .parquet).")

    # Either provide the tif directly, or let the script resolve it from dir/year/ssp
    p.add_argument("--tif", default="", type=str, help="Path to GLOBIO MSA GeoTIFF (overrides --globio-dir/--year/--ssp).")
    p.add_argument("--globio-dir", default="data/sources/GLOBIO", type=str, help="Directory containing MSA_*.tif files.")
    p.add_argument("--year", default=2015, type=int, help="Year (e.g. 2015, 2050).")
    p.add_argument("--ssp", default="", type=str, help="SSP number or label (e.g. 5 or SSP5).")

    p.add_argument("--lat-col", default="y_latitude", type=str)
    p.add_argument("--lon-col", default="x_longitude", type=str)
    p.add_argument("--out-col", default="MSA_GLOBIO", type=str)

    # IO options
    p.add_argument("--sheet", default=None, type=str, help="Excel sheet name (if input is xlsx/xls).")
    p.add_argument("--sep", default=None, type=str, help="CSV separator (default: auto-detect).")

    return p.parse_args()


def main() -> None:
    a = _parse_args()

    tif_path = Path(a.tif) if a.tif.strip() else None
    ssp = a.ssp.strip() or None

    cfg = AttachGlobioMSAConfig(
        tif_path=tif_path,
        globio_dir=Path(a.globio_dir),
        year=a.year,
        ssp=ssp,
        lat_col=a.lat_col,
        lon_col=a.lon_col,
        out_col=a.out_col,
    )

    run_attach_globio_file(
        input_path=Path(a.input),
        output_path=Path(a.out),
        cfg=cfg,
        sheet=a.sheet,
        sep=a.sep,
    )


if __name__ == "__main__":
    main()
