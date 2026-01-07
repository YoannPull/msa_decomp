# scripts/prepare/prepare_msa_lu.py
# -*- coding: utf-8 -*-

"""
CLI wrapper for preparing MSA_LU.

Usage:
  poetry run python scripts/prepare/prepare_msa_lu.py --scen 126 --year 2100
  poetry run python scripts/prepare/prepare_msa_lu.py --scen 370 --year 2030 --grid-ref data/datamart/MSA_ROAD/MSA_ROAD_370.nc
  poetry run python scripts/prepare/prepare_msa_lu.py --scen 585 --year 2050 --use-only-luh-mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carrefour.prepare.msa_lu import prepare_msa_lu


def _repo_root() -> Path:
    # scripts/prepare/prepare_msa_lu.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", type=int, required=True, help="Scenario: 126, 370, 585")
    ap.add_argument("--year", type=int, required=True, help="Year in [2015, 2100]")
    ap.add_argument("--grid-ref", type=str, default=None, help="Optional NetCDF path defining target lat/lon grid")
    ap.add_argument(
        "--use-only-luh-mask",
        action="store_true",
        help="If set: do NOT apply oceans.nc; only use LUH landfrac mask.",
    )
    args = ap.parse_args()

    out = prepare_msa_lu(
        repo_root=_repo_root(),
        scen=args.scen,
        year=args.year,
        grid_ref=Path(args.grid_ref) if args.grid_ref else None,
        use_only_luh_mask=args.use_only_luh_mask,
    )
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
