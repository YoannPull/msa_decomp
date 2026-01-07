# scripts/prepare/30_prepare_grip4_road.py
# -*- coding: utf-8 -*-

"""
CLI wrapper: GRIP4 ASCII -> data/datamart/ROAD/road_on_msa_grid.nc

Usage:
  poetry run python scripts/prepare/30_prepare_grip4_road.py \
    --grip4-asc data/sources/ROAD/GRIP4/grip4_total_dens_m_km2.asc \
    --oceans-nc data/datamart/oceans.nc \
    --out-nc    data/datamart/ROAD/road_on_msa_grid.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carrefour.prepare.road_grip4 import RoadPrepConfig, prepare_road_on_ocean_grid, write_netcdf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grip4-asc", type=Path, required=True)
    p.add_argument("--oceans-nc", type=Path, required=True)
    p.add_argument("--out-nc", type=Path, required=True)
    p.add_argument("--bin-threshold", type=float, default=0.0)
    p.add_argument("--resampling", type=str, default="average", choices=["nearest", "bilinear", "average", "mode"])
    p.add_argument("--land-threshold", type=float, default=0.5)
    args = p.parse_args()

    cfg = RoadPrepConfig(
        grip4_asc=args.grip4_asc,
        oceans_nc=args.oceans_nc,
        out_nc=args.out_nc,
        bin_threshold=args.bin_threshold,
        resampling=args.resampling,
        land_threshold=args.land_threshold,
    )

    ds = prepare_road_on_ocean_grid(cfg)
    args.out_nc.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf(ds, args.out_nc)
    print(f"[OK] wrote {args.out_nc}")


if __name__ == "__main__":
    main()
