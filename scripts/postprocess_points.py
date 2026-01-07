# scripts/postprocess_points.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from carrefour.postprocess_points import PostprocessConfig, postprocess_points_files


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocess point tables: fill ROAD NaNs, export CSVs, make plots.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more input tables (.parquet/.csv/.xlsx). Example: outputs/points_with_msa_126_2100.parquet ...",
    )
    p.add_argument("--out-dir", default="outputs/points_postprocess", type=str, help="Output directory.")

    p.add_argument("--lat-col", default="y_latitude", type=str)
    p.add_argument("--lon-col", default="x_longitude", type=str)
    p.add_argument("--scenario-col", default="scenario", type=str)

    p.add_argument("--road-col", default="MSA_ROAD", type=str)
    p.add_argument("--road-fill", default=1.0, type=float)

    p.add_argument("--topk", default=30, type=int)
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation.")

    return p.parse_args()


def main() -> None:
    a = _parse_args()

    cfg = PostprocessConfig(
        lat_col=a.lat_col,
        lon_col=a.lon_col,
        scenario_col=a.scenario_col,
        road_col=a.road_col,
        road_fill_value=a.road_fill,
        out_dir=Path(a.out_dir),
        topk=a.topk,
    )

    postprocess_points_files(
        inputs=[Path(x) for x in a.inputs],
        cfg=cfg,
        make_plots_flag=(not a.no_plots),
    )


if __name__ == "__main__":
    main()
