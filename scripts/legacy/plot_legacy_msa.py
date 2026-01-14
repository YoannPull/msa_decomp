# scripts/legacy/plot_legacy_msa.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from legacy_msa.legacy.plot_legacy_msa import PlotLegacyConfig, plot_legacy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LEGACY (Pierre) MSA maps.")
    p.add_argument("--legacy-root", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--year", required=True, type=int)

    p.add_argument("--scens", nargs="+", required=True)
    p.add_argument("--vars", nargs="+", required=True)

    p.add_argument("--cmap", default="msa", choices=["msa", "whitegreen", "greens"])
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--base-year", type=int, default=2015)
    p.add_argument("--engine", type=str, default=None)

    p.add_argument("--flip-lat", action="store_true")
    p.add_argument("--origin", default="upper", choices=["upper", "lower", "auto"])
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    cfg = PlotLegacyConfig(
        legacy_root=Path(a.legacy_root),
        out_dir=Path(a.out_dir),
        year=int(a.year),
        scens=[str(x) for x in a.scens],
        vars=[str(x) for x in a.vars],
        cmap=str(a.cmap),
        gamma=float(a.gamma),
        base_year=int(a.base_year),
        engine=a.engine,
        flip_lat=bool(a.flip_lat),
        origin=str(a.origin),
    )
    plot_legacy(cfg)


if __name__ == "__main__":
    main()
