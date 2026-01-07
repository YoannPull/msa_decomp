# MSA pipeline (LU, ROAD, CC, N, ENC, SQUARE + GLOBIO + LEGACY)

This repository builds **MSA (Mean Species Abundance)** rasters per pressure (**LU**, **ROAD**, **CC**, **N**, **ENC**) and a combined raster **MSA_SQUARE**.
It also supports:
- extracting MSA values at point locations (lat/lon CSV),
- attaching aggregated **GLOBIO MSA** (GeoTIFF),
- attaching and plotting a **legacy NetCDF datamart** (and checking square reconstruction).

---

## Requirements
- Python 3.12
- Poetry
- GNU Make

Project dependencies and config live in `pyproject.toml`.

---

## Install
```bash
poetry install
````

---

## Quick start (most common)

Set defaults via Make variables (examples below):

* `YEAR` (default: 2100)
* `SCEN` (default: 126)

### Build all MSA datamarts

```bash
make lu
make road
make cc
make n
make enc
make square
```

### Plot computed rasters

```bash
make plot-square YEAR=2100 SCEN=126
make plot-square-all YEAR=2100
```

---

## Data layout (expected)

* `data/sources/` : raw inputs (e.g., ROAD, GLOBIO, etc.)
* `data/datamart/` : computed NetCDF outputs per pressure + combined
* `data/datamart_legacy/` : legacy NetCDF datamart (optional)
* `data/private/` : private point files (e.g., a CSV with lat/lon)
* `outputs/plots/` : plots (computed datamart)
* `outputs/plots_legacy/` : plots (legacy datamart)
* `outputs/` : extracted point datasets (parquet/csv)

---

## Build steps

### Prepare ROAD input on the MSA grid

```bash
make prepare-road
```

### Build individual pressures

```bash
make lu
make road
make cc
make n
make enc
```

### Build combined MSA_SQUARE

```bash
make square
```

By default, the Makefile runs:

* `scripts/build_msa_square.py --fill-road-na-as-one --fill-cc-na-as-one`

---

## Extract MSA at point locations

Defaults:

* `POINTS=data/private/carrefour_loc.csv`
* `LAT_COL=y_latitude`
* `LON_COL=x_longitude`

Extract for one scenario:

```bash
make extract-points POINTS=path/to/points.csv SCEN=585 YEAR=2100
```

Extract for all scenarios (126/370/585):

```bash
make extract-points-all POINTS=path/to/points.csv YEAR=2100
```

Faster extraction (tune memory + engine):

```bash
make extract-points-fast POINTS=path/to/points.csv SCEN=126 YEAR=2100 MAX_FULL_LOAD_MB=600 ENGINE=h5netcdf
```

---

## Attach GLOBIO aggregated MSA (GeoTIFF)

Attach a given year (no SSP for 2015):

```bash
make attach-globio POINTS=path/to/points.csv GLOBIO_YEAR=2015
```

Attach a given year + one SSP (e.g., SSP5):

```bash
make attach-globio-ssp POINTS=path/to/points.csv GLOBIO_YEAR=2050 SSP=5
```

Run all SSPs for a given year:

```bash
make attach-globio-all-ssps POINTS=path/to/points.csv GLOBIO_YEAR=2050
```

Run all years + SSPs:

```bash
make attach-globio-all-years-ssps POINTS=path/to/points.csv
```

---

## Attach legacy NetCDF datamart (+ optional checks)

Attach legacy values for one scenario/year (writes parquet + csv):

```bash
make attach-legacy POINTS=path/to/points.csv SCEN=126 YEAR=2100
```

Notes:

* `LEGACY_LAT_FLIP=flip|none|auto` (default: `flip`)
* `DEBUG=1` for verbose logs
* `LEGACY_VERIFY=1` logs square reconstruction diagnostics
* `LEGACY_ADD_CHECK=1` writes `*_checked.(parquet|csv)` with reconstruction/diff/OK columns
* `LEGACY_CHECK_TOL=1e-3` tolerance for OK flag

Attach legacy for all scenarios (126/370/585):

```bash
make attach-legacy-all POINTS=path/to/points.csv YEAR=2100
```

Add check columns on existing legacy outputs:

```bash
make legacy-check SCEN=126 YEAR=2100
```

---

## Postprocess extracted point files

Typical use: merge/standardize outputs, fill ROAD NaNs, generate CSVs/plots.

For explicit inputs:

```bash
make postprocess INPUTS="a.parquet b.parquet"
```

For default 126/370/585 extracted files:

```bash
make postprocess-all YEAR=2100
```

Convenience targets:

```bash
make all-points POINTS=path/to/points.csv YEAR=2100
make all-legacy POINTS=path/to/points.csv YEAR=2100
```

---

## Plot legacy rasters

Plot selected variables:

```bash
make plot-legacy YEAR=2100 SCEN=126 VARS="MSA_SQUARE MSA_LU"
```

Plot all scenarios:

```bash
make plot-legacy-all YEAR=2100
```

---

## Plot computed rasters (per pressure)

Examples:

```bash
make plot-lu YEAR=2100 SCEN=126
make plot-road YEAR=2100 SCEN=126
make plot-cc YEAR=2100 SCEN=126
make plot-n YEAR=2100 SCEN=126
make plot-enc YEAR=2100 SCEN=126
make plot-square YEAR=2100 SCEN=126
```

All scenarios:

```bash
make plot-lu-all YEAR=2100
make plot-road-all YEAR=2100
make plot-cc-all YEAR=2100
make plot-n-all YEAR=2100
make plot-enc-all YEAR=2100
make plot-square-all YEAR=2100
```
