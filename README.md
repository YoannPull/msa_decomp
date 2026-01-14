# Legacy MSA — Attach & Plot

This repository contains the **legacy MSA workflow**:
- attach legacy **MSA_* variables** from a NetCDF datamart to **private point datasets** (lat/lon CSV),
- optionally **verify** the legacy `MSA_SQUARE` reconstruction and add **check columns**,
- plot legacy rasters for selected variables/scenarios/years.

> This repo does **not** build the “new” MSA rasters from public sources. It focuses only on the **legacy datamart**.

---

## What is “legacy MSA_SQUARE” here?

Legacy `MSA_SQUARE` is computed as:

\[
\mathrm{MSA\_SQUARE}
=\mathrm{OCEANS}\times\Big(
\mathrm{MSA}_{LU,art}
+\mathrm{MSA}_{LU,nat}\times \mathrm{MSA}_{NDEP}\times \mathrm{MSA}_{ENC}\times \mathrm{MSA}_{ROAD}\times \mathrm{MSA}_{CC}
\Big)
\]

- `OCEANS` is a land mask (0 on ocean, 1 on land).
- The “artificial” LU component is added separately; other pressures apply only on the “natural” fraction.

---

## Requirements
- Python 3.12
- Poetry
- GNU Make

Dependencies are defined in `pyproject.toml`.

---

## Install
```bash
poetry install
````

---

## Data layout (expected)

* `data/datamart_legacy/` : **legacy NetCDF datamart**
* `data/private/` : private point files (CSV with lat/lon)
* `outputs/output_legacy/` : attached point outputs (csv/parquet, checked variants)
* `outputs/plots_legacy/` : legacy raster plots

---

## Available years & scenarios

Legacy NetCDF files typically contain **annual data from 2015 to 2100** (inclusive), i.e. 86 years.

Scenarios follow CMIP6-style notation:

* `SCEN=126` (SSP1-2.6)
* `SCEN=370` (SSP3-7.0)
* `SCEN=585` (SSP5-8.5)

---

## Quick start

### Attach legacy MSA to a private CSV (CSV only)

Default private file:

* `POINTS=data/private/carrefour_loc.csv`
* `LAT_COL=y_latitude`
* `LON_COL=x_longitude`

Attach SSP126 (`SCEN=126`) for year 2025:

```bash
make attach-legacy-csv POINTS=data/private/carrefour_loc.csv SCEN=126 LEGACY_YEAR=2025
```

Attach to another private file:

```bash
make attach-legacy-csv POINTS=data/private/carrefour_full.csv SCEN=126 LEGACY_YEAR=2025
```

Outputs:

* `outputs/output_legacy/points_with_legacy_126_2025.csv`
* if checks enabled, also: `..._checked.csv`

---

## Attach legacy datamart (details)

Attach legacy (parquet + csv):

```bash
make attach-legacy POINTS=path/to/points.csv SCEN=126 LEGACY_YEAR=2100
```

Attach legacy (CSV only):

```bash
make attach-legacy-csv POINTS=path/to/points.csv SCEN=126 LEGACY_YEAR=2100
```

Attach legacy for all scenarios (126/370/585):

```bash
make attach-legacy-all POINTS=path/to/points.csv LEGACY_YEAR=2100
```

Attach legacy for all scenarios (CSV only):

```bash
make attach-legacy-all-csv POINTS=path/to/points.csv LEGACY_YEAR=2100
```

---

## Options & diagnostics

### Latitude orientation (important)

Legacy rasters may require a latitude flip to align correctly:

* `LEGACY_LAT_FLIP=flip|none|auto` (default: `flip`)

Example:

```bash
make attach-legacy-csv POINTS=path/to/points.csv SCEN=126 LEGACY_YEAR=2050 LEGACY_LAT_FLIP=flip
```

### Square reconstruction verification

* `LEGACY_VERIFY=1` prints reconstruction diagnostics (logs)
* `LEGACY_ADD_CHECK=1` writes `_checked.*` outputs with:

  * reconstructed square,
  * absolute difference,
  * OK flag (tolerance-based)

Tolerance:

* `LEGACY_CHECK_TOL=1e-3`

Example (CSV only + checks):

```bash
make attach-legacy-csv POINTS=path/to/points.csv SCEN=126 LEGACY_YEAR=2100 LEGACY_ADD_CHECK=1 LEGACY_CHECK_TOL=1e-3
```

Disable checks:

```bash
make attach-legacy-csv POINTS=path/to/points.csv SCEN=126 LEGACY_YEAR=2100 LEGACY_ADD_CHECK=0
```

Verbose logs:

```bash
make attach-legacy-csv POINTS=path/to/points.csv SCEN=126 LEGACY_YEAR=2100 DEBUG=1
```

---

## Plot legacy rasters

Plot selected variables:

```bash
make plot-legacy LEGACY_YEAR=2100 SCEN=126 VARS="MSA_SQUARE MSA_LU"
```

Plot full legacy set (default variables):

```bash
make plot-legacy LEGACY_YEAR=2100 SCEN=126
```

Plot all scenarios:

```bash
make plot-legacy-all LEGACY_YEAR=2100
```

Outputs are written to:

* `outputs/plots_legacy/`

---

## Common recipes

Attach SSP126 year 2025 on a private dataset (CSV only):

```bash
make attach-legacy-csv POINTS=data/private/carrefour_full.csv SCEN=126 LEGACY_YEAR=2025
```

Attach all scenarios for year 2050 (CSV only):

```bash
make attach-legacy-all-csv POINTS=data/private/carrefour_full.csv LEGACY_YEAR=2050
```

Plot legacy square (SSP370, 2100):

```bash
make plot-legacy LEGACY_YEAR=2100 SCEN=370 VARS="MSA_SQUARE"
```