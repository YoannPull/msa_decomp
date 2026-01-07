# Makefile — Carrefour MSA pipeline (LU + ROAD + CC + N + ENC + SQUARE + GLOBIO + LEGACY + POSTPROCESS)

PY := poetry run python

# -----------------------------------------------------------------------------
# Defaults (override from command line)
# -----------------------------------------------------------------------------
YEAR ?= 2100
SCEN ?= 126
DEBUG ?= 0

# Points extraction defaults
POINTS ?= data/private/carrefour_loc.csv
POINTS_OUT ?= outputs/points_with_msa_$(SCEN)_$(YEAR).parquet
LAT_COL ?= y_latitude
LON_COL ?= x_longitude

# Perf knobs for extraction
MAX_FULL_LOAD_MB ?= 300
ENGINE ?=

# GLOBIO defaults
GLOBIO_DIR ?= data/sources/GLOBIO
GLOBIO_YEAR ?= 2050
GLOBIO_SSPS ?= 1 3 5
GLOBIO_YEARS ?= 2015 2050
GLOBIO_OUTCOL ?= MSA_GLOBIO

# LEGACY (Pierre) defaults
LEGACY_ROOT ?= data/datamart_legacy
LEGACY_SUFFIX ?= _PIERRE
LEGACY_OUTDIR ?= outputs/output_legacy
LEGACY_MAX_FULL_LOAD_MB ?= 400
# IMPORTANT for Pierre legacy extraction: requires lat flip to match geography
LEGACY_LAT_FLIP ?= flip   # flip|none|auto

# LEGACY check columns
LEGACY_VERIFY ?= 1        # 1 -> run --verify-square (logs only)
LEGACY_ADD_CHECK ?= 1     # 1 -> add recon/diff/OK columns and write *_checked.*
LEGACY_CHECK_TOL ?= 1e-3  # tolerance for OK flag in *_checked.* outputs

# LEGACY plotting
LEGACY_PLOTS_DIR ?= outputs/plots_legacy
LEGACY_CMAP ?= msa
LEGACY_GAMMA ?= 1.0
LEGACY_ORIGIN ?= upper    # confirmed correct for legacy plots

# IMPORTANT: plot_legacy_msa.py requires --vars
# Override if your legacy dataset uses different names
LEGACY_VARS ?= MSA_SQUARE MSA_LU MSA_ROAD MSA_CC MSA_N MSA_ENC

# POSTPROCESS defaults
POST_DIR ?= outputs/points_postprocess
TOPK ?= 30
ROAD_COL ?= MSA_ROAD
ROAD_FILL ?= 1

.PHONY: help \
        prepare-road \
        lu road cc n enc square \
        extract-points extract-points-all extract-points-fast \
        attach-globio attach-globio-ssp attach-globio-all-ssps attach-globio-all-years-ssps \
        attach-legacy attach-legacy-all legacy-check \
        postprocess postprocess-all \
        plot-legacy plot-legacy-all \
        plot-lu plot-lu-all \
        plot-road plot-road-all \
        plot-cc plot-cc-all \
        plot-n plot-n-all \
        plot-enc plot-enc-all \
        plot-square plot-square-all \
        all-points all-legacy

help:
	@echo "Targets:"
	@echo "  make prepare-road"
	@echo "  make lu / road / cc / n / enc / square"
	@echo "  make extract-points POINTS=... SCEN=585 YEAR=2100"
	@echo "  make extract-points-all POINTS=... YEAR=2100"
	@echo "  make extract-points-fast ... MAX_FULL_LOAD_MB=600 ENGINE=h5netcdf"
	@echo "  make attach-globio POINTS=... GLOBIO_YEAR=2015"
	@echo "  make attach-globio-ssp POINTS=... GLOBIO_YEAR=2050 SSP=5"
	@echo "  make attach-globio-all-ssps POINTS=... GLOBIO_YEAR=2050"
	@echo "  make attach-globio-all-years-ssps POINTS=..."
	@echo "  make attach-legacy POINTS=... SCEN=585 YEAR=2100 LEGACY_LAT_FLIP=flip"
	@echo "    - add DEBUG=1 for verbose logs"
	@echo "    - LEGACY_VERIFY=1 logs square reconstruction metrics"
	@echo "    - LEGACY_ADD_CHECK=1 writes *_checked.(parquet|csv) with recon/diff/flags"
	@echo "  make legacy-check SCEN=126 YEAR=2100  -> add check columns on existing legacy outputs"
	@echo "  make attach-legacy-all POINTS=... YEAR=2100"
	@echo "  make plot-legacy SCEN=126 YEAR=2100 VARS='MSA_SQUARE MSA_LU'"
	@echo "  make plot-legacy-all YEAR=2100"
	@echo "  make postprocess INPUTS='a.parquet b.parquet ...'"
	@echo "  make postprocess-all YEAR=2100"
	@echo "  make all-points POINTS=... YEAR=2100   -> extract 126/370/585 then postprocess-all"
	@echo "  make all-legacy POINTS=... YEAR=2100   -> attach legacy 126/370/585 + check + plot legacy"

# -----------------------------------------------------------------------------
# Prepare datamart from public sources
# -----------------------------------------------------------------------------

prepare-road:
	$(PY) scripts/prepare/30_prepare_grip4_road.py \
	  --grip4-asc data/sources/ROAD/GRIP4/grip4_total_dens_m_km2.asc \
	  --oceans-nc data/datamart/oceans.nc \
	  --out-nc data/outputs/ROAD/road_on_msa_grid.nc

# -----------------------------------------------------------------------------
# Build MSA datamarts
# -----------------------------------------------------------------------------

lu:
	$(PY) scripts/build_msa_lu.py

road: prepare-road
	$(PY) scripts/build_msa_road.py

cc:
	$(PY) scripts/build_msa_cc.py

n:
	$(PY) scripts/build_msa_n.py

enc:
	$(PY) scripts/build_msa_enc.py

square:
	$(PY) scripts/build_msa_square.py --fill-road-na-as-one --fill-cc-na-as-one

# -----------------------------------------------------------------------------
# Extract MSA + components for point coordinates
# -----------------------------------------------------------------------------

extract-points:
	$(PY) scripts/extract_msa_points.py \
	  --input $(POINTS) \
	  --out $(POINTS_OUT) \
	  --scen $(SCEN) \
	  --year $(YEAR) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL)

extract-points-fast:
	$(PY) scripts/extract_msa_points.py \
	  --input $(POINTS) \
	  --out $(POINTS_OUT) \
	  --scen $(SCEN) \
	  --year $(YEAR) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --max-full-load-mb $(MAX_FULL_LOAD_MB) \
	  $(if $(strip $(ENGINE)),--engine $(ENGINE),)

extract-points-all:
	$(PY) scripts/extract_msa_points.py \
	  --input $(POINTS) \
	  --out outputs/points_with_msa_126_$(YEAR).parquet \
	  --scen 126 --year $(YEAR) --lat-col $(LAT_COL) --lon-col $(LON_COL)
	$(PY) scripts/extract_msa_points.py \
	  --input $(POINTS) \
	  --out outputs/points_with_msa_370_$(YEAR).parquet \
	  --scen 370 --year $(YEAR) --lat-col $(LAT_COL) --lon-col $(LON_COL)
	$(PY) scripts/extract_msa_points.py \
	  --input $(POINTS) \
	  --out outputs/points_with_msa_585_$(YEAR).parquet \
	  --scen 585 --year $(YEAR) --lat-col $(LAT_COL) --lon-col $(LON_COL)

# Handy: all scenarios points + postprocess in one go
all-points: extract-points-all postprocess-all

# -----------------------------------------------------------------------------
# Attach GLOBIO aggregated MSA (GeoTIFF)
# -----------------------------------------------------------------------------

attach-globio:
	$(PY) scripts/attach_globio_msa.py \
	  --input $(POINTS) \
	  --out outputs/points_globio_$(GLOBIO_YEAR).parquet \
	  --globio-dir $(GLOBIO_DIR) \
	  --year $(GLOBIO_YEAR) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --out-col $(GLOBIO_OUTCOL)

attach-globio-ssp:
	$(PY) scripts/attach_globio_msa.py \
	  --input $(POINTS) \
	  --out outputs/points_globio_$(GLOBIO_YEAR)_SSP$(SSP).parquet \
	  --globio-dir $(GLOBIO_DIR) \
	  --year $(GLOBIO_YEAR) \
	  --ssp $(SSP) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --out-col $(GLOBIO_OUTCOL)

attach-globio-all-ssps:
	@set -e; \
	for s in $(GLOBIO_SSPS); do \
	  echo "[RUN] GLOBIO year=$(GLOBIO_YEAR) SSP$$s"; \
	  $(MAKE) attach-globio-ssp POINTS="$(POINTS)" GLOBIO_DIR="$(GLOBIO_DIR)" GLOBIO_YEAR=$(GLOBIO_YEAR) SSP=$$s LAT_COL="$(LAT_COL)" LON_COL="$(LON_COL)" GLOBIO_OUTCOL="$(GLOBIO_OUTCOL)"; \
	done

attach-globio-all-years-ssps:
	@set -e; \
	for y in $(GLOBIO_YEARS); do \
	  if [ "$$y" = "2015" ]; then \
	    echo "[RUN] GLOBIO year=2015 (no SSP)"; \
	    $(MAKE) attach-globio POINTS="$(POINTS)" GLOBIO_DIR="$(GLOBIO_DIR)" GLOBIO_YEAR=2015 LAT_COL="$(LAT_COL)" LON_COL="$(LON_COL)" GLOBIO_OUTCOL="$(GLOBIO_OUTCOL)"; \
	  else \
	    echo "[RUN] GLOBIO year=$$y (SSPs: $(GLOBIO_SSPS))"; \
	    $(MAKE) attach-globio-all-ssps POINTS="$(POINTS)" GLOBIO_DIR="$(GLOBIO_DIR)" GLOBIO_YEAR=$$y LAT_COL="$(LAT_COL)" LON_COL="$(LON_COL)" GLOBIO_OUTCOL="$(GLOBIO_OUTCOL)"; \
	  fi; \
	done

# -----------------------------------------------------------------------------
# Attach Pierre legacy MSA_* (NetCDF datamart_legacy)
# -----------------------------------------------------------------------------

attach-legacy:
	@mkdir -p $(LEGACY_OUTDIR)

	# 1) Attach legacy (parquet)
	$(PY) scripts/legacy/attach_legacy_msa.py \
	  --input $(POINTS) \
	  --out $(LEGACY_OUTDIR)/points_with_legacy_$(SCEN)_$(YEAR).parquet \
	  --legacy-root $(LEGACY_ROOT) \
	  --scen $(SCEN) \
	  --year $(YEAR) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --suffix $(LEGACY_SUFFIX) \
	  --max-full-load-mb $(LEGACY_MAX_FULL_LOAD_MB) \
	  --lat-flip $(LEGACY_LAT_FLIP) \
	  $(if $(filter 1,$(DEBUG)),--debug,) \
	  $(if $(filter 1,$(LEGACY_VERIFY)),--verify-square,)

	# 2) Attach legacy (csv)
	$(PY) scripts/legacy/attach_legacy_msa.py \
	  --input $(POINTS) \
	  --out $(LEGACY_OUTDIR)/points_with_legacy_$(SCEN)_$(YEAR).csv \
	  --legacy-root $(LEGACY_ROOT) \
	  --scen $(SCEN) \
	  --year $(YEAR) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --suffix $(LEGACY_SUFFIX) \
	  --max-full-load-mb $(LEGACY_MAX_FULL_LOAD_MB) \
	  --lat-flip $(LEGACY_LAT_FLIP) \
	  $(if $(filter 1,$(DEBUG)),--debug,) \
	  $(if $(filter 1,$(LEGACY_VERIFY)),--verify-square,)

	# 3) Optionnel: ajouter colonnes de check (parquet + csv)
	@if [ "$(LEGACY_ADD_CHECK)" = "1" ]; then \
	  $(MAKE) legacy-check SCEN=$(SCEN) YEAR=$(YEAR); \
	fi

legacy-check:
	$(PY) scripts/legacy/add_square_check_column.py \
	  --in $(LEGACY_OUTDIR)/points_with_legacy_$(SCEN)_$(YEAR).parquet \
	  --out $(LEGACY_OUTDIR)/points_with_legacy_$(SCEN)_$(YEAR)_checked.parquet \
	  --suffix $(LEGACY_SUFFIX) \
	  --tol $(LEGACY_CHECK_TOL)

	$(PY) scripts/legacy/add_square_check_column.py \
	  --in $(LEGACY_OUTDIR)/points_with_legacy_$(SCEN)_$(YEAR).csv \
	  --out $(LEGACY_OUTDIR)/points_with_legacy_$(SCEN)_$(YEAR)_checked.csv \
	  --suffix $(LEGACY_SUFFIX) \
	  --tol $(LEGACY_CHECK_TOL)

attach-legacy-all:
	@set -e; \
	for s in 126 370 585; do \
	  echo "[RUN] attach-legacy scen=$$s year=$(YEAR) lat_flip=$(LEGACY_LAT_FLIP)"; \
	  $(MAKE) attach-legacy SCEN=$$s YEAR=$(YEAR) POINTS="$(POINTS)" LAT_COL="$(LAT_COL)" LON_COL="$(LON_COL)" \
	    LEGACY_ROOT="$(LEGACY_ROOT)" LEGACY_SUFFIX="$(LEGACY_SUFFIX)" LEGACY_OUTDIR="$(LEGACY_OUTDIR)" \
	    LEGACY_MAX_FULL_LOAD_MB="$(LEGACY_MAX_FULL_LOAD_MB)" LEGACY_LAT_FLIP="$(LEGACY_LAT_FLIP)" \
	    DEBUG="$(DEBUG)" LEGACY_VERIFY="$(LEGACY_VERIFY)" LEGACY_ADD_CHECK="$(LEGACY_ADD_CHECK)" LEGACY_CHECK_TOL="$(LEGACY_CHECK_TOL)"; \
	done

# Handy: attach legacy + plot legacy for all scens
all-legacy: attach-legacy-all plot-legacy-all

# -----------------------------------------------------------------------------
# Postprocess points (ROAD NaN->1 + CSVs + plots)
# -----------------------------------------------------------------------------

postprocess:
	@mkdir -p $(POST_DIR)
	$(PY) scripts/postprocess_points.py \
	  --inputs $(INPUTS) \
	  --out-dir $(POST_DIR) \
	  --road-col $(ROAD_COL) \
	  --road-fill $(ROAD_FILL) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --topk $(TOPK)

postprocess-all:
	@mkdir -p $(POST_DIR)
	$(PY) scripts/postprocess_points.py \
	  --inputs outputs/points_with_msa_126_$(YEAR).parquet outputs/points_with_msa_370_$(YEAR).parquet outputs/points_with_msa_585_$(YEAR).parquet \
	  --out-dir $(POST_DIR) \
	  --road-col $(ROAD_COL) \
	  --road-fill $(ROAD_FILL) \
	  --lat-col $(LAT_COL) \
	  --lon-col $(LON_COL) \
	  --topk $(TOPK)

# -----------------------------------------------------------------------------
# Plot LEGACY maps (Pierre datamart_legacy)
# -----------------------------------------------------------------------------
# Example:
#   make plot-legacy YEAR=2100 SCEN=126 VARS="MSA_SQUARE MSA_LU"
#   make plot-legacy-all YEAR=2100
#   make plot-legacy-all YEAR=2100 LEGACY_VARS="MSA_SQUARE"

plot-legacy:
	@mkdir -p $(LEGACY_PLOTS_DIR)
	$(PY) scripts/legacy/plot_legacy_msa.py \
	  --legacy-root $(LEGACY_ROOT) \
	  --out-dir $(LEGACY_PLOTS_DIR) \
	  --year $(YEAR) \
	  --scens $(SCEN) \
	  --vars $(if $(strip $(VARS)),$(VARS),$(LEGACY_VARS)) \
	  --cmap $(LEGACY_CMAP) \
	  --gamma $(LEGACY_GAMMA) \
	  --origin $(LEGACY_ORIGIN)

plot-legacy-all:
	@mkdir -p $(LEGACY_PLOTS_DIR)
	$(PY) scripts/legacy/plot_legacy_msa.py \
	  --legacy-root $(LEGACY_ROOT) \
	  --out-dir $(LEGACY_PLOTS_DIR) \
	  --year $(YEAR) \
	  --scens 126 370 585 \
	  --vars $(LEGACY_VARS) \
	  --cmap $(LEGACY_CMAP) \
	  --gamma $(LEGACY_GAMMA) \
	  --origin $(LEGACY_ORIGIN)

# -----------------------------------------------------------------------------
# Plotting (your computed datamart)
# -----------------------------------------------------------------------------

plot-lu:
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen $(SCEN)

plot-lu-all:
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen 585

plot-road:
	$(PY) scripts/plot_msa_road.py --year $(YEAR) --scen $(SCEN)

plot-road-all:
	$(PY) scripts/plot_msa_road.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_road.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_road.py --year $(YEAR) --scen 585

plot-cc:
	$(PY) scripts/plot_msa_cc.py --year $(YEAR) --scen $(SCEN)

plot-cc-all:
	$(PY) scripts/plot_msa_cc.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_cc.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_cc.py --year $(YEAR) --scen 585

plot-n:
	$(PY) scripts/plot_msa_n.py --year $(YEAR) --scen $(SCEN)

plot-n-all:
	$(PY) scripts/plot_msa_n.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_n.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_n.py --year $(YEAR) --scen 585

plot-enc:
	$(PY) scripts/plot_msa_enc.py --year $(YEAR) --scen $(SCEN)

plot-enc-all:
	$(PY) scripts/plot_msa_enc.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_enc.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_enc.py --year $(YEAR) --scen 585

plot-square:
	$(PY) scripts/plot_msa_square.py --year $(YEAR) --scen $(SCEN)

plot-square-all:
	$(PY) scripts/plot_msa_square.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_square.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_square.py --year $(YEAR) --scen 585
