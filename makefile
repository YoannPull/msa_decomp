# Makefile — Carrefour MSA pipeline (LU + ROAD + CC + N + ENC + SQUARE)
# Usage:
#   make help
#   make lu
#   make road
#   make cc
#   make n
#   make enc
#   make square
#   make prepare-road
#   make plot-lu YEAR=2100 SCEN=126
#   make plot-lu-all YEAR=2100
#   make plot-road YEAR=2100 SCEN=126
#   make plot-road-all YEAR=2100
#   make plot-cc YEAR=2100 SCEN=126
#   make plot-cc-all YEAR=2100
#   make plot-n YEAR=2100 SCEN=126
#   make plot-n-all YEAR=2100
#   make plot-enc YEAR=2100 SCEN=126
#   make plot-enc-all YEAR=2100
#   make plot-square YEAR=2100 SCEN=126
#   make plot-square-all YEAR=2100

PY := poetry run python

# Defaults (override from command line)
YEAR ?= 2100
SCEN ?= 126

.PHONY: help lu road cc n enc square \
        prepare-road \
        plot-lu plot-lu-all \
        plot-road plot-road-all \
        plot-cc plot-cc-all \
        plot-n plot-n-all \
        plot-enc plot-enc-all \
        plot-square plot-square-all

help:
	@echo "Targets:"
	@echo "  make lu                            -> build MSA_LU datamart (and oceans.nc)"
	@echo "  make road                          -> prepare ROAD inputs then build MSA_ROAD datamart"
	@echo "  make cc                            -> build MSA_CC datamart"
	@echo "  make n                             -> build MSA_N datamart"
	@echo "  make enc                           -> build MSA_ENC datamart"
	@echo "  make square                        -> build MSA_SQUARE datamart (combined MSA)"
	@echo "  make prepare-road                  -> GRIP4 -> data/datamart/ROAD/road_on_msa_grid.nc"
	@echo "  make plot-lu YEAR=2100 SCEN=126     -> plot one LU map to outputs/plots/"
	@echo "  make plot-lu-all YEAR=2100          -> plot LU maps for 126,370,585"
	@echo "  make plot-road YEAR=2100 SCEN=126   -> plot one ROAD map to outputs/plots/"
	@echo "  make plot-road-all YEAR=2100        -> plot ROAD maps for 126,370,585"
	@echo "  make plot-cc YEAR=2100 SCEN=126     -> plot one CC map to outputs/plots/"
	@echo "  make plot-cc-all YEAR=2100          -> plot CC maps for 126,370,585"
	@echo "  make plot-n YEAR=2100 SCEN=126      -> plot one N map to outputs/plots/"
	@echo "  make plot-n-all YEAR=2100           -> plot N maps for 126,370,585"
	@echo "  make plot-enc YEAR=2100 SCEN=126    -> plot one ENC map to outputs/plots/"
	@echo "  make plot-enc-all YEAR=2100         -> plot ENC maps for 126,370,585"
	@echo "  make plot-square YEAR=2100 SCEN=126 -> plot one SQUARE map to outputs/plots/"
	@echo "  make plot-square-all YEAR=2100      -> plot SQUARE maps for 126,370,585"

# ---- Prepare datamart from public sources ----

prepare-road:
	$(PY) scripts/prepare/30_prepare_grip4_road.py \
	  --grip4-asc data/sources/ROAD/GRIP4/grip4_total_dens_m_km2.asc \
	  --oceans-nc data/datamart/oceans.nc \
	  --out-nc data/outputs/ROAD/road_on_msa_grid.nc

# ---- Build MSA datamarts ----

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


# ---- Plotting ----

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
