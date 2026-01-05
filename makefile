# Makefile — Carrefour MSA pipeline (LU + ROAD + CC)
# Usage:
#   make help
#   make lu
#   make road
#   make cc
#   make plot-lu YEAR=2100 SCEN=126
#   make plot-lu-all YEAR=2100
#   make plot-road YEAR=2100 SCEN=126
#   make plot-road-all YEAR=2100
#   make plot-cc YEAR=2100 SCEN=126
#   make plot-cc-all YEAR=2100

PY := poetry run python

# Defaults (override from command line)
YEAR ?= 2100
SCEN ?= 126

.PHONY: help lu road cc plot-lu plot-lu-all plot-road plot-road-all plot-cc plot-cc-all

help:
	@echo "Targets:"
	@echo "  make lu                         -> build MSA_LU datamart (and oceans.nc)"
	@echo "  make road                       -> build MSA_ROAD datamart"
	@echo "  make cc                         -> build MSA_CC datamart"
	@echo "  make plot-lu YEAR=2100 SCEN=126  -> plot one LU map to outputs/plots/"
	@echo "  make plot-lu-all YEAR=2100       -> plot LU maps for 126,370,585"
	@echo "  make plot-road YEAR=2100 SCEN=126-> plot one ROAD map to outputs/plots/"
	@echo "  make plot-road-all YEAR=2100     -> plot ROAD maps for 126,370,585"
	@echo "  make plot-cc YEAR=2100 SCEN=126  -> plot one CC map to outputs/plots/"
	@echo "  make plot-cc-all YEAR=2100       -> plot CC maps for 126,370,585"

lu:
	$(PY) scripts/build_msa_lu.py

road:
	$(PY) scripts/build_msa_road.py

cc:
	$(PY) scripts/build_msa_cc.py

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
