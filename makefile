# Makefile — Carrefour MSA pipeline (LU + ROAD + CC + N)
# Usage:
#   make help
#   make lu
#   make road
#   make cc
#   make n
#   make plot-lu YEAR=2100 SCEN=126
#   make plot-lu-all YEAR=2100
#   make plot-road YEAR=2100 SCEN=126
#   make plot-road-all YEAR=2100
#   make plot-cc YEAR=2100 SCEN=126
#   make plot-cc-all YEAR=2100
#   make plot-n YEAR=2100 SCEN=126
#   make plot-n-all YEAR=2100

PY := poetry run python

# Defaults (override from command line)
YEAR ?= 2100
SCEN ?= 126

.PHONY: help lu road cc n \
        plot-lu plot-lu-all \
        plot-road plot-road-all \
        plot-cc plot-cc-all \
        plot-n plot-n-all

help:
	@echo "Targets:"
	@echo "  make lu                         -> build MSA_LU datamart (and oceans.nc)"
	@echo "  make road                       -> build MSA_ROAD datamart"
	@echo "  make cc                         -> build MSA_CC datamart"
	@echo "  make n                          -> build MSA_N datamart"
	@echo "  make plot-lu YEAR=2100 SCEN=126  -> plot one LU map to outputs/plots/"
	@echo "  make plot-lu-all YEAR=2100       -> plot LU maps for 126,370,585"
	@echo "  make plot-road YEAR=2100 SCEN=126-> plot one ROAD map to outputs/plots/"
	@echo "  make plot-road-all YEAR=2100     -> plot ROAD maps for 126,370,585"
	@echo "  make plot-cc YEAR=2100 SCEN=126  -> plot one CC map to outputs/plots/"
	@echo "  make plot-cc-all YEAR=2100       -> plot CC maps for 126,370,585"
	@echo "  make plot-n YEAR=2100 SCEN=126   -> plot one N map to outputs/plots/"
	@echo "  make plot-n-all YEAR=2100        -> plot N maps for 126,370,585"

lu:
	$(PY) scripts/build_msa_lu.py

road:
	$(PY) scripts/build_msa_road.py

cc:
	$(PY) scripts/build_msa_cc.py

n:
	$(PY) scripts/build_msa_n.py

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
