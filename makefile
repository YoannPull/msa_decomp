# Makefile — Carrefour MSA pipeline (LU only for now)
# Usage:
#   make help
#   make lu
#   make plot-lu YEAR=2100 SCEN=126
#   make plot-lu-all YEAR=2100

PY := poetry run python

# Defaults (override from command line)
YEAR ?= 2100
SCEN ?= 126

.PHONY: help lu plot-lu plot-lu-all

help:
	@echo "Targets:"
	@echo "  make lu                    -> compute MSA_LU datamart (and oceans.nc)"
	@echo "  make plot-lu YEAR=2100 SCEN=126   -> plot one LU map to outputs/plots/"
	@echo "  make plot-lu-all YEAR=2100        -> plot LU maps for SCEN=126,370,585"

lu:
	$(PY) scripts/build_msa_lu.py

plot-lu:
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen $(SCEN)

plot-lu-all:
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen 126
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen 370
	$(PY) scripts/plot_msa_lu.py --year $(YEAR) --scen 585
