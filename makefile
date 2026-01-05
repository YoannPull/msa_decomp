# Makefile — Carrefour MSA pipeline (minimal)
# Usage:
#   make help
#   make lu
#   make plot-lu

PY := poetry run python

.PHONY: help lu plot-lu

help:
	@echo "Targets:"
	@echo "  make lu       -> compute MSA_LU datamart (and oceans.nc)"
	@echo "  make plot-lu  -> plot MSA_LU and save to outputs/plots/"

lu:
	$(PY) scripts/build_msa_lu.py

plot-lu:
	$(PY) scripts/plot_msa_lu.py
