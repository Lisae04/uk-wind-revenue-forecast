PY ?= python

.PHONY: install test demo all bmus clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

demo:            ## offline run on synthetic data (no network, ~1 min)
	$(PY) scripts/run_pipeline.py --synthetic

all:             ## real data: Open-Meteo + Elexon (first run downloads and caches, ~30-60 min)
	$(PY) scripts/run_pipeline.py

bmus:            ## list Elexon wind BM units to pick reference farms
	$(PY) scripts/find_wind_bmus.py --check

clean:
	rm -rf data/processed/* results/*.csv results/*.json results/summary.md results/figures/*
