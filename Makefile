# Makefile - fifa-world-cup-model
# Ejecutar desde la raiz del proyecto

PY ?= python
ITER ?= 10000
MODEL ?= best
TRIALS ?= 100
SEEDS ?= 10
ENSEMBLE ?= 5

.PHONY: install features train evaluate ablation simulate ensemble sensitivity benchmark compare dashboard test all clean

install:
	$(PY) -m pip install -e .

features:
	$(PY) -m src.features.features

train:
	$(PY) -m src.models.train --trials $(TRIALS)

evaluate:
	$(PY) -m src.models.evaluate

ablation:
	$(PY) -m src.analysis.ablation --seeds $(SEEDS)

simulate:
	$(PY) -m src.simulation.simulate --iterations $(ITER) --model $(MODEL)

ensemble:
	$(PY) -m src.simulation.simulate --iterations $(ITER) --ensemble $(ENSEMBLE)

sensitivity:
	$(PY) -m src.analysis.sensitivity --iterations $(ITER) --model $(MODEL)

benchmark:
	$(PY) -m src.analysis.benchmark

compare:
	$(PY) -m src.analysis.compare_runs

dashboard:
	streamlit run src/visualization/dashboard.py

test:
	$(PY) -m pytest tests/ -v

all: features train evaluate ablation simulate sensitivity benchmark

clean:
	rm -rf data/processed/*.csv data/processed/models/*.joblib reports/figures/*.png
