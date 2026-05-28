# Makefile - fifa-world-cup-model
# Ejecutar desde la raiz del proyecto

PY ?= python
ITER ?= 10000
MODEL ?= xgboost_calibrated
TRIALS ?= 100

.PHONY: install features train evaluate ablation simulate sensitivity dashboard test all clean

install:
	$(PY) -m pip install -e .

features:
	$(PY) -m src.features.features

train:
	$(PY) -m src.models.train --trials $(TRIALS)

evaluate:
	$(PY) -m src.models.evaluate

ablation:
	$(PY) -m src.analysis.ablation

simulate:
	$(PY) -m src.simulation.simulate --iterations $(ITER) --model $(MODEL)

sensitivity:
	$(PY) -m src.analysis.sensitivity --iterations $(ITER) --model $(MODEL)

dashboard:
	streamlit run src/visualization/dashboard.py

test:
	$(PY) -m pytest tests/ -v

all: features train evaluate simulate sensitivity

clean:
	rm -rf data/processed/*.csv data/processed/models/*.joblib reports/figures/*.png
