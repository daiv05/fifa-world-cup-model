# El Oráculo del Balón — Predicción del Mundial 2026

Modelo de predicción para la FIFA World Cup 2026 basado en Machine Learning (XGBoost / LightGBM) y simulación Monte Carlo (10,000 iteraciones). Predice probabilidades de campeonato para los 48 equipos del torneo.

---

## Resultados principales

| Equipo | P(Campeón) | IC 90% |
|--------|:----------:|:------:|
| Spain | 15.2% | [14.6%, 15.8%] |
| France | 11.1% | [10.6%, 11.6%] |
| Argentina | 10.6% | [10.0%, 11.1%] |
| Brazil | 9.9% | [9.4%, 10.4%] |
| Germany | 6.5% | [6.1%, 6.9%] |

---

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                        # Datos crudos (no versionados)
│   │   ├── international_results.csv   # Auto-descargado de GitHub
│   │   ├── statsbomb_xg_by_team.csv    # Auto-generado por scraper.py
│   │   ├── squad_values.csv            # Snapshot manual Transfermarkt
│   │   ├── fifa_ranking.csv            # Ranking FIFA histórico
│   │   └── wc2026_fixture.csv          # Sorteo de los 48 equipos
│   └── processed/
│       ├── features.csv                  # Dataset de entrenamiento
│       ├── team_features.csv             # Features por equipo para simulación
│       ├── simulation_results.csv        # P(campeón) Monte Carlo
│       ├── tournament_progression.csv    # P(avanzar) por fase
│       ├── sensitivity_injuries.csv      # Análisis de sensibilidad
│       ├── model_evaluation.csv          # Log-loss / Brier por modelo
│       └── models/
│           ├── logreg_baseline.joblib
│           ├── xgboost.joblib
│           ├── lightgbm.joblib
│           ├── xgboost_calibrated.joblib
│           ├── xgboost_pre2022.joblib    # Modelo entrenado solo con date<2022
│           └── best_params_*.json        # Hiperparámetros de Optuna
├── notebooks/
│   └── 01_eda.ipynb                # Análisis exploratorio (10 secciones)
├── reports/figures/                 # SHAP, calibración, EDA, etc.
├── src/
│   ├── data/         # data_loader.py, scraper.py
│   ├── features/     # elo.py, time_decay.py, features.py
│   ├── models/       # train.py, evaluate.py
│   ├── simulation/   # tournament.py, simulate.py
│   ├── analysis/     # sensitivity.py
│   └── visualization/  # dashboard.py
├── tests/
├── conftest.py
├── pyproject.toml
├── Makefile
└── requirements.txt
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url>
cd fifa-world-cup-model

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS

# 3. Instalar el paquete en modo editable (preferido)
pip install -e .

# alternativa:
# pip install -r requirements.txt
```

---

## Pipeline de ejecución

Ejecutar desde la raíz del proyecto (todos los módulos son `src.*`):

```bash
make all          # features → train → evaluate → simulate → sensitivity
```

O paso a paso:

```bash
python -m src.features.features
python -m src.models.train --trials 100
python -m src.models.evaluate
python -m src.simulation.simulate --iterations 10000 --model xgboost_calibrated
python -m src.analysis.sensitivity --iterations 10000 --model xgboost_calibrated
streamlit run src/visualization/dashboard.py
```

### EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
# o ejecutar de un golpe:
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.executed.ipynb
```

El notebook genera ~15 figuras en `reports/figures/eda_*.png` que documentan
calidad de datos, distribución del target, análisis univariado por feature,
correlaciones, evolución del ELO, SHAP y limitaciones del dataset.

Outputs principales:
- `data/processed/features.csv` — dataset de entrenamiento.
- `data/processed/models/*.joblib` — LogReg, XGBoost, LightGBM, XGBoost calibrado, XGBoost pre-2022.
- `data/processed/model_evaluation.csv` — log-loss / Brier.
- `data/processed/simulation_results.csv` — P(campeón) con IC Clopper-Pearson.
- `data/processed/tournament_progression.csv` — P(avanzar) por fase.
- `data/processed/sensitivity_injuries.csv` — análisis de sensibilidad.
- `reports/figures/shap_summary.png`.

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Metodología

### Datos
| Fuente | Contenido | Partidos / Equipos |
|--------|-----------|-------------------|
| [martj42/international_results](https://github.com/martj42/international_results) | Resultados históricos 1872–2024 | ~47,000 partidos |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | xG por equipo (internacionales) | 109 equipos |
| Transfermarkt (snapshot manual, ver `scraper.SQUAD_VALUES_SNAPSHOT_DATE`) | Valor de mercado de plantilla | 60 equipos |
| FIFA Ranking (CSV histórico) | Posición y puntos por equipo y fecha | ~210 equipos |

### Features (7)
| Feature | Descripción | Justificación |
|---------|-------------|---------------|
| `elo_diff` | ELO local − ELO visitante | Métrica dinámica, superior al ranking FIFA estático |
| `squad_value_diff` | log(valor_local) − log(valor_visitante) | Proxy de calidad individual de la plantilla |
| `xg_avg_for` | xG promedio a favor: local − visitante | Eficiencia ofensiva reciente |
| `xg_avg_against` | xG promedio en contra: local − visitante | Solidez defensiva reciente |
| `travel_distance_home` | Distancia (km) de la capital del local a la sede real del partido (`country`). 0.0 si la sede no se puede geocodificar | Fatiga / desventaja de viaje |
| `travel_distance_away` | Ídem para el visitante | Ídem |
| `ranking_diff` | rank_visitante − rank_local (positivo = local mejor rankeado) | Captura cambios discretos del ranking FIFA |

**Decaimiento temporal:** cada partido tiene peso W(t) = e^(−0.002 · Δt) multiplicado por peso de clase balanceado (H≈49%, E≈21%, V≈30%).

### Modelos
- LogReg (baseline, escalado + balanceado).
- XGBoost — optimizado con Optuna (100 trials por default).
- LightGBM — optimizado con Optuna.
- XGBoost calibrado (isotonic) sobre validación temporal (2021).
- XGBoost pre-2022 — entrenado solo con `date < 2022-01-01` para validar el Mundial 2022 sin data leakage.

Hiperparámetros se cachean en `data/processed/models/best_params_{model}.json`.

Pesos de entrenamiento: `w = class_weight_balanced × time_decay`, re-normalizados por su media. Aborda el desbalance de clases (H:49% / D:21% / V:30%) sin sacrificar información temporal.

Las métricas exactas se generan con `make evaluate` y quedan en `data/processed/model_evaluation.csv`. La validación WC2022 se hace exclusivamente con `xgboost_pre2022`.

### Simulación Monte Carlo
- **10,000 iteraciones** del torneo completo (104 partidos c/u).
- Probabilidades simétricas: para cada par `(t1, t2)` se promedia `P(t1 vs t2)` con `P(t2 vs t1)` (invertida) para eliminar sesgo home/away. Excepción: anfitriones (USA, México, Canadá) reciben localía cuando juegan en su país.
- Goles modelados con Poisson independiente sobre `xg_for / xg_against` de los dos equipos. El outcome surge del marcador, no al revés.
- Desempate de grupos: puntos → diferencia de goles → goles a favor (FIFA 2026).
- Knockout: en caso de empate, penalty shootout (50/50).
- IC binomial: Clopper-Pearson (`scipy.stats.beta`).
- Tracking de avance por fase: `data/processed/tournament_progression.csv` (P de llegar a grupos / R32 / R16 / QF / SF / Final / Champion).

### Validación histórica (WC 2022)
El modelo `xgboost_pre2022` se entrena exclusivamente con `date < 2022-01-01` y se evalúa sobre el Mundial 2022. Las métricas exactas quedan en consola al ejecutar `make evaluate`.

---

## Limitaciones conocidas
- Los equipos debutantes (Uzbekistán, Curaçao, etc.) tienen muy pocos partidos históricos — ELO inicial por defecto (1500).
- Las lesiones de última hora no están modeladas de forma estructural, pero `src/analysis/sensitivity.py` simula escenarios `-30% squad_value` sobre el top-5 (ver `data/processed/sensitivity_injuries.csv`).
- El xG de StatsBomb cubre principalmente torneos UEFA/FIFA; equipos de otras confederaciones usan `1.2` por defecto (media global aprox.).
- `travel_distance = 0.0` cuando la sede del partido no se puede geocodificar (campo neutral / dato faltante).
- El snapshot Transfermarkt es manual (no scraping en vivo); fecha en `scraper.SQUAD_VALUES_SNAPSHOT_DATE`.

---

## Equipo
Proyecto desarrollado como trabajo final de curso · 2026  
Datos, Modelado y Simulación Monte Carlo.
