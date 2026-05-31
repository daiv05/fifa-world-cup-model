# Predicciones - Mundial 2026

Modelo de predicción para la FIFA World Cup 2026 basado en Machine Learning (XGBoost / LightGBM) y simulación Monte Carlo (10,000 iteraciones). Predice probabilidades de campeonato para los 48 equipos del torneo.

---

## Dashboard interactivo

Mediante Streamlit, se puede explorar el modelo, las probabilidades de cada equipo, la progresión por fases y el análisis de sensibilidad a lesiones.

[https://fifa-world-cup-model.streamlit.app/](https://fifa-world-cup-model.streamlit.app/)

---

## Resultados principales

| Equipo | P(Campeón) | IC 95% |
|--------|:----------:|:------:|
| Spain | 23.19% | [22.37%, 24.03%] |
| Argentina | 10.81% | [10.21%, 11.44%] |
| France | 10.15% | [9.56%, 10.76%] |
| Brazil | 6.46% | [5.99%, 6.96%] |
| England | 5.56% | [5.12%, 6.03%] |

---

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                        # Datos crudos
│   │   ├── international_results.csv   # Auto-descargado de GitHub
│   │   ├── statsbomb_xg_by_team.csv    # Auto-generado por scraper.py
│   │   ├── squad_values.csv            # Snapshot manual Transfermarkt
│   │   ├── fifa_ranking.csv            # Ranking FIFA histórico (hasta 2024-06)
│   │   ├── wc2026_fixture.csv          # Sorteo de los 48 equipos
│   │   └── new-data/international_results/
│   │       ├── goalscorers.csv         # Goleadores por partido (as-of features)
│   │       ├── shootouts.csv           # Tandas de penales
│   │       ├── former_names.csv        # Provenance de continuidad de ELO
│   │       └── fifa_rankings_2026.csv  # Snapshot ranking 2026-05-30
│   └── processed/
│       ├── features.csv                  # Dataset de entrenamiento
│       ├── team_features.csv             # Features por equipo para simulación
│       ├── simulation_results.csv        # P(campeón) Monte Carlo
│       ├── tournament_progression.csv    # P(avanzar) por fase
│       ├── sensitivity_injuries.csv      # Análisis de sensibilidad
│       ├── model_evaluation.csv          # Log-loss / Brier por modelo
│       ├── ablation_results.csv          # Ablación de grupos de features
│       └── models/
│           ├── logreg_baseline.joblib
│           ├── xgboost.joblib
│           ├── lightgbm.joblib
│           ├── xgboost_calibrated.joblib
│           ├── xgboost_pre2022.joblib    # Modelo entrenado solo con date<2022
│           └── best_params_*.json        # Hiperparámetros de Optuna
├── notebooks/
│   └── 01_eda.ipynb                # Análisis exploratorio
├── reports/figures/                 # SHAP, calibración, EDA
├── src/
│   ├── data/         # data_loader.py, scraper.py
│   ├── features/     # elo.py, time_decay.py, features.py
│   ├── models/       # train.py, evaluate.py
│   ├── simulation/   # tournament.py, simulate.py
│   ├── analysis/     # sensitivity.py, ablation.py
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
git clone https://github.com/daiv05/fifa-world-cup-model
cd fifa-world-cup-model

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS

# 3. Instalar el paquete en modo editable (preferido)
pip install -e .
```

---

## Pipeline de ejecución

Ejecutar desde la raíz del proyecto:

```bash
python -m src.features.features
python -m src.models.train --trials 100
python -m src.models.evaluate
python -m src.analysis.ablation
python -m src.simulation.simulate --iterations 10000 --model xgboost_calibrated
python -m src.analysis.sensitivity --iterations 10000 --model xgboost_calibrated
streamlit run src/visualization/dashboard.py
```

### EDA

Abrir notebook en VSCode con la extensión de Jupyter para ejecutar `notebooks/01_eda.ipynb`. El análisis exploratorio cubre calidad de datos, distribución del target, análisis univariado por feature, correlaciones, evolución del ELO, SHAP y limitaciones del dataset.

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
| [martj42/international_results](https://github.com/martj42/international_results) | Resultados históricos 1872-2026 | ~49,000 partidos |
| ↳ `goalscorers.csv` | Goleadores por partido (minuto, penal, autogol) | ~47,600 goles |
| ↳ `shootouts.csv` | Tandas de penales históricas | 677 tandas |
| ↳ `former_names.csv` | Mapeo de nombres históricos → actuales | provenance |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | xG por equipo (internacionales) | 109 equipos |
| Transfermarkt (snapshot manual) | Valor de mercado de plantilla | 60 equipos |
| FIFA Ranking (CSV histórico) | Posición y puntos por equipo y fecha | ~210 equipos, hasta 2024-06-20 |
| `fifa_rankings_2026.csv` (snapshot) | Ranking FIFA al 2026-05-30 | 211 equipos |

**Continuidad de ELO (former_names):** antes de calcular el ELO se unifican nombres
de federaciones sucesoras reconocidas por FIFA: Yugoslavia → Serbia y Checoslovaquia
→ Chequia (Eslovaquia y Montenegro se tratan como entidades nuevas). URSS→Rusia,
Zaire→DR Congo y Antillas→Curacao ya vienen unificados en la fuente.

**Snapshot de ranking 2026:** la serie histórica termina el 2024-06-20; se anexa el
snapshot del 2026-05-30 como un punto más. Es libre de leakage porque los consumidores
usan `merge_asof(backward)`/bisect: un partido en fecha D solo ve ranks con fecha ≤ D,
así que el snapshot solo aplica al horizonte de predicción (el propio Mundial). No se
deriva un `points_diff` porque la metodología de puntos FIFA cambió en 2018 (ruptura de
escala); el rank ordinal es robusto a ese cambio.

### Features (9)
Tres horizontes temporales distintos gobiernan el pipeline (`features.py`):
`ELO_HISTORY_START=None` (ELO sobre TODA la historia), `OUTPUT_ROW_START_YEAR=1993`
(primer año emitido como fila) y `TRAIN_MIN_YEAR=2010` (ventana de modelado). `FEATURE_COLS`
se define una sola vez en `features.py` y se importa en train/evaluate/ablation/simulate/dashboard.

| Feature | Descripción | Justificación |
|---------|-------------|---------------|
| `elo_diff` | ELO local − ELO visitante (ELO acumulado sobre la historia COMPLETA, incl. amistosos) | Métrica dinámica, superior al ranking FIFA estático |
| `squad_value_diff` | log(valor_local) − log(valor_visitante) | Proxy de calidad individual de la plantilla |
| `xg_avg_for` | xG promedio a favor: local − visitante | Eficiencia ofensiva reciente |
| `xg_avg_against` | xG promedio en contra: local − visitante | Solidez defensiva reciente |
| `travel_distance_diff` | dist(visitante→sede) − dist(local→sede), en km. Convención away−home; 0.0 si la sede no se geocodifica | Fatiga / desventaja de viaje; informativo también en partidos neutrales (Mundiales) |
| `ranking_diff` | rank_visitante − rank_local (positivo = local mejor rankeado) | Captura cambios discretos del ranking FIFA (refrescado al 2026-05-30) |
| `penalty_share_diff` | (goles de penal / totales) local − visitante, as-of-date | Perfil ofensivo / dependencia de penales |
| `striker_concentration_diff` | Herfindahl de goleadores, local − visitante, as-of-date | Dependencia de una figura vs. ataque distribuido |
| `shootout_winrate_diff` | winrate en tandas (shrinkage Bayes, prior 0.5, α=10) local − visitante, as-of-date | Desempate en eliminatorias |

Las features derivadas (3 últimas) se calculan **estrictamente as-of-date**
(`merge_asof(backward, allow_exact_matches=False)`): un partido en fecha D solo ve
eventos anteriores a D, sin leakage del propio partido.

> **Feature descartada:** `late_goal_ratio_diff` (goles tardíos) se evaluó y se eliminó
> por completo del pipeline. La ablación leave-one-out mostró aporte **negativo**
> (ruido neto): el modelo mejora sin ella (test log-loss 0.8582 → 0.8548).

**Decaimiento temporal:** cada partido tiene peso W(t) = e^(−0.001 · Δt) multiplicado por peso de clase balanceado (H≈49%, E≈21%, V≈30%).

### División del dataset

Split **temporal** implementado en `temporal_split` ([src/models/train.py](src/models/train.py)).

| Conjunto | Filtro de fecha | Uso |
|----------|-----------------|-----|
| **Train** | `date < 2021-01-01` | Entrenamiento de LogReg, XGBoost, LightGBM |
| **Validación** | `2021-01-01 ≤ date < 2022-01-01` | Calibración Platt (sigmoid) de XGBoost (sin leakage) |
| **Test** | `date ≥ 2022-01-01` | Evaluación final (`evaluate.py` reutiliza `temporal_split`) |

### Modelos
- LogReg (baseline, escalado + balanceado).
- XGBoost - optimizado con Optuna (100 trials por default).
- LightGBM - optimizado con Optuna.
- XGBoost calibrado con método Platt (sigmoid) sobre validación temporal (2021).
- XGBoost pre-2022 - entrenado solo con `date < 2022-01-01` para validar el Mundial 2022 sin data leakage.

### Estudio de ablación
`src/analysis/ablation.py` reentrena el XGBoost calibrado quitando grupos de features mientras mantiene fijos el split temporal, los pesos y los hiperparámetros óptimos, para aislar la contribución marginal de cada grupo sobre Log-Loss y Brier (resultados en `data/processed/ablation_results.csv`). Configuraciones evaluadas: completo, sin xG, sin `squad_value`, sin features derivadas, filas individuales para las features de aporte **marginal** (`travel_distance_diff`, `penalty_share_diff`, `shootout_winrate_diff`) que documentan su impacto, y sin features estáticas anacrónicas (xG+squad). Esta última cuantifica empíricamente el anacronismo descrito en *Limitaciones*.

### Simulación Monte Carlo
- **10,000 iteraciones** del torneo completo (104 partidos c/u).
- Probabilidades simétricas: para cada par `(t1, t2)` se promedia `P(t1 vs t2)` con `P(t2 vs t1)` (invertida) para eliminar sesgo home/away. Excepción: anfitriones (USA, México, Canadá) reciben localía cuando juegan en su país.
- Goles modelados con Poisson independiente sobre `xg_for / xg_against` de los dos equipos. El outcome surge del marcador, no al revés.
- Desempate de grupos: puntos - diferencia de goles - goles a favor (FIFA 2026).
- Knockout: en caso de empate, el desempate literal es 50/50. (La feature `shootout_winrate_diff` alimenta las probabilidades del modelo, no este coin-flip.)
- Tracking de avance por fase: `data/processed/tournament_progression.csv`.

### Validación histórica (WC 2022)
El modelo `xgboost_pre2022` se entrena exclusivamente con `date < 2022-01-01` y se evalúa sobre el Mundial 2022. Las métricas exactas quedan en consola al ejecutar la evaluación.

---

## Limitaciones conocidas
- Los equipos debutantes (Uzbekistán, Curaçao, etc.) tienen muy pocos partidos históricos - ELO inicial por defecto (1500).
- **Anacronismo de features estáticas (xG y squad_value):** ambos son snapshots (un valor por equipo, sin fecha) aplicados a TODOS los partidos históricos, lo que introduce leakage/anacronismo. No existe serie temporal histórica de estos datos, así que versionarlos sería fabricar información. El efecto está acotado por: time_decay (<2% pre-2010), `TRAIN_MIN_YEAR=2010` y el uso de diffs relativos. Su aporte se mide en la fila de ablación "Sin estáticas anacrónicas". El ELO **no** usa estos snapshots.
- Las lesiones de última hora no están modeladas de forma estructural, pero `src/analysis/sensitivity.py` simula escenarios `-30% squad_value` sobre el top-5 (ver `data/processed/sensitivity_injuries.csv`).
- El xG de StatsBomb cubre principalmente torneos UEFA/FIFA; equipos de otras confederaciones usan `1.2` por defecto (media global aproximada).
- **Cobertura de `travel_distance_diff`:** las coordenadas están precargadas solo para los 48 equipos del Mundial (`WC_TEAM_CAPITAL_COORDS`); los equipos históricos fuera del torneo no se geocodifican salvo `USE_NOMINATIM=1`, por lo que su diff queda en 0.0 en entrenamiento. La simulación (todos los equipos del WC) sí queda 100% poblada. Para cobertura histórica completa, ejecutar `features.py` con `USE_NOMINATIM=1` (poblará el cache geográfico vía Nominatim).
- **`shootout_winrate_diff`:** las potencias grandes tienen <5 tandas históricas, así que tras el shrinkage Bayes (prior 0.5) su valor es casi constante ~0.5. La señal se concentra en selecciones africanas/asiáticas con más tandas. Su aporte real se valida con la ablación "Sin shootout"; si no mejora el log-loss, considerar retirarla del set final.
- El ranking FIFA histórico termina el 2024-06-20; el periodo 2024-06 → 2026-05 solo se refresca con el snapshot puntual del 2026-05-30 (no hay serie mensual intermedia).
- El snapshot Transfermarkt es manual (no scraping en vivo); fecha en `scraper.SQUAD_VALUES_SNAPSHOT_DATE`.

---

## Hecho por
[David Deras](https://github.com/daiv05)
