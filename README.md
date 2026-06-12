# Predicciones - Mundial 2026

Modelo de predicción para la FIFA World Cup 2026 basado en Machine Learning y simulación Monte Carlo (10,000 iteraciones) sobre el bracket oficial del torneo. Predice probabilidades de campeonato para los 48 equipos.

> **v2 (2026-06):** revisión metodológica mayor tras una auditoría crítica
> (`reports/analisis_critico.md`, plan en `reports/plan_correcciones.md`):
> se eliminó el rebalanceo de clases (deformaba las probabilidades), se corrigió
> el bracket del Round of 32 al reglamento oficial FIFA, la validación WC2022 ya
> no usa features anacrónicas, la ablación reporta significancia con 10 semillas,
> y se añadió un benchmark contra el mercado de apuestas. Mejora neta verificada:
> log-loss de test 0.8547 → 0.8356 (Δ = −0.019, IC95 [−0.030, −0.009],
> bootstrap pareado).

---

## Dashboard interactivo

Mediante Streamlit, se puede explorar el modelo, las probabilidades de cada equipo, la progresión por fases y el análisis de sensibilidad a lesiones.

[https://fifa-world-cup-model.streamlit.app/](https://fifa-world-cup-model.streamlit.app/)

---

## Resultados principales

| Equipo | P(Campeón) | IC 95% (simulación) | Mercado (BetMGM 2026-06-10) |
|--------|:----------:|:------:|:------:|
| Spain | 27.46% | [26.59%, 28.35%] | 14.9% |
| Argentina | 13.78% | [13.11%, 14.47%] | 8.2% |
| France | 11.65% | [11.03%, 12.29%] | 13.7% |
| England | 7.73% | [7.21%, 8.27%] | 10.2% |
| Germany | 3.61% | [3.25%, 3.99%] | 5.5% |

**Lectura honesta de los intervalos:** el IC 95% reportado es **solo error de
muestreo Monte Carlo** (Clopper–Pearson, reducible con más iteraciones). No
incluye la incertidumbre del propio modelo; para eso ver
`simulation_ensemble.csv` (`--ensemble`), que reporta además el rango de
P(campeón) entre réplicas bootstrap del modelo. El modelo diverge del mercado
de apuestas en los extremos (sobrepondera al líder de ELO; ver
`data/processed/benchmark_market.csv` y la discusión en el paper).

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
│   │   ├── wc2026_bracket.csv          # Bracket OFICIAL FIFA (R32→final, sedes)
│   │   ├── market_odds_2026.csv        # Odds de mercado (benchmark externo)
│   │   └── new-data/international_results/
│   │       ├── goalscorers.csv         # Goleadores por partido (as-of features)
│   │       ├── shootouts.csv           # Tandas de penales
│   │       ├── former_names.csv        # Provenance de continuidad de ELO
│   │       └── fifa_rankings_2026.csv  # Snapshot ranking 2026-05-30
│   └── processed/
│       ├── features.csv                  # Dataset de entrenamiento
│       ├── team_features.csv             # Features por equipo para simulación
│       ├── simulation_results.csv        # P(campeón) Monte Carlo
│       ├── simulation_ensemble.csv       # P(campeón) + incertidumbre del modelo
│       ├── tournament_progression.csv    # P(avanzar) por fase
│       ├── sensitivity_injuries.csv      # Análisis de sensibilidad
│       ├── model_evaluation.csv          # Log-loss/Brier + IC bootstrap pareado
│       ├── ablation_results.csv          # Ablación multi-semilla con significancia
│       ├── benchmark_market.csv          # Modelo vs mercado de apuestas
│       ├── wc2022_validation.csv         # Validación out-of-time WC2022
│       ├── tune_elo_results.csv          # Grid de validación del ELO
│       ├── baseline_v1/                  # Resultados congelados pre-corrección
│       └── models/
│           ├── logreg_baseline.joblib
│           ├── xgboost.joblib / lightgbm.joblib
│           ├── xgboost_calibrated.joblib
│           ├── *_pre2022.joblib          # Pipeline sin leakage para validar WC2022
│           ├── best_model.json           # Modelo seleccionado por validación
│           └── best_params_*.json        # Hiperparámetros de Optuna
├── notebooks/
│   └── 01_eda.ipynb                # Análisis exploratorio
├── reports/
│   ├── figures/                    # SHAP, calibración, EDA
│   ├── analisis_critico.md         # Auditoría que motivó la v2
│   └── plan_correcciones.md        # Plan de implementación de la v2
├── src/
│   ├── data/         # data_loader.py, scraper.py
│   ├── features/     # elo.py, time_decay.py, features.py, derived_stats.py
│   ├── models/       # train.py, evaluate.py
│   ├── simulation/   # tournament.py, simulate.py
│   ├── analysis/     # sensitivity.py, ablation.py, benchmark.py,
│   │                 # compare_runs.py, tune_elo.py
│   └── visualization/  # dashboard.py
├── tests/
└── pyproject.toml / Makefile / requirements.txt
```

---

## Instalación

```bash
git clone https://github.com/daiv05/fifa-world-cup-model
cd fifa-world-cup-model
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -e .
```

---

## Pipeline de ejecución

```bash
python -m src.features.features
python -m src.models.train --trials 100 --n-jobs -1
python -m src.models.evaluate
python -m src.analysis.ablation --seeds 10
python -m src.simulation.simulate --iterations 10000          # usa best_model.json
python -m src.simulation.simulate --iterations 10000 --ensemble 5
python -m src.analysis.sensitivity --iterations 10000
python -m src.analysis.benchmark
streamlit run src/visualization/dashboard.py
```

Para comparar dos corridas (criterio de aceptación de cualquier cambio):

```bash
python -m src.analysis.compare_runs --a-dir data/processed/baseline_v1 --b-dir data/processed
```

---

## Metodología

### Datos
| Fuente | Contenido | Partidos / Equipos |
|--------|-----------|-------------------|
| [martj42/international_results](https://github.com/martj42/international_results) | Resultados históricos 1872-2026 | ~49,000 partidos |
| ↳ `goalscorers.csv` | Goleadores por partido (minuto, penal, autogol) | ~47,600 goles |
| ↳ `former_names.csv` | Mapeo de nombres históricos → actuales | provenance |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | xG por equipo (internacionales) | 109 equipos |
| Transfermarkt (snapshot manual) | Valor de mercado de plantilla | 62 equipos |
| FIFA Ranking (CSV histórico + snapshot 2026-05-30) | Posición por equipo y fecha | ~210 equipos |
| BetMGM (snapshot 2026-06-10) | Odds de título (benchmark externo) | 48 equipos |

### ELO (v2)
ELO propio sobre la historia completa (49k partidos, incl. amistosos con K
reducido), con **multiplicador por margen de victoria** (1.5 si dif=2,
(11+d)/8 si d≥3). La ventaja de local en el expected score está implementada y
parametrizada pero **desactivada por defecto**: el grid de validación
(`tune_elo.py`, train 2010-2018 / val 2019-2020) muestra que el margen mejora
el log-loss en todas las filas pero la ventaja de local no valida
(`data/processed/tune_elo_results.csv`).

### Features (7)
`FEATURE_COLS` se define una sola vez en `features.py`. Tres horizontes:
`ELO_HISTORY_START=None` (ELO sobre toda la historia), `OUTPUT_ROW_START_YEAR=1993`
(primera fila emitida) y `TRAIN_MIN_YEAR=2010` (ventana de modelado).

| Feature | Descripción |
|---------|-------------|
| `elo_diff` | ELO local − visitante (historia completa, margen de victoria) |
| `squad_value_diff` | log(valor_local) − log(valor_visitante) |
| `xg_avg_for` / `xg_avg_against` | xG promedio a favor / en contra (diffs) |
| `ranking_diff` | rank_visitante − rank_local (snapshot 2026 vía merge-asof backward) |
| `penalty_share_diff` | % goles de penal, as-of-date estricto |
| `striker_concentration_diff` | Herfindahl de goleadores sobre **ventana móvil de 4 años**, as-of-date |

> **Features retiradas en v2** (aplicando el criterio de retiro declarado):
> `travel_distance_diff` (83% ceros en train + shift de distribución en
> inferencia, aporte ~nulo), `shootout_winrate_diff` (casi constante tras el
> shrinkage, ablación dentro del ruido) y `late_goal_ratio_diff` (v1, ruido neto).
> El Herfindahl pasó de acumulado-de-por-vida (medía antigüedad del programa
> futbolístico) a ventana de 4 años (mide la estructura del ataque vigente).

**Pesos de entrenamiento (v2):** SOLO decaimiento temporal W(t) = e^(−0.001·Δt),
renormalizado a media 1. El rebalanceo de clases de la v1 se eliminó: deformaba
las probabilidades posteriores (sobrepredicción de empates ~+3.5 pp) y costaba
~0.02 de log-loss. Las probabilidades del modelo ahora reflejan las tasas base
reales (H≈47%, E≈21%, V≈32%).

### División del dataset y selección de modelo

| Conjunto | Filtro | Uso |
|----------|--------|-----|
| Train | `date < 2021-01-01` | Entrenamiento (LogReg, XGBoost, LightGBM) |
| Val-cal | primer 70% de 2021 | Calibración Platt del XGBoost |
| Val-sel | último 30% de 2021 | **Selección del modelo final** (sin sesgo hacia el calibrado) |
| Test | `date ≥ 2022-01-01` | Evaluación final, intocado por toda decisión |

El modelo final se elige por log-loss en val-sel y queda registrado en
`models/best_model.json`; `simulate.py` y `sensitivity.py` lo consumen por
defecto. En la corrida actual el seleccionado es **logreg_baseline**
(en test queda estadísticamente empatado con LightGBM: Δ = +0.0004,
IC95 [−0.007, +0.009]).

### Evaluación (test ≥ 2022, n = 2,208)

| Modelo | Log-Loss | Brier | Δ vs mejor (IC95) | ¿Significativo? |
|--------|:--------:|:-----:|:------------------:|:---:|
| LightGBM | 0.8356 | 0.1634 | — | — |
| LogReg | 0.8360 | 0.1635 | +0.0004 [−0.007, +0.009] | no |
| XGBoost | 0.8587 | 0.1678 | +0.023 [+0.015, +0.031] | sí |
| XGBoost-Cal | 0.8716 | 0.1699 | +0.036 [+0.026, +0.046] | sí |

Referencias: baseline de priors de clase ≈ 1.050; uniforme ln(3) ≈ 1.099;
mejor modelo v1 (con rebalanceo) = 0.8547.

### Simulación Monte Carlo (v2: bracket oficial)
- 10,000 iteraciones del torneo completo sobre el **bracket oficial FIFA 2026**
  (`data/raw/wc2026_bracket.csv`, transcrito del calendario publicado): 8
  ganadores de grupo vs mejores terceros (con pools de procedencia por partido),
  4 ganadores vs segundos, 4 cruces entre segundos. La v1 usaba un bracket
  inventado con cruces 3º-vs-3º que no existen en el reglamento.
- Asignación de terceros a slots por matching exacto que respeta los pools
  oficiales (la tabla FIFA de 495 combinaciones es una elección particular entre
  los matchings válidos).
- Desempates de grupo: puntos → DG → GF → enfrentamiento directo → **sorteo con
  el RNG** (la v1 dejaba el orden de inserción, un sesgo determinista). El fair
  play no es simulable y se omite.
- **Localía condicionada a la sede real**: un anfitrión solo recibe ventaja de
  local si el partido se juega en su país (México en una sede de EE.UU. va por
  probabilidades simétricas).
- Marcadores con **Poisson condicionado al outcome** sorteado (distribución
  conjunta truncada y renormalizada), en lugar de la reconciliación ad-hoc de la
  v1 que distorsionaba DG/GF justo donde deciden los desempates.
- Reproducible para cualquier número de workers (`SeedSequence.spawn`).

### Validación histórica (WC 2022) — sin leakage (v2)
El pipeline `_pre2022` se entrena solo con `date < 2022` **y sin las features
anacrónicas** (xG y squad_value, snapshots de ~2026): ni el modelo ni su vector
de entrada ven información posterior al cutoff. Resultado out-of-time honesto
sobre los partidos de 2022: **log-loss 0.985, accuracy 54.4%**
(`wc2022_validation.csv`). La cifra es peor que la del test contaminado —
exactamente lo que cabe esperar al retirar el leakage, y la razón de reportarla.

### Estudio de ablación (v2: multi-semilla)
`ablation.py --seeds 10` reentrena cada configuración con 10 semillas y reporta
media ± std más un bootstrap pareado del delta frente al modelo completo
(columna `significant`). Solo las filas significativas soportan conclusiones.

### Benchmark externo
`benchmark.py` compara P(campeón) contra las probabilidades implícitas
des-vigorizadas del mercado (BetMGM, 2026-06-10). El modelo diverge del mercado
en los extremos: sobrepondera al líder de ELO (España 27.5% vs 14.9%) e
infrapondera a Brasil/Portugal (ratio ≈ 0.33). Distancia de variación total
≈ 0.27. La divergencia se discute en el paper; el mercado es el baseline
estándar de esta literatura.

---

## Tests

```bash
python -m pytest tests/ -v   # 57+ tests
```

---

## Limitaciones conocidas
- Los equipos debutantes (Uzbekistán, Curaçao, etc.) tienen poco historial; su ELO parte de 1500 y converge lento.
- **Anacronismo de features estáticas (xG y squad_value)** en el pipeline principal: snapshots sin fecha aplicados a todos los partidos, incluido el test. Las métricas del test principal son por tanto una **cota optimista**; la cifra de referencia limpia es la validación WC2022 (pipeline `_pre2022`, que las excluye). Existe serie histórica de valores de Transfermarkt: versionarla es trabajo futuro.
- El modelo deposita casi toda la señal en `elo_diff` y produce favoritos más extremos que el mercado de apuestas (ver benchmark). Sin shrinkage hacia el consenso.
- Las lesiones no se modelan estructuralmente; `sensitivity.py` aplica un escenario agregado (squad −30%, xg_for −10%, ELO −25 pts) sobre el top-5.
- El xG de StatsBomb cubre principalmente torneos UEFA/FIFA; 13 de 48 equipos usan la media global 1.2.
- El ranking FIFA histórico termina en 2024-06; el periodo hasta 2026-05 solo tiene el snapshot puntual del 2026-05-30.
- La asignación de terceros usa un matching válido respecto a los pools oficiales, no la tabla exacta FIFA de 495 combinaciones (no publicada en forma compacta).

---

## Hecho por
[David Deras](https://github.com/daiv05)
