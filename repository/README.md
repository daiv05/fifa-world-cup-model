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
repository/
├── data/
│   ├── raw/                        # Datos crudos (no versionados)
│   │   ├── international_results.csv   # Auto-descargado de GitHub
│   │   ├── statsbomb_xg_by_team.csv    # Auto-generado por scraper.py
│   │   └── squad_values.csv            # Auto-generado (valores Transfermarkt)
│   └── processed/
│       ├── features.csv                # Dataset de entrenamiento (12,157 partidos)
│       ├── team_features.csv           # Features por equipo para simulación
│       ├── simulation_results.csv      # Output del Monte Carlo
│       ├── model_evaluation.csv        # Métricas log-loss / Brier por modelo
│       └── models/
│           ├── logreg_baseline.joblib
│           ├── xgboost.joblib
│           ├── lightgbm.joblib
│           └── xgboost_calibrated.joblib
├── notebooks/
│   └── 01_eda.ipynb                # (pendiente) Análisis exploratorio
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Carga y limpieza de datos históricos
│   │   └── scraper.py              # StatsBomb xG + valores de plantilla
│   ├── features/
│   │   ├── elo.py                  # Sistema ELO dinámico con K-factor variable
│   │   ├── time_decay.py           # Decaimiento exponencial W(t)=e^(-λΔt)
│   │   └── features.py             # Pipeline completo de features
│   ├── models/
│   │   ├── train.py                # Entrena LogReg, XGBoost, LightGBM + Optuna
│   │   └── evaluate.py             # Log-loss, Brier, SHAP, validación WC2022
│   ├── simulation/
│   │   ├── tournament.py           # Lógica del torneo: grupos, terceros, llaves
│   │   └── simulate.py             # Motor Monte Carlo con precomputed cache
│   └── visualization/
│       └── dashboard.py            # Dashboard Streamlit interactivo
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_simulation.py
├── conftest.py
└── requirements.txt
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url>
cd fifa-world-cup-model

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Instalar dependencias
pip install -r repository/requirements.txt
```

---

## Pipeline de ejecución

Ejecutar cada paso desde la raíz del proyecto (`fifa-world-cup-model/`):

### Paso 1 — Construir features
```bash
python -m repository.src.features.features
```
Genera `data/processed/features.csv` (12,157 partidos, ~2 min por geocoding).

### Paso 2 — Entrenar modelos
```bash
python -m repository.src.models.train
```
Guarda 4 modelos en `data/processed/models/`.

### Paso 3 — Evaluar modelos
```bash
python -m repository.src.models.evaluate
```
Imprime tabla log-loss / Brier y guarda `model_evaluation.csv` + `shap_summary.png`.

### Paso 4 — Simulación Monte Carlo
```bash
python -m repository.src.simulation.simulate --iterations 10000 --model xgboost_calibrated
```
Guarda `data/processed/simulation_results.csv`.

### Paso 5 — Dashboard
```bash
streamlit run repository/src/visualization/dashboard.py
```
Abre el dashboard en `http://localhost:8501`.

---

## Tests

```bash
python -m pytest repository/tests/ -v
```

---

## Metodología

### Datos
| Fuente | Contenido | Partidos / Equipos |
|--------|-----------|-------------------|
| [martj42/international_results](https://github.com/martj42/international_results) | Resultados históricos 1872–2024 | ~47,000 partidos |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | xG por equipo (internacionales) | 109 equipos |
| Transfermarkt (hardcoded) | Valor de mercado de plantilla | 60 equipos |

### Features (6)
| Feature | Descripción | Justificación |
|---------|-------------|---------------|
| `elo_diff` | ELO local − ELO visitante | Métrica dinámica, superior al ranking FIFA estático |
| `squad_value_diff` | log(valor_local) − log(valor_visitante) | Proxy de calidad individual de la plantilla |
| `xg_avg_for` | xG promedio a favor: local − visitante | Eficiencia ofensiva reciente |
| `xg_avg_against` | xG promedio en contra: local − visitante | Solidez defensiva reciente |
| `travel_distance_home` | Distancia del equipo local a las sedes (km) | Fatiga de viaje / ventaja de localía geográfica |
| `travel_distance_away` | Distancia del equipo visitante a las sedes (km) | Ídem para el visitante |

**Decaimiento temporal:** cada partido tiene peso W(t) = e^(−0.002 · Δt) multiplicado por peso de clase balanceado (H≈49%, E≈21%, V≈30%).

### Modelos
| Modelo | Log-Loss (test) | Brier Score | WC2022 Accuracy |
|--------|:---------------:|:-----------:|:---------------:|
| LightGBM | 0.6843 | 0.1309 | — |
| XGBoost | 0.7296 | 0.1380 | — |
| XGBoost (calibrado) | — | — | **69.2%** |
| Regresión Logística | 0.8915 | 0.1741 | — |

Hiperparámetros optimizados con **Optuna** (búsqueda bayesiana, 100 trials).  
Pesos de entrenamiento: `w = class_weight_balanced × time_decay` — aborda el desbalance (H:49%, D:21%, V:30%) sin sacrificar información temporal.

### Simulación Monte Carlo
- **10,000 iteraciones** del torneo completo (104 partidos c/u).
- Optimización: todas las probabilidades (48×47 = 2,256 matchups) se precomputan en un único batch antes del loop → lookup O(1) por partido.
- Desempate de grupos: puntos → diferencia de goles → goles a favor (FIFA 2026).
- Fase eliminatoria: en caso de empate, penalty shootout (50/50).

### Validación histórica (WC 2022)
Entrenando únicamente con datos anteriores a 2022:
- **Accuracy: 66.6%** sobre partidos de 2022
- **Log-Loss: 0.7952**

---

## Limitaciones conocidas
- Los equipos debutantes (Uzbekistán, Curaçao, etc.) tienen muy pocos partidos históricos → ELO inicial por defecto (1500).
- Las lesiones de última hora no están modeladas.
- El xG de StatsBomb cubre principalmente torneos UEFA/FIFA; equipos de otras confederaciones usan valores por defecto.
- `travel_distance = −1.0` para equipos cuya geocodificación falla (se pasa al modelo tal cual, consistente con el entrenamiento).

---

## Equipo
Proyecto desarrollado como trabajo final de curso · 2026  
Datos, Modelado y Simulación Monte Carlo.
