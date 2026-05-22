# Plan de Desarrollo Integral: El Oráculo del Balón (Mundial 2026)

Este documento detalla el cronograma integral, técnico y operativo para resolver el desafío de predicción del Mundial 2026\. El plan está estructurado para ejecutarse de manera altamente modular, distribuyendo la carga de forma balanceada para los tres miembros del equipo en frentes paralelos (Datos, Modelado Predictivo y Simulación). Se ha diseñado priorizando un entorno de desarrollo **Windows-first**, conservando una estructura modular que facilite la posterior ejecución multiplataforma.

## Fase 1: Arquitectura de Datos e Ingeniería de Características

**Objetivo:** Construir un dataset robusto y limpio a nivel de "partido", combinando fuentes históricas, APIs y web scraping para generar variables de alto poder predictivo.

| Tarea | Descripción Técnica | Fuentes, Páginas y Programas   |
| :---- | :---- | :---- |
| 1.1 Pipeline Base y Limpieza | Ingesta de resultados históricos (1872-2024). Tratamiento de valores nulos, estandarización de nombres de países y filtro de competiciones relevantes. | **Fuentes (Datasets):** Kaggle (*WC2026 Match Probability Baseline Dataset*, *FIFA World Cup Matches 1974-2022*), GitHub (*martj42/international\_results*, *openfootball/worldcup*). **Programas:** Python, Pandas, Jupyter Notebook. |
| 1.2 Scraping y APIs (Features) | Extracción de estadísticas avanzadas: valor de mercado de las plantillas y métricas de rendimiento avanzadas como xG (Goles Esperados) o progresiones de pase. | **Fuentes:** StatsBomb Open Data (xG y métricas tácticas gratuitas, ~4,000 partidos internacionales). **Páginas (Scraping):** FBref.com (métricas tácticas avanzadas), Transfermarkt (valor económico del equipo). **Programas:** `soccerdata` (wrapper Python para FBref y Transfermarkt — gestiona rate-limits y parsing automáticamente), `statsbombpy` (cliente oficial de StatsBomb). |
| 1.3 Ingeniería de Variables | Cálculo iterativo del Rating ELO histórico por equipo. Implementación de una función de decaimiento exponencial (Time Decay) para dar más peso a partidos recientes. | **Páginas de Referencia:** EloRatings.net (para alinear fórmulas matemáticas del sistema ELO). **Programas:** Python, NumPy (operaciones vectoriales), SciPy. |

## Fase 2: Modelado Predictivo (Clasificación Multiclase)

**Objetivo:** Entrenar algoritmos de clasificación a nivel partido (Gana, Pierde, Empata) que devuelvan probabilidades altamente calibradas sin sobreajustar (overfitting).

| Tarea | Descripción Técnica | Fuentes, Páginas y Programas   |
| :---- | :---- | :---- |
| 2.1 Análisis Exploratorio (EDA) | Análisis de correlación de las nuevas variables. Verificación de distribuciones y justificación estadística de las 6+ features elegidas. | **Programas:** Jupyter Notebook, Pandas. **Librerías Visuales:** Seaborn, Matplotlib, ydata-profiling (para reportes HTML automáticos del dataset). |
| 2.2 Entrenamiento de Modelos | Entrenamiento cruzado de tres modelos: Regresión Logística (baseline), XGBoost (objetivo `multi:softprob`) y LightGBM (más rápido que XGBoost, ideal para iteraciones de hiperparámetros). Ajuste de desbalance de clases (victorias ~45%, empates ~25%) con `imbalanced-learn` (SMOTE o class weights). | **Programas/Frameworks:** Scikit-Learn (LogisticRegression), XGBoost (XGBClassifier), LightGBM (LGBMClassifier), `imbalanced-learn`. **Herramientas:** `Optuna` como optimizador principal de hiperparámetros (3x más eficiente que GridSearchCV con búsqueda bayesiana). |
| 2.3 Evaluación y Calibración | Evaluación rigurosa de las probabilidades devueltas por los modelos. Calibración explícita de probabilidades con `CalibratedClassifierCV` (Platt scaling o regresión isotónica). Análisis de interpretabilidad de features con SHAP (valores SHAP por variable). Prueba de validación utilizando datos aislados del Mundial 2022. | **Programas (Métricas):** Scikit-Learn (`log_loss`, `brier_score_loss`, `CalibratedClassifierCV`), `shap` (para interpretabilidad — directamente citable en el paper IEEE). |

## Fase 3: Motor de Simulación de Monte Carlo

**Objetivo:** Desarrollar la lógica del torneo y procesar las $\\ge 10,000$ iteraciones con alta eficiencia computacional, gestionando cuidadosamente las limitaciones de RAM y CPU físicas.

| Tarea | Descripción Técnica | Fuentes, Páginas y Programas   |
| :---- | :---- | :---- |
| 3.1 Lógica del Torneo 2026 | Programar el reglamento exacto: 12 grupos de 4 equipos, pase de los 8 mejores terceros, llaves eliminatorias de dieciseisavos hasta la gran final (104 partidos por simulación). | **Fuentes:** Documentación oficial y reglamento de la FIFA sobre criterios de desempate en fase de grupos. **Programas:** Python (Implementación de lógica orientada a objetos). |
| 3.2 Monte Carlo Optimizado | Ejecutar el bucle de miles de iteraciones del torneo. La carga real (10,000 × 104 partidos = ~1,040,000 simulaciones) es manejable en Python con la estrategia correcta de optimización, sin necesidad de cambiar de lenguaje. **Estrategia por capas:** (1) NumPy vectorizado con batch simulation como base; (2) `numba` con `@njit` para JIT compilation (50-100x sobre Python puro) si el tiempo de ejecución supera los 30 segundos; (3) Rust + PyO3 solo si las pruebas de rendimiento demuestran que las capas anteriores son insuficientes. | **Programas:** NumPy (vectorización), `numba` (JIT compilation con `@njit`), `joblib` (paralelización de procesos si se requiere). |

## Fase 4: Consolidación y Entregables (Reporte IEEE)

**Objetivo:** Empaquetar el trabajo bajo los estrictos lineamientos de la rúbrica para asegurar la puntuación máxima de 100/100, con énfasis en la reproducibilidad técnica.

| Tarea | Descripción Técnica | Fuentes, Páginas y Programas   |
| :---- | :---- | :---- |
| 4.1 Redacción del Paper (IEEE) | Estructurar las secciones requeridas. Detallar las limitaciones del modelo (ej. factor caos por lesiones, falta de historial sólido en debutantes como Uzbekistán). | **Fuentes:** Directrices de autor y plantilla oficial de la *IEEE*. **Programas:** LaTeX (mediante plataformas colaborativas como Overleaf) o Microsoft Word. Herramientas como Draw.io o Lucidchart (para esquemas de arquitectura de datos). |
| 4.2 Repositorio y Reproducibilidad | Limpieza del código. Generar archivos de entorno virtual, documentar rutas y asegurar un README explicativo. Tests de validación del pipeline de datos. Dashboard interactivo para presentación de resultados. | **Páginas:** GitHub. **Programas/Herramientas:** Git, VS Code. Archivos `requirements.txt` para dependencias. `python-dotenv` para gestión segura de API keys. `pytest` + `Great Expectations` para tests y validación del pipeline de datos. `Streamlit` para dashboard interactivo con resultados del torneo simulado. |

## Estado de Implementación (Mayo 2026)

| Ítem | Estado | Notas |
|:-----|:------:|:------|
| Pipeline base (resultados 1872-2024) | ✅ | `load_international_results()` — 12,157 partidos |
| StatsBomb xG | ✅ | 109 equipos con xG real |
| Valores de plantilla | ✅ | Hardcoded (60 equipos); csv override soportado |
| FBref scraping | ⚠️ | `get_fbref_stats()` funciona pero **no se consume** en features.py |
| Kaggle datasets (WC 1974-2022, FIFA ranking) | ⚠️ | Loaders implementados pero **no integrados** al pipeline |
| ELO histórico | ✅ | K-factor variable por torneo |
| Time Decay (λ=0.002) | ✅ | Half-life ≈ 1 año |
| Feature: squad_value_diff | ✅ | log(home\_val) − log(away\_val) |
| Feature: % jugadores Top-5 ligas | ❌ | **Pendiente** — mencionado en REQUERIMENT.md pero no implementado |
| Regresión Logística (baseline) | ✅ | |
| XGBoost (multi:softprob) | ✅ | Guardado en `models/xgboost.joblib` |
| LightGBM | ✅ | Mejor log-loss en test: 0.6621 |
| Optuna (sustituye GridSearchCV) | ✅ | Más eficiente que GridSearchCV |
| CalibratedClassifierCV | ✅ | cv=5 (post sklearn 1.5+) |
| imbalanced-learn / SMOTE | ❌ | **Pendiente** — mencionado en plan, no implementado |
| Validación WC2022 | ✅ | Accuracy 66.6% sobre partidos 2022 |
| SHAP analysis | ✅ | `shap_summary.png` generado |
| Monte Carlo 10,000 iter. | ✅ | Precomputed cache (O(1) por partido) |
| Intervalos de confianza | ✅ | Bootstrap binomial corregido |
| Streamlit dashboard | ✅ | 3 páginas; requiere `simulation_results.csv` |
| pytest (23 tests) | ✅ | 23/23 passing |
| Great Expectations | ❌ | **Pendiente** |
| Notebooks EDA / Modeling | ❌ | **Pendiente** |
| Paper IEEE | ❌ | **Pendiente** — redactar en Overleaf |

## Cronograma

* **Bloque 1:** Configuración del entorno en Windows. Extracción masiva de datos mediante las APIs (API-Football) y Scraping (FBref, Transfermarkt). Entrega del pipeline de datos base.  
* **Bloque 2:** Programación de variables complejas en NumPy/Pandas (ELO, Time Decay). Análisis Exploratorio de Datos (EDA) inicial en Jupyter.  
* **Bloque 3:** Entrenamiento, validación cruzada (Optuna) y evaluación de XGBoost. Desarrollo de las reglas del torneo en código.  
* **Bloque 4:** Ejecución de la simulación de Monte Carlo. Recopilación de resultados, limpieza del código en GitHub y redacción final del reporte técnico formato IEEE en Overleaf.