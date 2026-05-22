# Requerimientos Técnicos para el Proyecto de Predicción del Mundial 2026

### Fase 1: Arquitectura de Datos e Ingeniería de Características (25 puntos)

La rúbrica exige múltiples fuentes de datos y al menos 6 características justificadas. El enfoque debe ser construir un pipeline de extracción automatizado.

* **Fuentes de Datos (El Pipeline):**
1. **Base Histórica:** Utiliza el dataset de *International Football Results (1872-2024)* de GitHub. Te dará la volumetría necesaria.
2. **Datos Recientes y Estadísticas:** Implementa scripts de scraping (con BeautifulSoup o Selenium) para *FBref* y extraer xG (Goles Esperados) y posesión. Usa *Transfermarkt* para obtener el valor de mercado total de cada selección (un proxy excelente de la calidad individual).

* **Ingeniería de Características (Features Clave):**
1. **Diferencial de ELO:** Calcula el sistema de puntuación ELO dinámico antes de cada partido. Es mucho más predictivo que el Ranking FIFA estático.
2. **Calidad de Plantilla:** Sumatoria del valor de mercado de los 26 convocados (normalizado) y porcentaje de jugadores en las "Top 5 Ligas".
3. **Métricas Tácticas (Últimos 2 años):** Promedio de xG a favor y en contra.
4. **Fatiga de Viaje / Localía:** Distancia geográfica al país anfitrión o ventaja de jugar en el continente americano.
5. **Degradación Temporal (Time Decay):** Esta es una exigencia crítica de la rúbrica. Los partidos históricos deben pesar menos en el entrenamiento. Define una función de decaimiento exponencial donde el peso $W$ de una muestra está dado por:

$$W(t) = e^{-\lambda \Delta t}$$

Donde $\Delta t$ es el tiempo transcurrido desde el partido y $\lambda$ es la tasa de decaimiento a calibrar.

### Fase 2: Modelado Matemático (15 puntos)

El problema de negocio se enmarca como clasificación multiclase a nivel de partido: `[Gana Equipo A, Empate, Gana Equipo B]`.

* **Selección de Modelos:** La rúbrica pide comparar al menos dos.
    * **XGBoost (`XGBClassifier`):** Configurado con el objetivo `multi:softprob` para que devuelva un vector de probabilidades continuas, no una clase discreta. Maneja muy bien las relaciones no lineales (ej. un ELO muy alto vs un ELO muy bajo).
    * **Regresión Logística Multinomial:** Servirá como tu modelo base (baseline).

* **Métricas de Evaluación:** Abandona el *Accuracy*. Las métricas a optimizar y reportar deben evaluar la calibración de las probabilidades:
    * **Log-Loss (Entropía Cruzada):** Penaliza severamente predicciones con alta confianza que resultan ser incorrectas.
    * **Brier Score:** Mide la diferencia cuadrática media entre la probabilidad predicha y el resultado real.

* **Estrategia de Entrenamiento:** Utiliza *GridSearchCV* o *RandomizedSearchCV* para afinar hiperparámetros. Adopta una metodología iterativa (como CRISP-DM) para pivotar rápidamente entre el ajuste del modelo y la adición de nuevas variables en el EDA inicial.

### Fase 3: El Motor de Simulación de Monte Carlo (35 puntos)

Esta es la fase computacionalmente más intensiva. Simular 104 partidos del nuevo formato de 48 equipos (12 grupos de 4; avanzan los 2 primeros y los 8 mejores terceros), multiplicado por $\ge 10,000$ iteraciones, equivale a más de 1.04 millones de simulaciones de partidos.

* **Arquitectura de Alta Precisión y Rendimiento:**
    * **Vectorización:** Evita los bucles `for` anidados en Python estándar. Utiliza operaciones matriciales con NumPy para resolver los 10,000 torneos en paralelo.
    * **Delegación de Procesamiento:** Para manejar esta carga computacional sin saturar la VRAM ni provocar latencias extremas en la ejecución de las pruebas, puedes mantener el modelo predictivo en Python (scikit-learn/XGBoost) y delegar la lógica bruta de la simulación iterativa a un microservicio o un binario precompilado (por ejemplo, desarrollando el motor central de Monte Carlo en Rust e invocándolo desde Python mediante FFI o subprocesos). Esto acelerará drásticamente los tiempos de prueba.

* **Reglas de Sorteo:** Implementa la lógica exacta del Mundial 2026. Usa las probabilidades devueltas por XGBoost para hacer un `np.random.choice` ponderado para cada partido.
* **Resultados:** Acumula las victorias totales del torneo para cada equipo. Extrae el Top 5 calculando los intervalos de confianza (percentiles 5% y 95%) del número de veces que llegaron a la final o se coronaron campeones.

### Fase 4: Flujo de Trabajo y Entregables (25 puntos)

Para estructurar este nivel de complejidad técnica, el proyecto se divide naturalmente en tres frentes paralelos, ideal para un equipo de trabajo multidisciplinario de tres miembros:

1. **Ingeniería de Datos (Pipeline y EDA):** Encargado del scraping, limpieza de datos, cálculo del ELO histórico y la redacción del análisis exploratorio.
2. **Modelado de Machine Learning:** Encargado del entrenamiento de XGBoost y Regresión Logística, ajuste de hiperparámetros, aplicación del *Time Decay* y validación histórica prediciendo el Mundial 2022.
3. **Ingeniería de Simulación:** Encargado del desarrollo del algoritmo de Monte Carlo, la lógica estructural de los 48 equipos, el manejo de desempates de la fase de grupos y la optimización del rendimiento del script final.

* **El Repositorio:** Mantén una arquitectura modular desde el día uno. Crea carpetas separadas para `/data` (crudos y procesados), `/notebooks` (EDA), y `/src` (código fuente modularizado: `data_loader.py`, `features.py`, `train.py`, `simulate.py`). Incluye un `requirements.txt` impecable.
* **El Documento IEEE:** Redacten el análisis como un *paper* de investigación. Dediquen una sección exclusiva a detallar cómo la falta de datos históricos de los debutantes o las lesiones imprevistas (el factor caos) actúan como limitantes del modelo.

---