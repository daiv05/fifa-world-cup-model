# Análisis crítico del proyecto «Predicción Probabilística del Mundial FIFA 2026»

> **ESTADO (2026-06-10): CORREGIDO.** Las recomendaciones de este análisis se
> implementaron según `plan_correcciones.md` (fases F0–F6). Resultado verificado
> con bootstrap pareado: log-loss de test 0.8547 → 0.8360 (Δ = −0.019,
> IC95 [−0.029, −0.008]). Los resultados pre-corrección quedan congelados en
> `data/processed/baseline_v1/`. Este documento se conserva como registro de la
> auditoría original; los números que cita corresponden a la v1.

**Alcance:** revisión académica del código (`src/`), los datos procesados (`data/processed/`), el README y el paper (`reports/paper/paper.tex`). Todos los hallazgos cuantitativos fueron **verificados empíricamente** ejecutando código contra el entorno del proyecto (`.venv`, 44 tests pasan). Fecha de revisión: 2026-06-10.

---

## 1. Resumen y fortalezas

El proyecto es notablemente superior al promedio de trabajos de curso: split temporal estricto train/val/test, calibración Platt sobre hold-out, features de eventos calculadas as-of-date con `merge_asof(allow_exact_matches=False)` (verificado: sin leakage del propio partido), simulación Monte Carlo reproducible independiente del paralelismo, estudio de ablación, análisis de sensibilidad, suite de tests, y —raro de ver— una sección de limitaciones honesta tanto en README como en el paper. La nota metodológica sobre la escala del Brier (promedio one-vs-rest vs. suma multiclase) es correcta y bienvenida.

Dicho esto, la auditoría encontró **un error metodológico central que invalida la tabla de evaluación tal como se interpreta**, **un error estructural en el bracket de la simulación que contradice una afirmación literal del paper**, y varias discrepancias y sobreinterpretaciones de menor rango. Se detallan en orden de severidad.

---

## 2. Hallazgo crítico A — El rebalanceo de clases contradice el objetivo del modelo y degrada todos los resultados reportados

**Severidad: alta. Afecta: tabla de evaluación, selección de modelo, hiperparámetros, simulación.**

El proyecto declara que su objetivo son *probabilidades calibradas* (se evalúa con Log-Loss/Brier, se calibra con Platt). Sin embargo, todos los modelos se entrenan con `class_weight="balanced"` (`compute_combined_weights`, [train.py:43-61](../src/models/train.py)). Rebalancear clases deforma deliberadamente las probabilidades posteriores alejándolas de las tasas base reales (los empates, ~21 %, se sobreponderan ~1.6×). Es una técnica para problemas de *clasificación dura* con clases raras; para predicción probabilística es directamente contraproducente.

Verificación empírica (mismo split, mismos `best_params_xgboost.json`, mismo test ≥2022, n=2 208):

| Configuración | Log-Loss test |
|---|---|
| **XGBoost best_params, SIN pesos, sin calibrar** | **0.8348** |
| XGBoost best_params, sin pesos, calibrado | 0.8499 |
| LogReg 9 features, sin pesos | 0.8407 |
| **LogReg con SOLO `elo_diff`, sin pesos** | **0.8417** |
| XGBoost-Cal (mejor modelo reportado) | 0.8547 |
| XGBoost reportado (con pesos) | 0.8657 |
| LogReg reportado (con pesos) | 0.8832 |

Consecuencias:

1. **Una regresión logística de una sola variable (`elo_diff`), sin pesos ni tuning, supera al mejor modelo del paper** (0.8417 < 0.8547). El sistema completo de 9 features + Optuna 100 trials + calibración no logra batir al baseline más trivial posible una vez que se elimina el rebalanceo.
2. La mejora atribuida a la calibración Platt (0.866→0.855, presentada en §VI del paper como corrección de "sobreconfianza") es en realidad **reparación parcial de un daño autoinfligido por los pesos**: el modelo sin pesos no necesita esa reparación (0.8348 sin calibrar) y la calibración incluso lo empeora (0.8499).
3. Incluso tras calibrar, el sesgo persiste: P(empate) media predicha del XGB-Cal en test = 0.246 vs. frecuencia real 0.212; P(local) 0.442 vs. 0.466. La figura de calibración del paper muestra el síntoma ("Draw sub-calibrada") pero el diagnóstico ofrecido (baja correlación features-empate) omite la causa principal.
4. Optuna optimiza `neg_log_loss` en CV **con los pesos dentro del fit**, de modo que los hiperparámetros "óptimos" lo son para el objetivo distorsionado.
5. La simulación Monte Carlo consume `xgboost_calibrated`, así que las probabilidades de campeonato heredan la distorsión residual (empates inflados en fase de grupos alteran puntos esperados y clasificación de terceros).

El ranking *interno* de los cuatro modelos reportados sí es estadísticamente significativo (bootstrap pareado, 2 000 réplicas: ΔLL XGB-Cal vs LogReg = −0.028, IC95 [−0.041, −0.015]), pero los cuatro están dominados por sus variantes sin pesos. La conclusión del paper («XGBoost calibrado es el mejor modelo») es un artefacto del esquema de ponderación, no una propiedad de los modelos.

**Nota:** el `time_decay` como peso muestral es defendible; el problema es específicamente el componente `balanced`. Lo correcto sería entrenar sin rebalanceo (o re-ponderar solo el decay) y dejar que la calibración corrija lo que quede.

---

## 3. Hallazgo crítico B — El bracket del Round of 32 no es el reglamento FIFA 2026

**Severidad: alta. Afecta: todas las probabilidades de progresión y campeonato; contradice una contribución declarada del paper.**

El paper afirma (§I y §V) que el motor «implementa **fielmente** el reglamento 2026». El bracket implementado ([tournament.py:36-41](../src/simulation/tournament.py)) es:

```python
KNOCKOUT_BRACKET_ORDER = [
    ("A1","B2"), ("C1","D2"), ... 12 cruces 1º-vs-2º ...,
    ("3rd_1","3rd_2"), ("3rd_3","3rd_4"), ("3rd_5","3rd_6"), ("3rd_7","3rd_8"),
]
```

Es decir: 12 partidos *ganador vs. segundo* y 4 partidos *tercero vs. tercero*. En la asignación oficial FIFA del Mundial 2026, **los 8 mejores terceros se emparejan contra ganadores de grupo** (8 de los 12 ganadores juegan contra terceros), 4 ganadores juegan contra segundos y los 8 segundos restantes se cruzan entre sí; no existe ningún partido tercero-vs-tercero, y la asignación de terceros a llaves depende de la combinación de grupos de procedencia.

Distorsiones inducidas:

- **Exactamente 4 terceros llegan siempre a octavos** (en la realidad pueden ser entre 0 y 8). Esto infla sistemáticamente la progresión de equipos medianos/débiles que clasifican terceros.
- Los ganadores de grupo enfrentan rivales más duros (segundos en vez de terceros) en R32, **penalizando a los favoritos** — el efecto se propaga a P(campeón).
- La estructura de cuadrantes (qué semifinal toca a quién) tampoco corresponde al calendario oficial, lo que afecta a las probabilidades condicionales de cruces tempranos entre favoritos.

Adicionalmente, el desempate de grupos implementa solo puntos→DG→GF ([tournament.py:44-45](../src/simulation/tournament.py)); el reglamento FIFA continúa con enfrentamiento directo, fair play y sorteo. Con empate total, `sort_values` deja el orden de inserción (determinista, no aleatorio), introduciendo un micro-sesgo sistemático por orden de listado en el grupo en lugar del sorteo que prescribe FIFA.

---

## 4. Hallazgo C — El anacronismo de xG/squad_value contamina las *métricas*, no solo el entrenamiento

**Severidad: media-alta. Afecta: tabla de evaluación, ablación, y la afirmación «validación WC2022 sin data leakage».**

El proyecto documenta con honestidad que xG y `squad_value` son snapshots de ~2026 aplicados a toda la historia. Pero el tratamiento argumental (time decay, recorte 2010, diffs relativos) solo mitiga el efecto sobre el *entrenamiento*. Lo que no se reconoce es que:

1. **El conjunto de test (2022–2026) también recibe esos snapshots.** Las métricas reportadas (LL 0.8547) evalúan un modelo que, para predecir un partido de 2022, dispone del valor de plantilla y el xG observados *después* de ese partido. El "aporte" de las features estáticas medido en la ablación (+0.0041 al quitarlas) es indistinguible de leakage medido como señal.
2. **La «validación histórica WC2022 sin data leakage» (§VI del paper, README) es falsa en sentido estricto:** el *modelo* `xgboost_pre2022` se entrena con `date < 2022`, pero sus *features* de entrada incluyen los snapshots 2026. Solo el target está libre de contaminación; el vector de entrada no. La validación honesta requeriría reconstruir las features con información disponible a 2021 (squad values de Transfermarkt sí existen históricamente, contra lo que afirma la limitación 3 del paper — Transfermarkt publica valores con fecha y hay archivos/snapshots históricos).

La dirección del sesgo favorece al modelo (las features "saben" qué equipos terminaron siendo buenos en 2026), así que las métricas reportadas son cotas optimistas.

---

## 5. Hallazgo D — Las conclusiones de la ablación no están soportadas estadísticamente

**Severidad: media.**

La tabla de ablación se basa en **una sola corrida** por configuración. El propio texto admite que deltas ≤ |0.0008| son "ruido de una sola corrida", pero luego presenta +0.0040 (xG) y +0.0035 (derivadas) como contribuciones reales y concluye en el paper que «el grupo de mayor aporte son las features estáticas». Mi bootstrap pareado sobre el test (n=2 208) da IC95 de ±0.008–0.013 para diferencias de log-loss entre modelos *ya entrenados*; a eso hay que sumar la varianza de reentrenamiento (semilla, submuestreo de XGBoost). Deltas de 0.004 están dentro del ruido conjunto. Las filas de la ablación son descriptivas, no evidencia; harían falta múltiples semillas y un test pareado por configuración.

Esto importa porque el "hallazgo más relevante" del paper (§VIII: `striker_concentration_diff` como segunda feature más informativa) descansa sobre SHAP + esta ablación. Ver Hallazgo E.

---

## 6. Hallazgo E — Sobreinterpretación de `striker_concentration_diff`

**Severidad: media (interpretativa).**

El paper interpreta el Herfindahl de goleadores como «estructura del ataque, eje ortogonal al nivel del equipo». Tres objeciones:

1. **Es un acumulado de por vida** (`_cumulative_herfindahl` suma goles desde el inicio del dataset de goleadores), no una ventana reciente. Para selecciones con 100+ años de historia el índice es una variable casi estática que mide *antigüedad y profundidad histórica del programa futbolístico*, no la estructura del ataque actual. Un debutante con 30 goles históricos repartidos entre 6 jugadores tendrá H alto por pura varianza muestral, no por "dependencia de una figura".
2. La propia matriz de correlación del paper muestra r≈−0.37 con la fuerza del equipo: la feature es en parte un **proxy redundante de fuerza/tradición**, y SHAP reparte crédito de forma inestable entre features colineales (elo, ranking, concentración). Atribuirle un "eje ortogonal" contradice la evidencia de colinealidad presentada dos secciones antes.
3. El soporte de ablación para "derivadas" (+0.0035) no es estadísticamente distinguible de cero (Hallazgo D).

La afirmación causal-estructural debería rebajarse a hipótesis.

---

## 7. Hallazgo F — La incertidumbre comunicada es solo ruido Monte Carlo

**Severidad: media (comunicación de resultados).**

Los IC Clopper–Pearson del README/paper (España 23.72 % [22.89, 24.57]) cuantifican únicamente el error de muestreo de 10 000 iteraciones — un error reducible arbitrariamente corriendo más simulaciones. **No incluyen** incertidumbre de parámetros del modelo, de los hiperparámetros, del ELO, de las features imputadas (13/48 equipos con xG por defecto), ni del propio bracket. Presentar un intervalo de ±0.8 pp sugiere una precisión que el sistema no tiene; la incertidumbre epistémica real es un orden de magnitud mayor. Lo académicamente correcto sería etiquetarlos como "IC de simulación" y/o propagar incertidumbre del modelo (p. ej., bootstrap de entrenamiento o ensembles).

Relacionado: España con 23.7 % está muy por encima del consenso de mercados de apuestas y de modelos públicos comparables (típicamente ~15–17 % para el favorito). No es un error per se, pero un paper que reporta un favorito tan destacado debería contrastar contra un benchmark externo (odds implícitas, que además son el baseline estándar en esta literatura — Dixon-Coles mismo se evalúa contra el mercado) o explicar la divergencia. La causa probable es mecánica: ELO de España (2113) 65 puntos sobre el segundo, sin shrinkage, amplificado en 7 rondas.

---

## 8. Hallazgos menores y discrepancias código-documentación

1. **Localía de anfitriones sin verificar sede.** README: los anfitriones reciben localía «cuando juegan en su país». El código ([tournament.py:64-78](../src/simulation/tournament.py)) la aplica en *cualquier* partido del anfitrión, sin comprobar la sede. México hereda la ventaja de local aprendida del dataset (Azteca, altitud, eliminatorias CONCACAF) incluso en partidos que el fixture real ubicaría en EE.UU./Canadá. Además, en fase eliminatoria la localía de México/Canadá deja de aplicar en la realidad a partir de ciertas rondas (sedes en EE.UU.).
2. **Shift de distribución en `travel_distance_diff`.** En entrenamiento la feature es 0.0 en el 83.4 % de las filas (verificado); en simulación está 100 % poblada con valores de miles de km, y además se calcula contra la *mínima* distancia a cualquiera de las 3 sedes, no contra la sede real del partido. El modelo ve en inferencia una distribución que casi nunca vio en entrenamiento. (Mitigante: SHAP ≈ 0.03, la feature es casi inerte — lo que a su vez cuestiona conservarla.)
3. **Incoherencia outcome/marcador en la simulación.** El outcome se sortea del clasificador y los goles de un Poisson independiente que luego se "reconcilia" (g1=g2+1, etc., [tournament.py:134-139](../src/simulation/tournament.py)). Las probabilidades implícitas del Poisson no coinciden con las del modelo, y la reconciliación trunca/desplaza la distribución de GD y GF que alimenta los desempates de grupo — precisamente el mecanismo que decide los mejores terceros. Un Dixon-Coles (ya identificado como trabajo futuro) o un Poisson condicionado al outcome serían consistentes.
4. **ELO sin ventaja de local ni margen de goles.** El ELO implementado omite los dos ajustes estándar del World Football Elo: +100 (aprox.) al local en el expected score y multiplicador por diferencia de goles. Sin el ajuste de localía, los equipos que juegan muchos partidos en casa (anfitriones de torneos, eliminatorias asimétricas) acumulan rating inflado; ese sesgo entra directo en `elo_diff`, la feature dominante.
5. **Constantes de imputación inconsistentes:** xG por defecto 1.2 en features pero `league_xg=1.25` en el Poisson de la simulación; rank por defecto 78; squad por defecto 50 M€ — valores razonables pero sin justificación empírica documentada.
6. **K-factors ad-hoc:** la tabla `K_FACTORS` (60/50/40/20) no cita fuente ni se valida (p. ej., optimizando K por log-loss del propio ELO).
7. **Sensibilidad a lesiones poco informativa por construcción.** El stress test perturba la feature con menor SHAP (`squad_value_diff`, 0.04), así que el resultado (Δ ≤ 0.4 pp) estaba predeterminado por el diseño; el paper lo reconoce a medias, pero presentarlo como "análisis de sensibilidad a lesiones" sobrevende lo que mide. Una lesión real movería ELO-proxy/xG, no solo el valor de mercado.
8. **`shootout_winrate_diff` casi constante** (~0.5 tras shrinkage para las potencias, 5/48 equipos exactamente en 0.5) y su ablación (+0.0008) está dentro del ruido declarado por los propios autores; el criterio que ellos mismos fijan en README («si no mejora el log-loss, considerar retirarla») se cumple y la feature sigue en el set.

---

## 9. Lo que se verificó y está correcto

Para balance: (a) el split temporal no tiene solapamiento y `evaluate.py` reutiliza la misma función; (b) el merge as-of de ranking y eventos es genuinamente backward-only; (c) la simetría home/away y la reordenación de columnas `[2,1,0]` en `build_predict_fn` son correctas; (d) las probabilidades de campeón suman 100.0 %; (e) la implementación de Clopper–Pearson es correcta; (f) la reproducibilidad multi-worker vía `SeedSequence.spawn` es real; (g) los 44 tests pasan; (h) la aclaración de escala del Brier es correcta; (i) el ELO se acumula sobre el universo completo incluyendo amistosos con K reducido, como se documenta.

---

## 10. Recomendaciones priorizadas

1. **Eliminar `class_weight="balanced"`** del entrenamiento (conservando el time decay si se desea), re-correr Optuna y regenerar toda la cadena. Esperable: log-loss ≈ 0.83–0.84 y un cambio material en las probabilidades simuladas. Reescribir §VI del paper: la narrativa de la calibración cambia por completo.
2. **Corregir `KNOCKOUT_BRACKET_ORDER`** al bracket oficial FIFA 2026 (terceros vs. ganadores, cruces 2º-vs-2º, asignación por combinación de grupos) o, como mínimo, rebajar la afirmación «implementa fielmente el reglamento».
3. **Reconstruir la validación WC2022 con features as-of 2021** (snapshots históricos de Transfermarkt existen) y reportar esas métricas como la validación sin leakage; reclasificar las métricas actuales como optimistas.
4. **Repetir la ablación con ≥10 semillas** y reportar media ± desviación; eliminar las conclusiones de contribución que no superen el ruido.
5. **Añadir benchmark externo** (odds de mercado o ranking-only) y etiquetar los IC como error de simulación, no incertidumbre del pronóstico.
6. Menores: condicionar la localía a la sede real del fixture, unificar 1.2/1.25, añadir ventaja de local al ELO, retirar `shootout_winrate_diff` y `travel_distance_diff` según el propio criterio declarado.

---

*Análisis generado mediante auditoría de código y experimentos de re-entrenamiento controlados sobre `data/processed/features.csv` con el entorno `.venv` del proyecto.*
