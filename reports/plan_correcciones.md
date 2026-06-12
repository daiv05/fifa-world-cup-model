# Plan de implementación — Correcciones y mejoras del modelo WC2026

> **ESTADO (2026-06-10): EJECUTADO (F0–F6 completas).** Desviaciones respecto
> al plan, todas guiadas por la evidencia obtenida durante la ejecución:
> (1) la ventaja de local del ELO (2.1) se implementó pero quedó desactivada
> — el grid de `tune_elo.py` no la validó (HA=0 gana); (2) el modelo final
> seleccionado por validación resultó ser la regresión logística, así que el
> ensemble (5.4) usa réplicas bootstrap del train en lugar de semillas;
> (3) los snapshots históricos de Transfermarkt (3.1) no se capturaron — se
> aplicó el fallback previsto: el pipeline pre-2022 excluye las features
> anacrónicas; (4) la ablación multi-semilla reveló que ninguna feature más
> allá de elo_diff aporta señal significativa, hallazgo que reescribió las
> conclusiones del paper. Mejora neta: LL test 0.8547 → 0.8360 (significativa).

**Origen:** hallazgos de `reports/analisis_critico.md` (2026-06-10).
**Principio rector del orden:** Optuna (100 trials × 2 modelos × 2 pipelines) es el paso más caro de la cadena. Todas las decisiones que alteran las entradas del entrenamiento (pesos, set de features, ELO) se toman en las Fases 1–2 y se paga **un solo re-entrenamiento completo** en la Fase 3. Las fases 4–5 (simulación y rigor estadístico) consumen los modelos ya finales. La Fase 6 actualiza la documentación al final, cuando los números son definitivos.

```
F0 (harness) ─→ F1 (pesos) ─→ F2 (features/ELO) ─→ F3 (retrain único)
                                                        │
                              F4 (motor de simulación) ─┤  (F4 es paralelizable con F1–F3:
                                                        │   no depende de los modelos)
                                          F5 (rigor estadístico) ─→ F6 (docs/paper)
```

---

## Fase 0 — Harness de regresión y congelado del baseline

*Objetivo: poder afirmar con evidencia "esto mejoró" en cada fase posterior.*

| # | Tarea | Detalle |
|---|---|---|
| 0.1 | Congelar baseline actual | Copiar `data/processed/model_evaluation.csv`, `ablation_results.csv`, `simulation_results.csv`, `tournament_progression.csv` y `models/*.joblib` a `data/processed/baseline_v1/` (o tag git `v1-baseline`). |
| 0.2 | Script de comparación | Nuevo `src/analysis/compare_runs.py`: dado dos directorios de resultados, imprime Δlog-loss con bootstrap pareado (reutilizar la lógica del bootstrap del análisis: 2 000 réplicas, IC95) y Δ de probabilidades de campeón top-10. Es la herramienta de aceptación de todas las fases. |
| 0.3 | Test de no-regresión del pipeline | Test pytest que corre `build_match_features` sobre un fixture sintético pequeño y compara contra un snapshot dorado, para detectar cambios accidentales de esquema durante las fases siguientes. |

**Criterio de salida:** `compare_runs.py` reproduce contra el baseline los números del análisis crítico (LL 0.8547, etc.).

---

## Fase 1 — Eliminar el rebalanceo de clases (Hallazgo A) ⚠️ máxima prioridad

*Archivos: `src/models/train.py`, `src/analysis/ablation.py`.*

| # | Tarea | Detalle |
|---|---|---|
| 1.1 | Reescribir `compute_combined_weights` | Eliminar el componente `class_weight="balanced"`. Conservar **solo** el time decay, renormalizado por su media. Mantener la firma para no romper consumidores; deprecar el parámetro implícito con un comentario que cite el porqué (las probabilidades posteriores deben reflejar las tasas base reales del problema). |
| 1.2 | Decisión documentada: ¿calibrar o no? | Evidencia del análisis: sin pesos, el XGBoost **sin calibrar** (0.8348) supera al calibrado (0.8499). Plan: mantener la calibración Platt como *artefacto a evaluar* pero seleccionar el modelo final por log-loss en **validación** (no en test), de modo que la elección calibrado-vs-crudo sea un paso de selección legítimo. Implementar en `train.py`: tras entrenar ambos, evaluar en val 2021 y guardar `best_model.json` con el nombre del ganador; `simulate.py` lee ese puntero por defecto. |
| 1.3 | Verificación rápida pre-Optuna | Antes de lanzar los 100 trials: reentrenar con `best_params` actuales sin pesos y confirmar con `compare_runs.py` la mejora esperada (~0.83–0.84 en test). Si no se reproduce, detener y diagnosticar. |
| 1.4 | Sincronizar la ablación | `ablation.py` usa `compute_combined_weights`; hereda el cambio automáticamente. Verificar que no haya otra copia de la lógica de pesos (grep `compute_class_weight`). |

**Criterio de salida:** con hiperparámetros viejos y sin pesos, test LL ≤ 0.845 y P(empate) media predicha dentro de ±0.015 de la frecuencia real del test.

**No incluido a propósito:** re-correr Optuna aquí — se hace una sola vez en la Fase 3, después de fijar el set de features.

---

## Fase 2 — Set de features y ELO definitivos (Hallazgos E, 8.2, 8.4, 8.6, 8.8)

*Archivos: `src/features/elo.py`, `src/features/features.py`, `src/features/derived_stats.py`.*

| # | Tarea | Detalle |
|---|---|---|
| 2.1 | ELO con ventaja de local | En `calculate_elo_ratings`: sumar `HOME_ADVANTAGE` (inicializar en 100, estándar World Football Elo) al rating del local en `_expected_score` **solo si `neutral == False`** (la columna existe en la fuente martj42; hoy no se usa). Esto corrige el rating inflado de equipos con muchos partidos en casa. |
| 2.2 | ELO con margen de goles | Multiplicador estándar World Football Elo sobre K: `1.0` si dif=1, `1.5` si dif=2, `(11+dif)/8` si dif≥3. |
| 2.3 | Validar K y λ empíricamente | Mini-estudio (script `src/analysis/tune_elo.py`): grid sobre `HOME_ADVANTAGE ∈ {0, 50, 100, 150}` y K-scale, evaluando log-loss de la probabilidad ELO pura sobre 2010–2020 (nunca sobre test). Documentar el resultado; fija defaults con evidencia en lugar de constantes ad-hoc. |
| 2.4 | Retirar features según el criterio ya declarado | Eliminar de `FEATURE_COLS`: `shootout_winrate_diff` (casi constante tras shrinkage, Δ dentro del ruido) y `travel_distance_diff` (83 % ceros en train + shift de distribución en inferencia, SHAP 0.03). El README ya establecía el criterio de retiro; aplicarlo. Conservar el código de cálculo (lo usa la simulación para nada → limpiar también `_team_pair_to_feature_dict`). |
| 2.5 | `striker_concentration` con ventana | Reemplazar el Herfindahl acumulado de por vida por uno sobre **ventana móvil de 4 años** (as-of, mismo `merge_asof` backward). Mide estructura de ataque *actual* en vez de antigüedad del programa futbolístico. Mantener el acumulado como columna alternativa para que la ablación de F5 compare ambos. |
| 2.6 | Unificar constantes de imputación | Una sola constante `LEAGUE_AVG_XG = 1.2` exportada desde `features.py`, consumida por la imputación y por `_sample_goals_poisson` (hoy 1.25). |
| 2.7 | Actualizar tests | `tests/test_features.py` y `test_derived_stats.py`: casos para ELO con localía/margen (partido neutral vs. no neutral), Herfindahl con ventana, y esquema de `FEATURE_COLS` reducido (7 features). `assert_model_feature_count` ya protege el resto de la cadena. |

**Criterio de salida:** `python -m src.features.features` regenera `features.csv`/`team_features.csv` con el nuevo esquema; tests pasan; el LL del ELO puro (2.3) mejora respecto al ELO sin localía.

---

## Fase 3 — Re-entrenamiento único + validación WC2022 sin leakage (Hallazgo C)

*Archivos: `src/models/train.py`, `src/models/evaluate.py`, `src/data/scraper.py`, nuevo dato histórico.*

| # | Tarea | Detalle |
|---|---|---|
| 3.1 | Snapshots históricos de Transfermarkt | Crear `data/raw/squad_values_history.csv` con esquema `(team, snapshot_date, squad_value_eur)`: como mínimo un snapshot ~2021-11 (pre-WC2022) además del actual. Transfermarkt publica valores con fecha (la limitación del paper que dice "no existe serie histórica" es incorrecta para squad_value). Si el scraping histórico no es viable, captura manual de los 32 equipos del WC2022 — es un esfuerzo acotado (~1 h). |
| 3.2 | Features versionadas por fecha | En `build_match_features`: si existe la serie histórica, `squad_value_diff` se asigna por `merge_asof(backward)` sobre `snapshot_date` en lugar del map estático. Para xG: si no hay serie histórica viable, **excluir `xg_avg_*` del pipeline pre-2022** (el doble pipeline ya existe; pasar un flag `feature_subset`) en vez de fingir que el snapshot 2026 es válido en 2021. |
| 3.3 | Re-entrenamiento completo | `python -m src.models.train --trials 100` con todo lo anterior integrado. Un solo pase: pipeline principal + pipeline pre-2022. Guardar como `models/` nuevos (el baseline quedó congelado en F0). |
| 3.4 | Validación WC2022 honesta | `validate_wc2022` debe evaluar el modelo pre-2022 sobre features **as-of 2021** (squad histórico, sin xG anacrónico). Reportar ambas cifras —contaminada y limpia— en consola y CSV (`wc2022_validation.csv`) para cuantificar el efecto del leakage en lugar de solo eliminarlo. |
| 3.5 | Etiquetar el test set principal | En `evaluate.py` y docs: las métricas del test ≥2022 con snapshot 2026 se reportan como "cota optimista (features parcialmente anacrónicas)"; la cifra de referencia del paper pasa a ser la de 3.4. |

**Criterio de salida:** existe una métrica WC2022 donde *ni el modelo ni las features* ven información posterior a 2021-12-31, y `compare_runs.py` documenta el gap contaminada-vs-limpia.

---

## Fase 4 — Motor de simulación fiel al reglamento (Hallazgos B, 8.1, 8.3) — paralelizable con F1–F3

*Archivos: `src/simulation/tournament.py`, `src/simulation/simulate.py`, nuevo `data/raw/wc2026_bracket.csv`.*

| # | Tarea | Detalle |
|---|---|---|
| 4.1 | Bracket oficial como **dato**, no como constante | Crear `data/raw/wc2026_bracket.csv` transcribiendo el calendario oficial FIFA (partidos 73–88 del Round of 32, con sede y la procedencia de cada slot: `1A`, `2C`, `3rd(pool)`), más la **tabla de asignación de terceros** del reglamento (la asignación a llaves depende de la combinación de los 8 grupos de procedencia, estilo Euro/anexo FIFA). Fuente: reglamento FIFA 2026 + match schedule publicado. **No inventar el bracket de memoria: transcribir y citar.** |
| 4.2 | Implementar asignación de terceros | Función `allocate_thirds(qualified_groups: frozenset) -> dict[slot, group]` que aplica la tabla 4.1. Estructura del R32 real: los 8 mejores terceros se cruzan con ganadores de grupo; existen cruces 2º-vs-2º; **ningún 3º-vs-3º**. Eliminar `KNOCKOUT_BRACKET_ORDER` hardcodeado. |
| 4.3 | Cuadrantes y rondas siguientes | Codificar la progresión real de llaves (qué ganador de R32 cruza con cuál en octavos, etc.) desde el mismo CSV, garantizando que los cuadrantes coinciden con el calendario oficial. |
| 4.4 | Desempates de grupo completos | Tras puntos→DG→GF: enfrentamiento directo entre empatados (puntos/DG/GF del subconjunto), y como último recurso **sorteo con el `rng` de la iteración** (hoy es orden de inserción determinista — micro-sesgo). Fair play no es simulable: documentarlo como aproximación y saltarlo. |
| 4.5 | Localía condicionada a la sede real | El CSV 4.1 trae la sede de cada partido. `_host_advantage_probs(t1, t2, venue_country)`: aplicar localía solo si el anfitrión juega en su propio país. Para fase de grupos, las sedes de los anfitriones son conocidas (juegan en casa); en eliminatorias depende del slot, que el CSV resuelve. |
| 4.6 | Marcadores coherentes con el outcome | Reemplazar la "reconciliación" ad-hoc por **Poisson condicionado**: muestrear `(g1, g2)` de la distribución conjunta Poisson *condicionada al outcome sorteado* (rejection sampling acotado o muestreo directo sobre una grilla 0–10 normalizada, que es barato y exacto). Elimina la distorsión de GD/GF que alimenta los desempates. Dixon-Coles queda como mejora opcional (F7). |
| 4.7 | Tests del motor | Tests nuevos: (i) con 8 terceros conocidos, la asignación reproduce la tabla oficial; (ii) ningún emparejamiento 3º-vs-3º; (iii) el desempate total usa el rng (dos seeds → órdenes distintos); (iv) la distribución de GD condicionada a "gana t1" tiene soporte solo en GD>0; (v) localía no se aplica a México en sede EE.UU. |

**Criterio de salida:** los 44+ tests pasan; una corrida de validación muestra que la fracción de terceros en octavos ya no es la constante 4/16 sino una distribución.

---

## Fase 5 — Rigor estadístico y comunicación de incertidumbre (Hallazgos D, F)

*Archivos: `src/analysis/ablation.py`, `src/simulation/simulate.py`, nuevo `src/analysis/benchmark.py`.*

| # | Tarea | Detalle |
|---|---|---|
| 5.1 | Ablación multi-semilla | `ablation.py --seeds 10`: reentrenar cada configuración con 10 semillas (`random_state` del XGBoost), reportar media ± std del LL y un **bootstrap pareado** del Δ vs. completo (reutilizar `compare_runs.py`). Columna `significant` (IC95 del Δ no cruza 0). Las conclusiones del paper se reescriben solo sobre filas significativas. Incluir fila para Herfindahl ventana-vs-acumulado (decisión 2.5). |
| 5.2 | Bootstrap en la evaluación principal | `evaluate.py`: añadir IC95 bootstrap pareado a `model_evaluation.csv` (diferencias entre modelos), no solo puntos. |
| 5.3 | Benchmark externo | `benchmark.py`: comparar P(campeón) del modelo contra probabilidades implícitas de mercado (odds de cierre de un agregador, snapshot manual en `data/raw/market_odds_2026.csv`, con fecha y fuente citada; des-vigorish proporcional). Reportar divergencia por equipo y log-score relativo donde sea evaluable. Es el baseline estándar de la literatura (Dixon-Coles se valida contra mercado). |
| 5.4 | Incertidumbre del modelo en la simulación | Implementar `--ensemble K` en `simulate.py`: K modelos reentrenados con semillas distintas (reusar los de 5.1), simulación estratificada (N/K iteraciones por modelo). Reportar **dos intervalos**: IC de simulación (Clopper–Pearson, reducible) e IC entre-modelos (percentiles de P(campeón) sobre los K modelos, irreducible). Renombrar en outputs: `ci_sim_*` y `ci_model_*`. |
| 5.5 | Sensibilidad a lesiones rediseñada | En vez de perturbar solo `squad_value` (la feature más débil — resultado predeterminado), el escenario de lesión perturba conjuntamente: squad_value −30 %, xg_for −10 %, ELO −25 pts (aprox. literatura de ausencia de jugadores clave). Documentar que es un stress test agregado, no modelado estructural. |

**Criterio de salida:** la tabla de ablación tiene columnas de significancia; `simulation_results.csv` distingue las dos fuentes de incertidumbre; existe comparación contra mercado con fuente citada.

---

## Fase 6 — Documentación y paper

*Archivos: `README.md`, `reports/paper/paper.tex`.*

| # | Tarea | Detalle |
|---|---|---|
| 6.1 | Regenerar todos los números | Tabla de evaluación, ablación, top-10, progresión de anfitriones, sensibilidad — todo cambia tras F1–F5. Verificar consistencia README ↔ paper ↔ CSVs con un script (`src/analysis/check_docs_sync.py` opcional, o revisión manual con checklist). |
| 6.2 | Reescrituras obligadas del paper | (i) §Modelado: eliminar la justificación del rebalanceo; explicar por qué se entrena sobre la distribución natural citando el objetivo de calibración. (ii) §Evaluación: nueva narrativa de la calibración (ya no "corrige sobreconfianza" si el modelo crudo gana). (iii) §Simulación: describir el bracket oficial y la asignación de terceros; retirar la palabra "fielmente" si queda cualquier aproximación (fair play). (iv) §Limitaciones: corregir la afirmación de que no existe serie histórica de squad_value; reclasificar las métricas con snapshot como cota optimista. (v) IC: etiquetar como error de simulación + intervalo entre-modelos. (vi) Abstract y conclusiones: números nuevos; rebajar `striker_concentration` a hipótesis si 5.1 no la confirma. |
| 6.3 | README | Mismos cambios en versión corta + actualizar pipeline de ejecución con los flags nuevos (`--ensemble`, `--seeds`). |
| 6.4 | Limpieza final | Borrar `data/processed/baseline_v1/` o archivarlo fuera del repo; verificar que el dashboard (`dashboard.py`) consume las columnas renombradas (`ci_sim_*`). |

---

## Orden de ejecución recomendado y esfuerzo estimado

| Fase | Esfuerzo | Bloquea a | Nota |
|---|---|---|---|
| F0 | 0.5 día | todas | trivial pero imprescindible |
| F1 | 0.5 día | F3 | el fix de mayor impacto/esfuerzo del proyecto |
| F2 | 1.5–2 días | F3 | la tarea 2.3 (tune ELO) es la más abierta; timeboxear |
| F4 | 1.5–2 días | F5.4 | **arrancar en paralelo con F1–F2**; la transcripción del bracket oficial (4.1) es trabajo de datos, no de código |
| F3 | 1 día + cómputo | F5 | un solo Optuna completo (≈ horas de CPU); 3.1 incluye captura manual de datos |
| F5 | 1.5 días + cómputo | F6 | 5.1 y 5.4 comparten los reentrenos multi-semilla: implementarlos juntos |
| F6 | 1 día | — | no empezar antes de tener números finales |

**Total: ~7–9 días de trabajo efectivo** más tiempo de cómputo (Optuna + 10 semillas; usar `--n-jobs -1` asumiendo la pérdida de bit-reproducibilidad del TPE, que F0 mitiga al fijar criterios de aceptación por métrica y no por igualdad exacta).

## Riesgos y decisiones abiertas

1. **El bracket oficial (4.1) es el único punto con riesgo de error de transcripción** — mitigación: test 4.7(i) contra un caso publicado y doble verificación contra dos fuentes (reglamento + schedule).
2. **Puede que tras F1–F2 el modelo calibrado pierda contra el crudo** (es lo que sugiere la evidencia): la decisión 1.2 (selección en validación) resuelve esto sin juicio manual, pero cambia la narrativa del paper — asumirlo desde ya.
3. **Si los snapshots históricos de Transfermarkt no son recuperables** para los 32 equipos del WC2022, el fallback es excluir `squad_value_diff` del pipeline pre-2022 (igual que xG), dejando la validación limpia aunque con menos features.
4. **Las probabilidades de España bajarán** (menos empates inflados + bracket más favorable a ganadores ya no, al contrario: ganadores enfrentan terceros → favoritos suben en R32 pero pierden el cruce blando 3º-vs-3º desaparecido...). La dirección neta es incierta; el benchmark 5.3 es el árbitro, no la intuición.
