# 1. Estás haciendo doble weighting en LogReg

Aquí:

```python
class_weight="balanced"
```

Y además:

```python
clf__sample_weight = weights
```

Pero `weights` YA incluye:

```python
compute_class_weight("balanced", ...)
```

Entonces LogReg está recibiendo:

* class weights,
* Y sample weights balanceados.

Eso puede distorsionar el entrenamiento.

## Recomendación

Quitar:

```python
class_weight="balanced"
```

del `LogisticRegression`.

Deja solo:

```python
sample_weight=weights
```

Porque ya estás manejando el balance manualmente.

---

# 2. Tu calibración isotónica probablemente sobreajusta

Esto es probablemente EL mayor problema.

Tienes:

```python
method="isotonic"
```

Para multiclase + ~1 año de validación.

Isotonic:

* necesita MUCHOS datos,
* y en fútbol suele sobreajustar.

Por eso ves:

| Modelo  | LogLoss |
| ------- | ------- |
| XGB     | 0.8948  |
| XGB-Cal | 1.0242  |

Eso es textbook overfitting de calibración.

---

# 3. Prueba sigmoid en vez de isotonic

Cambia:

```python
method="sigmoid"
```

Muy probablemente vas a obtener:

* mejor log loss,
* calibración más estable,
* menos probabilidades extremas.

En datasets medianos, sigmoid suele ganar.

---

# 4. Tu espacio de búsqueda todavía permite sobreajuste

Aquí:

```python
max_depth: 3-7
```

7 ya es bastante alto para fútbol.

Y además no estás regulando:

* `min_child_weight`
* `gamma`

Eso es importante.

---

# 5. Te faltan hiperparámetros críticos en XGBoost

Agrega:

```python
"min_child_weight": trial.suggest_float("min_child_weight", 1, 15),
"gamma": trial.suggest_float("gamma", 0, 5),
```

Y quizá:

```python
"max_delta_step": trial.suggest_float("max_delta_step", 0, 10),
```

Especialmente útil para clases desbalanceadas.

---

# 6. Falta early stopping

Ahora mismo haces:

```python
model.fit(X, y)
```

sin validation set.

Entonces:

* Optuna evalúa CV,
* pero cada modelo individual puede sobreentrenarse.

---

# 7. El problema MÁS serio: leakage en Optuna CV

Aquí hay algo importante.

Tu `_cv_score` usa:

```python
cross_val_score(model, X, y, cv=TimeSeriesSplit)
```

PERO:

el modelo YA fue entrenado antes:

```python
model = train_xgboost(X, y, weights, params)
```

y luego haces CV sobre ese modelo entrenado.

Eso es incorrecto.

---

# 8. Estás entrenando antes del cross validation

Este es probablemente el bug principal.

Actualmente:

```python
model = train_xgboost(...)
return _cv_score(model, X, y, weights)
```

Pero `cross_val_score` espera un estimador NO entrenado.

Debe ser:

```python
model = XGBClassifier(**params)
```

sin `.fit()`.

Entonces sklearn hará fit correctamente en cada fold.

---

# 9. Esto puede explicar MUCHAS cosas

Porque ahora:

* el modelo ya viene entrenado con TODO el dataset,
* luego CV reutiliza/copias del estimador,
* resultados de Optuna pueden quedar contaminados.

Eso puede producir:

* tuning raro,
* hiperparámetros inestables,
* mala generalización.

---

# 10. Cómo debería verse

En Optuna:

```python
if model_type == "xgboost":
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **params
    )
```

NO:

```python
train_xgboost(...)
```

---

# 11. También revisaría esto

Tu weighting temporal + class balancing puede estar exagerando pesos.

Haz debug:

```python
print(weights_train.min(), weights_train.max())
```

Si tienes:

* weights > 5-10,
* probablemente estás metiendo demasiado ruido.

---
