"""
Modelos: Regresión Logística (baseline), XGBoost, LightGBM.
Optimización de hiperparámetros con Optuna. Calibración de probabilidades.

Split temporal:
  - train: date < 2021-01-01
  - val/calibración: 2021-01-01 <= date < 2022-01-01
  - test: date >= 2022-01-01
"""

import argparse
import json
import joblib
import numpy as np
import optuna
import pandas as pd
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Fuente única de verdad: FEATURE_COLS y el piso de la ventana de modelado se
# definen en features.py (módulo sin dependencias ML). Se re-exportan aquí para
# que los consumidores existentes (`from src.models.train import FEATURE_COLS`,
# p.ej. ablation.py) sigan funcionando.
from src.features.features import FEATURE_COLS, TRAIN_MIN_YEAR

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path(__file__).parents[2] / "data" / "processed" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")


def compute_combined_weights(
    y: np.ndarray,
    time_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pesos muestrales para el entrenamiento: SOLO time_decay, renormalizado por
    su media para que el peso promedio sea ~1.0.

    Nota metodológica (corrección v2): la versión anterior multiplicaba además
    por `class_weight="balanced"`. Rebalancear clases deforma las probabilidades
    posteriores alejándolas de las tasas base reales (los empates ~21% se
    sobreponderaban ~1.6x), lo que es contraproducente cuando el objetivo del
    modelo son probabilidades calibradas evaluadas con log-loss/Brier. La
    auditoría empírica mostró que el mismo XGBoost sin rebalanceo mejora el
    log-loss de test de 0.8547 a ~0.835 y elimina la sobrepredicción de empates.
    El parámetro `y` se conserva en la firma por compatibilidad.
    """
    if time_weights is not None:
        w = time_weights.astype(np.float32).copy()
        w /= w.mean()
        return w
    return np.ones(len(y), dtype=np.float32)


def temporal_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
):
    """
    Devuelve (train_mask, val_mask, test_mask) sobre `df["date"]`.
    """
    dates = pd.to_datetime(df["date"])
    train_mask = dates < train_end
    val_mask = (dates >= train_end) & (dates < val_end)
    test_mask = dates >= val_end
    return train_mask.values, val_mask.values, test_mask.values


def _cv_score(model, X, y, weights, cv=5) -> float:
    # TimeSeriesSplit preserva el orden temporal: cada fold entrena con el
    # pasado y valida en un bloque futuro contiguo. Coherente con el split
    # externo train/val/test por fecha - evita que Optuna elija
    # hiperparámetros bajo un supuesto i.i.d. que el diseño externo niega.
    splitter = TimeSeriesSplit(n_splits=cv)
    kwargs = {"cv": splitter, "scoring": "neg_log_loss"}
    if weights is not None:
        # sklearn >=1.6 usa `params`; versiones previas usan `fit_params`.
        try:
            scores = cross_val_score(model, X, y, params={"sample_weight": weights}, **kwargs)
        except TypeError:
            scores = cross_val_score(model, X, y, fit_params={"sample_weight": weights}, **kwargs)
    else:
        scores = cross_val_score(model, X, y, **kwargs)
    return float(scores.mean())


def train_baseline(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0,
        )),
    ])
    if isinstance(X, np.ndarray):
        cols = feature_names or FEATURE_COLS
        X = pd.DataFrame(X, columns=cols)

    fit_params = {"clf__sample_weight": weights} if weights is not None else {}
    pipe.fit(X, y, **fit_params)
    return pipe


def train_xgboost(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    params: dict | None = None,
) -> XGBClassifier:
    default_params = dict(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    if params:
        default_params.update(params)
    model = XGBClassifier(**default_params)
    model.fit(X, y, sample_weight=weights)
    return model


def train_lightgbm(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> LGBMClassifier:
    default_params = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        default_params.update(params)
    model = LGBMClassifier(**default_params)

    if isinstance(X, np.ndarray):
        cols = feature_names or FEATURE_COLS
        X = pd.DataFrame(X, columns=cols)

    model.fit(X, y, sample_weight=weights)
    return model


def run_optuna_study(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    n_trials: int = 100,
    model_type: str = "xgboost",
    n_jobs: int = 1,
) -> tuple[dict, optuna.Study]:
    """
    `n_jobs` controla la paralelización de TRIALS de Optuna:
      - n_jobs=1 (default): búsqueda en serie, 100% reproducible con seed=42.
      - n_jobs>1 o -1: trials en paralelo (~3-4x más rápido en datos pequeños),
        pero el TPE deja de ser determinista (el orden de trials varía), así que
        best_params no es bit-reproducible entre corridas. Para evitar
        oversubscripción, cada modelo se fuerza a n_jobs=1 cuando se paraleliza.
    """
    parallel = n_jobs != 1
    model_threads = 1 if parallel else -1

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_jobs": model_threads,
        }
        if model_type == "xgboost":
            params["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 15.0)
            params["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
            model = train_xgboost(X, y, weights, params)
        else:
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
            params["min_split_gain"] = trial.suggest_float("min_split_gain", 0.0, 1.0)
            model = train_lightgbm(X, y, weights, params)
        return _cv_score(model, X, y, weights)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_type}_study",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    # `n_jobs` no debe contaminar el best_params guardado (es de runtime, no del modelo).
    best = {k: v for k, v in study.best_params.items() if k != "n_jobs"}
    return best, study


def calibrate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Calibra un modelo ya entrenado sobre el set de validación.
    sklearn >=1.6 quitó cv="prefit"; ahora se usa FrozenEstimator.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def save_model(model, name: str) -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Modelo guardado: {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def assert_model_feature_count(model, expected: int = len(FEATURE_COLS), name: str = "modelo") -> None:
    """
    Verifica que el modelo espera exactamente `expected` features. Convierte un
    desync silencioso (p.ej. modelo entrenado con un esquema viejo de features
    cargado contra el vector nuevo) en un error explícito. Tras cambiar el set de
    features hay que reentrenar (`make clean && make all`).
    """
    n = getattr(model, "n_features_in_", None)
    if n is None and hasattr(model, "named_steps"):  # sklearn Pipeline (logreg)
        for step in model.named_steps.values():
            n = getattr(step, "n_features_in_", None)
            if n is not None:
                break
    if n is not None and n != expected:
        raise ValueError(
            f"{name} espera {n} features pero FEATURE_COLS tiene {expected}. "
            f"Reentrena los modelos: `make clean && make all`."
        )


def save_best_params(best_params: dict, model_name: str) -> Path:
    path = MODELS_DIR / f"best_params_{model_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2)
    print(f"Best params guardados: {path}")
    return path


def _full_training_pipeline(
    df: pd.DataFrame,
    trials: int,
    cutoff: pd.Timestamp | None = None,
    suffix: str = "",
    min_year: int | None = TRAIN_MIN_YEAR,
    optuna_jobs: int = 1,
    feature_cols: list[str] | None = None,
) -> None:
    """
    Si `cutoff` se pasa, se entrena solo con date < cutoff y los modelos
    se guardan con `suffix` (p.ej. "_pre2022").

    `feature_cols` permite entrenar con un subconjunto de FEATURE_COLS. El
    pipeline pre-2022 lo usa para EXCLUIR las features anacrónicas (xG y
    squad_value, snapshots de ~2026): un modelo "pre-2022" cuyo vector de
    entrada contiene información observada en 2026 no es una validación
    out-of-time honesta, aunque el target sea limpio.

    `min_year` recorta el dataset por abajo (default 2010). Razones:
      - los features estáticos (xg_*, squad_value, ranking moderno) no son
        representativos del fútbol pre-2010.
      - TimeSeriesSplit sobre datos multi-década promedia regímenes
        incomparables y empuja a hiperparámetros sub-óptimos.
      - el time_decay con lambda=0.001 ya hace que los partidos pre-2010
        pesen <2%, así que el recorte explícito no pierde señal real.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    df = df.dropna(subset=cols + ["target"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if min_year is not None:
        before = len(df)
        df = df[df["date"].dt.year >= min_year].reset_index(drop=True)
        print(f"  Recorte temporal (>= {min_year}): {before:,} -> {len(df):,} partidos")
    # Ordenar por fecha es requisito para TimeSeriesSplit dentro del CV de Optuna.
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    if cutoff is not None:
        df = df[df["date"] < cutoff].reset_index(drop=True)

    train_mask, val_mask, test_mask = temporal_split(df)
    print(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

    if val_mask.sum() == 0:
        # Para entrenamiento con cutoff = 2022 no hay val_mask 2021 dentro
        # del split estándar; se usa el último 15% del train como validación.
        n = train_mask.sum()
        split_at = int(n * 0.85)
        train_idx = np.where(train_mask)[0]
        val_mask = np.zeros_like(train_mask)
        val_mask[train_idx[split_at:]] = True
        train_mask = train_mask & ~val_mask
        print(f"  Sin val_mask 2021 - fallback 85/15 dentro de train: "
              f"Train={train_mask.sum():,}, Val={val_mask.sum():,}")

    X_all = df[cols].astype(np.float32)
    y_all = df["target"].values.astype(int)
    tw_all = (
        df["time_weight"].values.astype(np.float32)
        if "time_weight" in df.columns else None
    )

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    tw_train = tw_all[train_mask] if tw_all is not None else None

    weights_train = compute_combined_weights(y_train, tw_train)
    print(f"  Pesos combinados - media: {weights_train.mean():.3f}, max: {weights_train.max():.3f}")

    # Baseline
    print("Entrenando baseline (LogReg)...")
    baseline = train_baseline(X_train, y_train, weights_train)
    save_model(baseline, f"logreg_baseline{suffix}")

    # XGBoost con Optuna
    print(f"Optuna XGBoost ({trials} trials)...")
    best_xgb, _ = run_optuna_study(X_train, y_train, weights_train, n_trials=trials, model_type="xgboost", n_jobs=optuna_jobs)
    save_best_params(best_xgb, f"xgboost{suffix}")
    xgb_model = train_xgboost(X_train, y_train, weights_train, best_xgb)
    save_model(xgb_model, f"xgboost{suffix}")

    # LightGBM con Optuna
    print(f"Optuna LightGBM ({trials} trials)...")
    best_lgb, _ = run_optuna_study(X_train, y_train, weights_train, n_trials=trials, model_type="lightgbm", n_jobs=optuna_jobs)
    save_best_params(best_lgb, f"lightgbm{suffix}")
    lgb_model = train_lightgbm(X_train, y_train, weights_train, best_lgb)
    save_model(lgb_model, f"lightgbm{suffix}")

    # Calibración + selección del modelo final, ambas sobre validación (no test).
    #
    # La calibración Platt se ajusta SOLO sobre el primer 70% temporal de val;
    # el 30% restante (val_sel) queda virgen para decidir calibrado-vs-crudo
    # sin sesgo (comparar el calibrado sobre los mismos datos donde se ajustó
    # la sigmoide lo favorecería espuriamente). La decisión se persiste en
    # best_model{suffix}.json y simulate.py la consume por defecto.
    if X_val.size > 0:
        from sklearn.metrics import log_loss as _log_loss

        val_idx = np.where(val_mask)[0]
        cal_end = int(len(val_idx) * 0.70)
        cal_idx, sel_idx = val_idx[:cal_end], val_idx[cal_end:]
        X_cal, y_cal = X_all.iloc[cal_idx], y_all[cal_idx]
        X_sel, y_sel = X_all.iloc[sel_idx], y_all[sel_idx]

        # Sigmoid (Platt) en vez de isotonic: con ~1 año de validación e
        # isotonic 3-clase, isotonic sobreajusta y degrada log-loss en test.
        print("Calibrando XGBoost (sigmoid/Platt, prefit) sobre el 70% de validación...")
        xgb_cal = calibrate_model(xgb_model, X_cal, y_cal, method="sigmoid")
        save_model(xgb_cal, f"xgboost_calibrated{suffix}")

        candidates = {
            f"logreg_baseline{suffix}": baseline,
            f"xgboost{suffix}": xgb_model,
            f"lightgbm{suffix}": lgb_model,
            f"xgboost_calibrated{suffix}": xgb_cal,
        }
        val_scores = {
            name: round(float(_log_loss(y_sel, m.predict_proba(X_sel), labels=[0, 1, 2])), 4)
            for name, m in candidates.items()
        }
        best_name = min(val_scores, key=val_scores.get)
        print(f"  Log-loss en val_sel (30% final de val): {val_scores}")
        print(f"  Modelo seleccionado: {best_name}")
        pointer_path = MODELS_DIR / f"best_model{suffix}.json"
        pointer_path.write_text(
            json.dumps({"best_model": best_name, "val_sel_log_loss": val_scores}, indent=2),
            encoding="utf-8",
        )
        print(f"  Selección guardada en {pointer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento WC2026")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials por modelo")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Si se pasa (YYYY-MM-DD), entrena solo con date < cutoff "
                             "y guarda con suffix _pre<YYYY>")
    parser.add_argument("--min-year", type=int, default=TRAIN_MIN_YEAR,
                        help=f"Año mínimo a incluir en el entrenamiento (default "
                             f"{TRAIN_MIN_YEAR}). Pasa 0 para usar todo el histórico.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Workers para los trials de Optuna (default 1 = serie, "
                             "reproducible). >1 o -1 acelera ~3-4x pero el TPE deja de "
                             "ser determinista (best_params no bit-reproducibles).")
    args = parser.parse_args()

    from src.features.features import PROCESSED_DIR
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    min_year = args.min_year if args.min_year and args.min_year > 0 else None

    print(f"\n=== Pipeline principal (sin cutoff) ===")
    _full_training_pipeline(df, trials=args.trials, suffix="", min_year=min_year,
                             optuna_jobs=args.n_jobs)

    cutoff_pre22 = pd.Timestamp(args.cutoff) if args.cutoff else pd.Timestamp("2022-01-01")
    suffix = f"_pre{cutoff_pre22.year}"
    from src.features.features import ANACHRONISTIC_FEATURE_COLS
    clean_cols = [c for c in FEATURE_COLS if c not in ANACHRONISTIC_FEATURE_COLS]
    print(f"\n=== Pipeline pre-{cutoff_pre22.year} (validación WC sin leakage) ===")
    print(f"  Features (sin anacrónicas xG/squad): {clean_cols}")
    _full_training_pipeline(df, trials=args.trials, cutoff=cutoff_pre22, suffix=suffix,
                             min_year=min_year, optuna_jobs=args.n_jobs,
                             feature_cols=clean_cols)

    print("\nOK - modelos guardados en", MODELS_DIR)
