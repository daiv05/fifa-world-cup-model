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
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path(__file__).parents[2] / "data" / "processed" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "elo_diff",
    "squad_value_diff",
    "xg_avg_for",
    "xg_avg_against",
    "travel_distance_home",
    "travel_distance_away",
    "ranking_diff",
]

TRAIN_END = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-01-01")


def compute_combined_weights(
    y: np.ndarray,
    time_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Combina pesos balanceados de clase con time_decay y re-normaliza por su
    media para que el peso promedio sea ~1.0 (evita que XGBoost interprete
    todos los partidos como muy importantes o muy poco importantes).
    """
    classes = np.unique(y)
    cw_values = compute_class_weight("balanced", classes=classes, y=y)
    cw_map = dict(zip(classes, cw_values))
    class_w = np.array([cw_map[yi] for yi in y], dtype=np.float32)

    if time_weights is not None:
        combined = class_w * time_weights.astype(np.float32)
        combined /= combined.mean()
        return combined
    return class_w


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


def _cv_score_pruned(
    trial,
    params: dict,
    X,
    y: np.ndarray,
    weights: np.ndarray | None,
    model_type: str,
    cv: int = 5,
) -> float:
    """
    CV temporal con reporte por fold para habilitar el pruning de Optuna.

    TimeSeriesSplit preserva el orden temporal: cada fold entrena con el pasado
    y valida en un bloque futuro contiguo, coherente con el split externo por
    fecha. Tras cada fold se reporta el neg_log_loss medio acumulado y se
    consulta `trial.should_prune()`: los trials claramente peores que la mediana
    se abortan temprano (MedianPruner), ahorrando folds inútiles.

    Devuelve el neg_log_loss medio (a maximizar). Las columnas de probabilidad
    se reordenan a [0,1,2] para que `log_loss` sea robusto si algún fold no ve
    las tres clases.
    """
    splitter = TimeSeriesSplit(n_splits=cv)
    # Mantener X como DataFrame con nombres de columna en fit y predict evita el
    # warning "X does not have valid feature names" y es coherente con cómo se
    # entrenan los modelos finales y se consultan en la simulación.
    X_df = X if hasattr(X, "iloc") else pd.DataFrame(np.asarray(X), columns=FEATURE_COLS)
    y_arr = np.asarray(y)
    w_arr = np.asarray(weights) if weights is not None else None

    fold_scores: list[float] = []
    for step, (tr_idx, va_idx) in enumerate(splitter.split(X_df)):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
        w_tr = w_arr[tr_idx] if w_arr is not None else None

        if model_type == "xgboost":
            model = train_xgboost(X_tr, y_tr, w_tr, params)
        else:
            model = train_lightgbm(X_tr, y_tr, w_tr, params)

        proba = model.predict_proba(X_va)
        proba_full = np.zeros((proba.shape[0], 3), dtype=float)
        for col, cls in enumerate(model.classes_):
            proba_full[:, int(cls)] = proba[:, col]
        # Renormaliza a suma 1 (float64): corrige el error de redondeo de las
        # probabilidades float32 del modelo, que para ciertos hiperparámetros
        # excede la tolerancia de log_loss y dispararía un UserWarning.
        proba_full /= proba_full.sum(axis=1, keepdims=True)
        fold_scores.append(-log_loss(y_va, proba_full, labels=[0, 1, 2]))

        trial.report(float(np.mean(fold_scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


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
) -> tuple[dict, optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        if model_type == "xgboost":
            params["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 15.0)
            params["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
        else:
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
            params["min_split_gain"] = trial.suggest_float("min_split_gain", 0.0, 1.0)
        return _cv_score_pruned(trial, params, X, y, weights, model_type)

    # MedianPruner: aborta un trial cuando su score intermedio cae por debajo de
    # la mediana de los trials previos en el mismo fold. n_startup_trials evita
    # podar antes de tener una base; n_warmup_steps no poda en los primeros folds.
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_type}_study",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Optuna {model_type}: {n_pruned}/{len(study.trials)} trials podados")
    return study.best_params, study


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
    min_year: int | None = 2010,
    reuse_params: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    Si `cutoff` se pasa, se entrena solo con date < cutoff y los modelos
    se guardan con `suffix` (p.ej. "_pre2022").

    `reuse_params` permite saltarse la optimización Optuna reutilizando
    hiperparámetros ya hallados (claves "xgboost" y "lightgbm"). Se usa para
    el pipeline pre-cutoff: los parámetros del pipeline principal se ajustan
    con CV sobre datos < 2021 (subconjunto de lo permitido aquí, < cutoff),
    así que reutilizarlos NO introduce leakage del test y evita repetir una
    búsqueda redundante. Devuelve los best_params efectivamente usados.

    `min_year` recorta el dataset por abajo (default 2010). Razones:
      - los features estáticos (xg_*, squad_value, ranking moderno) no son
        representativos del fútbol pre-2010.
      - TimeSeriesSplit sobre datos multi-década promedia regímenes
        incomparables y empuja a hiperparámetros sub-óptimos.
      - el time_decay con lambda=0.001 ya hace que los partidos pre-2010
        pesen <2%, así que el recorte explícito no pierde señal real.
    """
    df = df.dropna(subset=FEATURE_COLS + ["target"]).copy()
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

    X_all = df[FEATURE_COLS].astype(np.float32)
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

    # XGBoost: Optuna o reutilización de hiperparámetros
    if reuse_params and "xgboost" in reuse_params:
        best_xgb = reuse_params["xgboost"]
        print("Reutilizando best_params XGBoost (sin Optuna)...")
    else:
        print(f"Optuna XGBoost ({trials} trials)...")
        best_xgb, _ = run_optuna_study(X_train, y_train, weights_train, n_trials=trials, model_type="xgboost")
    save_best_params(best_xgb, f"xgboost{suffix}")
    xgb_model = train_xgboost(X_train, y_train, weights_train, best_xgb)
    save_model(xgb_model, f"xgboost{suffix}")

    # LightGBM: Optuna o reutilización de hiperparámetros
    if reuse_params and "lightgbm" in reuse_params:
        best_lgb = reuse_params["lightgbm"]
        print("Reutilizando best_params LightGBM (sin Optuna)...")
    else:
        print(f"Optuna LightGBM ({trials} trials)...")
        best_lgb, _ = run_optuna_study(X_train, y_train, weights_train, n_trials=trials, model_type="lightgbm")
    save_best_params(best_lgb, f"lightgbm{suffix}")
    lgb_model = train_lightgbm(X_train, y_train, weights_train, best_lgb)
    save_model(lgb_model, f"lightgbm{suffix}")

    # Calibración sobre val (no leakage)
    if X_val.size > 0:
        # Sigmoid (Platt) en vez de isotonic: con ~1 año de validación e
        # isotonic 3-clase, isotonic sobreajusta y degrada log-loss en test.
        print("Calibrando XGBoost (sigmoid/Platt, prefit) sobre validación temporal...")
        xgb_cal = calibrate_model(xgb_model, X_val, y_val, method="sigmoid")
        save_model(xgb_cal, f"xgboost_calibrated{suffix}")

    return {"xgboost": best_xgb, "lightgbm": best_lgb}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento WC2026")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials por modelo")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Si se pasa (YYYY-MM-DD), entrena solo con date < cutoff "
                             "y guarda con suffix _pre<YYYY>")
    parser.add_argument("--min-year", type=int, default=2010,
                        help="Año mínimo a incluir en el entrenamiento (default 2010). "
                             "Pasa 0 para usar todo el histórico.")
    args = parser.parse_args()

    from src.features.features import PROCESSED_DIR
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    min_year = args.min_year if args.min_year and args.min_year > 0 else None

    print(f"\n=== Pipeline principal (sin cutoff) ===")
    best_params = _full_training_pipeline(df, trials=args.trials, suffix="", min_year=min_year)

    cutoff_pre22 = pd.Timestamp(args.cutoff) if args.cutoff else pd.Timestamp("2022-01-01")
    suffix = f"_pre{cutoff_pre22.year}"
    print(f"\n=== Pipeline pre-{cutoff_pre22.year} (para validar WC sin leakage) ===")
    # Reutiliza los hiperparámetros del pipeline principal (leakage-safe: se
    # ajustaron con CV sobre datos < 2021, subconjunto de date < cutoff) para
    # evitar repetir Optuna y ~reducir a la mitad el tiempo de entrenamiento.
    _full_training_pipeline(df, trials=args.trials, cutoff=cutoff_pre22, suffix=suffix,
                             min_year=min_year, reuse_params=best_params)

    print("\nOK - modelos guardados en", MODELS_DIR)
