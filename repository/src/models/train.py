"""
Entrenamiento de modelos de clasificación multiclase para predicción de partidos.
Modelos: Regresión Logística (baseline), XGBoost, LightGBM.
Optimización de hiperparámetros con Optuna. Calibración de probabilidades.
"""

import sys
import joblib
import numpy as np
import optuna
import pandas as pd
from pathlib import Path

_repo_root = Path(__file__).parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def compute_combined_weights(
    y: np.ndarray,
    time_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Combina pesos de clase balanceados con pesos de decaimiento temporal.

    Estrategia: w_i = class_weight[y_i] × time_decay_i
    Esto aborda el desbalance (H:~49%, D:~21%, A:~31%) sin descartar información temporal.
    Si time_weights es None, devuelve solo los pesos de clase.
    """
    classes = np.unique(y)
    cw_values = compute_class_weight("balanced", classes=classes, y=y)
    cw_map = dict(zip(classes, cw_values))
    class_w = np.array([cw_map[yi] for yi in y], dtype=np.float32)

    if time_weights is not None:
        combined = class_w * time_weights.astype(np.float32)
        # Normaliza para que la media sea 1 (evita escalar el learning rate efectivo)
        combined /= combined.mean()
        return combined
    return class_w


def _cv_score(model, X, y, weights, cv=5) -> float:
    """Log-loss negado con CV estratificado (más bajo = mejor)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y,
        cv=skf,
        scoring="neg_log_loss",
        fit_params={"sample_weight": weights} if weights is not None else {},
    )
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
            class_weight="balanced",   # fallback; sample_weight tiene precedencia
        )),
    ])
    # Entrenar con DataFrame para que StandardScaler almacene los nombres de columna
    # y no produzca UserWarning cuando se prediga con DataFrames con nombres.
    if isinstance(X, np.ndarray):
        cols = feature_names or FEATURE_COLS
        X = pd.DataFrame(X, columns=cols)

    fit_params = {"clf__sample_weight": weights} if weights is not None else {}
    pipe.fit(X, y, **fit_params)
    return pipe


def train_xgboost(
    X: np.ndarray,
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

    # LightGBM 4.x lanza UserWarning si se entrena con DataFrame y se predice con
    # numpy (o viceversa). Pasar siempre un DataFrame con nombres garantiza
    # consistencia en training y prediction.
    if isinstance(X, np.ndarray):
        cols = feature_names or FEATURE_COLS
        X = pd.DataFrame(X, columns=cols)

    model.fit(X, y, sample_weight=weights)
    return model


def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    n_trials: int = 100,
    model_type: str = "xgboost",
) -> tuple[dict, optuna.Study]:
    """
    Busca los mejores hiperparámetros para XGBoost o LightGBM.
    Devuelve (best_params, study).
    """
    def objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            model = train_xgboost(X, y, weights, params)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            model = train_lightgbm(X, y, weights, params)

        score = _cv_score(model, X, y, weights)
        return score

    study = optuna.create_study(direction="maximize", study_name=f"{model_type}_study")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study


def calibrate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Calibra las probabilidades del modelo con Platt scaling (sigmoid) o regresión isotónica.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
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


if __name__ == "__main__":
    from src.features.features import PROCESSED_DIR

    print("Cargando features...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)
    time_weights = df["time_weight"].values.astype(np.float32) if "time_weight" in df.columns else None

    # Combina time decay con class_weight balanceado para tratar el desbalance
    # Distribución: Home Win ~49%, Away Win ~31%, Draw ~21%
    weights = compute_combined_weights(y, time_weights)
    print(f"  Pesos combinados — media: {weights.mean():.3f}, max: {weights.max():.3f}")

    print("Entrenando modelo baseline (Regresión Logística)...")
    baseline = train_baseline(X, y, weights)
    save_model(baseline, "logreg_baseline")

    print("Entrenando XGBoost...")
    xgb_model = train_xgboost(X, y, weights)
    save_model(xgb_model, "xgboost")

    print("Entrenando LightGBM...")
    lgb_model = train_lightgbm(X, y, weights)
    save_model(lgb_model, "lightgbm")

    print("Calibrando XGBoost (Platt scaling)...")
    split = int(len(X) * 0.8)
    xgb_cal = calibrate_model(xgb_model, X[split:], y[split:], method="sigmoid")
    save_model(xgb_cal, "xgboost_calibrated")

    print("OK — modelos guardados en", MODELS_DIR)
