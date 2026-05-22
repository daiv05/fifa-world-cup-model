"""
Evaluación de modelos: log-loss, Brier score, curvas de calibración y análisis SHAP.
Incluye validación histórica sobre el Mundial 2022.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

_repo_root = Path(__file__).parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

FEATURE_COLS = [
    "elo_diff",
    "squad_value_diff",
    "xg_avg_for",
    "xg_avg_against",
    "travel_distance_home",
    "travel_distance_away",
    "ranking_diff",
]

CLASS_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def _to_df(X: np.ndarray) -> pd.DataFrame:
    """Convierte numpy array a DataFrame con nombres de features.
    Evita el UserWarning de LightGBM cuando predict recibe numpy
    pero el modelo fue entrenado con un DataFrame con nombres de columna."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=FEATURE_COLS)


def evaluate_all(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Calcula log-loss y Brier score macro para cada modelo.

    Parámetros
    ----------
    models : dict[name -> fitted_model]
    X_test, y_test : datos de prueba

    Devuelve
    --------
    DataFrame con columnas [model, log_loss, brier_score]
    """
    records = []
    X_df = _to_df(X_test)
    for name, model in models.items():
        proba = model.predict_proba(X_df)
        ll = log_loss(y_test, proba)

        # Brier score macro: promedio sobre las 3 clases
        brier = np.mean([
            brier_score_loss(
                (y_test == cls).astype(int),
                proba[:, i],
            )
            for i, cls in enumerate(sorted(np.unique(y_test)))
        ])
        records.append({"model": name, "log_loss": round(ll, 4), "brier_score": round(brier, 4)})

    return pd.DataFrame(records).sort_values("log_loss")


def plot_calibration_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> plt.Figure:
    """
    Grafica curvas de calibración (reliability diagrams) para cada clase y modelo.
    """
    classes = sorted(np.unique(y_test))
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    X_df = _to_df(X_test)
    for ax, cls in zip(axes, classes):
        for name, model in models.items():
            proba = model.predict_proba(X_df)
            class_idx = list(model.classes_).index(cls) if hasattr(model, "classes_") else cls
            prob_pos = proba[:, class_idx]
            fraction_of_positives, mean_predicted = calibration_curve(
                (y_test == cls).astype(int), prob_pos, n_bins=n_bins
            )
            ax.plot(mean_predicted, fraction_of_positives, marker="o", label=name)

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_title(f"Calibración — {CLASS_NAMES.get(cls, cls)}")
        ax.set_xlabel("Probabilidad predicha promedio")
        ax.set_ylabel("Fracción de positivos")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def shap_analysis(model, X_train: np.ndarray, feature_names: list[str] | None = None):
    """
    Genera análisis SHAP de importancia de features.
    Devuelve el objeto shap.Explanation y muestra el summary plot.
    """
    import shap

    if feature_names is None:
        feature_names = FEATURE_COLS

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train, check_additivity=False)

    print("Generando SHAP summary plot...")
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    print("  Guardado en shap_summary.png")

    return shap_values


def validate_wc2022(
    wc_matches_df: pd.DataFrame | None,
    features_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Validación histórica: evalúa el modelo sobre los partidos del Mundial 2022.

    Si wc_matches_df no es None: hace inner join exacto (date, home_team, away_team)
    para aislar los partidos reales del WC 2022 dentro de features_df, descartando
    amistosos, clasificatorias y otras competiciones del mismo año.

    Uso recomendado en __main__:
        all_matches = load_international_results()
        wc22_actual = all_matches[
            (all_matches["tournament"] == "FIFA World Cup")
            & (all_matches["date"].astype(str).str.startswith("2022"))
        ]
        validate_wc2022(wc22_actual, df, model)

    Si wc_matches_df es None: fallback a filtrar todos los partidos del año 2022.

    Devuelve DataFrame con [date, home_team, away_team, pred_class, true_class,
    correct, confidence].
    """
    if wc_matches_df is not None and not wc_matches_df.empty:
        wc22_raw = wc_matches_df[
            wc_matches_df["date"].astype(str).str.startswith("2022")
        ][["date", "home_team", "away_team"]].copy()
        wc22_raw["_date_str"] = wc22_raw["date"].astype(str).str[:10]
        features_copy = features_df.copy()
        features_copy["_date_str"] = features_copy["date"].astype(str).str[:10]
        wc22 = features_copy.merge(
            wc22_raw[["_date_str", "home_team", "away_team"]],
            on=["_date_str", "home_team", "away_team"],
            how="inner",
        ).drop(columns="_date_str")
        print(f"  Partidos WC 2022 encontrados por join exacto: {len(wc22)}")
    else:
        date_mask = features_df["date"].astype(str).str.startswith("2022")
        wc22 = features_df[date_mask].copy()
        print(f"  Partidos 2022 (fallback por año): {len(wc22)}")

    if wc22.empty:
        print("No hay datos del año 2022 en el dataset de features.")
        return pd.DataFrame()

    X_wc22 = _to_df(wc22[FEATURE_COLS].fillna(0).values.astype(np.float32))
    y_wc22 = wc22["target"].values.astype(int)

    proba = model.predict_proba(X_wc22)
    preds = proba.argmax(axis=1)

    result = wc22[["date", "home_team", "away_team"]].copy()
    result["pred_class"] = preds
    result["true_class"] = y_wc22
    result["correct"] = preds == y_wc22
    result["confidence"] = proba.max(axis=1).round(3)

    accuracy = result["correct"].mean()
    ll = log_loss(y_wc22, proba)
    print(f"WC2022 Accuracy: {accuracy:.1%}  |  Log-Loss: {ll:.4f}")
    return result


if __name__ == "__main__":
    from pathlib import Path
    from src.models.train import load_model, FEATURE_COLS
    from src.features.features import PROCESSED_DIR

    df = pd.read_csv(PROCESSED_DIR / "features.csv").dropna(subset=FEATURE_COLS + ["target"])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)

    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    models = {
        "LogReg": load_model("logreg_baseline"),
        "XGBoost": load_model("xgboost_calibrated"),
        "LightGBM": load_model("lightgbm"),
    }

    print("\n=== Tabla de evaluación ===")
    eval_df = evaluate_all(models, X_test, y_test)
    print(eval_df.to_string(index=False))
    from src.features.features import PROCESSED_DIR as _EVAL_DIR
    eval_df.to_csv(_EVAL_DIR / "model_evaluation.csv", index=False)
    print(f"  Guardado en {_EVAL_DIR / 'model_evaluation.csv'}")

    print("\n=== SHAP Analysis (XGBoost) ===")
    xgb_raw = load_model("xgboost")
    shap_analysis(xgb_raw, X[:split])

    print("\n=== Validación WC2022 ===")
    # Carga los 64 partidos reales del Mundial 2022 desde international_results
    # (filtrado por tournament="FIFA World Cup" + año 2022) para hacer join exacto.
    # wc_matches_1974_2022.csv tiene datos incorrectos para 2022, no se usa aquí.
    from src.data.data_loader import load_international_results
    all_matches = load_international_results()
    wc22_actual = all_matches[
        (all_matches["tournament"] == "FIFA World Cup")
        & (all_matches["date"].astype(str).str.startswith("2022"))
    ].copy()
    print(f"  Partidos del Mundial 2022 (fuente: international_results): {len(wc22_actual)}")
    wc22_results = validate_wc2022(wc22_actual, df, models["XGBoost"])
    print(wc22_results.head(10).to_string(index=False))
