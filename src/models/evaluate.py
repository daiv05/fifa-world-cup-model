import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

REPORTS_DIR = Path(__file__).parents[2] / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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


def _to_df(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=FEATURE_COLS)


def evaluate_all(
    models: dict,
    X_test,
    y_test: np.ndarray,
) -> pd.DataFrame:
    records = []
    X_df = _to_df(X_test)
    for name, model in models.items():
        proba = model.predict_proba(X_df)
        ll = log_loss(y_test, proba)
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
    X_test,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> plt.Figure:
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
        ax.set_title(f"Calibración - {CLASS_NAMES.get(cls, cls)}")
        ax.set_xlabel("Probabilidad predicha promedio")
        ax.set_ylabel("Fracción de positivos")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def shap_analysis(model, X_train, feature_names: list[str] | None = None):
    import shap

    if feature_names is None:
        feature_names = FEATURE_COLS

    plt.close("all")
    X_df = _to_df(X_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df, check_additivity=False)

    print("Generando SHAP summary plot...")
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    if hasattr(values, "ndim") and values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=2)
        max_val = float(np.max(mean_abs)) if mean_abs.size else 0.0
        shap.summary_plot(
            mean_abs,
            X_df,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title("Importancia SHAP agregada (mean |SHAP|)")
    else:
        mean_abs = np.abs(values).mean(axis=0) if hasattr(values, "ndim") else np.array([])
        max_val = float(np.max(mean_abs)) if mean_abs.size else 0.0
        shap.summary_plot(
            shap_values,
            X_df,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )

    ax = plt.gca()
    if max_val > 0:
        ax.set_xlim(0, max_val * 1.15)

    fig = plt.gcf()
    fig.set_size_inches(7.5, 4.5)
    fig.tight_layout()
    out_path = REPORTS_DIR / "shap_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Guardado en {out_path}")

    paper_figs_dir = REPORTS_DIR.parent / "paper" / "figs"
    if paper_figs_dir.exists():
        paper_path = paper_figs_dir / "shap_summary.png"
        fig.savefig(paper_path, dpi=200, bbox_inches="tight")
        print(f"  Copiado en {paper_path}")

    plt.close(fig)

    return shap_values


def validate_wc2022(
    features_df: pd.DataFrame,
    model_pre2022,
) -> pd.DataFrame:
    """
    Evalúa el modelo `xgboost_pre2022` (entrenado solo con date < 2022)
    sobre todos los partidos de 2022 - el Mundial 2022 incluido.
    """
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wc22 = df[df["date"].dt.year == 2022].dropna(subset=FEATURE_COLS + ["target"]).copy()
    if wc22.empty:
        print("No hay partidos de 2022 en features.csv")
        return pd.DataFrame()

    X = _to_df(wc22[FEATURE_COLS].values.astype(np.float32))
    y = wc22["target"].values.astype(int)
    proba = model_pre2022.predict_proba(X)
    preds = proba.argmax(axis=1)

    result = wc22[["date", "home_team", "away_team"]].copy()
    result["pred_class"] = preds
    result["true_class"] = y
    result["correct"] = preds == y
    result["confidence"] = proba.max(axis=1).round(3)

    accuracy = result["correct"].mean()
    ll = log_loss(y, proba)
    print(f"WC/2022 Accuracy (modelo pre-2022): {accuracy:.1%}  |  Log-Loss: {ll:.4f}")
    return result


if __name__ == "__main__":
    from src.models.train import load_model, FEATURE_COLS, temporal_split
    from src.features.features import PROCESSED_DIR

    df = pd.read_csv(PROCESSED_DIR / "features.csv").dropna(subset=FEATURE_COLS + ["target"])
    df["date"] = pd.to_datetime(df["date"])

    _, _, test_mask = temporal_split(df)
    X_test = df.loc[test_mask, FEATURE_COLS].values.astype(np.float32)
    y_test = df.loc[test_mask, "target"].values.astype(int)

    print(f"Test set (date >= 2022): {len(y_test):,} partidos")

    models = {
        "LogReg": load_model("logreg_baseline"),
        "XGBoost": load_model("xgboost"),
        "XGBoost-Cal": load_model("xgboost_calibrated"),
        "LightGBM": load_model("lightgbm"),
    }

    print("\n=== Evaluación sobre test temporal (>= 2022) ===")
    eval_df = evaluate_all(models, X_test, y_test)
    print(eval_df.to_string(index=False))
    eval_df.to_csv(PROCESSED_DIR / "model_evaluation.csv", index=False)
    print(f"Guardado en {PROCESSED_DIR / 'model_evaluation.csv'}")

    print("\n=== Calibration curves ===")
    fig = plot_calibration_curves(models, X_test, y_test)
    fig.savefig(REPORTS_DIR / "calibration_curves.png", dpi=150, bbox_inches="tight")
    print(f"  Guardado en {REPORTS_DIR / 'calibration_curves.png'}")
    plt.close(fig)

    print("\n=== SHAP Analysis (XGBoost) ===")
    train_mask, _, _ = temporal_split(df)
    X_train = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    xgb_raw = load_model("xgboost")
    shap_analysis(xgb_raw, X_train[:5000])  # limitar para velocidad

    print("\n=== Validación WC2022 (modelo pre-2022) ===")
    try:
        xgb_pre22 = load_model("xgboost_pre2022")
    except FileNotFoundError:
        print("xgboost_pre2022 no encontrado - ejecuta `python -m src.models.train --cutoff 2022-01-01`")
    else:
        wc22_results = validate_wc2022(df, xgb_pre22)
        if not wc22_results.empty:
            print(wc22_results.head(10).to_string(index=False))
