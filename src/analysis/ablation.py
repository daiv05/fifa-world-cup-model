"""
Ablation study: impacto marginal de grupos de features sobre Log-Loss y Brier.

Para aislar la contribucion de cada grupo se mantienen fijos el split temporal,
los pesos (time decay) y los hiperparametros ya optimizados del
XGBoost (best_params_xgboost.json); lo unico que cambia entre filas es el
conjunto de features con el que se reentrena y recalibra (Platt/sigmoid).
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.features.features import PROCESSED_DIR, DERIVED_FEATURE_COLS
from src.models.train import (
    FEATURE_COLS,
    MODELS_DIR,
    calibrate_model,
    compute_combined_weights,
    temporal_split,
    train_xgboost,
)

# Features estáticas anacrónicas: xG y squad_value son snapshots aplicados a toda
# la historia (ver nota en features.py). Esta fila cuantifica empíricamente su
# aporte para juzgar si la limitación es material.
_ANACHRONISTIC = [c for c in FEATURE_COLS if c.startswith("xg_") or c == "squad_value_diff"]

# Estudio de ablación. `late_goal_ratio_diff`, `travel_distance_diff` y
# `shootout_winrate_diff` se eliminaron por completo del pipeline en v2
# (aporte dentro del ruido o negativo); ya no tienen fila propia.
ABLATIONS: dict[str, list[str]] = {
    f"Completo ({len(FEATURE_COLS)} features)": list(FEATURE_COLS),
    "Sin xG": [c for c in FEATURE_COLS if not c.startswith("xg_")],
    "Sin squad_value": [c for c in FEATURE_COLS if c != "squad_value_diff"],
    "Sin features derivadas": [c for c in FEATURE_COLS if c not in DERIVED_FEATURE_COLS],
    "Sin penalty_share (marginal)": [c for c in FEATURE_COLS if c != "penalty_share_diff"],
    "Sin estáticas anacrónicas (xG+squad)": [c for c in FEATURE_COLS if c not in _ANACHRONISTIC],
    "Solo elo_diff": ["elo_diff"],
}


def _per_sample_losses(model, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[np.ndarray, float]:
    proba = np.clip(model.predict_proba(X_test), 1e-15, 1.0)
    losses = -np.log(proba[np.arange(len(y_test)), y_test])
    brier = float(np.mean([
        brier_score_loss((y_test == cls).astype(int), proba[:, i])
        for i, cls in enumerate(sorted(np.unique(y_test)))
    ]))
    return losses, brier


def _paired_bootstrap_delta(
    base: np.ndarray, other: np.ndarray, n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float, float, bool]:
    """Delta = mean(other - base) con IC95 bootstrap pareado por partido."""
    d = other - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


def run_ablation(
    df: pd.DataFrame,
    min_year: int | None = 2010,
    n_seeds: int = 10,
) -> pd.DataFrame:
    """
    Corrección v2 (rigor): cada configuración se reentrena con `n_seeds`
    semillas y se reporta media ± std del log-loss, más un bootstrap pareado
    (por partido de test, sobre la pérdida promediada entre semillas) del delta
    frente a la configuración completa. La columna `significant` marca si el
    IC95 del delta excluye el cero: solo esas filas soportan conclusiones.
    """
    df = df.dropna(subset=FEATURE_COLS + ["target"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if min_year is not None:
        df = df[df["date"].dt.year >= min_year]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    train_mask, val_mask, test_mask = temporal_split(df)
    y_all = df["target"].values.astype(int)
    tw_all = df["time_weight"].values.astype(np.float32) if "time_weight" in df.columns else None

    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    tw_train = tw_all[train_mask] if tw_all is not None else None
    weights_train = compute_combined_weights(y_train, tw_train)

    best_params_path = MODELS_DIR / "best_params_xgboost.json"
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))

    records = []
    avg_losses: dict[str, np.ndarray] = {}
    for label, cols in ABLATIONS.items():
        X = df[cols].astype(np.float32)
        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

        seed_lls, seed_briers, loss_vectors = [], [], []
        for seed in range(n_seeds):
            params = {**best_params, "random_state": seed}
            xgb = train_xgboost(X_train, y_train, weights_train, params)
            xgb_cal = calibrate_model(xgb, X_val, y_val, method="sigmoid")
            losses, brier = _per_sample_losses(xgb_cal, X_test, y_test)
            loss_vectors.append(losses)
            seed_lls.append(float(losses.mean()))
            seed_briers.append(brier)

        avg_losses[label] = np.mean(loss_vectors, axis=0)
        records.append({
            "config": label,
            "n_features": len(cols),
            "log_loss_mean": round(float(np.mean(seed_lls)), 4),
            "log_loss_std": round(float(np.std(seed_lls)), 4),
            "brier_mean": round(float(np.mean(seed_briers)), 4),
        })
        print(f"  {label:40s} | feat={len(cols)} | "
              f"LL={np.mean(seed_lls):.4f} +/- {np.std(seed_lls):.4f}")

    out = pd.DataFrame(records)
    base_label = out.iloc[0]["config"]
    deltas, los, his, sigs = [], [], [], []
    for label in out["config"]:
        if label == base_label:
            deltas.append(0.0); los.append(0.0); his.append(0.0); sigs.append(False)
            continue
        d, lo, hi, sig = _paired_bootstrap_delta(avg_losses[base_label], avg_losses[label])
        deltas.append(round(d, 4)); los.append(round(lo, 4)); his.append(round(hi, 4)); sigs.append(sig)
    out["delta_log_loss"] = deltas
    out["delta_ci_low"] = los
    out["delta_ci_high"] = his
    out["significant"] = sigs
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study WC2026")
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--seeds", type=int, default=10,
                        help="Reentrenos por configuración (default 10). Las "
                             "conclusiones solo se sostienen sobre filas con "
                             "significant=True.")
    args = parser.parse_args()
    min_year = args.min_year if args.min_year and args.min_year > 0 else None

    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    print(f"\n=== Ablation study (XGBoost calibrado, test >= 2022, {args.seeds} seeds) ===")
    result = run_ablation(df, min_year=min_year, n_seeds=args.seeds)

    out_path = PROCESSED_DIR / "ablation_results.csv"
    result.to_csv(out_path, index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nGuardado en {out_path}")
