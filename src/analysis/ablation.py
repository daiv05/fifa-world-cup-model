"""
Ablation study: impacto marginal de grupos de features sobre Log-Loss y Brier.

Para aislar la contribucion de cada grupo se mantienen fijos el split temporal,
los pesos (time decay + balanced) y los hiperparametros ya optimizados del
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

# Estudio de ablación. Las features de aporte MARGINAL (según leave-one-out:
# travel_distance_diff, penalty_share_diff, shootout_winrate_diff) se documentan
# con su propia fila "Sin <feature>" para cuantificar su impacto individual.
# `late_goal_ratio_diff` se eliminó por completo del pipeline (era ruido neto).
ABLATIONS: dict[str, list[str]] = {
    f"Completo ({len(FEATURE_COLS)} features)": list(FEATURE_COLS),
    "Sin xG": [c for c in FEATURE_COLS if not c.startswith("xg_")],
    "Sin squad_value": [c for c in FEATURE_COLS if c != "squad_value_diff"],
    "Sin features derivadas": [c for c in FEATURE_COLS if c not in DERIVED_FEATURE_COLS],
    "Sin travel_distance (marginal)": [c for c in FEATURE_COLS if c != "travel_distance_diff"],
    "Sin penalty_share (marginal)": [c for c in FEATURE_COLS if c != "penalty_share_diff"],
    "Sin shootout (marginal)": [c for c in FEATURE_COLS if c != "shootout_winrate_diff"],
    "Sin estáticas anacrónicas (xG+squad)": [c for c in FEATURE_COLS if c not in _ANACHRONISTIC],
}


def _metrics(model, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[float, float]:
    proba = model.predict_proba(X_test)
    ll = log_loss(y_test, proba)
    brier = float(np.mean([
        brier_score_loss((y_test == cls).astype(int), proba[:, i])
        for i, cls in enumerate(sorted(np.unique(y_test)))
    ]))
    return float(ll), brier


def run_ablation(df: pd.DataFrame, min_year: int | None = 2010) -> pd.DataFrame:
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
    for label, cols in ABLATIONS.items():
        X = df[cols].astype(np.float32)
        X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

        xgb = train_xgboost(X_train, y_train, weights_train, best_params)
        xgb_cal = calibrate_model(xgb, X_val, y_val, method="sigmoid")
        ll, brier = _metrics(xgb_cal, X_test, y_test)
        records.append({
            "config": label,
            "n_features": len(cols),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
        })
        print(f"  {label:24s} | features={len(cols)} | LogLoss={ll:.4f} | Brier={brier:.4f}")

    out = pd.DataFrame(records)
    base = out.iloc[0]
    out["delta_log_loss"] = (out["log_loss"] - base["log_loss"]).round(4)
    out["delta_brier"] = (out["brier_score"] - base["brier_score"]).round(4)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study WC2026")
    parser.add_argument("--min-year", type=int, default=2010)
    args = parser.parse_args()
    min_year = args.min_year if args.min_year and args.min_year > 0 else None

    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    print("\n=== Ablation study (XGBoost calibrado, test >= 2022) ===")
    result = run_ablation(df, min_year=min_year)

    out_path = PROCESSED_DIR / "ablation_results.csv"
    result.to_csv(out_path, index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nGuardado en {out_path}")
