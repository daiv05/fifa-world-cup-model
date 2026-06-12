"""
Harness de comparación entre dos corridas del pipeline (p.ej. baseline_v1 vs
actual). Es el criterio de aceptación de cada fase del plan de correcciones:
ningún cambio se da por bueno sin un delta de log-loss con IC bootstrap pareado.

Uso típico:
  python -m src.analysis.compare_runs \
      --a-dir data/processed/baseline_v1 --b-dir data/processed \
      --a-model xgboost_calibrated --b-model xgboost_calibrated

Cada lado usa su propio features.csv y su propio modelo; las features que el
modelo espera se leen de `feature_names_in_` (los modelos se ajustan con
DataFrames), así que los dos lados pueden tener esquemas distintos. El bootstrap
se parea por (date, home_team, away_team) sobre la intersección de partidos de
test, de modo que la comparación es manzanas-con-manzanas aunque cambien las
features.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2022-01-01")
N_BOOT = 2_000
BOOT_SEED = 0


def _model_feature_cols(model) -> list[str]:
    """Extrae los nombres de features con los que se ajustó el modelo."""
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):  # Pipeline (logreg)
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    if names is None:
        from src.features.features import FEATURE_COLS
        return list(FEATURE_COLS)
    return list(names)


def per_sample_log_loss(run_dir: Path, model_name: str) -> pd.DataFrame:
    """
    Devuelve un DataFrame indexado por (date, home_team, away_team) con la
    log-loss por partido del modelo sobre el test temporal (date >= 2022).
    """
    model = joblib.load(run_dir / "models" / f"{model_name}.joblib")
    cols = _model_feature_cols(model)

    df = pd.read_csv(run_dir / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=cols + ["target"])
    df = df[df["date"] >= TEST_START].copy()

    X = df[cols].astype(np.float32)
    y = df["target"].values.astype(int)
    proba = np.clip(model.predict_proba(X), 1e-15, 1.0)
    df["loss"] = -np.log(proba[np.arange(len(y)), y])

    out = df[["date", "home_team", "away_team", "loss"]].copy()
    # En caso de duplicados exactos (no debería haberlos), conservar el primero.
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
    return out.set_index(["date", "home_team", "away_team"])


def paired_bootstrap(
    loss_a: pd.Series,
    loss_b: pd.Series,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Δ = media(b) - media(a) con IC95 bootstrap pareado (negativo = b mejor)."""
    d = (loss_b - loss_a).values
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_matches": n,
        "ll_a": float(loss_a.mean()),
        "ll_b": float(loss_b.mean()),
        "delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def compare_models(
    a_dir: Path, b_dir: Path, a_model: str, b_model: str,
) -> dict:
    la = per_sample_log_loss(a_dir, a_model)["loss"]
    lb = per_sample_log_loss(b_dir, b_model)["loss"]
    common = la.index.intersection(lb.index)
    if len(common) == 0:
        raise ValueError("Sin partidos de test en común entre las dos corridas.")
    return paired_bootstrap(la.loc[common], lb.loc[common])


def compare_simulations(a_dir: Path, b_dir: Path, top: int = 10) -> pd.DataFrame | None:
    pa, pb = a_dir / "simulation_results.csv", b_dir / "simulation_results.csv"
    if not (pa.exists() and pb.exists()):
        return None
    a = pd.read_csv(pa)[["team", "champion_pct"]].rename(columns={"champion_pct": "pct_a"})
    b = pd.read_csv(pb)[["team", "champion_pct"]].rename(columns={"champion_pct": "pct_b"})
    m = a.merge(b, on="team", how="outer").fillna(0.0)
    m["delta_pp"] = (m["pct_b"] - m["pct_a"]).round(2)
    return (
        m.sort_values("pct_b", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación pareada entre corridas")
    parser.add_argument("--a-dir", type=str, default="data/processed/baseline_v1")
    parser.add_argument("--b-dir", type=str, default="data/processed")
    parser.add_argument("--a-model", type=str, default="xgboost_calibrated")
    parser.add_argument("--b-model", type=str, default="xgboost_calibrated")
    args = parser.parse_args()

    a_dir, b_dir = Path(args.a_dir), Path(args.b_dir)

    print(f"A: {a_dir} :: {args.a_model}")
    print(f"B: {b_dir} :: {args.b_model}")

    r = compare_models(a_dir, b_dir, args.a_model, args.b_model)
    verdict = "SIGNIFICATIVO" if r["significant"] else "no significativo"
    print(
        f"\nTest pareado sobre {r['n_matches']:,} partidos (date >= 2022):\n"
        f"  LL A = {r['ll_a']:.4f} | LL B = {r['ll_b']:.4f}\n"
        f"  dLL (B-A) = {r['delta']:+.4f}  IC95 [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        f"  -> {verdict}"
    )
    if r["delta"] < 0:
        print("  B es mejor (menor log-loss).")
    elif r["delta"] > 0:
        print("  A es mejor (menor log-loss).")

    sims = compare_simulations(a_dir, b_dir)
    if sims is not None:
        print("\nDelta P(campeon) top-10 (pp, B - A):")
        print(sims.to_string(index=False))
