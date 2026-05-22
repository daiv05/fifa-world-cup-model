"""
Motor de simulación Monte Carlo para el Mundial 2026.
Estrategia por capas: NumPy vectorizado → numba @njit si es necesario.
"""

import sys
import argparse
import time
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# Garantiza que repository/ esté en sys.path sin importar desde dónde se ejecute
_repo_root = Path(__file__).parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.simulation.tournament import GROUPS_2026, ALL_TEAMS, simulate_full_tournament

RESULTS_DIR = Path(__file__).parents[2] / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_predict_fn(model, team_features: pd.DataFrame):
    """
    Construye una función de predicción a partir de un modelo sklearn/XGBoost.
    Precomputa todas las probabilidades de enfrentamientos posibles en un único
    batch para minimizar llamadas al modelo durante la simulación.
    predict_fn(home, away, features) -> np.array([p_away, p_draw, p_home])
    """
    feat_map = team_features.set_index("team").to_dict("index") if not team_features.empty else {}
    teams = list(feat_map.keys())

    # Build feature matrix for all directed pairs in one shot
    pairs = [(h, a) for h in teams for a in teams if h != a]
    rows = []
    for home, away in pairs:
        h = feat_map[home]
        a = feat_map[away]
        rows.append([
            h.get("elo", 1500.0) - a.get("elo", 1500.0),
            np.log1p(h.get("squad_value_eur", 1e7)) - np.log1p(a.get("squad_value_eur", 1e7)),
            h.get("xg_for", 1.2) - a.get("xg_for", 1.2),
            h.get("xg_against", 1.2) - a.get("xg_against", 1.2),
            h.get("travel_distance", 5000.0),
            a.get("travel_distance", 5000.0),
        ])

    # Usar DataFrame con nombres de columna para evitar el UserWarning de LightGBM
    # "X does not have valid feature names, but LGBMClassifier was fitted with feature names"
    _FEATURE_COLS = ["elo_diff", "squad_value_diff", "xg_avg_for",
                     "xg_avg_against", "travel_distance_home", "travel_distance_away"]
    X_all = pd.DataFrame(rows, columns=_FEATURE_COLS).astype(np.float32)

    # Model returns [p_away, p_draw, p_home] (class order 0,1,2).
    # tournament.py expects [p_home, p_draw, p_away] (outcome choice [2,1,0]).
    # Reorder: flip index 0 ↔ 2 so the cache stores [p_home, p_draw, p_away].
    probas_raw = model.predict_proba(X_all)
    probas = probas_raw[:, [2, 1, 0]]

    # Cache: (home, away) -> probability array [p_home, p_draw, p_away]
    cache: dict[tuple[str, str], np.ndarray] = {
        pair: probas[i] for i, pair in enumerate(pairs)
    }
    default_proba = np.array([1 / 3, 1 / 3, 1 / 3])

    def predict_fn(home: str, away: str, _features: dict = None) -> np.ndarray:
        # Returns [p_home, p_draw, p_away] as expected by tournament.py
        return cache.get((home, away), default_proba)

    return predict_fn


def _dummy_predict_fn(home: str, away: str, _features: dict = None) -> np.ndarray:
    """Predictor de fallback con probabilidades fijas [p_home=0.40, p_draw=0.25, p_away=0.35]."""
    return np.array([0.40, 0.25, 0.35])


def run_simulation(
    n_iterations: int = 10_000,
    model=None,
    team_features: pd.DataFrame | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Ejecuta la simulación Monte Carlo del torneo completo.

    Devuelve DataFrame con columnas:
      team, champion_count, champion_pct, champion_ci_low, champion_ci_high
    """
    np.random.seed(seed)

    if model is not None and team_features is not None:
        predict_fn = build_predict_fn(model, team_features)
    else:
        print("Usando predictor dummy (ELO simple). Proporciona un modelo para resultados reales.")
        predict_fn = _dummy_predict_fn

    champions: Counter = Counter()

    # Mide la primera iteración para estimar si se necesita numba
    start = time.perf_counter()
    _ = simulate_full_tournament(predict_fn, {})
    first_iter_ms = (time.perf_counter() - start) * 1000

    estimated_total_s = first_iter_ms * n_iterations / 1000
    print(f"Primera iteración: {first_iter_ms:.1f}ms — estimado total: {estimated_total_s:.0f}s")

    if estimated_total_s > 30:
        print("Tiempo estimado > 30s. Considera activar numba (ver simulate_numba_ready.py)")

    for _ in tqdm(range(n_iterations), desc="Simulando torneos"):
        champion = simulate_full_tournament(predict_fn, {})
        champions[champion] += 1

    results = []
    for team in ALL_TEAMS:
        c = champions[team]
        results.append({
            "team": team,
            "champion_count": c,
            "champion_pct": round(c / n_iterations * 100, 2),
            # Bootstrap CI: simulate 10_000 tournament seasons and take 5th/95th percentile
            # of the champion proportion. Uses binomial(n, p) / n to get proportions.
            "champion_ci_low": round(np.percentile(
                np.random.binomial(n_iterations, max(c / n_iterations, 1e-9), 10_000) / n_iterations * 100, 5
            ), 2),
            "champion_ci_high": round(np.percentile(
                np.random.binomial(n_iterations, min(c / n_iterations, 1 - 1e-9), 10_000) / n_iterations * 100, 95
            ), 2),
        })

    df = pd.DataFrame(results).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, filename: str = "simulation_results.csv") -> Path:
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"Resultados guardados en {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor Monte Carlo — Mundial 2026")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--model", type=str, default=None, help="Nombre del modelo en data/processed/models/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = None
    team_features = None

    if args.model:
        from src.models.train import load_model
        model = load_model(args.model)
        features_path = RESULTS_DIR / "team_features.csv"
        if features_path.exists():
            team_features = pd.read_csv(features_path)

    print(f"\nEjecutando {args.iterations:,} iteraciones del Mundial 2026...")
    results = run_simulation(args.iterations, model=model, team_features=team_features, seed=args.seed)

    print("\n=== TOP 10 candidatos al campeonato ===")
    print(results.head(10).to_string(index=False))

    save_results(results)
