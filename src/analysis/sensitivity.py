"""
Análisis de sensibilidad: simula el efecto de "lesiones" reduciendo el
squad_value_eur de un equipo en un porcentaje dado, y compara la
probabilidad de campeonato resultante contra el escenario base.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.simulation.simulate import run_simulation, RESULTS_DIR


def simulate_injury_scenario(
    team: str,
    squad_reduction_pct: float,
    base_team_features: pd.DataFrame,
    model,
    n_iterations: int = 10_000,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Reduce `squad_value_eur` del `team` en `squad_reduction_pct` (e.g. 0.30
    para -30%), corre la simulación y devuelve el DataFrame de campeones.
    """
    tf = base_team_features.copy()
    # Ensure float dtype so percentage reductions do not raise on assignment.
    tf["squad_value_eur"] = tf["squad_value_eur"].astype(float)
    mask = tf["team"] == team
    if not mask.any():
        raise ValueError(f"Equipo no encontrado en team_features: {team}")
    tf.loc[mask, "squad_value_eur"] = tf.loc[mask, "squad_value_eur"] * (1 - squad_reduction_pct)

    champions_df, _ = run_simulation(
        n_iterations=n_iterations, model=model, team_features=tf, seed=seed, n_jobs=n_jobs,
    )
    return champions_df


def run_sensitivity_top_n(
    base_champions_df: pd.DataFrame,
    base_team_features: pd.DataFrame,
    model,
    top_n: int = 5,
    reduction: float = 0.30,
    n_iterations: int = 10_000,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Aplica reducción de squad a cada uno de los `top_n` candidatos y
    compara su nuevo P(campeón) con el base."""
    base_pct = base_champions_df.set_index("team")["champion_pct"].to_dict()
    top_teams = base_champions_df.head(top_n)["team"].tolist()
    rows = []
    for team in top_teams:
        print(f"\n--- Escenario: {team} con squad -{int(reduction*100)}% ---")
        scenario_df = simulate_injury_scenario(
            team, reduction, base_team_features, model,
            n_iterations=n_iterations, seed=seed, n_jobs=n_jobs,
        )
        new_pct = scenario_df.set_index("team")["champion_pct"].get(team, 0.0)
        rows.append({
            "team": team,
            "squad_reduction_pct": reduction,
            "champion_pct_base": base_pct.get(team, 0.0),
            "champion_pct_injury": new_pct,
            "delta_champion_pct": round(new_pct - base_pct.get(team, 0.0), 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensibilidad a lesiones - top 5")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--model", type=str, default="xgboost_calibrated")
    parser.add_argument("--reduction", type=float, default=0.30)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Workers para el Monte Carlo (default -1 = todos los cores).")
    args = parser.parse_args()

    from src.models.train import load_model
    model = load_model(args.model)

    tf_path = RESULTS_DIR / "team_features.csv"
    if not tf_path.exists():
        raise SystemExit(f"Falta {tf_path}. Ejecuta `python -m src.features.features` primero.")
    team_features = pd.read_csv(tf_path)

    # Escenario base
    base_path = RESULTS_DIR / "simulation_results.csv"
    if base_path.exists():
        base_champ = pd.read_csv(base_path)
        print(f"Usando escenario base desde {base_path}")
    else:
        print("Generando escenario base...")
        base_champ, _ = run_simulation(
            n_iterations=args.iterations, model=model,
            team_features=team_features, seed=args.seed, n_jobs=args.n_jobs,
        )

    df = run_sensitivity_top_n(
        base_champ, team_features, model,
        top_n=args.top_n, reduction=args.reduction,
        n_iterations=args.iterations, seed=args.seed, n_jobs=args.n_jobs,
    )
    print("\n=== Resultado ===")
    print(df.to_string(index=False))

    out_path = RESULTS_DIR / "sensitivity_injuries.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")
