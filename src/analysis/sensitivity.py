"""
Análisis de sensibilidad: simula el efecto de una crisis de lesiones sobre un
equipo y compara su P(campeón) contra el escenario base.

Corrección v2: la versión anterior solo perturbaba squad_value_eur — la
feature de MENOR importancia SHAP del modelo — por lo que el resultado
("efecto modesto") estaba predeterminado por el diseño. El escenario v2
perturba conjuntamente los tres canales por los que una crisis de lesiones
afecta al equipo en el espacio de features del modelo:
  - squad_value_eur: -reduction (default -30%)
  - xg_for: -10% (menor producción ofensiva sin titulares)
  - elo: -25 puntos (aprox. del impacto de bajas clave en ratings de equipo)
Sigue siendo un stress test agregado, NO un modelado estructural de bajas
individuales (limitación documentada).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.simulation.simulate import run_simulation, RESULTS_DIR


XG_REDUCTION = 0.10
ELO_REDUCTION_PTS = 25.0


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
    Aplica el escenario de lesiones conjunto al `team` (squad -reduction,
    xg_for -10%, elo -25 pts), corre la simulación y devuelve los campeones.
    """
    tf = base_team_features.copy()
    for col in ("squad_value_eur", "xg_for", "elo"):
        tf[col] = tf[col].astype(float)
    mask = tf["team"] == team
    if not mask.any():
        raise ValueError(f"Equipo no encontrado en team_features: {team}")
    tf.loc[mask, "squad_value_eur"] *= (1 - squad_reduction_pct)
    tf.loc[mask, "xg_for"] *= (1 - XG_REDUCTION)
    tf.loc[mask, "elo"] -= ELO_REDUCTION_PTS

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
    parser.add_argument("--model", type=str, default="best",
                        help="'best' usa la selección por validación (best_model.json).")
    parser.add_argument("--reduction", type=float, default=0.30)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Workers para el Monte Carlo (default -1 = todos los cores).")
    args = parser.parse_args()

    from src.models.train import load_model
    model_name = args.model
    if model_name == "best":
        import json
        pointer = RESULTS_DIR / "models" / "best_model.json"
        if pointer.exists():
            model_name = json.loads(pointer.read_text(encoding="utf-8"))["best_model"]
            print(f"Modelo seleccionado por validación: {model_name}")
        else:
            model_name = "xgboost_calibrated"
    model = load_model(model_name)

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
