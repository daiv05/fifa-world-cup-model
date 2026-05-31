"""
Motor de simulación Monte Carlo para el Mundial 2026.
"""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from scipy.stats import beta
from joblib import Parallel, delayed

from src.simulation.tournament import (
    GROUPS_2026, ALL_TEAMS, PHASES,
    simulate_full_tournament,
)
from src.features.features import FEATURE_COLS

RESULTS_DIR = Path(__file__).parents[2] / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _team_pair_to_feature_dict(h: dict, a: dict) -> dict[str, float]:
    """
    Construye el vector de features para un par (home `h`, away `a`) a partir de
    sus filas en team_features, como un dict {columna -> valor}. El DataFrame
    final se reindexa por `FEATURE_COLS`, de modo que el orden de columnas en la
    simulación SIEMPRE coincide con el del entrenamiento aunque cambie el set de
    features (evita el bug clásico de desalineación train/inferencia).

    Convención de signo de los diffs: home - away (salvo ranking_diff, que es
    away - home por diseño histórico: positivo = local mejor rankeado).

    travel_distance_diff usa host_distance (distancia a la sede más cercana del
    Mundial) como away - home, misma convención que a nivel de partido.
    """
    return {
        "elo_diff": h.get("elo", 1500.0) - a.get("elo", 1500.0),
        "squad_value_diff": (
            np.log1p(h.get("squad_value_eur", 1e7))
            - np.log1p(a.get("squad_value_eur", 1e7))
        ),
        "xg_avg_for": h.get("xg_for", 1.2) - a.get("xg_for", 1.2),
        "xg_avg_against": h.get("xg_against", 1.2) - a.get("xg_against", 1.2),
        "travel_distance_diff": a.get("host_distance", 5000.0) - h.get("host_distance", 5000.0),
        "ranking_diff": a.get("rank", 78) - h.get("rank", 78),
        # Features derivadas (diff home - away), desde columnas per-equipo de team_features.
        "penalty_share_diff": h.get("penalty_share", 0.07) - a.get("penalty_share", 0.07),
        "striker_concentration_diff": (
            h.get("striker_concentration", 0.4) - a.get("striker_concentration", 0.4)
        ),
        "shootout_winrate_diff": h.get("shootout_winrate", 0.5) - a.get("shootout_winrate", 0.5),
    }


def build_predict_fn(model, team_features: pd.DataFrame):
    """
    Precalcula `predict_proba` para los 48*47 = 2256 pares ordenados de
    equipos del torneo. Devuelve una función que hace lookup O(1) por par.
    `team_features` debe traer las columnas por equipo: elo, squad_value_eur,
    xg_for, xg_against, host_distance, rank.
    """
    feat_map = team_features.set_index("team").to_dict("index") if not team_features.empty else {}
    teams = list(feat_map.keys())
    pairs = [(h, a) for h in teams for a in teams if h != a]
    rows = [
        _team_pair_to_feature_dict(feat_map[home], feat_map[away])
        for home, away in pairs
    ]

    # Reindexar por FEATURE_COLS garantiza el orden y la identidad de columnas.
    X_all = pd.DataFrame(rows)[FEATURE_COLS].astype(np.float32)
    probas_raw = model.predict_proba(X_all)
    # El modelo devuelve [class0=away_win, class1=draw, class2=home_win].
    # `predict_fn` debe devolver [home_win, draw, away_win].
    probas = probas_raw[:, [2, 1, 0]]

    cache: dict[tuple[str, str], np.ndarray] = {
        pair: probas[i] for i, pair in enumerate(pairs)
    }
    default_proba = np.array([1 / 3, 1 / 3, 1 / 3])

    def predict_fn(home: str, away: str) -> np.ndarray:
        return cache.get((home, away), default_proba)

    return predict_fn


def _build_team_xg(team_features: pd.DataFrame) -> dict[str, dict[str, float]]:
    if team_features is None or team_features.empty:
        return {}
    return team_features.set_index("team")[["xg_for", "xg_against"]].to_dict("index")


def _dummy_predict_fn(home: str, away: str) -> np.ndarray:
    return np.array([0.40, 0.25, 0.35])


def _clopper_pearson(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC bilateral Clopper-Pearson (1-alpha) sobre proporción c/n, en %."""
    lo = beta.ppf(alpha / 2, c, n - c + 1) if c > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, c + 1, n - c) if c < n else 1.0
    return float(lo * 100), float(hi * 100)


def _simulate_block(n_block: int, predict_fn, team_xg, seed_seq) -> dict[str, dict[str, int]]:
    """
    Corre `n_block` torneos con un Generator propio sembrado de `seed_seq`
    (np.random.SeedSequence). Devuelve los conteos por fase. Cada bloque es
    independiente y determinista dado su seed_seq -> el resultado total es
    reproducible para CUALQUIER número de workers.
    """
    rng = np.random.default_rng(seed_seq)
    counts: dict[str, dict[str, int]] = {p: defaultdict(int) for p in PHASES}
    for _ in range(n_block):
        result = simulate_full_tournament(predict_fn, team_xg, rng=rng)
        for phase in PHASES:
            if phase == "champion":
                counts["champion"][result["champion"]] += 1
            else:
                for team in result[phase]:
                    counts[phase][team] += 1
    return counts


def run_simulation(
    n_iterations: int = 10_000,
    model=None,
    team_features: pd.DataFrame | None = None,
    seed: int = 42,
    n_jobs: int = -1,
    n_blocks: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (champions_df, progression_df):
      - champions_df: team, champion_count, champion_pct, champion_ci_low/high
      - progression_df: team + porcentaje + IC por fase (group_stage, R32, R16, QF, SF, Final, Champion)

    Las iteraciones (independientes) se reparten en `n_blocks` bloques fijos y se
    corren en paralelo (`n_jobs`). Cada bloque usa un stream RNG independiente
    derivado de `SeedSequence(seed).spawn(n_blocks)`, así que el resultado es
    reproducible sin importar cuántos workers se usen.
    """
    if model is not None and team_features is not None:
        predict_fn = build_predict_fn(model, team_features)
        team_xg = _build_team_xg(team_features)
    else:
        print("Usando predictor dummy. Proporciona un modelo para resultados reales.")
        predict_fn = _dummy_predict_fn
        team_xg = {}

    n_blocks = max(1, min(n_blocks, n_iterations))
    block_sizes = [len(b) for b in np.array_split(np.arange(n_iterations), n_blocks)]
    seed_seqs = np.random.SeedSequence(seed).spawn(n_blocks)

    block_results = Parallel(n_jobs=n_jobs)(
        delayed(_simulate_block)(bs, predict_fn, team_xg, ss)
        for bs, ss in zip(block_sizes, seed_seqs)
    )

    # Merge de conteos por fase.
    phase_counts: dict[str, dict[str, int]] = {p: defaultdict(int) for p in PHASES}
    for counts in block_results:
        for phase in PHASES:
            for team, n in counts[phase].items():
                phase_counts[phase][team] += n

    # ----- Tabla de campeones (con IC Clopper-Pearson) -----
    champ_rows = []
    for team in ALL_TEAMS:
        c = phase_counts["champion"][team]
        lo, hi = _clopper_pearson(c, n_iterations)
        champ_rows.append({
            "team": team,
            "champion_count": c,
            "champion_pct": round(c / n_iterations * 100, 2),
            "champion_ci_low": round(lo, 2),
            "champion_ci_high": round(hi, 2),
        })
    champions_df = (
        pd.DataFrame(champ_rows)
        .sort_values("champion_pct", ascending=False)
        .reset_index(drop=True)
    )

    # ----- Tabla de avance por fase -----
    prog_rows = []
    for team in ALL_TEAMS:
        row = {"team": team}
        for phase in PHASES:
            c = phase_counts[phase][team]
            lo, hi = _clopper_pearson(c, n_iterations)
            row[f"{phase}_pct"] = round(c / n_iterations * 100, 2)
            row[f"{phase}_ci_low"] = round(lo, 2)
            row[f"{phase}_ci_high"] = round(hi, 2)
        prog_rows.append(row)
    progression_df = (
        pd.DataFrame(prog_rows)
        .sort_values("champion_pct", ascending=False)
        .reset_index(drop=True)
    )

    return champions_df, progression_df


def save_results(
    champions_df: pd.DataFrame,
    progression_df: pd.DataFrame,
) -> tuple[Path, Path]:
    p_champ = RESULTS_DIR / "simulation_results.csv"
    p_prog = RESULTS_DIR / "tournament_progression.csv"
    champions_df.to_csv(p_champ, index=False)
    progression_df.to_csv(p_prog, index=False)
    print(f"Campeones    - {p_champ}")
    print(f"Progresión   - {p_prog}")
    return p_champ, p_prog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor Monte Carlo - Mundial 2026")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--model", type=str, default=None,
                        help="Nombre del modelo en data/processed/models/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Workers para el Monte Carlo (default -1 = todos los cores). "
                             "El resultado es reproducible para cualquier valor.")
    args = parser.parse_args()

    model = None
    team_features = None
    if args.model:
        from src.models.train import load_model, assert_model_feature_count
        model = load_model(args.model)
        assert_model_feature_count(model, name=args.model)
        features_path = RESULTS_DIR / "team_features.csv"
        if features_path.exists():
            team_features = pd.read_csv(features_path)

    print(f"\nEjecutando {args.iterations:,} iteraciones del Mundial 2026...")
    champions_df, progression_df = run_simulation(
        args.iterations,
        model=model,
        team_features=team_features,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    print("\n=== TOP 10 candidatos al campeonato ===")
    print(champions_df.head(10).to_string(index=False))

    print("\n=== TOP 10 - Probabilidad de llegar a la final ===")
    print(
        progression_df[["team", "final_pct", "champion_pct"]]
        .sort_values("final_pct", ascending=False)
        .head(10).to_string(index=False)
    )

    save_results(champions_df, progression_df)
