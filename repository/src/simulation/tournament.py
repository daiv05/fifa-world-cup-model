"""
Lógica del torneo FIFA Mundial 2026.
12 grupos de 4 equipos. Avanzan los 2 primeros de cada grupo + 8 mejores terceros (32 total).
Llaves eliminatorias de 32 hasta la final.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

_FIXTURE_CSV = Path(__file__).parents[2] / "data" / "raw" / "wc2026_fixture.csv"


def _load_groups(fixture_path: Path) -> dict[str, list[str]]:
    """
    Carga los grupos del fixture CSV y aplica estandarización de nombres de equipo.
    Los nombres pasan por TEAM_NAME_ALIASES ("USA" → "United States", etc.).
    Para actualizar el torneo: reemplaza data/raw/wc2026_fixture.csv y re-ejecuta.
    """
    from src.data.data_loader import load_wc2026_fixture
    df = load_wc2026_fixture(fixture_path)
    groups: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        groups.setdefault(row["group"], []).append(row["team"])
    return groups


GROUPS_2026: dict[str, list[str]] = _load_groups(_FIXTURE_CSV)

ALL_TEAMS: list[str] = [team for teams in GROUPS_2026.values() for team in teams]

KNOCKOUT_BRACKET_ORDER = [
    ("A1", "B2"), ("C1", "D2"), ("E1", "F2"), ("G1", "H2"),
    ("I1", "J2"), ("K1", "L2"), ("A2", "B1"), ("C2", "D1"),
    ("E2", "F1"), ("G2", "H1"), ("I2", "J1"), ("K2", "L1"),
    ("3rd_1", "3rd_2"), ("3rd_3", "3rd_4"), ("3rd_5", "3rd_6"), ("3rd_7", "3rd_8"),
]


def _points_tiebreak(table: pd.DataFrame) -> pd.DataFrame:
    """Ordena la tabla de grupo por: puntos → diferencia de goles → goles a favor."""
    return table.sort_values(
        ["points", "gd", "gf"], ascending=False
    ).reset_index(drop=True)


def simulate_group_stage(
    teams: list[str],
    predict_fn,
    group_features: dict,
) -> pd.DataFrame:
    """
    Simula los 6 partidos de un grupo de 4 equipos.

    Parámetros
    ----------
    teams        : 4 equipos del grupo
    predict_fn   : función(home, away, features) -> np.array([p_home_win, p_draw, p_away_win])
    group_features : dict con features actuales por equipo

    Devuelve
    --------
    DataFrame con columnas [team, points, gw, gd_count, gl, gf, ga, gd]
    ordenado de 1° a 4°.
    """
    standings = {
        t: {"points": 0, "gw": 0, "gd_count": 0, "gl": 0, "gf": 0, "ga": 0}
        for t in teams
    }

    for home, away in combinations(teams, 2):
        probs = predict_fn(home, away, group_features)
        outcome = np.random.choice([2, 1, 0], p=probs)

        if outcome == 2:
            standings[home]["points"] += 3
            standings[home]["gw"] += 1
            standings[away]["gl"] += 1
            hg, ag = _sample_goals(probs[2]), _sample_goals(probs[0])
            if hg <= ag:
                hg = ag + 1
        elif outcome == 0:
            standings[away]["points"] += 3
            standings[away]["gw"] += 1
            standings[home]["gl"] += 1
            hg, ag = _sample_goals(probs[2]), _sample_goals(probs[0])
            if ag <= hg:
                ag = hg + 1
        else:
            standings[home]["points"] += 1
            standings[away]["points"] += 1
            standings[home]["gd_count"] += 1
            standings[away]["gd_count"] += 1
            hg = ag = _sample_goals(0.4)

        standings[home]["gf"] += hg
        standings[home]["ga"] += ag
        standings[away]["gf"] += ag
        standings[away]["ga"] += hg

    table = pd.DataFrame([
        {"team": t, **v, "gd": v["gf"] - v["ga"]}
        for t, v in standings.items()
    ])
    return _points_tiebreak(table)


def _sample_goals(win_prob: float) -> int:
    """Muestra goles de una distribución Poisson simplificada según la prob de ganar."""
    lam = max(0.3, 1.0 + win_prob * 1.5)
    return int(np.random.poisson(lam))


def select_best_thirds(all_group_results: dict[str, pd.DataFrame]) -> list[str]:
    """
    Selecciona los 8 mejores terceros entre los 12 grupos.
    Criterio FIFA: puntos → dif. goles → goles a favor.
    """
    thirds = []
    for group, table in all_group_results.items():
        if len(table) >= 3:
            row = table.iloc[2].to_dict()
            row["group"] = group
            thirds.append(row)

    thirds_df = pd.DataFrame(thirds)
    thirds_df = thirds_df.sort_values(
        ["points", "gd", "gf"], ascending=False
    ).head(8)
    return thirds_df["team"].tolist()


def build_knockout_bracket(
    group_results: dict[str, pd.DataFrame],
    best_thirds: list[str],
) -> list[tuple[str, str]]:
    """
    Construye las 16 llaves del Round of 32 a partir de los clasificados.
    Devuelve lista de (equipo_local, equipo_visitante).
    """
    slots: dict[str, str] = {}
    for group, table in group_results.items():
        slots[f"{group}1"] = table.iloc[0]["team"]
        slots[f"{group}2"] = table.iloc[1]["team"]
    for i, team in enumerate(best_thirds, 1):
        slots[f"3rd_{i}"] = team

    bracket = []
    for h_slot, a_slot in KNOCKOUT_BRACKET_ORDER:
        home = slots.get(h_slot, "TBD")
        away = slots.get(a_slot, "TBD")
        bracket.append((home, away))
    return bracket


def simulate_knockout_round(
    bracket: list[tuple[str, str]],
    predict_fn,
    features: dict,
) -> list[str]:
    """
    Simula una ronda eliminatoria. En caso de empate hay tiempo extra y penaltis (50/50).
    Devuelve la lista de ganadores (siguiente ronda).
    """
    winners = []
    for home, away in bracket:
        if home == "TBD" or away == "TBD":
            winners.append(home if away == "TBD" else away)
            continue

        probs = predict_fn(home, away, features)
        outcome = np.random.choice([2, 1, 0], p=probs)
        if outcome == 2:
            winners.append(home)
        elif outcome == 0:
            winners.append(away)
        else:
            winners.append(np.random.choice([home, away]))

    return winners


def simulate_full_tournament(predict_fn, features: dict) -> str:
    """
    Simula un torneo completo del Mundial 2026.
    Devuelve el nombre del equipo campeón.
    """
    all_group_results: dict[str, pd.DataFrame] = {}
    for group, teams in GROUPS_2026.items():
        all_group_results[group] = simulate_group_stage(teams, predict_fn, features)

    best_thirds = select_best_thirds(all_group_results)
    bracket = build_knockout_bracket(all_group_results, best_thirds)

    remaining = bracket
    while len(remaining) > 1:
        winners = simulate_knockout_round(remaining, predict_fn, features)
        remaining = list(zip(winners[::2], winners[1::2]))

    if len(remaining) == 1:
        home, away = remaining[0]
        probs = predict_fn(home, away, features)
        outcome = np.random.choice([2, 1, 0], p=probs)
        if outcome == 2:
            return home
        if outcome == 0:
            return away
        return np.random.choice([home, away])

    return remaining[0] if isinstance(remaining[0], str) else remaining[0][0]
