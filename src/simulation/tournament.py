"""
Lógica del torneo FIFA Mundial 2026.
12 grupos de 4. Avanzan 2 primeros de cada grupo + 8 mejores terceros = 32 equipos.
Llaves eliminatorias R32 - R16 - QF - SF - Final.

Las fases trackeadas: "group_stage", "round_of_32", "round_of_16",
"quarterfinals", "semifinals", "final", "champion".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

_FIXTURE_CSV = Path(__file__).parents[2] / "data" / "raw" / "wc2026_fixture.csv"

HOST_COUNTRIES = {"United States", "Mexico", "Canada"}
PHASES = [
    "group_stage", "round_of_32", "round_of_16",
    "quarterfinals", "semifinals", "final", "champion",
]


def _load_groups(fixture_path: Path) -> dict[str, list[str]]:
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
    return table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)


def _symmetric_probs(predict_fn, t1: str, t2: str) -> np.ndarray:
    """
    Promedia P(t1 vs t2) con la versión invertida P(t2 vs t1) para eliminar
    el sesgo home/away cuando no hay localía real. Devuelve [P(win_t1),
    P(draw), P(win_t2)].
    """
    p_fwd = predict_fn(t1, t2)              # [home_win, draw, away_win] desde la perspectiva t1=home
    p_rev = predict_fn(t2, t1)              # [home_win, draw, away_win] desde la perspectiva t2=home
    p_t1_win = (p_fwd[0] + p_rev[2]) / 2.0
    p_draw   = (p_fwd[1] + p_rev[1]) / 2.0
    p_t2_win = (p_fwd[2] + p_rev[0]) / 2.0
    out = np.array([p_t1_win, p_draw, p_t2_win])
    out /= out.sum()
    return out


def _host_advantage_probs(predict_fn, t1: str, t2: str) -> np.ndarray:
    """
    Si exactamente uno de los dos equipos es anfitrión, lo usamos como home
    (sin promediar) para reflejar la localía real. Si ambos o ninguno son
    anfitriones, se usa el promedio simétrico.
    """
    t1_host = t1 in HOST_COUNTRIES
    t2_host = t2 in HOST_COUNTRIES
    if t1_host and not t2_host:
        return predict_fn(t1, t2)
    if t2_host and not t1_host:
        # Invertir orientación para devolver siempre [P(t1_win), P(draw), P(t2_win)]
        rev = predict_fn(t2, t1)
        return np.array([rev[2], rev[1], rev[0]])
    return _symmetric_probs(predict_fn, t1, t2)


def _sample_goals_poisson(
    t1: str,
    t2: str,
    team_xg: dict[str, dict[str, float]],
    league_xg: float = 1.25,
    rng=None,
) -> tuple[int, int]:
    """
    Modela goles con Poisson independiente: λ_t1 = xg_for(t1) * xg_against(t2) / league_xg.
    El outcome surge del marcador, no al revés. `rng` es un np.random.Generator
    (o el módulo np.random si None) para soportar streams reproducibles en paralelo.
    """
    rng = rng if rng is not None else np.random
    xg_for_1 = team_xg.get(t1, {}).get("xg_for", league_xg)
    xg_for_2 = team_xg.get(t2, {}).get("xg_for", league_xg)
    xg_ag_1  = team_xg.get(t1, {}).get("xg_against", league_xg)
    xg_ag_2  = team_xg.get(t2, {}).get("xg_against", league_xg)

    lam_1 = max(0.05, xg_for_1 * xg_ag_2 / league_xg)
    lam_2 = max(0.05, xg_for_2 * xg_ag_1 / league_xg)
    g1 = int(rng.poisson(lam_1))
    g2 = int(rng.poisson(lam_2))
    return g1, g2


def _outcome_from_probs(probs: np.ndarray, rng=None) -> int:
    """Devuelve 2=t1 wins, 1=draw, 0=t2 wins según probs=[t1, draw, t2]."""
    rng = rng if rng is not None else np.random
    return int(rng.choice([2, 1, 0], p=probs))


def simulate_group_stage(
    teams: list[str],
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng=None,
) -> pd.DataFrame:
    """
    Simula los 6 partidos del grupo. Los outcomes se sortean con probs
    simétricas (sin sesgo home/away); los goles se samplean con Poisson
    sobre xG del equipo, y se corrige el marcador si contradice el outcome.
    """
    team_xg = team_xg or {}
    standings = {
        t: {"points": 0, "gf": 0, "ga": 0}
        for t in teams
    }
    for t1, t2 in combinations(teams, 2):
        probs = _host_advantage_probs(predict_fn, t1, t2)
        outcome = _outcome_from_probs(probs, rng)
        g1, g2 = _sample_goals_poisson(t1, t2, team_xg, rng=rng)

        # Reconciliar marcador con outcome sorteado
        if outcome == 2 and g1 <= g2:
            g1 = g2 + 1
        elif outcome == 0 and g2 <= g1:
            g2 = g1 + 1
        elif outcome == 1 and g1 != g2:
            g2 = g1  # forzar empate al marcador del t1

        if outcome == 2:
            standings[t1]["points"] += 3
        elif outcome == 0:
            standings[t2]["points"] += 3
        else:
            standings[t1]["points"] += 1
            standings[t2]["points"] += 1

        standings[t1]["gf"] += g1
        standings[t1]["ga"] += g2
        standings[t2]["gf"] += g2
        standings[t2]["ga"] += g1

    table = pd.DataFrame([
        {"team": t, **v, "gd": v["gf"] - v["ga"]}
        for t, v in standings.items()
    ])
    return _points_tiebreak(table)


def select_best_thirds(all_group_results: dict[str, pd.DataFrame]) -> list[str]:
    thirds = []
    for group, table in all_group_results.items():
        if len(table) >= 3:
            row = table.iloc[2].to_dict()
            row["group"] = group
            thirds.append(row)

    if not thirds:
        return []

    thirds_df = pd.DataFrame(thirds)
    thirds_df = thirds_df.sort_values(["points", "gd", "gf"], ascending=False).head(8)
    return thirds_df["team"].tolist()


def build_knockout_bracket(
    group_results: dict[str, pd.DataFrame],
    best_thirds: list[str],
) -> list[tuple[str, str]]:
    slots: dict[str, str] = {}
    for group, table in group_results.items():
        slots[f"{group}1"] = table.iloc[0]["team"]
        slots[f"{group}2"] = table.iloc[1]["team"]
    for i, team in enumerate(best_thirds, 1):
        slots[f"3rd_{i}"] = team

    bracket = []
    for h_slot, a_slot in KNOCKOUT_BRACKET_ORDER:
        bracket.append((slots.get(h_slot, "TBD"), slots.get(a_slot, "TBD")))
    return bracket


def simulate_knockout_round(
    pairs: list[tuple[str, str]],
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng=None,
) -> list[str]:
    team_xg = team_xg or {}
    rng_ = rng if rng is not None else np.random
    winners = []
    for t1, t2 in pairs:
        if t1 == "TBD" or t2 == "TBD":
            winners.append(t1 if t2 == "TBD" else t2)
            continue
        probs = _host_advantage_probs(predict_fn, t1, t2)
        outcome = _outcome_from_probs(probs, rng)
        if outcome == 2:
            winners.append(t1)
        elif outcome == 0:
            winners.append(t2)
        else:
            winners.append(str(rng_.choice([t1, t2])))  # penalty shootout 50/50
    return winners


def simulate_full_tournament(
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng=None,
) -> dict:
    """
    Simula el torneo completo y devuelve un dict con los equipos que
    llegaron a cada fase y el campeón.

    Estructura: {
        "group_stage": set,        # 48 equipos
        "round_of_32": set,        # 32 equipos clasificados
        "round_of_16": set,        # 16 ganadores R32
        "quarterfinals": set,      # 8
        "semifinals": set,         # 4
        "final": set,              # 2
        "champion": str,
    }
    """
    progression = {phase: set() for phase in PHASES if phase != "champion"}
    progression["group_stage"] = set(ALL_TEAMS)

    group_results: dict[str, pd.DataFrame] = {}
    for group, teams in GROUPS_2026.items():
        group_results[group] = simulate_group_stage(teams, predict_fn, team_xg, rng)

    best_thirds = select_best_thirds(group_results)
    bracket = build_knockout_bracket(group_results, best_thirds)

    # round_of_32: los 32 que entran al bracket
    for h, a in bracket:
        if h != "TBD":
            progression["round_of_32"].add(h)
        if a != "TBD":
            progression["round_of_32"].add(a)

    # Avanzar fases hasta que quede 1 campeón
    current_pairs = bracket
    phase_after_pairs = ["round_of_16", "quarterfinals", "semifinals", "final"]
    phase_idx = 0

    while len(current_pairs) >= 1:
        winners = simulate_knockout_round(current_pairs, predict_fn, team_xg, rng)
        if phase_idx < len(phase_after_pairs):
            for w in winners:
                progression[phase_after_pairs[phase_idx]].add(w)
        phase_idx += 1

        if len(winners) == 1:
            progression["champion"] = winners[0]
            return progression
        current_pairs = list(zip(winners[::2], winners[1::2]))

    # Fallback defensivo
    progression["champion"] = "TBD"
    return progression
