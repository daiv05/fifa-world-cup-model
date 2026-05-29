"""
Lógica del torneo FIFA Mundial 2026.
12 grupos de 4. Avanzan 2 primeros de cada grupo + 8 mejores terceros = 32 equipos.
Llaves eliminatorias R32 - R16 - QF - SF - Final.

Las fases trackeadas: "group_stage", "round_of_32", "round_of_16",
"quarterfinals", "semifinals", "final", "champion".

Diseño de rendimiento y reproducibilidad
----------------------------------------
El núcleo de simulación (funciones `_*_fast`) trabaja con estructuras nativas
(dicts/listas) y un `numpy.random.Generator` explícito, evitando dos cuellos
de botella: la construcción de `DataFrame` por grupo en el bucle caliente y la
dependencia del RNG global mutable de NumPy. La API pública
(`simulate_group_stage`, `select_best_thirds`, `simulate_full_tournament`) se
conserva como capa delgada sobre ese núcleo para compatibilidad y pruebas.
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

# Orden de desempate FIFA 2026: puntos -> diferencia de goles -> goles a favor.
def _standing_sort_key(s: dict) -> tuple[int, int, int]:
    return (s["points"], s["gd"], s["gf"])


# --------------------------------------------------------------------------- #
# Probabilidades (puras, sin RNG)
# --------------------------------------------------------------------------- #
def _symmetric_probs(predict_fn, t1: str, t2: str) -> np.ndarray:
    """
    Promedia P(t1 vs t2) con la versión invertida P(t2 vs t1) para eliminar
    el sesgo home/away cuando no hay localía real. Devuelve [P(win_t1),
    P(draw), P(win_t2)].
    """
    p_fwd = predict_fn(t1, t2)              # [home_win, draw, away_win] con t1=home
    p_rev = predict_fn(t2, t1)              # [home_win, draw, away_win] con t2=home
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
    anfitriones, se usa el promedio simétrico. Devuelve [P(t1), P(draw), P(t2)].
    """
    t1_host = t1 in HOST_COUNTRIES
    t2_host = t2 in HOST_COUNTRIES
    if t1_host and not t2_host:
        return predict_fn(t1, t2)
    if t2_host and not t1_host:
        rev = predict_fn(t2, t1)
        return np.array([rev[2], rev[1], rev[0]])
    return _symmetric_probs(predict_fn, t1, t2)


# --------------------------------------------------------------------------- #
# Núcleo de muestreo (Generator explícito, sin pandas)
# --------------------------------------------------------------------------- #
def _sample_outcome(probs, rng: np.random.Generator) -> int:
    """Devuelve 2=t1 gana, 1=empate, 0=t2 gana según probs=[t1, draw, t2].

    Usa una sola extracción uniforme con umbral acumulado: más rápido que
    `Generator.choice(..., p=...)` y equivalente en distribución.
    """
    r = rng.random()
    if r < probs[0]:
        return 2
    if r < probs[0] + probs[1]:
        return 1
    return 0


def _poisson_goals(
    t1: str,
    t2: str,
    team_xg: dict[str, dict[str, float]],
    rng: np.random.Generator,
    league_xg: float = 1.25,
) -> tuple[int, int]:
    """Goles con Poisson independiente: λ_t1 = xg_for(t1)·xg_against(t2)/league_xg."""
    xg_for_1 = team_xg.get(t1, {}).get("xg_for", league_xg)
    xg_for_2 = team_xg.get(t2, {}).get("xg_for", league_xg)
    xg_ag_1  = team_xg.get(t1, {}).get("xg_against", league_xg)
    xg_ag_2  = team_xg.get(t2, {}).get("xg_against", league_xg)
    lam_1 = max(0.05, xg_for_1 * xg_ag_2 / league_xg)
    lam_2 = max(0.05, xg_for_2 * xg_ag_1 / league_xg)
    return int(rng.poisson(lam_1)), int(rng.poisson(lam_2))


def _simulate_group_fast(
    teams: list[str],
    predict_fn,
    team_xg: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> list[dict]:
    """Simula los 6 partidos del grupo y devuelve la tabla (lista de dicts)
    ordenada por el desempate FIFA. El outcome se sortea con probs simétricas;
    los goles con Poisson, reconciliando el marcador con el outcome."""
    standings = {t: {"team": t, "points": 0, "gf": 0, "ga": 0} for t in teams}
    for t1, t2 in combinations(teams, 2):
        probs = _host_advantage_probs(predict_fn, t1, t2)
        outcome = _sample_outcome(probs, rng)
        g1, g2 = _poisson_goals(t1, t2, team_xg, rng)

        # Reconciliar marcador con outcome sorteado
        if outcome == 2 and g1 <= g2:
            g1 = g2 + 1
        elif outcome == 0 and g2 <= g1:
            g2 = g1 + 1
        elif outcome == 1 and g1 != g2:
            g2 = g1

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

    table = list(standings.values())
    for s in table:
        s["gd"] = s["gf"] - s["ga"]
    table.sort(key=_standing_sort_key, reverse=True)
    return table


def _select_best_thirds_fast(group_results: dict[str, list[dict]]) -> list[str]:
    thirds = [standings[2] for standings in group_results.values() if len(standings) >= 3]
    thirds.sort(key=_standing_sort_key, reverse=True)
    return [s["team"] for s in thirds[:8]]


def _build_bracket_fast(
    group_results: dict[str, list[dict]],
    best_thirds: list[str],
) -> list[tuple[str, str]]:
    slots: dict[str, str] = {}
    for group, standings in group_results.items():
        slots[f"{group}1"] = standings[0]["team"]
        slots[f"{group}2"] = standings[1]["team"]
    for i, team in enumerate(best_thirds, 1):
        slots[f"3rd_{i}"] = team
    return [(slots.get(h, "TBD"), slots.get(a, "TBD")) for h, a in KNOCKOUT_BRACKET_ORDER]


def _simulate_knockout_fast(
    pairs: list[tuple[str, str]],
    predict_fn,
    team_xg: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> list[str]:
    winners = []
    for t1, t2 in pairs:
        if t1 == "TBD" or t2 == "TBD":
            winners.append(t1 if t2 == "TBD" else t2)
            continue
        probs = _host_advantage_probs(predict_fn, t1, t2)
        outcome = _sample_outcome(probs, rng)
        if outcome == 2:
            winners.append(t1)
        elif outcome == 0:
            winners.append(t2)
        else:
            winners.append(t1 if rng.random() < 0.5 else t2)  # penales 50/50
    return winners


def _simulate_full_fast(
    predict_fn,
    team_xg: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict:
    """Núcleo del torneo completo. Devuelve el dict de progresión por fase."""
    progression = {phase: set() for phase in PHASES if phase != "champion"}
    progression["group_stage"] = set(ALL_TEAMS)

    group_results = {
        group: _simulate_group_fast(teams, predict_fn, team_xg, rng)
        for group, teams in GROUPS_2026.items()
    }
    best_thirds = _select_best_thirds_fast(group_results)
    bracket = _build_bracket_fast(group_results, best_thirds)

    for h, a in bracket:
        if h != "TBD":
            progression["round_of_32"].add(h)
        if a != "TBD":
            progression["round_of_32"].add(a)

    current_pairs = bracket
    phase_after_pairs = ["round_of_16", "quarterfinals", "semifinals", "final"]
    phase_idx = 0
    while len(current_pairs) >= 1:
        winners = _simulate_knockout_fast(current_pairs, predict_fn, team_xg, rng)
        if phase_idx < len(phase_after_pairs):
            progression[phase_after_pairs[phase_idx]].update(winners)
        phase_idx += 1
        if len(winners) == 1:
            progression["champion"] = winners[0]
            return progression
        current_pairs = list(zip(winners[::2], winners[1::2]))

    progression["champion"] = "TBD"
    return progression


# --------------------------------------------------------------------------- #
# API pública (capa delgada sobre el núcleo; mantiene compatibilidad y tests)
# --------------------------------------------------------------------------- #
def _points_tiebreak(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)


def simulate_group_stage(
    teams: list[str],
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Versión pública: devuelve la tabla del grupo como `DataFrame` ordenado."""
    rng = rng if rng is not None else np.random.default_rng()
    table = _simulate_group_fast(teams, predict_fn, team_xg or {}, rng)
    return pd.DataFrame(table, columns=["team", "points", "gf", "ga", "gd"])


def select_best_thirds(all_group_results: dict[str, pd.DataFrame]) -> list[str]:
    """Versión pública sobre `DataFrame` (los terceros mejor clasificados)."""
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


def simulate_full_tournament(
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Simula el torneo completo y devuelve un dict con los equipos que
    llegaron a cada fase y el campeón:
    {"group_stage": set(48), "round_of_32": set(32), "round_of_16": set(16),
     "quarterfinals": set(8), "semifinals": set(4), "final": set(2),
     "champion": str}
    """
    rng = rng if rng is not None else np.random.default_rng()
    return _simulate_full_fast(predict_fn, team_xg or {}, rng)
