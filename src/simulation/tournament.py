"""
Lógica del torneo FIFA Mundial 2026 (corrección v2: bracket oficial).

12 grupos de 4. Avanzan los 2 primeros de cada grupo + los 8 mejores terceros
= 32 equipos. El bracket del Round of 32 se carga de
`data/raw/wc2026_bracket.csv`, transcrito del calendario oficial FIFA
(partidos 73-104; ver Wikipedia "2026 FIFA World Cup knockout stage" y
fifa.com): 8 ganadores de grupo enfrentan a los mejores terceros (con pools de
procedencia por partido), 4 ganadores enfrentan a segundos y los 8 segundos
restantes se cruzan entre sí. No existe ningún cruce tercero-vs-tercero (la
versión v1 inventaba 4, distorsionando toda la progresión).

La asignación de terceros a slots respeta los pools oficiales por partido
mediante un matching exacto (backtracking); la tabla FIFA de 495 combinaciones
es una elección particular entre los matchings válidos, no publicada en forma
compacta, así que se usa el primer matching válido en orden determinista
(aproximación estructuralmente fiel: pools respetados, nunca contra el ganador
del propio grupo).

Desempates de grupo (reglamento FIFA art. 13): puntos -> dif. de goles ->
goles a favor -> enfrentamiento directo entre empatados (puntos/DG/GF del
subconjunto) -> sorteo (rng). El fair play no es simulable y se omite
(documentado como aproximación).

Fases trackeadas: "group_stage", "round_of_32", "round_of_16",
"quarterfinals", "semifinals", "final", "champion".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from math import factorial

from src.features.features import LEAGUE_AVG_XG

_FIXTURE_CSV = Path(__file__).parents[2] / "data" / "raw" / "wc2026_fixture.csv"
_BRACKET_CSV = Path(__file__).parents[2] / "data" / "raw" / "wc2026_bracket.csv"

HOST_COUNTRIES = {"United States", "Mexico", "Canada"}
PHASES = [
    "group_stage", "round_of_32", "round_of_16",
    "quarterfinals", "semifinals", "final", "champion",
]
# Fase a la que entra el GANADOR de un partido de cada fase del bracket.
_NEXT_PHASE = {
    "round_of_32": "round_of_16",
    "round_of_16": "quarterfinals",
    "quarterfinals": "semifinals",
    "semifinals": "final",
    "final": "champion",
}


def _load_groups(fixture_path: Path) -> dict[str, list[str]]:
    from src.data.data_loader import load_wc2026_fixture
    df = load_wc2026_fixture(fixture_path)
    groups: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        groups.setdefault(row["group"], []).append(row["team"])
    return groups


def _load_bracket(bracket_path: Path = _BRACKET_CSV) -> list[dict]:
    df = pd.read_csv(bracket_path)
    return df.to_dict("records")


GROUPS_2026: dict[str, list[str]] = _load_groups(_FIXTURE_CSV)
ALL_TEAMS: list[str] = [team for teams in GROUPS_2026.values() for team in teams]
BRACKET: list[dict] = _load_bracket()

# Pools de terceros por partido del R32: {match -> set(grupos elegibles)}.
THIRD_SLOT_POOLS: dict[int, set[str]] = {
    row["match"]: set(row["away_slot"].split(":", 1)[1])
    for row in BRACKET
    if isinstance(row["away_slot"], str) and row["away_slot"].startswith("3:")
}


# --------------------------------------------------------------------------- #
# Probabilidades por partido
# --------------------------------------------------------------------------- #
def _symmetric_probs(predict_fn, t1: str, t2: str) -> np.ndarray:
    """
    Promedia P(t1 vs t2) con la versión invertida P(t2 vs t1) para eliminar
    el sesgo home/away cuando no hay localía real. Devuelve [P(win_t1),
    P(draw), P(win_t2)].
    """
    p_fwd = predict_fn(t1, t2)              # [home_win, draw, away_win] perspectiva t1=home
    p_rev = predict_fn(t2, t1)              # [home_win, draw, away_win] perspectiva t2=home
    p_t1_win = (p_fwd[0] + p_rev[2]) / 2.0
    p_draw   = (p_fwd[1] + p_rev[1]) / 2.0
    p_t2_win = (p_fwd[2] + p_rev[0]) / 2.0
    out = np.array([p_t1_win, p_draw, p_t2_win])
    out /= out.sum()
    return out


def _host_advantage_probs(
    predict_fn, t1: str, t2: str, venue_country: str | None = None,
) -> np.ndarray:
    """
    Localía condicionada a la SEDE real (corrección v2): un anfitrión solo
    recibe ventaja de local si el partido se juega en su propio país.

    - venue_country=None (fase de grupos): los anfitriones juegan todos sus
      partidos de grupo en casa por calendario, así que basta con que el
      equipo sea anfitrión.
    - venue_country dado (eliminatorias): se exige venue == país del equipo.
      México en una sede de EE.UU. NO recibe localía.
    """
    def _is_home(team: str) -> bool:
        if team not in HOST_COUNTRIES:
            return False
        return venue_country is None or venue_country == team

    t1_home, t2_home = _is_home(t1), _is_home(t2)
    if t1_home and not t2_home:
        return predict_fn(t1, t2)
    if t2_home and not t1_home:
        rev = predict_fn(t2, t1)
        return np.array([rev[2], rev[1], rev[0]])
    return _symmetric_probs(predict_fn, t1, t2)


def _outcome_from_probs(probs: np.ndarray, rng=None) -> int:
    """Devuelve 2=t1 wins, 1=draw, 0=t2 wins según probs=[t1, draw, t2]."""
    rng = rng if rng is not None else np.random
    return int(rng.choice([2, 1, 0], p=probs))


# --------------------------------------------------------------------------- #
# Marcadores: Poisson condicionado al outcome (corrección v2)
# --------------------------------------------------------------------------- #
_MAX_GOALS = 10
_FACTORIALS = np.array([factorial(i) for i in range(_MAX_GOALS + 1)], dtype=np.float64)


def _poisson_pmf_vector(lam: float) -> np.ndarray:
    i = np.arange(_MAX_GOALS + 1)
    return np.exp(-lam) * lam ** i / _FACTORIALS


def build_goal_sampler(team_xg: dict[str, dict[str, float]], league_xg: float = LEAGUE_AVG_XG):
    """
    Devuelve sample(t1, t2, outcome, rng) -> (g1, g2): marcador muestreado de la
    distribución conjunta Poisson independiente CONDICIONADA al outcome sorteado
    (2 = gana t1, 1 = empate, 0 = gana t2), sobre la grilla 0..10 goles.

    Esto reemplaza la "reconciliación" ad-hoc de la v1 (g1 = g2 + 1, etc.), que
    truncaba/desplazaba la distribución de DG y GF que alimenta los desempates
    de grupo. Las distribuciones condicionadas se cachean por (t1, t2, outcome):
    en un torneo solo hay ~72 pares de grupo x 3 outcomes.
    """
    team_xg = team_xg or {}
    cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    grid_i, grid_j = np.meshgrid(
        np.arange(_MAX_GOALS + 1), np.arange(_MAX_GOALS + 1), indexing="ij"
    )
    masks = {
        2: grid_i > grid_j,
        1: grid_i == grid_j,
        0: grid_i < grid_j,
    }

    def _conditional(t1: str, t2: str, outcome: int):
        key = (t1, t2, outcome)
        if key not in cache:
            xg_for_1 = team_xg.get(t1, {}).get("xg_for", league_xg)
            xg_for_2 = team_xg.get(t2, {}).get("xg_for", league_xg)
            xg_ag_1 = team_xg.get(t1, {}).get("xg_against", league_xg)
            xg_ag_2 = team_xg.get(t2, {}).get("xg_against", league_xg)
            lam_1 = max(0.05, xg_for_1 * xg_ag_2 / league_xg)
            lam_2 = max(0.05, xg_for_2 * xg_ag_1 / league_xg)

            joint = np.outer(_poisson_pmf_vector(lam_1), _poisson_pmf_vector(lam_2))
            joint = joint * masks[outcome]
            total = joint.sum()
            if total <= 0:  # degenerado (no debería ocurrir con lam >= 0.05)
                joint = masks[outcome].astype(np.float64)
                total = joint.sum()
            p = (joint / total).ravel()
            cache[key] = (p, grid_i.ravel(), grid_j.ravel())
        return cache[key]

    def sample(t1: str, t2: str, outcome: int, rng) -> tuple[int, int]:
        p, gi, gj = _conditional(t1, t2, outcome)
        k = rng.choice(len(p), p=p)
        return int(gi[k]), int(gj[k])

    return sample


# --------------------------------------------------------------------------- #
# Fase de grupos: simulación + desempates FIFA
# --------------------------------------------------------------------------- #
def _rank_group(
    standings: dict[str, dict],
    results: list[tuple[str, str, int, int]],
    rng=None,
) -> pd.DataFrame:
    """
    Ordena el grupo según el reglamento FIFA: puntos -> DG -> GF ->
    enfrentamiento directo entre los empatados (puntos/DG/GF del subconjunto)
    -> sorteo. `results` son tuplas (t1, t2, g1, g2). El fair play se omite
    (no simulable).
    """
    rng_ = rng if rng is not None else np.random
    table = pd.DataFrame([
        {"team": t, **v, "gd": v["gf"] - v["ga"]}
        for t, v in standings.items()
    ])

    # Clave primaria global.
    table = table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)

    # Resolver bloques de empate total con head-to-head y, si persiste, sorteo.
    keys = list(zip(table["points"], table["gd"], table["gf"]))
    final_order: list[int] = []
    i = 0
    while i < len(table):
        j = i
        while j < len(table) and keys[j] == keys[i]:
            j += 1
        block = list(range(i, j))
        if len(block) > 1:
            teams = [table.iloc[b]["team"] for b in block]
            sub = {t: {"points": 0, "gf": 0, "ga": 0} for t in teams}
            for t1, t2, g1, g2 in results:
                if t1 in sub and t2 in sub:
                    sub[t1]["gf"] += g1; sub[t1]["ga"] += g2
                    sub[t2]["gf"] += g2; sub[t2]["ga"] += g1
                    if g1 > g2:
                        sub[t1]["points"] += 3
                    elif g2 > g1:
                        sub[t2]["points"] += 3
                    else:
                        sub[t1]["points"] += 1; sub[t2]["points"] += 1
            # Orden por sub-tabla; el desempate final es sorteo (no orden de
            # inserción, que introduciría un sesgo determinista).
            tiebreak = {
                t: (sub[t]["points"], sub[t]["gf"] - sub[t]["ga"], sub[t]["gf"],
                    float(rng_.random()) if hasattr(rng_, "random") else float(np.random.random()))
                for t in teams
            }
            ordered = sorted(teams, key=lambda t: tiebreak[t], reverse=True)
            pos = {t: block[k] for k, t in enumerate(ordered)}
            # Reordenar el bloque según head-to-head/sorteo.
            block_rows = sorted(range(i, j), key=lambda b: pos[table.iloc[b]["team"]])
            final_order.extend(block_rows)
        else:
            final_order.extend(block)
        i = j

    return table.iloc[final_order].reset_index(drop=True)


def _simulate_group_stage_full(
    teams: list[str],
    predict_fn,
    goal_sampler,
    rng=None,
) -> tuple[pd.DataFrame, list[tuple[str, str, int, int]]]:
    """
    Simula los 6 partidos del grupo. El outcome se sortea con las probs del
    modelo (simétricas salvo localía real) y el marcador con Poisson
    condicionado al outcome. Devuelve (tabla ordenada, resultados).
    """
    standings = {t: {"points": 0, "gf": 0, "ga": 0} for t in teams}
    results: list[tuple[str, str, int, int]] = []

    for t1, t2 in combinations(teams, 2):
        probs = _host_advantage_probs(predict_fn, t1, t2)  # grupos: venue implícito
        outcome = _outcome_from_probs(probs, rng)
        g1, g2 = goal_sampler(t1, t2, outcome, rng if rng is not None else np.random)

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
        results.append((t1, t2, g1, g2))

    return _rank_group(standings, results, rng), results


def simulate_group_stage(
    teams: list[str],
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng=None,
) -> pd.DataFrame:
    """API pública (compatible): devuelve solo la tabla ordenada."""
    sampler = build_goal_sampler(team_xg or {})
    table, _ = _simulate_group_stage_full(teams, predict_fn, sampler, rng)
    return table


# --------------------------------------------------------------------------- #
# Mejores terceros: ranking + asignación a slots oficiales
# --------------------------------------------------------------------------- #
def rank_thirds(
    all_group_results: dict[str, pd.DataFrame],
    rng=None,
) -> list[tuple[str, str]]:
    """
    Devuelve [(group, team)] de los 8 mejores terceros, rankeados por
    puntos -> DG -> GF -> sorteo (reglamento FIFA; fair play omitido).
    """
    rng_ = rng if rng is not None else np.random
    thirds = []
    for group, table in all_group_results.items():
        if len(table) >= 3:
            row = table.iloc[2]
            thirds.append({
                "group": group, "team": row["team"],
                "points": row["points"], "gd": row["gd"], "gf": row["gf"],
                "lot": float(rng_.random()) if hasattr(rng_, "random") else float(np.random.random()),
            })
    if not thirds:
        return []
    df = pd.DataFrame(thirds).sort_values(
        ["points", "gd", "gf", "lot"], ascending=False
    ).head(8)
    return list(zip(df["group"], df["team"]))


def select_best_thirds(all_group_results: dict[str, pd.DataFrame]) -> list[str]:
    """API pública (compatible): solo los nombres de los 8 mejores terceros."""
    return [team for _, team in rank_thirds(all_group_results)]


def allocate_thirds_to_slots(
    qualified_groups: list[str],
    pools: dict[int, set[str]] | None = None,
) -> dict[int, str] | None:
    """
    Asigna los 8 grupos de los terceros clasificados a los 8 partidos del R32
    con slot de tercero, respetando el pool de grupos elegibles de cada partido
    (transcrito del calendario oficial). Matching exacto por backtracking,
    procesando primero los slots más restringidos; determinista (grupos en
    orden alfabético). Devuelve {match -> group} o None si no hay matching.
    """
    pools = pools if pools is not None else THIRD_SLOT_POOLS
    qualified = set(qualified_groups)
    slot_options = {
        m: sorted(pool & qualified) for m, pool in pools.items()
    }
    order = sorted(slot_options, key=lambda m: (len(slot_options[m]), m))

    assignment: dict[int, str] = {}
    used: set[str] = set()

    def bt(k: int) -> bool:
        if k == len(order):
            return True
        m = order[k]
        for g in slot_options[m]:
            if g not in used:
                assignment[m] = g
                used.add(g)
                if bt(k + 1):
                    return True
                used.discard(g)
                del assignment[m]
        return False

    return assignment if bt(0) else None


# --------------------------------------------------------------------------- #
# Torneo completo sobre el bracket oficial
# --------------------------------------------------------------------------- #
def simulate_full_tournament(
    predict_fn,
    team_xg: dict[str, dict[str, float]] | None = None,
    rng=None,
    goal_sampler=None,
) -> dict:
    """
    Simula el torneo completo sobre el bracket oficial FIFA 2026 y devuelve un
    dict con los equipos que alcanzaron cada fase y el campeón.
    """
    rng_ = rng if rng is not None else np.random
    sampler = goal_sampler or build_goal_sampler(team_xg or {})

    progression: dict = {phase: set() for phase in PHASES if phase != "champion"}
    progression["group_stage"] = set(ALL_TEAMS)

    # --- Fase de grupos ---
    group_tables: dict[str, pd.DataFrame] = {}
    for group, teams in GROUPS_2026.items():
        table, _results = _simulate_group_stage_full(teams, predict_fn, sampler, rng)
        group_tables[group] = table

    # Slots 1X / 2X.
    slots: dict[str, str] = {}
    for group, table in group_tables.items():
        slots[f"1{group}"] = table.iloc[0]["team"]
        slots[f"2{group}"] = table.iloc[1]["team"]

    # Mejores terceros y su asignación a partidos (pools oficiales).
    thirds = rank_thirds(group_tables, rng)
    third_team_by_group = dict(thirds)
    alloc = allocate_thirds_to_slots([g for g, _ in thirds])
    if alloc is None:
        # Fallback defensivo: asignación greedy ignorando pools (no debería
        # ocurrir: los pools oficiales admiten matching para las combinaciones
        # alcanzables).
        groups_left = [g for g, _ in thirds]
        alloc = {m: groups_left[i] for i, m in enumerate(sorted(THIRD_SLOT_POOLS))}

    third_by_match = {m: third_team_by_group[g] for m, g in alloc.items()}

    # --- Bracket: resolver partidos en orden ---
    winners: dict[str, str] = {}

    def _resolve(slot: str, match: int) -> str:
        if slot.startswith("3:"):
            return third_by_match[match]
        if slot.startswith("W"):
            return winners[slot]
        return slots[slot]

    for row in BRACKET:
        match = int(row["match"])
        phase = row["phase"]
        t1 = _resolve(str(row["home_slot"]), match)
        t2 = _resolve(str(row["away_slot"]), match)
        venue = row.get("venue_country")

        progression[phase].add(t1)
        progression[phase].add(t2)

        probs = _host_advantage_probs(predict_fn, t1, t2, venue_country=venue)
        outcome = _outcome_from_probs(probs, rng)
        if outcome == 2:
            winner = t1
        elif outcome == 0:
            winner = t2
        else:
            winner = str(rng_.choice([t1, t2]))  # tanda de penales 50/50
        winners[f"W{match}"] = winner

        if _NEXT_PHASE[phase] == "champion":
            progression["champion"] = winner
        # En las demás fases el ganador queda registrado cuando juegue su
        # partido de la fase siguiente (todo partido agrega a sus participantes).

    return progression
