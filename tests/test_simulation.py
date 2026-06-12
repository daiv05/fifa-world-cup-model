import numpy as np
import pytest
from scipy.stats import binomtest

from src.simulation.tournament import (
    GROUPS_2026,
    ALL_TEAMS,
    PHASES,
    BRACKET,
    THIRD_SLOT_POOLS,
    simulate_group_stage,
    select_best_thirds,
    simulate_full_tournament,
    allocate_thirds_to_slots,
    build_goal_sampler,
    _host_advantage_probs,
)
from src.simulation.simulate import _clopper_pearson


def uniform_predict_fn(home, away):
    return np.array([1 / 3, 1 / 3, 1 / 3])


def home_always_wins(home, away):
    return np.array([1.0, 0.0, 0.0])  # [home_win, draw, away_win]


def biased_first_team_wins(home, away):
    """Predictor que SIEMPRE da ventaja al `home` (75%)."""
    return np.array([0.75, 0.10, 0.15])


def test_all_48_teams_defined():
    total = sum(len(t) for t in GROUPS_2026.values())
    assert total == 48, f"Se esperaban 48 equipos, se encontraron {total}"


def test_group_stage_returns_4_rows():
    group = list(GROUPS_2026.values())[0]
    table = simulate_group_stage(group, uniform_predict_fn)
    assert len(table) == 4


def test_group_stage_points_non_negative():
    group = list(GROUPS_2026.values())[0]
    table = simulate_group_stage(group, uniform_predict_fn)
    assert (table["points"] >= 0).all()


def test_group_stage_total_points_range():
    group = list(GROUPS_2026.values())[0]
    for _ in range(10):
        table = simulate_group_stage(group, uniform_predict_fn)
        total = table["points"].sum()
        # 6 partidos: cada decisivo = 3 pts, cada empate = 2 pts - [12, 18]
        assert 12 <= total <= 18, f"Total inesperado: {total}"


def test_best_thirds_returns_8():
    all_results = {
        g: simulate_group_stage(teams, uniform_predict_fn)
        for g, teams in GROUPS_2026.items()
    }
    thirds = select_best_thirds(all_results)
    assert len(thirds) == 8


def test_best_thirds_are_valid_teams():
    all_set = set(ALL_TEAMS)
    all_results = {
        g: simulate_group_stage(teams, uniform_predict_fn)
        for g, teams in GROUPS_2026.items()
    }
    thirds = select_best_thirds(all_results)
    for t in thirds:
        assert t in all_set


def test_full_tournament_returns_progression_dict():
    result = simulate_full_tournament(uniform_predict_fn)
    assert "champion" in result
    assert result["champion"] in ALL_TEAMS
    for phase in PHASES:
        if phase != "champion":
            assert phase in result


def test_full_tournament_phase_sizes():
    """Cada fase debe tener el número correcto de equipos."""
    result = simulate_full_tournament(uniform_predict_fn)
    assert len(result["group_stage"]) == 48
    assert len(result["round_of_32"]) == 32
    assert len(result["round_of_16"]) == 16
    assert len(result["quarterfinals"]) == 8
    assert len(result["semifinals"]) == 4
    assert len(result["final"]) == 2
    assert result["champion"] in result["final"]


def test_simulation_reproducible_with_seed():
    np.random.seed(0)
    r1 = simulate_full_tournament(uniform_predict_fn)
    np.random.seed(0)
    r2 = simulate_full_tournament(uniform_predict_fn)
    assert r1["champion"] == r2["champion"]


def test_probabilities_sum_to_one():
    probs = uniform_predict_fn("A", "B")
    assert abs(probs.sum() - 1.0) < 1e-6


def test_no_home_away_bias_in_groups():
    """
    Con un predictor que asigna 75% al 'home', la simulación NO debe darle
    sistemáticamente más victorias al primer equipo alfabético del par,
    porque tournament.py usa probs simétricas (promedio fwd/rev) cuando
    ninguno es anfitrión.
    """
    group_no_host = [t for t in list(GROUPS_2026.values())[2] if t not in ("United States", "Mexico", "Canada")]
    if len(group_no_host) < 2:
        pytest.skip("Grupo con anfitriones - no se puede testear sesgo neutro")

    np.random.seed(42)
    # Contar puntos del primer equipo alfabético del grupo vs último
    first = sorted(group_no_host)[0]
    last = sorted(group_no_host)[-1]
    wins_first = 0
    wins_last = 0
    n = 500
    for _ in range(n):
        table = simulate_group_stage(group_no_host, biased_first_team_wins)
        pts = table.set_index("team")["points"].to_dict()
        if pts.get(first, 0) > pts.get(last, 0):
            wins_first += 1
        elif pts.get(last, 0) > pts.get(first, 0):
            wins_last += 1
    # Sin sesgo, la diferencia debería estar centrada en 0
    diff = abs(wins_first - wins_last)
    # Con n=500 y predictor 50/50 efectivo, diff > 100 sería sesgo claro
    assert diff < 100, f"Posible sesgo home/away: first={wins_first}, last={wins_last}"


def test_bracket_structure_official():
    """El bracket cargado debe tener la estructura oficial FIFA 2026: 16
    partidos de R32 (8 con slot de tercero, 4 ganador-vs-segundo, 4
    segundo-vs-segundo), sin ningún cruce tercero-vs-tercero."""
    r32 = [r for r in BRACKET if r["phase"] == "round_of_32"]
    assert len(r32) == 16
    n_third, n_1v2, n_2v2 = 0, 0, 0
    for r in r32:
        h, a = str(r["home_slot"]), str(r["away_slot"])
        assert not (h.startswith("3:") and a.startswith("3:")), "cruce 3º-vs-3º"
        if a.startswith("3:"):
            assert h.startswith("1"), "los terceros solo enfrentan ganadores"
            n_third += 1
        elif h.startswith("1"):
            n_1v2 += 1
        else:
            assert h.startswith("2") and a.startswith("2")
            n_2v2 += 1
    assert (n_third, n_1v2, n_2v2) == (8, 4, 4)
    # Los pools nunca incluyen al grupo del propio ganador.
    for r in r32:
        a = str(r["away_slot"])
        if a.startswith("3:"):
            winner_group = str(r["home_slot"])[1]
            assert winner_group not in a.split(":")[1]


def test_allocate_thirds_respects_pools():
    qualified = ["A", "B", "C", "D", "F", "G", "K", "L"]
    alloc = allocate_thirds_to_slots(qualified)
    assert alloc is not None
    assert sorted(alloc.values()) == sorted(qualified)
    for match, group in alloc.items():
        assert group in THIRD_SLOT_POOLS[match]


def test_allocate_thirds_many_combinations():
    """Toda combinación muestreada de 8 grupos debe admitir matching válido."""
    from itertools import combinations as combos
    groups = list("ABCDEFGHIJKL")
    rng = np.random.default_rng(7)
    all_combos = list(combos(groups, 8))
    for idx in rng.choice(len(all_combos), size=60, replace=False):
        q = list(all_combos[idx])
        alloc = allocate_thirds_to_slots(q)
        assert alloc is not None, f"sin matching para {q}"
        for match, group in alloc.items():
            assert group in THIRD_SLOT_POOLS[match]


def test_conditional_goal_sampler_consistent_with_outcome():
    sampler = build_goal_sampler({})
    rng = np.random.default_rng(0)
    for outcome, check in [(2, lambda a, b: a > b),
                           (1, lambda a, b: a == b),
                           (0, lambda a, b: a < b)]:
        for _ in range(200):
            g1, g2 = sampler("X", "Y", outcome, rng)
            assert check(g1, g2), f"outcome={outcome} dio {g1}-{g2}"


def test_host_advantage_requires_own_venue():
    """México solo recibe localía si la sede es México."""
    def predict(home, away):
        return np.array([0.7, 0.2, 0.1])  # fuerte ventaja al home

    in_mexico = _host_advantage_probs(predict, "Mexico", "Spain", venue_country="Mexico")
    in_usa = _host_advantage_probs(predict, "Mexico", "Spain", venue_country="United States")
    symmetric = (0.7 + 0.1) / 2
    assert abs(in_mexico[0] - 0.7) < 1e-9          # localía plena
    assert abs(in_usa[0] - symmetric) < 1e-9       # simétrico: sin localía
    # Fase de grupos (venue None): el anfitrión juega en casa por calendario.
    group_stage = _host_advantage_probs(predict, "Mexico", "Spain", venue_country=None)
    assert abs(group_stage[0] - 0.7) < 1e-9


def test_group_tiebreak_uses_rng_not_insertion_order():
    """Con un predictor 100% empates 0-0, el orden del grupo debe variar entre
    seeds (sorteo), no quedar fijo al orden de inserción."""
    def all_draws(home, away):
        return np.array([0.0, 1.0, 0.0])

    teams = list(GROUPS_2026.values())[5]
    orders = set()
    for seed in range(10):
        rng = np.random.default_rng(seed)
        table = simulate_group_stage(teams, all_draws, rng=rng)
        orders.add(tuple(table["team"]))
    assert len(orders) > 1, "el desempate total no usa el rng"


def test_clopper_pearson_basic():
    # Default alpha=0.05 (IC 95%, estándar en la literatura)
    lo, hi = _clopper_pearson(100, 1000)
    # 10% ± algo razonable
    assert lo < 10.0 < hi
    assert 7.5 < lo < 10.0
    assert 10.0 < hi < 12.5


def test_clopper_pearson_edges():
    lo, hi = _clopper_pearson(0, 1000)
    assert lo == 0.0 and hi > 0.0
    lo, hi = _clopper_pearson(1000, 1000)
    assert hi == 100.0 and lo < 100.0
