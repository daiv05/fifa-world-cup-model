import numpy as np
import pytest
from scipy.stats import binomtest

from src.simulation.tournament import (
    GROUPS_2026,
    ALL_TEAMS,
    PHASES,
    simulate_group_stage,
    select_best_thirds,
    simulate_full_tournament,
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
        # 6 partidos: cada decisivo = 3 pts, cada empate = 2 pts → [12, 18]
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
        pytest.skip("Grupo con anfitriones — no se puede testear sesgo neutro")

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


def test_clopper_pearson_basic():
    lo, hi = _clopper_pearson(100, 1000, alpha=0.10)
    # 10% ± algo razonable
    assert lo < 10.0 < hi
    assert 7.5 < lo < 10.0
    assert 10.0 < hi < 12.5


def test_clopper_pearson_edges():
    lo, hi = _clopper_pearson(0, 1000)
    assert lo == 0.0 and hi > 0.0
    lo, hi = _clopper_pearson(1000, 1000)
    assert hi == 100.0 and lo < 100.0
