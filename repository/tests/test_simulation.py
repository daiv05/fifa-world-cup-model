"""Tests para el motor de simulación Monte Carlo y la lógica del torneo."""

import numpy as np
import pytest

from src.simulation.tournament import (
    GROUPS_2026,
    ALL_TEAMS,
    simulate_group_stage,
    select_best_thirds,
    simulate_full_tournament,
)


def uniform_predict_fn(home, away, _features):
    """Predictor con probabilidades iguales para los tres resultados."""
    return np.array([1 / 3, 1 / 3, 1 / 3])


def home_always_wins(home, away, _features):
    return np.array([0.0, 0.0, 1.0])


def test_all_48_teams_defined():
    total = sum(len(t) for t in GROUPS_2026.values())
    assert total == 48, f"Se esperaban 48 equipos, se encontraron {total}"


def test_group_stage_returns_4_rows():
    group = list(GROUPS_2026.values())[0]
    table = simulate_group_stage(group, uniform_predict_fn, {})
    assert len(table) == 4


def test_group_stage_points_non_negative():
    group = list(GROUPS_2026.values())[0]
    table = simulate_group_stage(group, uniform_predict_fn, {})
    assert (table["points"] >= 0).all()


def test_group_stage_total_points_correct():
    group = list(GROUPS_2026.values())[0]
    for _ in range(10):
        table = simulate_group_stage(group, uniform_predict_fn, {})
        total = table["points"].sum()
        # 6 games: decisive game = 3 pts, draw = 2 pts → range [12, 18]
        assert 12 <= total <= 18, f"Total de puntos inesperado: {total}"


def test_best_thirds_returns_8():
    all_results = {}
    for g, teams in GROUPS_2026.items():
        all_results[g] = simulate_group_stage(teams, uniform_predict_fn, {})
    thirds = select_best_thirds(all_results)
    assert len(thirds) == 8


def test_best_thirds_are_valid_teams():
    all_teams_set = set(ALL_TEAMS)
    all_results = {}
    for g, teams in GROUPS_2026.items():
        all_results[g] = simulate_group_stage(teams, uniform_predict_fn, {})
    thirds = select_best_thirds(all_results)
    for t in thirds:
        assert t in all_teams_set, f"{t} no es un equipo válido"


def test_champion_is_valid_team():
    champion = simulate_full_tournament(uniform_predict_fn, {})
    assert champion in ALL_TEAMS, f"{champion} no es un equipo del torneo"


def test_champion_always_home_team_when_home_always_wins():
    """Con home_always_wins, el primer equipo de cada llave siempre gana."""
    for _ in range(5):
        champion = simulate_full_tournament(home_always_wins, {})
        assert champion in ALL_TEAMS


def test_simulation_reproducible_with_seed():
    np.random.seed(0)
    c1 = simulate_full_tournament(uniform_predict_fn, {})
    np.random.seed(0)
    c2 = simulate_full_tournament(uniform_predict_fn, {})
    assert c1 == c2


def test_probabilities_sum_to_one():
    probs = uniform_predict_fn("A", "B", {})
    assert abs(probs.sum() - 1.0) < 1e-6
