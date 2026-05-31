import numpy as np
import pandas as pd

from src.features.derived_stats import (
    bayesian_shootout_winrate,
    compute_team_goal_stats_asof,
    compute_team_shootout_stats_asof,
    attach_goal_stat_diffs,
    attach_shootout_stat_diff,
    SHOOTOUT_PRIOR,
)


def test_shrinkage_zero_samples_returns_prior():
    assert bayesian_shootout_winrate(0, 0) == SHOOTOUT_PRIOR


def test_shrinkage_monotonic_and_bounded():
    # Con n creciente y todas victorias, el winrate crece pero nunca llega a 1.
    wrs = [bayesian_shootout_winrate(n, n) for n in [1, 5, 20, 100]]
    assert all(0.5 < w < 1.0 for w in wrs)
    assert wrs[0] < wrs[1] < wrs[2] < wrs[3]


def test_herfindahl_single_vs_two_scorers():
    # Equipo A: un solo goleador en dos fechas -> concentración 1.0.
    # Equipo B: dos goleadores distintos, uno por fecha -> 0.5.
    g = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01",
                                 "2020-01-01", "2020-02-01"]),
        "home_team": ["A", "A", "B", "B"],
        "away_team": ["X", "X", "Y", "Y"],
        "team": ["A", "A", "B", "B"],
        "scorer": ["a1", "a1", "b1", "b2"],
        "minute": [10, 20, 10, 20],
        "own_goal": [False, False, False, False],
        "penalty": [False, False, False, False],
    })
    stats = compute_team_goal_stats_asof(g)
    a_last = stats[stats["team"] == "A"].iloc[-1]
    b_last = stats[stats["team"] == "B"].iloc[-1]
    assert abs(a_last["striker_concentration"] - 1.0) < 1e-9
    assert abs(b_last["striker_concentration"] - 0.5) < 1e-9


def test_goal_stats_asof_is_strictly_prior():
    """
    El estado para un partido en fecha D NO debe incluir los goles de D.
    Construimos un equipo cuyo único gol tardío/penal ocurre el 2020-02-01 y
    verificamos que el diff para un partido ese mismo día usa el fill neutral
    (estado estrictamente anterior, sin ese gol).
    """
    g = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "home_team": ["A", "A"],
        "away_team": ["X", "X"],
        "team": ["A", "A"],
        "scorer": ["a1", "a2"],
        "minute": [10, 88],          # el de 2020-02-01 es tardío
        "own_goal": [False, False],
        "penalty": [False, True],    # el de 2020-02-01 es penal
    })
    # Partido de A el 2020-02-01: debe ver SOLO el gol del 2020-01-01.
    match = pd.DataFrame({
        "date": pd.to_datetime(["2020-02-01"]),
        "home_team": ["A"],
        "away_team": ["Z"],  # Z sin datos -> fill
    })
    diffs = attach_goal_stat_diffs(match, g)
    # A as-of (estrictamente antes de 2020-02-01) tiene 1 gol, no tardío, no penal:
    # late_goal_ratio=0, penalty_share=0. Z usa fill. diff = 0 - fill.
    # Lo esencial: el penal del 2020-02-01 NO se cuenta (sería leakage).
    # penalty_share de A as-of = 0, así que el diff es -(fill) y NO refleja el penal.
    assert diffs["penalty_share_diff"][0] <= 0  # A no tiene penales aún
    # Si hubiera leakage, A tendría penalty_share=0.5 y el diff sería positivo.


def test_shootout_winrate_asof_and_shrinkage():
    s = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01"]),
        "home_team": ["A", "A", "A"],
        "away_team": ["B", "C", "D"],
        "winner": ["A", "A", "A"],
        "first_shooter": [None, None, None],
    })
    stats = compute_team_shootout_stats_asof(s)
    a = stats[stats["team"] == "A"].sort_values("date")
    assert list(a["shootout_n"]) == [1, 2, 3]
    assert list(a["shootout_wins"]) == [1, 2, 3]

    # Partido de A después de las 3 tandas: winrate con shrinkage, < 1.0.
    match = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01"]),
        "home_team": ["A"],
        "away_team": ["Z"],  # sin tandas -> prior 0.5
    })
    diff = attach_shootout_stat_diff(match, s)
    wr_a = bayesian_shootout_winrate(3, 3)
    assert abs(diff[0] - (wr_a - 0.5)) < 1e-9
    assert wr_a < 1.0
