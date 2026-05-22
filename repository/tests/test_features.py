"""Tests para ELO, time decay y pipeline de features."""

import numpy as np
import pandas as pd
import pytest

from src.features.elo import calculate_elo_ratings, INITIAL_RATING
from src.features.time_decay import (
    compute_time_decay_weights,
    lambda_to_halflife_years,
    halflife_years_to_lambda,
)


@pytest.fixture
def three_matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        "home_team": ["Brazil", "Argentina", "Brazil"],
        "away_team": ["Argentina", "Germany", "Germany"],
        "home_score": [2, 1, 0],
        "away_score": [1, 0, 1],
        "tournament": ["Copa América", "FIFA World Cup", "Friendly"],
    })


def test_elo_output_columns(three_matches):
    result = calculate_elo_ratings(three_matches)
    for col in ["home_elo_before", "away_elo_before", "home_elo_after", "away_elo_after"]:
        assert col in result.columns


def test_elo_initial_rating(three_matches):
    result = calculate_elo_ratings(three_matches)
    first = result.iloc[0]
    assert first["home_elo_before"] == INITIAL_RATING
    assert first["away_elo_before"] == INITIAL_RATING


def test_elo_winner_gains_points(three_matches):
    result = calculate_elo_ratings(three_matches)
    first = result.iloc[0]
    assert first["home_elo_after"] > first["home_elo_before"]
    assert first["away_elo_after"] < first["away_elo_before"]


def test_elo_ratings_are_finite(three_matches):
    result = calculate_elo_ratings(three_matches)
    for col in ["home_elo_after", "away_elo_after"]:
        assert np.all(np.isfinite(result[col].values))


def test_time_decay_range():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2020-01-01", "2010-01-01"]))
    weights = compute_time_decay_weights(dates, lambda_=0.002)
    assert np.all(weights > 0)
    assert np.all(weights <= 1.0)


def test_time_decay_monotone():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2020-01-01", "2015-01-01"]))
    weights = compute_time_decay_weights(dates, lambda_=0.002)
    assert weights[0] >= weights[1] >= weights[2]


def test_time_decay_recent_is_max():
    today = pd.Timestamp.today()
    dates = pd.Series([today, today - pd.Timedelta(days=365)])
    weights = compute_time_decay_weights(dates)
    assert weights[0] > weights[1]


def test_halflife_lambda_roundtrip():
    lambda_ = 0.002
    years = lambda_to_halflife_years(lambda_)
    recovered = halflife_years_to_lambda(years)
    assert abs(recovered - lambda_) < 1e-10
