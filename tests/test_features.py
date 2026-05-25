import numpy as np
import pandas as pd
import pytest

from src.features.elo import calculate_elo_ratings, INITIAL_RATING
from src.features.time_decay import (
    compute_time_decay_weights,
    lambda_to_halflife_years,
    halflife_years_to_lambda,
    REFERENCE_DATE,
    DEFAULT_LAMBDA,
)
from src.features.features import (
    encode_target,
    _vectorized_elo_diff,
    build_match_features,
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
        "country": ["Brazil", "Argentina", "Brazil"],
        "neutral": [False, False, False],
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
    weights = compute_time_decay_weights(dates, lambda_=DEFAULT_LAMBDA)
    assert np.all(weights > 0)
    assert np.all(weights <= 1.0)


def test_time_decay_monotone():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2020-01-01", "2015-01-01"]))
    weights = compute_time_decay_weights(dates, lambda_=DEFAULT_LAMBDA)
    assert weights[0] >= weights[1] >= weights[2]


def test_time_decay_uses_reference_date_default():
    """Sin argumento, usa REFERENCE_DATE (determinista, no today())."""
    dates = pd.Series([REFERENCE_DATE - pd.Timedelta(days=10)])
    w_default = compute_time_decay_weights(dates)
    w_explicit = compute_time_decay_weights(dates, reference_date=REFERENCE_DATE)
    assert np.allclose(w_default, w_explicit)


def test_halflife_lambda_roundtrip():
    lambda_ = DEFAULT_LAMBDA
    years = lambda_to_halflife_years(lambda_)
    recovered = halflife_years_to_lambda(years)
    assert abs(recovered - lambda_) < 1e-10


def test_encode_target_vectorized(three_matches):
    target = encode_target(three_matches)
    # row 0: 2-1 home wins - 2; row 1: 1-0 home wins - 2; row 2: 0-1 away wins - 0
    assert target[0] == 2
    assert target[1] == 2
    assert target[2] == 0


def test_vectorized_elo_diff_matches_pointwise(three_matches):
    elo_df = calculate_elo_ratings(three_matches)
    diff = _vectorized_elo_diff(three_matches, elo_df)
    # Para cada fila, debe ser home_elo_before - away_elo_before del elo_df
    for i in range(len(three_matches)):
        expected = elo_df.iloc[i]["home_elo_before"] - elo_df.iloc[i]["away_elo_before"]
        assert abs(diff[i] - expected) < 1e-9


def test_build_match_features_no_nan(three_matches):
    feats = build_match_features(three_matches, year_cutoff=2019)
    required = [
        "elo_diff", "squad_value_diff", "xg_avg_for", "xg_avg_against",
        "travel_distance_home", "travel_distance_away", "ranking_diff",
        "time_weight", "target",
    ]
    for col in required:
        assert col in feats.columns
        assert feats[col].notna().all(), f"NaN encontrado en {col}"
