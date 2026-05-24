import pandas as pd
import pytest

from src.data.data_loader import (
    standardize_team_names,
    filter_relevant_matches,
    TEAM_NAME_ALIASES,
)

@pytest.fixture
def sample_matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2010-06-01", "1985-03-15"]),
        "home_team": ["USA", "IR Iran", "Brazil"],
        "away_team": ["Korea Republic", "Germany", "Argentina"],
        "home_score": [1, 0, 2],
        "away_score": [0, 1, 1],
        "tournament": ["FIFA World Cup", "FIFA World Cup", "Friendly"],
    })

def test_standardize_replaces_aliases(sample_matches):
    result = standardize_team_names(sample_matches)
    assert "United States" in result["home_team"].values
    assert "Iran" in result["home_team"].values
    assert "South Korea" in result["away_team"].values


def test_standardize_preserves_unknown_names(sample_matches):
    result = standardize_team_names(sample_matches)
    assert "Brazil" in result["home_team"].values
    assert "Germany" in result["away_team"].values


def test_filter_year_cutoff(sample_matches):
    standardize_team_names(sample_matches)
    filtered = filter_relevant_matches(sample_matches, year_cutoff=2000)
    assert all(filtered["date"].dt.year >= 2000)
    assert len(filtered) == 2


def test_filter_relevant_tournaments(sample_matches):
    filtered = filter_relevant_matches(sample_matches, year_cutoff=1980)
    assert len(filtered) == 2
    assert all(
        any(k.lower() in t.lower() for k in ["World Cup", "Euro", "Copa", "Africa", "Asian", "Gold", "Nations"])
        for t in filtered["tournament"]
    )


def test_team_aliases_dict_has_no_duplicates():
    assert len(TEAM_NAME_ALIASES) == len(set(TEAM_NAME_ALIASES.keys()))
