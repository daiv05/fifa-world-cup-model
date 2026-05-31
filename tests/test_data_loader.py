import pandas as pd
import pytest

from src.data.data_loader import (
    standardize_team_names,
    filter_relevant_matches,
    build_ranking_dict,
    get_ranking_at_date,
    load_fifa_ranking,
    load_fifa_rankings_2026,
    TEAM_NAME_ALIASES,
    FORMER_NAME_MAP,
)
from src.features.time_decay import SNAPSHOT_DATE

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


# --------------------------------------------------------------------------- #
# Acción 5: continuidad histórica del ELO (former_names curado)
# --------------------------------------------------------------------------- #
def test_former_names_mapping_curated():
    df = pd.DataFrame({
        "home_team": ["Yugoslavia", "Czechoslovakia", "Soviet Union", "Zaire"],
        "away_team": ["Netherlands Antilles", "Brazil", "Germany", "France"],
    })
    out = standardize_team_names(df)
    assert list(out["home_team"]) == ["Serbia", "Czech Republic", "Russia", "DR Congo"]
    assert out["away_team"].iloc[0] == "Curacao"


def test_cabo_verde_alias():
    df = pd.DataFrame({"team": ["Cabo Verde"]})
    assert standardize_team_names(df)["team"].iloc[0] == "Cape Verde"


def test_former_name_map_targets_are_canonical():
    # Ningún destino debe ser a su vez un nombre histórico a remapear.
    assert set(FORMER_NAME_MAP.values()).isdisjoint(set(FORMER_NAME_MAP.keys()))


# --------------------------------------------------------------------------- #
# Acción 2: snapshot del ranking FIFA 2026 (leak-free)
# --------------------------------------------------------------------------- #
def test_snapshot_2026_parsing():
    snap = load_fifa_rankings_2026()
    assert list(snap.columns) == ["team", "rank", "total_points", "rank_date"]
    assert len(snap) > 200
    assert (snap["rank_date"] == SNAPSHOT_DATE).all()
    assert snap["rank"].min() == 1


def test_ranking_series_extends_to_snapshot():
    r = load_fifa_ranking()
    assert r["rank_date"].max() == SNAPSHOT_DATE


def test_ranking_snapshot_is_leak_free():
    r = load_fifa_ranking()
    rd = build_ranking_dict(r)
    # Un lookup en el horizonte del Mundial refleja el snapshot...
    assert get_ranking_at_date(rd, "France", pd.Timestamp("2026-06-15")) == 1
    # ...pero un lookup histórico NO es afectado por el slice futuro.
    rank_2015 = get_ranking_at_date(rd, "France", pd.Timestamp("2015-01-01"))
    assert rank_2015 != 1
    assert rank_2015 > 0
