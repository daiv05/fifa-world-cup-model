import bisect
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

TEAM_NAME_ALIASES: dict[str, str] = {
    # Nombres en datos históricos - nombre canónico del proyecto
    "USA": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia": "Bosnia & Herzegovina",
    "Czechia": "Czech Republic",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
    "Venezuela (Bolivarian Republic)": "Venezuela",
    "China PR": "China",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Antigua and Barbuda": "Antigua & Barbuda",
}

RELEVANT_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "Confederations Cup",
    "Nations League",
    "UEFA Nations League",
    "CONMEBOL",
    "CONCACAF Nations League",
}


def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("home_team", "away_team", "team"):
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_ALIASES)
    return df


def filter_relevant_matches(df: pd.DataFrame, year_cutoff: int = 1990) -> pd.DataFrame:
    df = df[df["date"].dt.year >= year_cutoff].copy()
    if "tournament" in df.columns:
        mask = df["tournament"].apply(
            lambda t: any(k.lower() in t.lower() for k in RELEVANT_TOURNAMENTS)
        )
        df = df[mask].copy()
    return df.reset_index(drop=True)


def load_international_results(use_cache: bool = True) -> pd.DataFrame:
    cache_path = RAW_DIR / "international_results.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        df = pd.read_csv(INTERNATIONAL_RESULTS_URL, parse_dates=["date"])
        df.to_csv(cache_path, index=False)
    df = standardize_team_names(df)
    return df


def load_wc_matches(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "wc_matches_1974_2022.csv"
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else False)
    df = standardize_team_names(df)
    return df


def load_fifa_ranking(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "fifa_ranking.csv"
    df = pd.read_csv(path)
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df = df.rename(columns={"country_full": "team"})
    df = standardize_team_names(df)          # ahora actúa sobre columna "team"
    return df[["team", "rank", "total_points", "rank_date"]]


def build_ranking_dict(ranking_df: pd.DataFrame) -> dict[str, list[tuple]]:
    result: dict[str, list[tuple]] = {}
    for team, grp in ranking_df.groupby("team"):
        sorted_grp = grp.dropna(subset=["rank"]).sort_values("rank_date")
        if sorted_grp.empty:
            continue
        result[team] = list(zip(sorted_grp["rank_date"], sorted_grp["rank"].astype(int)))
    return result


def get_ranking_at_date(
    ranking_dict: dict,
    team: str,
    date: pd.Timestamp,
    default_rank: int = 78,
) -> int:
    entries = ranking_dict.get(team, [])
    if not entries:
        return default_rank
    idx = bisect.bisect_right(entries, (date, float("inf"))) - 1
    return entries[idx][1] if idx >= 0 else default_rank


def load_wc2026_fixture(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "wc2026_fixture.csv"
    df = pd.read_csv(path)
    df = standardize_team_names(df)
    return df


if __name__ == "__main__":
    print("Descargando resultados internacionales...")
    df = load_international_results(use_cache=False)
    print(f"Total partidos: {len(df):,}")

    df_filtered = filter_relevant_matches(df, year_cutoff=1990)
    print(f"Partidos relevantes desde 1990: {len(df_filtered):,}")
    print(f"Torneos únicos: {df_filtered['tournament'].nunique()}")
    print("OK")
