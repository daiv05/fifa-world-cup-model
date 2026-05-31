import bisect
import pandas as pd
from pathlib import Path

from src.features.features import OUTPUT_ROW_START_YEAR
from src.features.time_decay import SNAPSHOT_DATE

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

NEW_DATA_DIR = RAW_DIR / "new-data" / "international_results"
FIFA_RANKINGS_2026_PATH = NEW_DATA_DIR / "fifa_rankings_2026.csv"

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
    # El snapshot FIFA 2026 usa "Cabo Verde"; el resto del proyecto usa "Cape Verde".
    "Cabo Verde": "Cape Verde",
}

# Mapeo curado de nombres históricos -> entidad actual, para que el ELO acumule
# de forma continua a través de cambios de nombre/federación. Se aplica ANTES de
# calcular el ELO (vía standardize_team_names en load_international_results).
#
# Solo se incluyen sucesiones limpias y reconocidas por FIFA, relevantes para el
# horizonte de modelado:
#   - Yugoslavia (SFR, 1920-1992) -> Serbia. FIFA reconoce a Serbia como sucesora
#     de Yugoslavia y Serbia y Montenegro. No hay coexistencia real: Yugoslavia
#     termina en 1992 y la cadena moderna de "Serbia" empieza en 1994. Montenegro
#     se trata como entidad NUEVA desde 2007 (no se mapea).
#   - Czechoslovakia (1903-1993) -> Czech Republic. Sucesor reconocido por FIFA;
#     Eslovaquia se trata como entidad NUEVA desde 1993 (no se mapea). Sin
#     solapamiento de fechas con "Czech Republic" (que empieza en 1994).
#
# Las entradas restantes son no-ops sobre el export cacheado actual (que ya usa
# los nombres modernos: URSS->Russia, Zaire->DR Congo, Antillas->Curacao), pero
# se documentan y mantienen para robustez si la fuente cruda se reemplaza por un
# export con nombres históricos. Provenance: data/raw/new-data/international_results/former_names.csv
FORMER_NAME_MAP: dict[str, str] = {
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Soviet Union": "Russia",
    "CIS": "Russia",
    "Zaire": "DR Congo",
    "Zaïre": "DR Congo",
    "Congo-Kinshasa": "DR Congo",
    "Netherlands Antilles": "Curacao",
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
    """
    Normaliza nombres de equipos. Aplica primero los alias de grafía
    (TEAM_NAME_ALIASES) y luego el mapeo de continuidad histórica
    (FORMER_NAME_MAP) para que el ELO acumule de forma continua a través de
    sucesiones de federación reconocidas por FIFA.
    """
    for col in ("home_team", "away_team", "team"):
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_ALIASES).replace(FORMER_NAME_MAP)
    return df


def filter_relevant_matches(df: pd.DataFrame, year_cutoff: int = OUTPUT_ROW_START_YEAR) -> pd.DataFrame:
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


def load_goalscorers(path: str | Path | None = None) -> pd.DataFrame:
    """Goleadores históricos (un registro por gol). Nombres normalizados."""
    if path is None:
        path = NEW_DATA_DIR / "goalscorers.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = standardize_team_names(df)  # actúa sobre home_team, away_team y team
    return df


def load_shootouts(path: str | Path | None = None) -> pd.DataFrame:
    """Tandas de penales históricas. Nombres normalizados (incl. `winner`)."""
    if path is None:
        path = NEW_DATA_DIR / "shootouts.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = standardize_team_names(df)
    # `winner` también puede contener nombres históricos.
    if "winner" in df.columns:
        df["winner"] = df["winner"].replace(TEAM_NAME_ALIASES).replace(FORMER_NAME_MAP)
    return df


def load_wc_matches(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "wc_matches_1974_2022.csv"
    header_cols = pd.read_csv(path, nrows=0).columns
    parse_dates = ["date"] if "date" in header_cols else False
    df = pd.read_csv(path, parse_dates=parse_dates)
    df = standardize_team_names(df)
    return df


def load_fifa_rankings_2026(path: str | Path | None = None) -> pd.DataFrame:
    """
    Carga el snapshot del ranking FIFA al 2026-05-30 (SNAPSHOT_DATE) y lo
    devuelve con el MISMO esquema que la serie histórica:
        team, rank, total_points, rank_date

    Notas de parseo del archivo crudo `fifa_rankings_2026.csv`:
      - Cada fila trae una coma final de más (5 campos para un header de 4), por
        lo que se nombran las columnas explícitamente y se descarta el sobrante.
      - El archivo está en latin-1 (p.ej. "Türkiye" se corrompe en utf-8).
      - La columna "upcoming" es ruido de scraping y se descarta.
      - El archivo NO trae fecha; se le asigna SNAPSHOT_DATE.
    """
    if path is None:
        path = FIFA_RANKINGS_2026_PATH
    df = pd.read_csv(
        path,
        header=0,
        names=["rank", "team", "upcoming", "points", "_junk"],
        usecols=["rank", "team", "points"],
        encoding="latin-1",
    )
    df = df.dropna(subset=["rank", "team"])
    df["rank"] = df["rank"].astype(int)
    df = df.rename(columns={"points": "total_points"})
    df["rank_date"] = SNAPSHOT_DATE
    df = standardize_team_names(df)
    return df[["team", "rank", "total_points", "rank_date"]]


def load_fifa_ranking(path: str | Path | None = None, include_2026_snapshot: bool = True) -> pd.DataFrame:
    """
    Serie temporal del ranking FIFA. La fuente histórica `fifa_ranking.csv`
    termina el 2024-06-20; cuando `include_2026_snapshot=True` se anexa el
    snapshot 2026-05-30 (`fifa_rankings_2026.csv`) como un slice más.

    Esto refresca el ranking en el horizonte de predicción sin leakage: los
    consumidores usan `merge_asof(direction="backward")` / bisect, así que un
    partido en fecha D solo ve ranks con rank_date <= D. El slice de 2026 solo
    aplica a partidos en/después de 2026-05-30 (es decir, el propio Mundial).
    """
    if path is None:
        path = RAW_DIR / "fifa_ranking.csv"
    df = pd.read_csv(path)
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df = df.rename(columns={"country_full": "team"})
    df = standardize_team_names(df)          # ahora actúa sobre columna "team"
    df = df[["team", "rank", "total_points", "rank_date"]]

    if include_2026_snapshot and FIFA_RANKINGS_2026_PATH.exists():
        snap = load_fifa_rankings_2026()
        df = pd.concat([df, snap], ignore_index=True)

    return df.sort_values("rank_date").reset_index(drop=True)


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
