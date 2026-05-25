"""
Combina ELO, time decay, valor de plantilla, xG, distancia geográfica y ranking FIFA
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.features.elo import calculate_elo_ratings, INITIAL_RATING
from src.features.time_decay import compute_time_decay_weights, REFERENCE_DATE, DEFAULT_LAMBDA

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WC2026_HOST_CITIES = {
    "United States": (38.9, -77.0),
    "Canada": (45.4, -75.7),
    "Mexico": (19.4, -99.1),
}
DEFAULT_HOST_COORDS = (34.0, -100.0)

WC_TEAM_CAPITAL_COORDS: dict[str, tuple[float, float]] = {

    # Grupo A
    "Mexico":               (19.43, -99.13),   # Ciudad de México
    "South Africa":         (-25.74, 28.19),   # Pretoria
    "South Korea":          (37.57, 126.98),   # Seúl
    "Czech Republic":       (50.08, 14.44),    # Praga
    # Grupo B
    "Canada":               (45.42, -75.70),   # Ottawa
    "Bosnia & Herzegovina": (43.85, 18.36),    # Sarajevo
    "Qatar":                (25.29, 51.53),    # Doha
    "Switzerland":          (46.95, 7.45),     # Berna
    # Grupo C
    "Brazil":               (-15.78, -47.93),  # Brasilia
    "Morocco":              (34.02, -6.84),    # Rabat
    "Haiti":                (18.54, -72.34),   # Puerto Príncipe
    "Scotland":             (55.95, -3.19),    # Edimburgo
    # Grupo D
    "United States":        (38.89, -77.03),   # Washington D.C.
    "Paraguay":             (-25.28, -57.64),  # Asunción
    "Australia":            (-35.28, 149.13),  # Canberra
    "Turkey":               (39.93, 32.86),    # Ankara
    # Grupo E
    "Germany":              (52.52, 13.41),    # Berlín
    "Curacao":              (12.11, -68.93),   # Willemstad
    "Ivory Coast":          (5.36, -4.01),     # Abiyán
    "Ecuador":              (-0.23, -78.52),   # Quito
    # Grupo F
    "Netherlands":          (52.37, 4.90),     # Ámsterdam
    "Japan":                (35.69, 139.69),   # Tokio
    "Sweden":               (59.33, 18.06),    # Estocolmo
    "Tunisia":              (36.82, 10.17),    # Túnez
    # Grupo G
    "Belgium":              (50.85, 4.35),     # Bruselas
    "Egypt":                (30.04, 31.24),    # El Cairo
    "Iran":                 (35.69, 51.39),    # Teherán
    "New Zealand":          (-41.29, 174.78),  # Wellington
    # Grupo H
    "Spain":                (40.41, -3.70),    # Madrid
    "Cape Verde":           (14.93, -23.51),   # Praia
    "Saudi Arabia":         (24.69, 46.72),    # Riad
    "Uruguay":              (-34.90, -56.19),  # Montevideo
    # Grupo I
    "France":               (48.85, 2.35),     # París
    "Senegal":              (14.71, -17.47),   # Dakar
    "Iraq":                 (33.34, 44.40),    # Bagdad
    "Norway":               (59.91, 10.75),    # Oslo
    # Grupo J
    "Argentina":            (-34.61, -58.38),  # Buenos Aires
    "Algeria":              (36.74, 3.06),     # Argel
    "Austria":              (48.21, 16.37),    # Viena
    "Jordan":               (31.95, 35.93),    # Amán
    # Grupo K
    "Portugal":             (38.72, -9.14),    # Lisboa
    "DR Congo":             (-4.32, 15.32),    # Kinshasa
    "Uzbekistan":           (41.30, 69.24),    # Taskent
    "Colombia":             (4.71, -74.07),    # Bogotá
    # Grupo L
    "England":              (51.51, -0.13),    # Londres
    "Croatia":              (45.81, 15.98),    # Zagreb
    "Ghana":                (5.55, -0.20),     # Acra
    "Panama":               (8.99, -79.52),    # Ciudad de Panamá
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _get_team_coords(team: str) -> tuple[float, float] | None:
    """
    Devuelve coordenadas (lat, lon) de la capital del equipo.
    Primero consulta WC_TEAM_CAPITAL_COORDS (instantáneo, sin red) para los 48
    clasificados al WC 2026. Para equipos históricos fuera del torneo usa
    Nominatim como fallback.
    """
    if team in WC_TEAM_CAPITAL_COORDS:
        return WC_TEAM_CAPITAL_COORDS[team]
    try:
        from geopy.geocoders import Nominatim
        import time

        geolocator = Nominatim(user_agent="wc2026_model")
        location = geolocator.geocode(team, timeout=5)
        if location:
            time.sleep(0.5)
            return (location.latitude, location.longitude)
    except Exception:
        pass
    return None


def _get_country_coords(country: str, cache: dict) -> tuple[float, float] | None:
    """Coordenadas del país donde se juega el partido. Reusa _get_team_coords
    porque la mayoría de equipos coincide con su país."""
    if country in cache:
        return cache[country]
    coords = _get_team_coords(country)
    cache[country] = coords
    return coords


def compute_host_distance_wc2026(team: str, coords_cache: dict) -> float:
    """
    Distancia mínima desde la capital del equipo hasta cualquiera de las
    sedes del Mundial 2026 (USA, Canadá, México). Usada para `team_features`.
    """
    if team not in coords_cache:
        coords_cache[team] = _get_team_coords(team)
    team_coords = coords_cache.get(team)
    if team_coords is None:
        return 0.0
    distances = [
        _haversine_km(team_coords[0], team_coords[1], lat, lon)
        for lat, lon in WC2026_HOST_CITIES.values()
    ]
    return float(min(distances))


def encode_target(df: pd.DataFrame) -> np.ndarray:
    """
    Codifica el resultado del partido:
      2 - victoria local, 1 - empate, 0 - victoria visitante
    """
    home = df["home_score"].values
    away = df["away_score"].values
    return np.select(
        [home > away, home == away],
        [2, 1],
        default=0,
    ).astype(int)


def _vectorized_elo_diff(
    matches_df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> np.ndarray:
    """
    Devuelve un vector elo_diff (home_elo_before - away_elo_before) para cada
    fila de `matches_df`, hecho merge contra `elo_df` (output de
    calculate_elo_ratings) por (date, home_team, away_team).
    """
    key_cols = ["date", "home_team", "away_team"]
    elo_keyed = elo_df[key_cols + ["home_elo_before", "away_elo_before"]].copy()
    elo_keyed["date"] = pd.to_datetime(elo_keyed["date"])

    left = matches_df[key_cols].copy()
    left["date"] = pd.to_datetime(left["date"])
    left["_row"] = np.arange(len(left))

    merged = left.merge(elo_keyed, on=key_cols, how="left")
    # Para los partidos sin entrada en elo_df (no debería pasar si elo_df
    # se construyó con los mismos matches), fallback a INITIAL_RATING.
    merged["home_elo_before"] = merged["home_elo_before"].fillna(INITIAL_RATING)
    merged["away_elo_before"] = merged["away_elo_before"].fillna(INITIAL_RATING)
    merged = merged.sort_values("_row")
    return (merged["home_elo_before"] - merged["away_elo_before"]).values


def _vectorized_travel_distances(
    matches_df: pd.DataFrame,
    team_coords_cache: dict,
    country_coords_cache: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada partido, calcula distancia desde la capital del equipo a la
    sede real del partido (`country`). Si el campo `neutral` es True o el
    país no se puede geocodificar, devuelve 0.0 (campo neutral).
    """
    n = len(matches_df)
    home_dist = np.zeros(n, dtype=np.float64)
    away_dist = np.zeros(n, dtype=np.float64)

    has_country = "country" in matches_df.columns
    has_neutral = "neutral" in matches_df.columns

    homes = matches_df["home_team"].values
    aways = matches_df["away_team"].values
    countries = matches_df["country"].values if has_country else [None] * n
    neutrals = matches_df["neutral"].values if has_neutral else [False] * n

    for i in range(n):
        if not has_country or neutrals[i] or pd.isna(countries[i]):
            continue
        venue = _get_country_coords(countries[i], country_coords_cache)
        if venue is None:
            continue

        h_coords = team_coords_cache.get(homes[i])
        if h_coords is None:
            h_coords = _get_team_coords(homes[i])
            team_coords_cache[homes[i]] = h_coords
        a_coords = team_coords_cache.get(aways[i])
        if a_coords is None:
            a_coords = _get_team_coords(aways[i])
            team_coords_cache[aways[i]] = a_coords

        if h_coords is not None:
            home_dist[i] = _haversine_km(h_coords[0], h_coords[1], venue[0], venue[1])
        if a_coords is not None:
            away_dist[i] = _haversine_km(a_coords[0], a_coords[1], venue[0], venue[1])

    return home_dist, away_dist


def data_quality_report(df: pd.DataFrame, name: str = "dataset") -> None:
    """Imprime un reporte simple de calidad del dataset."""
    print(f"\n=== Data quality report: {name} ===")
    print(f"  Filas: {len(df):,}")
    print(f"  Columnas: {list(df.columns)}")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print(f"  Nulos por columna:")
        for c, n in nulls.items():
            print(f"    {c}: {n}")
    else:
        print("  Sin nulos.")
    dups = df.duplicated().sum()
    print(f"  Duplicados exactos: {dups}")
    if "date" in df.columns:
        d = pd.to_datetime(df["date"])
        print(f"  Rango de fechas: {d.min().date()} - {d.max().date()}")
    for col in ("home_team", "away_team", "team"):
        if col in df.columns:
            print(f"  Equipos únicos en {col}: {df[col].nunique()}")
    print()


def build_match_features(
    matches_df: pd.DataFrame,
    xg_df: pd.DataFrame | None = None,
    squad_df: pd.DataFrame | None = None,
    ranking_df: pd.DataFrame | None = None,
    lambda_decay: float = DEFAULT_LAMBDA,
    year_cutoff: int = 1993,
) -> pd.DataFrame:
    """
    Construye el dataset de features a nivel de partido.

    Columnas de salida:
      date, home_team, away_team, elo_diff, squad_value_diff, xg_avg_for,
      xg_avg_against, travel_distance_home, travel_distance_away,
      ranking_diff, time_weight, target
    """
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= year_cutoff].dropna(
        subset=["home_score", "away_score"]
    ).reset_index(drop=True)

    print("Calculando ELO histórico (una sola pasada cronológica)...")
    all_for_elo = matches_df.copy()
    all_for_elo["date"] = pd.to_datetime(all_for_elo["date"])
    all_for_elo = all_for_elo.sort_values("date")
    elo_df = calculate_elo_ratings(all_for_elo)

    print("Mergeando ELO por partido (vectorizado)...")
    df["elo_diff"] = _vectorized_elo_diff(df, elo_df)

    print("Calculando time decay weights...")
    df["time_weight"] = compute_time_decay_weights(df["date"], lambda_=lambda_decay)

    if xg_df is not None and not xg_df.empty:
        xg_map_for = xg_df.set_index("team")["xg_for"].to_dict()
        xg_map_against = xg_df.set_index("team")["xg_against"].to_dict()
        df["xg_avg_for"] = (
            df["home_team"].map(xg_map_for).fillna(1.2)
            - df["away_team"].map(xg_map_for).fillna(1.2)
        )
        df["xg_avg_against"] = (
            df["home_team"].map(xg_map_against).fillna(1.2)
            - df["away_team"].map(xg_map_against).fillna(1.2)
        )
    else:
        df["xg_avg_for"] = 0.0
        df["xg_avg_against"] = 0.0

    if squad_df is not None and not squad_df.empty:
        val_map = squad_df.set_index("team")["squad_value_eur"].to_dict()
        log_val = {k: np.log1p(v) for k, v in val_map.items()}
        default_val = np.log1p(squad_df["squad_value_eur"].median())
        df["squad_value_diff"] = (
            df["home_team"].map(log_val).fillna(default_val)
            - df["away_team"].map(log_val).fillna(default_val)
        )
    else:
        df["squad_value_diff"] = 0.0

    if ranking_df is not None and not ranking_df.empty:
        from src.data.data_loader import build_ranking_dict, get_ranking_at_date
        ranking_dict = build_ranking_dict(ranking_df)

        print("Calculando ranking_diff...")
        ranks_h = df.apply(
            lambda r: get_ranking_at_date(ranking_dict, r["home_team"], r["date"]),
            axis=1,
        )
        ranks_a = df.apply(
            lambda r: get_ranking_at_date(ranking_dict, r["away_team"], r["date"]),
            axis=1,
        )
        df["ranking_diff"] = (ranks_a - ranks_h).astype(float)
    else:
        df["ranking_diff"] = 0.0

    print("Calculando distancias de viaje (sede real del partido)...")
    team_coords_cache: dict = {}
    country_coords_cache: dict = {}
    # Precalentar cache con equipos del WC 2026
    for t in WC_TEAM_CAPITAL_COORDS:
        team_coords_cache[t] = WC_TEAM_CAPITAL_COORDS[t]
    home_dist, away_dist = _vectorized_travel_distances(
        df, team_coords_cache, country_coords_cache
    )
    df["travel_distance_home"] = home_dist
    df["travel_distance_away"] = away_dist

    df["target"] = encode_target(df)

    feature_cols = [
        "date", "home_team", "away_team",
        "elo_diff", "squad_value_diff",
        "xg_avg_for", "xg_avg_against",
        "travel_distance_home", "travel_distance_away",
        "ranking_diff",
        "time_weight", "target",
    ]
    result = df[[c for c in feature_cols if c in df.columns]]
    return result


def build_team_features_for_simulation(
    matches_df: pd.DataFrame,
    xg_df: pd.DataFrame | None = None,
    squad_df: pd.DataFrame | None = None,
    ranking_df: pd.DataFrame | None = None,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """
    Builds a per-team feature row for the Monte Carlo simulation.
    Columns: team, elo, squad_value_eur, xg_for, xg_against, host_distance, rank
    """
    matches_df = matches_df.copy()
    matches_df["date"] = pd.to_datetime(matches_df["date"])

    elo_df = calculate_elo_ratings(matches_df.sort_values("date"))
    ref_date = REFERENCE_DATE

    # Construir un mapping team - último ELO antes de ref_date
    elo_long = pd.concat([
        elo_df[["date", "home_team", "home_elo_after"]].rename(
            columns={"home_team": "team", "home_elo_after": "elo"}
        ),
        elo_df[["date", "away_team", "away_elo_after"]].rename(
            columns={"away_team": "team", "away_elo_after": "elo"}
        ),
    ])
    elo_long = elo_long[elo_long["date"] <= ref_date].sort_values("date")
    last_elo = elo_long.groupby("team")["elo"].last().to_dict()

    all_teams = teams or sorted(
        set(matches_df["home_team"]) | set(matches_df["away_team"])
    )

    sq_map = (
        squad_df.set_index("team")["squad_value_eur"].to_dict()
        if squad_df is not None and not squad_df.empty
        else {}
    )
    xg_for_map = (
        xg_df.set_index("team")["xg_for"].to_dict()
        if xg_df is not None and not xg_df.empty
        else {}
    )
    xg_against_map = (
        xg_df.set_index("team")["xg_against"].to_dict()
        if xg_df is not None and not xg_df.empty
        else {}
    )

    if ranking_df is not None and not ranking_df.empty:
        from src.data.data_loader import build_ranking_dict, get_ranking_at_date
        ranking_dict = build_ranking_dict(ranking_df)
        get_rank = lambda team: get_ranking_at_date(ranking_dict, team, ref_date)
    else:
        get_rank = lambda team: 78

    coords_cache: dict = dict(WC_TEAM_CAPITAL_COORDS)
    records = []
    for team in all_teams:
        records.append({
            "team": team,
            "elo": last_elo.get(team, INITIAL_RATING),
            "squad_value_eur": sq_map.get(team, 50_000_000),
            "xg_for": xg_for_map.get(team, 1.2),
            "xg_against": xg_against_map.get(team, 1.2),
            "host_distance": compute_host_distance_wc2026(team, coords_cache),
            "rank": get_rank(team),
        })

    return pd.DataFrame(records)


def save_features(df: pd.DataFrame, filename: str = "features.csv") -> Path:
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"Features guardadas en {path}")
    return path


if __name__ == "__main__":
    from src.data.data_loader import (
        load_international_results, filter_relevant_matches, load_fifa_ranking,
    )
    from src.data.scraper import get_statsbomb_xg_by_team, get_squad_values

    print("Cargando datos históricos...")
    matches = load_international_results()
    matches = filter_relevant_matches(matches, year_cutoff=1993)
    data_quality_report(matches, "international_results (filtrado)")

    xg_df = get_statsbomb_xg_by_team()
    squad_df = get_squad_values()
    ranking_df = load_fifa_ranking()
    print(f"  Ranking FIFA cargado: {len(ranking_df):,} filas, "
          f"{ranking_df['team'].nunique()} equipos únicos")

    print("Construyendo features...")
    features = build_match_features(
        matches, xg_df=xg_df, squad_df=squad_df, ranking_df=ranking_df,
    )
    print(f"  Partidos con features: {len(features):,}")
    print(f"  Distribución del target:\n{features['target'].value_counts()}")
    data_quality_report(features, "features")
    save_features(features)

    from src.simulation.tournament import GROUPS_2026
    wc_teams = [t for ts in GROUPS_2026.values() for t in ts]
    print("Construyendo team_features para simulación...")
    team_feats = build_team_features_for_simulation(
        matches, xg_df=xg_df, squad_df=squad_df, ranking_df=ranking_df, teams=wc_teams,
    )
    save_features(team_feats, "team_features.csv")
    print(team_feats.to_string(index=False))
