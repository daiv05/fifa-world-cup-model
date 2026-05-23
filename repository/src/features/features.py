"""
Pipeline de ingeniería de características.
Combina ELO, time decay, valor de plantilla, xG, distancia geográfica
y ranking FIFA en un único DataFrame listo para entrenamiento.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Garantiza que repository/ esté en sys.path sin importar desde dónde se ejecute
_repo_root = Path(__file__).parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.features.elo import calculate_elo_ratings, get_elo_at_date, INITIAL_RATING
from src.features.time_decay import compute_time_decay_weights

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WC2026_HOST_CITIES = {
    "United States": (38.9, -77.0),
    "Canada": (45.4, -75.7),
    "Mexico": (19.4, -99.1),
}
DEFAULT_HOST_COORDS = (34.0, -100.0)

# Coordenadas de las capitales (o ciudades principales) de los 48 selecciones
# clasificadas al Mundial 2026. Lookup instantáneo; evita dependencia de Nominatim
# para los equipos del torneo, donde geocoding fallaba en ~98 % de los casos.
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
    "Ivory Coast":          (5.36, -4.01),     # Abiyán (sede de gobierno)
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
    # 1. Dict hardcodeado: cobertura 100 % para los 48 equipos WC 2026
    if team in WC_TEAM_CAPITAL_COORDS:
        return WC_TEAM_CAPITAL_COORDS[team]
    # 2. Fallback Nominatim para equipos históricos de entrenamiento
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        import time

        geolocator = Nominatim(user_agent="wc2026_model")
        location = geolocator.geocode(team, timeout=5)
        if location:
            return (location.latitude, location.longitude)
        time.sleep(0.5)
    except Exception:
        pass
    return None


def compute_travel_distance(team: str, coords_cache: dict) -> float:
    """
    Calcula la distancia mínima desde el país del equipo hasta cualquiera
    de las ciudades sede del Mundial 2026 (USA, Canadá, México).
    """
    if team not in coords_cache:
        coords = _get_team_coords(team)
        coords_cache[team] = coords

    team_coords = coords_cache.get(team)
    if team_coords is None:
        return -1.0

    distances = [
        _haversine_km(team_coords[0], team_coords[1], lat, lon)
        for lat, lon in WC2026_HOST_CITIES.values()
    ]
    return float(min(distances))


def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Codifica el resultado del partido como clase entera:
      2 → victoria local, 1 → empate, 0 → victoria visitante
    """
    def _encode(row):
        if row["home_score"] > row["away_score"]:
            return 2
        if row["home_score"] == row["away_score"]:
            return 1
        return 0

    return df.apply(_encode, axis=1).astype(int)


def build_match_features(
    matches_df: pd.DataFrame,
    xg_df: pd.DataFrame | None = None,
    squad_df: pd.DataFrame | None = None,
    ranking_df: pd.DataFrame | None = None,
    lambda_decay: float = 0.002,
    year_cutoff: int = 1993,
) -> pd.DataFrame:
    """
    Construye el dataset de features a nivel de partido.

    Columnas de salida:
      elo_diff, squad_value_diff (log), xg_avg_for, xg_avg_against,
      travel_distance_home, travel_distance_away, ranking_diff, time_weight, target

    Parámetros
    ----------
    matches_df   : resultados históricos (date, home_team, away_team,
                   home_score, away_score, tournament)
    xg_df        : xG promedio por equipo (team, xg_for, xg_against)
    squad_df     : valor de plantilla (team, squad_value_eur)
    ranking_df   : ranking FIFA histórico; salida de load_fifa_ranking()
                   (team, rank, rank_date). Si None, ranking_diff = 0.
    lambda_decay : tasa de decaimiento temporal
    year_cutoff  : año mínimo de partidos a incluir
    """
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= year_cutoff].dropna(
        subset=["home_score", "away_score"]
    ).reset_index(drop=True)

    print("Calculando ELO histórico...")
    elo_df = calculate_elo_ratings(
        matches_df[matches_df["date"].dt.year >= 1872].sort_values("date")
        if "date" in matches_df.columns else df
    )

    print("Calculando time decay weights...")
    df["time_weight"] = compute_time_decay_weights(df["date"], lambda_=lambda_decay)

    print("Uniendo ELO por partido...")
    def elo_diff(row):
        h = get_elo_at_date(elo_df, row["home_team"], row["date"])
        a = get_elo_at_date(elo_df, row["away_team"], row["date"])
        return h - a

    df["elo_diff"] = df.apply(elo_diff, axis=1)

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

        def _rank_diff(row):
            h = get_ranking_at_date(ranking_dict, row["home_team"], row["date"])
            a = get_ranking_at_date(ranking_dict, row["away_team"], row["date"])
            return a - h   # positivo = local mejor rankeado (menor número = mejor)

        print("Calculando ranking_diff...")
        df["ranking_diff"] = df.apply(_rank_diff, axis=1)
    else:
        df["ranking_diff"] = 0.0

    print("Calculando distancias de viaje...")
    coords_cache: dict = {}
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    for team in all_teams:
        compute_travel_distance(team, coords_cache)

    df["travel_distance_home"] = df["home_team"].map(
        lambda t: compute_travel_distance(t, coords_cache)
    )
    df["travel_distance_away"] = df["away_team"].map(
        lambda t: compute_travel_distance(t, coords_cache)
    )

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
    Columns: team, elo, squad_value_eur, xg_for, xg_against, travel_distance, rank
    """
    matches_df = matches_df.copy()
    matches_df["date"] = pd.to_datetime(matches_df["date"])

    elo_df = calculate_elo_ratings(matches_df.sort_values("date"))
    ref_date = pd.Timestamp("2026-06-11")  # WC 2026 kickoff

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

    # Ranking FIFA: usar la posición más reciente disponible (hasta fecha WC kickoff)
    if ranking_df is not None and not ranking_df.empty:
        from src.data.data_loader import build_ranking_dict, get_ranking_at_date
        ranking_dict = build_ranking_dict(ranking_df)
        get_rank = lambda team: get_ranking_at_date(ranking_dict, team, ref_date)
    else:
        get_rank = lambda team: 78  # mediana de 156 equipos

    coords_cache: dict = {}
    records = []
    for team in all_teams:
        records.append({
            "team": team,
            "elo": get_elo_at_date(elo_df, team, ref_date),
            "squad_value_eur": sq_map.get(team, 50_000_000),
            "xg_for": xg_for_map.get(team, 1.2),
            "xg_against": xg_against_map.get(team, 1.2),
            "travel_distance": compute_travel_distance(team, coords_cache),
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
    save_features(features)

    from src.simulation.tournament import GROUPS_2026
    wc_teams = [t for ts in GROUPS_2026.values() for t in ts]
    print("Construyendo team_features para simulación...")
    team_feats = build_team_features_for_simulation(
        matches, xg_df=xg_df, squad_df=squad_df, ranking_df=ranking_df, teams=wc_teams,
    )
    save_features(team_feats, "team_features.csv")
    print(team_feats.to_string(index=False))
