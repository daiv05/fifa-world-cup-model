"""
Combina ELO, time decay, valor de plantilla, xG, distancia geográfica y ranking FIFA
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

USE_NOMINATIM = os.environ.get("USE_NOMINATIM", "0") == "1"

from src.features.elo import calculate_elo_ratings, INITIAL_RATING
from src.features.time_decay import (
    compute_time_decay_weights, REFERENCE_DATE, SNAPSHOT_DATE, DEFAULT_LAMBDA,
)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GEO_CACHE_PATH = PROCESSED_DIR / "geo_coords_cache.json"

# --------------------------------------------------------------------------- #
# Constantes de horizonte temporal del pipeline.
#
# Hay TRES cortes distintos, con semánticas diferentes; no deben colapsarse en
# un único número:
#
#   ELO_HISTORY_START   -> universo sobre el que se acumula el ELO. None = TODA
#                          la historia disponible (todos los torneos, incl.
#                          amistosos, desde 1872). El ELO necesita la historia
#                          completa para estar bien calibrado; recortarlo
#                          reiniciaría los ratings y perdería señal.
#   OUTPUT_ROW_START_YEAR -> primer año en que un partido se EMITE como fila de
#                          entrenamiento. Pre-1993 la densidad de fixtures y la
#                          metodología de ranking son poco representativas, así
#                          que esos partidos alimentan el ELO (warm-up) pero no
#                          se usan como filas.
#   TRAIN_MIN_YEAR      -> piso de la ventana de MODELADO (train.py/ablation.py).
#                          Las features estáticas modernas (xG, squad_value,
#                          ranking actual) no representan el fútbol pre-2010, y
#                          el time_decay ya pondera esos partidos <2%.
# --------------------------------------------------------------------------- #
ELO_HISTORY_START: int | None = None
OUTPUT_ROW_START_YEAR = 1993
TRAIN_MIN_YEAR = 2010

# Fuente única de verdad para el conjunto de features de entrada al modelo.
# Importado por train.py, evaluate.py, ablation.py, simulate.py y dashboard.py
# para evitar listas duplicadas que se desincronicen.
#
# Retiradas en v2 (aplicando el criterio de retiro ya declarado en README):
#   - travel_distance_diff: 83% ceros en entrenamiento pero 100% poblada en
#     inferencia (shift de distribución train->simulación) y aporte ~nulo
#     (SHAP 0.03, ablación dentro del ruido).
#   - shootout_winrate_diff: casi constante (~0.5 tras shrinkage para las
#     potencias) y ablación dentro del ruido de una corrida.
FEATURE_COLS = [
    "elo_diff",
    "squad_value_diff",
    "xg_avg_for",
    "xg_avg_against",
    "ranking_diff",
    # Features derivadas de eventos (goleadores), as-of-date, leak-free.
    "penalty_share_diff",
    "striker_concentration_diff",
]

# Subconjunto de FEATURE_COLS proveniente de derived_stats (para fallbacks/ablación).
DERIVED_FEATURE_COLS = [
    "penalty_share_diff",
    "striker_concentration_diff",
]

# Media global aproximada de xG por partido. Única constante de imputación,
# compartida por la construcción de features y el Poisson de la simulación
# (antes había un 1.2 y un 1.25 inconsistentes).
LEAGUE_AVG_XG = 1.2

# Subconjunto de FEATURE_COLS con anacronismo conocido: snapshots estáticos
# (un valor por equipo, sin fecha) aplicados a toda la historia. El pipeline
# pre-2022 (validación WC2022 sin leakage) los excluye.
ANACHRONISTIC_FEATURE_COLS = [
    "squad_value_diff",
    "xg_avg_for",
    "xg_avg_against",
]

# Esquema completo del CSV a nivel de partido (features.csv).
MATCH_OUTPUT_COLS = [
    "date", "home_team", "away_team",
    *FEATURE_COLS,
    "time_weight", "target",
]


def _load_geo_cache() -> dict[str, tuple[float, float] | None]:
    if not GEO_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(GEO_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, tuple[float, float] | None] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        else:
            out[k] = (float(v[0]), float(v[1]))
    return out


def _save_geo_cache(cache: dict[str, tuple[float, float] | None]) -> None:
    serializable = {k: (list(v) if v is not None else None) for k, v in cache.items()}
    GEO_CACHE_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

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


def _haversine_km(lat1, lon1, lat2, lon2):
    """Acepta escalares o arrays numpy; devuelve el mismo tipo."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _get_team_coords(team: str) -> tuple[float, float] | None:
    """
    Devuelve coordenadas (lat, lon) de la capital del equipo.
    Primero consulta WC_TEAM_CAPITAL_COORDS (instantáneo, sin red) para los 48
    clasificados al WC 2026. Para equipos históricos fuera del torneo usa
    Nominatim como fallback SOLO si USE_NOMINATIM=1 (opt-in, por la latencia
    de red con sleep de 0.5s por petición).
    """
    if team in WC_TEAM_CAPITAL_COORDS:
        return WC_TEAM_CAPITAL_COORDS[team]
    if not USE_NOMINATIM:
        return None
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


def _resolve_coords_for_names(
    names: list[str],
    coords_cache: dict[str, tuple[float, float] | None],
) -> None:
    """
    Llena `coords_cache` con coordenadas para cada nombre único en `names`.
    Solo va a Nominatim para nombres que ni están en cache ni en
    WC_TEAM_CAPITAL_COORDS. Mutates `coords_cache` en sitio.
    """
    for name in names:
        if name in coords_cache:
            continue
        if name in WC_TEAM_CAPITAL_COORDS:
            coords_cache[name] = WC_TEAM_CAPITAL_COORDS[name]
            continue
        coords_cache[name] = _get_team_coords(name)


def _vectorized_travel_distance_diff(
    matches_df: pd.DataFrame,
    team_coords_cache: dict,
    country_coords_cache: dict,
) -> np.ndarray:
    """
    Distancia de viaje diferencial por partido: `dist(away->sede) - dist(home->sede)`
    (convención away - home), en km.

    Cambios de rigor frente a la versión anterior (que devolvía home/away por
    separado y los forzaba a 0 en partidos neutrales):
      - NO se anula en partidos neutrales. En un partido neutral ambos equipos
        viajan de verdad, así que la diferencia es señal real (justo el caso de
        los Mundiales, ~87% neutrales). La versión vieja dejaba la feature en 0
        precisamente donde más importa.
      - Para un partido de local normal, dist(home->sede)≈0, así que el diff ≈
        dist(away). Sigue siendo informativo y con signo consistente.
      - Si la sede (`country`) no se puede geocodificar, el diff de esa fila es
        0.0 (sin sede no se puede fabricar señal).

    Resolución de coordenadas: una sola pasada por nombres únicos (cache
    persistente en disco). Haversine totalmente vectorizado.
    """
    n = len(matches_df)
    diff = np.zeros(n, dtype=np.float64)

    if "country" not in matches_df.columns:
        return diff

    homes = matches_df["home_team"].astype(str).values
    aways = matches_df["away_team"].astype(str).values
    countries = matches_df["country"].astype(object).values

    # Único gate: que la sede sea geocodificable. El flag `neutral` ya NO anula.
    active = ~pd.isna(countries)
    if not active.any():
        return diff

    unique_teams = pd.unique(np.concatenate([homes[active], aways[active]]))
    unique_countries = pd.unique(countries[active].astype(str))

    _resolve_coords_for_names(list(unique_teams), team_coords_cache)
    _resolve_coords_for_names(list(unique_countries), country_coords_cache)

    def _coord_arrays(names: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
        lat_map = {k: (v[0] if v is not None else np.nan) for k, v in cache.items()}
        lon_map = {k: (v[1] if v is not None else np.nan) for k, v in cache.items()}
        s = pd.Series(names)
        return (
            np.asarray(s.map(lat_map), dtype=np.float64).copy(),
            np.asarray(s.map(lon_map), dtype=np.float64).copy(),
        )

    home_lat, home_lon = _coord_arrays(homes, team_coords_cache)
    away_lat, away_lon = _coord_arrays(aways, team_coords_cache)
    venue_lat, venue_lon = _coord_arrays(countries.astype(str), country_coords_cache)

    inactive = ~active
    home_lat[inactive] = np.nan
    away_lat[inactive] = np.nan
    venue_lat[inactive] = np.nan

    h = _haversine_km(home_lat, home_lon, venue_lat, venue_lon)
    a = _haversine_km(away_lat, away_lon, venue_lat, venue_lon)
    home_dist = np.where(np.isnan(h), 0.0, h)
    away_dist = np.where(np.isnan(a), 0.0, a)

    return away_dist - home_dist


def _vectorized_ranking_diff(
    matches_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    default_rank: int = 78,
) -> np.ndarray:
    """
    Calcula `away_rank - home_rank` para cada partido usando `pd.merge_asof`,
    evitando el O(n) en Python puro del bisect por fila.
    """
    r = ranking_df.dropna(subset=["rank"])[["team", "rank_date", "rank"]].copy()
    r["rank_date"] = pd.to_datetime(r["rank_date"])
    r["rank"] = r["rank"].astype(float)
    r = r.sort_values("rank_date").rename(columns={"rank_date": "date"})

    left = matches_df[["date", "home_team", "away_team"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    left["_row"] = np.arange(len(left))

    def _lookup(team_col: str) -> np.ndarray:
        side = left[["_row", "date", team_col]].rename(columns={team_col: "team"})
        side = side.sort_values("date")
        merged = pd.merge_asof(
            side, r, on="date", by="team", direction="backward",
        )
        s = merged.set_index("_row")["rank"].reindex(np.arange(len(left)))
        return s.fillna(default_rank).to_numpy()

    ranks_h = _lookup("home_team")
    ranks_a = _lookup("away_team")
    return (ranks_a - ranks_h).astype(float)


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
    year_cutoff: int = OUTPUT_ROW_START_YEAR,
    elo_matches_df: pd.DataFrame | None = None,
    goalscorers_df: pd.DataFrame | None = None,
    shootouts_df: pd.DataFrame | None = None,
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye el dataset de features a nivel de partido.

    Parámetros clave:
      matches_df: partidos que se EMITEN como filas (ya pueden venir filtrados
        a torneos relevantes). Se recortan además a `year >= year_cutoff`.
      elo_matches_df: universo sobre el que se acumula el ELO. Debe ser un
        SUPERCONJUNTO de matches_df (idealmente la historia cruda completa: todos
        los torneos incl. amistosos, todos los años) para que los ratings estén
        bien calibrados. Si es None, se usa `matches_df` (compat. retro / tests).

    Columnas de salida (MATCH_OUTPUT_COLS):
      date, home_team, away_team, elo_diff, squad_value_diff, xg_avg_for,
      xg_avg_against, ranking_diff, penalty_share_diff,
      striker_concentration_diff, time_weight, target
    """
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= year_cutoff].dropna(
        subset=["home_score", "away_score"]
    ).reset_index(drop=True)

    steps = tqdm(
        total=5,
        desc="build_match_features",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}",
    )

    steps.set_description("ELO histórico")
    # El ELO se acumula sobre el universo completo (elo_matches_df), no sobre el
    # subconjunto filtrado de filas a emitir. Así los amistosos y la historia
    # pre-cutoff contribuyen al rating (warm-up) sin emitirse como filas.
    # Si el caller ya lo calculó (elo_df), se reutiliza para evitar recomputarlo.
    if elo_df is None:
        all_for_elo = (elo_matches_df if elo_matches_df is not None else matches_df).copy()
        all_for_elo["date"] = pd.to_datetime(all_for_elo["date"])
        all_for_elo = all_for_elo.sort_values("date")
        elo_df = calculate_elo_ratings(all_for_elo)
    steps.update(1)

    steps.set_description("Merge ELO + time decay")
    df["elo_diff"] = _vectorized_elo_diff(df, elo_df)
    df["time_weight"] = compute_time_decay_weights(df["date"], lambda_=lambda_decay)
    steps.update(1)

    steps.set_description("xG / squad value")
    # LIMITACIÓN CONOCIDA (anacronismo): xg_df y squad_df son snapshots ESTÁTICOS
    # (un valor por equipo, sin fecha) que se aplican a TODOS los partidos
    # históricos. Un partido de 2012 recibe el xG/valor de 2026 -> leakage/
    # anacronismo. No existe serie temporal histórica de estos datos, así que
    # versionarlos sería fabricar información (se evita). El efecto está acotado
    # por: (a) time_decay pondera <2% los partidos pre-2010, (b) TRAIN_MIN_YEAR
    # restringe el modelado a una ventana reciente, (c) son features relativas
    # (diffs). Su aporte se cuantifica en la fila de ablación "Sin estáticas
    # anacrónicas (xG+squad)". El ELO NO usa estos snapshots.
    if xg_df is not None and not xg_df.empty:
        xg_map_for = xg_df.set_index("team")["xg_for"].to_dict()
        xg_map_against = xg_df.set_index("team")["xg_against"].to_dict()
        df["xg_avg_for"] = (
            df["home_team"].map(xg_map_for).fillna(LEAGUE_AVG_XG)
            - df["away_team"].map(xg_map_for).fillna(LEAGUE_AVG_XG)
        )
        df["xg_avg_against"] = (
            df["home_team"].map(xg_map_against).fillna(LEAGUE_AVG_XG)
            - df["away_team"].map(xg_map_against).fillna(LEAGUE_AVG_XG)
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
    steps.update(1)

    steps.set_description("ranking_diff")
    if ranking_df is not None and not ranking_df.empty:
        df["ranking_diff"] = _vectorized_ranking_diff(df, ranking_df)
    else:
        df["ranking_diff"] = 0.0
    steps.update(1)

    steps.set_description("features derivadas (as-of)")
    # Diffs derivados de eventos (goleadores), estrictamente as-of-date.
    # Si no se proveen los datos, se emiten en 0.0 para mantener el esquema
    # estable (el diff neutral de dos equipos sin datos es 0).
    # Nota v2: travel_distance_diff y shootout_winrate_diff se retiraron del
    # set de features (ver comentario en FEATURE_COLS); ya no se calculan aquí.
    if goalscorers_df is not None and not goalscorers_df.empty:
        from src.features.derived_stats import attach_goal_stat_diffs
        for col, arr in attach_goal_stat_diffs(df, goalscorers_df).items():
            df[col] = arr
    else:
        for col in ("penalty_share_diff", "striker_concentration_diff"):
            df[col] = 0.0
    steps.update(1)
    steps.set_description("target + finalización")
    steps.close()

    df["target"] = encode_target(df)

    result = df[[c for c in MATCH_OUTPUT_COLS if c in df.columns]]
    return result


def build_team_features_for_simulation(
    matches_df: pd.DataFrame,
    xg_df: pd.DataFrame | None = None,
    squad_df: pd.DataFrame | None = None,
    ranking_df: pd.DataFrame | None = None,
    teams: list[str] | None = None,
    elo_matches_df: pd.DataFrame | None = None,
    goalscorers_df: pd.DataFrame | None = None,
    shootouts_df: pd.DataFrame | None = None,
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Builds a per-team feature row for the Monte Carlo simulation.
    Columns: team, elo, squad_value_eur, xg_for, xg_against, host_distance, rank,
             penalty_share, striker_concentration, shootout_winrate

    `elo_matches_df` es el universo para el ELO (superconjunto, historia cruda
    completa). Si es None se usa `matches_df`. El ELO de cada equipo y todas las
    features derivadas se toman as-of SNAPSHOT_DATE (fecha real del snapshot de
    datos), no de REFERENCE_DATE (horizonte de decay).
    """
    matches_df = matches_df.copy()
    matches_df["date"] = pd.to_datetime(matches_df["date"])

    # Reutilizar elo_df si el caller ya lo calculó (evita recomputar el ELO).
    if elo_df is None:
        elo_source = (elo_matches_df if elo_matches_df is not None else matches_df).copy()
        elo_source["date"] = pd.to_datetime(elo_source["date"])
        elo_df = calculate_elo_ratings(elo_source.sort_values("date"))
    ref_date = SNAPSHOT_DATE

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

    # Features derivadas per-equipo, as-of SNAPSHOT_DATE (ref_date).
    if goalscorers_df is not None and not goalscorers_df.empty:
        from src.features.derived_stats import team_goal_stats_at_date
        goal_stats = team_goal_stats_at_date(goalscorers_df, ref_date)
    else:
        goal_stats = {}
    if shootouts_df is not None and not shootouts_df.empty:
        from src.features.derived_stats import team_shootout_winrate_at_date
        shootout_wr = team_shootout_winrate_at_date(shootouts_df, ref_date)
    else:
        shootout_wr = {}

    persistent_cache = _load_geo_cache()
    coords_cache: dict = dict(persistent_cache)
    coords_cache.update(WC_TEAM_CAPITAL_COORDS)
    records = []
    for team in all_teams:
        gs = goal_stats.get(team, {})
        records.append({
            "team": team,
            "elo": last_elo.get(team, INITIAL_RATING),
            "squad_value_eur": sq_map.get(team, 50_000_000),
            "xg_for": xg_for_map.get(team, LEAGUE_AVG_XG),
            "xg_against": xg_against_map.get(team, LEAGUE_AVG_XG),
            "host_distance": compute_host_distance_wc2026(team, coords_cache),
            "rank": get_rank(team),
            "penalty_share": gs.get("penalty_share", 0.07),
            "striker_concentration": gs.get("striker_concentration", 0.4),
            "shootout_winrate": shootout_wr.get(team, 0.5),
        })

    merged_cache = {**persistent_cache, **coords_cache}
    if len(merged_cache) != len(persistent_cache):
        _save_geo_cache(merged_cache)

    return pd.DataFrame(records)


def save_features(df: pd.DataFrame, filename: str = "features.csv") -> Path:
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"Features guardadas en {path}")
    return path


if __name__ == "__main__":
    from src.data.data_loader import (
        load_international_results, filter_relevant_matches, load_fifa_ranking,
        load_goalscorers, load_shootouts,
    )
    from src.data.scraper import get_statsbomb_xg_by_team, get_squad_values

    print("Cargando datos históricos...")
    # raw_all = universo COMPLETO para el ELO (todos los torneos incl. amistosos,
    # todos los años). matches = subconjunto filtrado que se emite como filas.
    raw_all = load_international_results()
    data_quality_report(raw_all, "international_results (universo ELO, sin filtrar)")
    matches = filter_relevant_matches(raw_all, year_cutoff=OUTPUT_ROW_START_YEAR)
    data_quality_report(matches, "international_results (filtrado, filas a emitir)")
    n_friendly = int((raw_all["tournament"] == "Friendly").sum()) if "tournament" in raw_all else 0
    print(f"  ELO usa historia completa: {len(raw_all):,} partidos "
          f"({len(raw_all)/max(len(matches),1):.1f}x las filas), incl. {n_friendly:,} amistosos.")

    xg_df = get_statsbomb_xg_by_team()
    squad_df = get_squad_values()
    ranking_df = load_fifa_ranking()
    print(f"  Ranking FIFA cargado: {len(ranking_df):,} filas, "
          f"{ranking_df['team'].nunique()} equipos únicos, "
          f"hasta {ranking_df['rank_date'].max().date()}")
    goalscorers_df = load_goalscorers()
    shootouts_df = load_shootouts()
    print(f"  Goleadores: {len(goalscorers_df):,} goles | Tandas: {len(shootouts_df):,}")

    # ELO calculado UNA sola vez sobre el universo completo y reutilizado por
    # ambos builders (evita la doble pasada de ELO).
    print("Calculando ELO (historia completa)...")
    elo_all = calculate_elo_ratings(
        raw_all.assign(date=pd.to_datetime(raw_all["date"])).sort_values("date")
    )

    print("Construyendo features...")
    features = build_match_features(
        matches, xg_df=xg_df, squad_df=squad_df, ranking_df=ranking_df,
        goalscorers_df=goalscorers_df, shootouts_df=shootouts_df,
        elo_df=elo_all,
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
        elo_df=elo_all,
        goalscorers_df=goalscorers_df, shootouts_df=shootouts_df,
    )
    save_features(team_feats, "team_features.csv")
    print(team_feats.to_string(index=False))
