import bisect
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

INITIAL_RATING = 1500.0

K_FACTORS: dict[str, float] = {
    "FIFA World Cup": 60,
    "Confederations Cup": 50,
    "UEFA Euro": 50,
    "Copa América": 50,
    "Africa Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "CONCACAF Gold Cup": 50,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "Nations League": 40,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 35


def _get_k(tournament: str) -> float:
    for key, k in K_FACTORS.items():
        if key.lower() in tournament.lower():
            return k
    return DEFAULT_K


def _build_k_lookup(tournaments: pd.Series) -> dict[str, float]:
    """Memoiza _get_k por torneo único (evita escanear K_FACTORS por cada fila)."""
    return {t: _get_k(str(t)) for t in tournaments.dropna().unique()}


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _result_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def calculate_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    ELO secuencial sobre los partidos ordenados por fecha. El loop es
    inherentemente secuencial (cada rating depende del estado previo), pero se
    itera sobre arrays de numpy en vez de `iterrows()` (que crea una Series por
    fila) y se memoiza el K-factor por torneo, reduciendo el overhead por fila.
    """
    df = matches_df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)

    dates = df["date"].to_numpy()
    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    home_scores = pd.to_numeric(df["home_score"], errors="coerce").to_numpy()
    away_scores = pd.to_numeric(df["away_score"], errors="coerce").to_numpy()
    if "tournament" in df.columns:
        tournaments = df["tournament"].fillna("Friendly").to_numpy()
        k_lookup = _build_k_lookup(df["tournament"].fillna("Friendly"))
    else:
        tournaments = np.full(len(df), "Friendly", dtype=object)
        k_lookup = {"Friendly": _get_k("Friendly")}

    records = []
    for i in tqdm(range(len(df)), total=len(df), desc="ELO", unit="match"):
        h_goal, a_goal = home_scores[i], away_scores[i]
        if np.isnan(h_goal) or np.isnan(a_goal):
            continue

        home, away = homes[i], aways[i]
        r_h = ratings[home]
        r_a = ratings[away]

        exp_h = _expected_score(r_h, r_a)
        exp_a = 1.0 - exp_h
        res_h, res_a = _result_score(int(h_goal), int(a_goal))

        k = k_lookup.get(tournaments[i], DEFAULT_K)
        new_r_h = r_h + k * (res_h - exp_h)
        new_r_a = r_a + k * (res_a - exp_a)

        records.append({
            "date": dates[i],
            "home_team": home,
            "away_team": away,
            "home_elo_before": r_h,
            "away_elo_before": r_a,
            "home_elo_after": new_r_h,
            "away_elo_after": new_r_a,
        })

        ratings[home] = new_r_h
        ratings[away] = new_r_a

    return pd.DataFrame(records)


def get_current_elo(elo_df: pd.DataFrame) -> dict[str, float]:
    latest = elo_df.sort_values("date")
    home_last = latest.groupby("home_team")["home_elo_after"].last()
    away_last = latest.groupby("away_team")["away_elo_after"].last()
    combined = pd.concat([home_last.rename("elo"), away_last.rename("elo")])
    return combined.groupby(combined.index).last().to_dict()


def get_elo_at_date(
    elo_df: pd.DataFrame,
    team: str,
    date: pd.Timestamp,
) -> float:
    home_mask = elo_df["home_team"] == team
    away_mask = elo_df["away_team"] == team

    home_entries = elo_df[home_mask][["date", "home_elo_before"]].rename(
        columns={"home_elo_before": "elo"}
    )
    away_entries = elo_df[away_mask][["date", "away_elo_before"]].rename(
        columns={"away_elo_before": "elo"}
    )

    entries = pd.concat([home_entries, away_entries]).sort_values("date")
    if entries.empty:
        return INITIAL_RATING

    dates = entries["date"].tolist()
    idx = bisect.bisect_left(dates, date)
    if idx == 0:
        return INITIAL_RATING
    return float(entries.iloc[idx - 1]["elo"])
