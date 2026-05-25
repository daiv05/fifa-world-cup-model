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


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _result_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def calculate_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ELO", unit="match"):
        home = row["home_team"]
        away = row["away_team"]
        tournament = row.get("tournament", "Friendly")

        r_h = ratings[home]
        r_a = ratings[away]

        try:
            h_goals = int(row["home_score"])
            a_goals = int(row["away_score"])
        except (ValueError, TypeError):
            continue

        exp_h = _expected_score(r_h, r_a)
        exp_a = 1.0 - exp_h
        res_h, res_a = _result_score(h_goals, a_goals)

        k = _get_k(tournament)
        new_r_h = r_h + k * (res_h - exp_h)
        new_r_a = r_a + k * (res_a - exp_a)

        records.append({
            "date": row["date"],
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
