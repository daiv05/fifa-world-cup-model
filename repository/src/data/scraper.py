import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Fecha del snapshot manual de valores Transfermarkt
SQUAD_VALUES_SNAPSHOT_DATE = "2026-05"


def get_statsbomb_matches() -> pd.DataFrame:
    import statsbombpy.sb as sb

    cache_path = RAW_DIR / "statsbomb_matches.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    competitions = sb.competitions()
    intl_comps = competitions[
        competitions["competition_name"].str.contains(
            "World Cup|Euro|Copa America|Africa Cup|Asian Cup|Gold Cup",
            case=False,
            na=False,
        )
    ]

    all_matches = []
    for _, row in intl_comps.iterrows():
        try:
            matches = sb.matches(
                competition_id=row["competition_id"],
                season_id=row["season_id"],
            )
            matches["competition_name"] = row["competition_name"]
            matches["season_name"] = row["season_name"]
            all_matches.append(matches)
        except Exception:
            continue

    if not all_matches:
        return pd.DataFrame()

    df = pd.concat(all_matches, ignore_index=True)
    df.to_csv(cache_path, index=False)
    return df


def get_statsbomb_xg_by_team() -> pd.DataFrame:
    import statsbombpy.sb as sb

    cache_path = RAW_DIR / "statsbomb_xg_by_team.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    matches = get_statsbomb_matches()
    if matches.empty:
        return pd.DataFrame()

    home_col = next((c for c in ("home_team_name", "home_team") if c in matches.columns), None)
    away_col = next((c for c in ("away_team_name", "away_team") if c in matches.columns), None)
    if not home_col or not away_col or "home_score" not in matches.columns:
        return pd.DataFrame()

    records = []
    n_total = len(matches)
    n_event_ok = 0

    for i, (_, row) in enumerate(matches.iterrows(), 1):
        match_id = row["match_id"]
        home_team = row[home_col]
        away_team = row[away_col]

        if i % 50 == 0:
            print(f"  StatsBomb eventos: {i}/{n_total} partidos procesados "
                  f"({n_event_ok} con xG real)...")

        try:
            events = sb.events(match_id=match_id)
            shots = events[events["type"] == "Shot"].copy()

            if shots.empty or "shot_statsbomb_xg" not in shots.columns:
                raise ValueError("sin datos xG")

            if (shots["team"].dtype == object
                    and len(shots) > 0
                    and isinstance(shots["team"].iloc[0], dict)):
                shots["team"] = shots["team"].apply(
                    lambda t: t.get("name", "") if isinstance(t, dict) else t
                )

            xg_map = shots.groupby("team")["shot_statsbomb_xg"].sum().to_dict()

            h_xg = xg_map.get(home_team)
            a_xg = xg_map.get(away_team)
            if h_xg is None or a_xg is None:
                raise ValueError(f"equipo no encontrado en eventos: "
                                 f"{home_team!r} / {away_team!r} vs {list(xg_map)[:4]}")

            n_event_ok += 1
        except Exception:
            h_xg = row.get("home_score", 1.0)
            a_xg = row.get("away_score", 1.0)

        records.append({"team": home_team, "xg_for": h_xg, "xg_against": a_xg})
        records.append({"team": away_team, "xg_for": a_xg, "xg_against": h_xg})

    print(f"  StatsBomb: {n_event_ok}/{n_total} partidos con xG real de eventos "
          f"({n_total - n_event_ok} con fallback a goles)")

    df = pd.DataFrame(records)
    agg = (
        df.groupby("team")
        .agg(
            xg_for=("xg_for", "mean"),
            xg_against=("xg_against", "mean"),
            n_matches=("xg_for", "count"),
        )
        .reset_index()
    )
    agg.to_csv(cache_path, index=False)
    return agg


# Snapshot manual de valores Transfermarkt (fecha: ver SQUAD_VALUES_SNAPSHOT_DATE).
# No es scraping en vivo: si se necesitan datos actualizados, regenerar este dict
# manualmente o editar directamente data/raw/squad_values.csv.
_SQUAD_VALUES_EUR: dict[str, int] = {
    "France": 1_470_000_000,
    "England": 1_320_000_000,
    "Brazil": 905_700_000,
    "Germany": 1_010_000_000,
    "Spain": 1_310_000_000,
    "Portugal": 905_000_000,
    "Argentina": 762_000_000,
    "Netherlands": 763_000_000,
    "Belgium": 558_200_000,
    "Italy": 833_500_000,
    "United States": 356_700_000,
    "Colombia": 296_450_000,
    "Norway": 586_500_000,
    "Japan": 264_050_000,
    "Croatia": 357_300_000,
    "Uruguay": 363_000_000,
    "Serbia": 260_500_000,
    "South Korea": 142_300_000,
    "Morocco": 235_800_000,
    "Senegal": 464_300_000,
    "Australia": 51_330_000,
    "Mexico": 83_600_000,
    "Ecuador": 366_200_000,
    "Ivory Coast": 516_900_000,
    "Nigeria": 160_700_000,
    "Egypt": 136_200_000,
    "Peru": 28_900_000,
    "Chile": 75_200_000,
    "Algeria": 227_850_000,
    "Iran": 36_550_000,
    "Cameroon": 197_600_000,
    "Venezuela": 62_730_000,
    "Saudi Arabia": 27_630_000,
    "DR Congo": 149_250_000,
    "South Africa": 52_700_000,
    "Slovenia": 143_850_000,
    "Qatar": 17_930_000,
    "Honduras": 16_300_000,
    "Panama": 31_350_000,
    "Jamaica": 54_000_000,
    "New Zealand": 31_700_000,
    "Iraq": 19_280_000,
    "Canada": 129_550_000,
    "Turkey": 525_200_000,
    "Switzerland": 317_600_000,
    "Sweden": 435_380_000,
    "Austria": 258_300_000,
    "Scotland": 207_830_000,
    "Czech Republic": 196_430_000,
    "Ghana": 289_180_000,
    "Tunisia": 69_550_000,
    "Paraguay": 137_300_000,
    "Bosnia & Herzegovina": 133_400_000,
    "Cape Verde": 56_380_000,
    "Jordan": 16_230_000,
    "Uzbekistan": 79_130_000,
    "Curacao": 27_780_000,
    "Haiti": 55_730_000,
    "Guatemala": 5_480_000,
    "Bahrain": 6_230_000,
    "Cuba": 2_100_000,
    "Kenya": 3_130_000,
}


def get_squad_values(teams: list[str] | None = None) -> pd.DataFrame:
    """
    Devuelve un DataFrame con (team, squad_value_eur) a partir del snapshot
    manual `_SQUAD_VALUES_EUR` (Transfermarkt, fecha SQUAD_VALUES_SNAPSHOT_DATE).
    No realiza scraping en vivo.
    """
    cache_path = RAW_DIR / "squad_values.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  squad_values.csv cargado desde caché ({len(df)} equipos)")
        return df

    records = [
        {"team": team, "squad_value_eur": val}
        for team, val in _SQUAD_VALUES_EUR.items()
    ]
    df = pd.DataFrame(records).drop_duplicates(subset="team")

    if teams is not None:
        df = df[df["team"].isin(teams)]

    df.to_csv(cache_path, index=False)
    print(f"  squad_values.csv generado desde snapshot manual "
          f"({SQUAD_VALUES_SNAPSHOT_DATE}, {len(df)} equipos).")
    print(f"  Para actualizar valores: edita _SQUAD_VALUES_EUR o {cache_path}")
    return df


if __name__ == "__main__":
    print("Descargando datos de StatsBomb...")
    xg_df = get_statsbomb_xg_by_team()
    print(f"  Equipos con xG: {len(xg_df)}")

    print("Cargando snapshot de valores de plantilla (Transfermarkt)...")
    squad_df = get_squad_values()
    print(f"  Equipos con valor: {len(squad_df)}")
    print("OK")
