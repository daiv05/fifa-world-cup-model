"""
Features derivadas de datos de eventos (goleadores y tandas de penales),
calculadas estrictamente as-of-date para garantizar ausencia de leakage.

Fuentes (data/raw/new-data/international_results/):
  - goalscorers.csv: un registro por gol (date, home_team, away_team, team,
    scorer, minute, own_goal, penalty).
  - shootouts.csv: un registro por tanda (date, home_team, away_team, winner,
    first_shooter).

Métricas por equipo (luego diferenciadas home - away a nivel de partido):
  - penalty_share          : goles de penal / goles totales.
  - striker_concentration  : índice de Herfindahl sobre la participación de
                             goleadores (1.0 = depende de un solo goleador).
  - shootout_winrate       : tasa de victoria en tandas, con shrinkage bayesiano
                             hacia 0.5 (prior Beta) para muestras pequeñas.

Nota: `late_goal_ratio` (goles min>=75 / totales) se evaluó y descartó por
completo: la ablación leave-one-out mostró aporte NEGATIVO (ruido neto).

Garantía leak-free: las tablas as-of guardan el estado ACUMULADO al cierre de
cada fecha; al unirlas a los partidos se usa `merge_asof(direction="backward",
allow_exact_matches=False)`, de modo que un partido en fecha D solo ve eventos
estrictamente anteriores a D (nunca el propio partido).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shrinkage bayesiano del winrate de tandas: posterior mean de un prior
# Beta(alpha*mu0, alpha*(1-mu0)) actualizado con Binomial(wins, n):
#   winrate = (wins + alpha*mu0) / (n + alpha)
# alpha = fuerza del prior (pseudo-muestras). Con alpha=10 y mu0=0.5, equipos
# con <5 tandas (caso de las grandes potencias) quedan ~0.5 (sin sobreconfianza),
# mientras que equipos con >=20 tandas conservan señal.
SHOOTOUT_ALPHA = 10.0
SHOOTOUT_PRIOR = 0.5

# Nombres de las columnas de feature a nivel de equipo y sus diffs de partido.
GOAL_STAT_COLS = ["penalty_share", "striker_concentration"]
SHOOTOUT_STAT_COL = "shootout_winrate"


def bayesian_shootout_winrate(
    wins: np.ndarray | float,
    n: np.ndarray | float,
    alpha: float = SHOOTOUT_ALPHA,
    mu0: float = SHOOTOUT_PRIOR,
) -> np.ndarray | float:
    """Winrate de tandas con shrinkage hacia `mu0`. n=0 -> mu0."""
    return (np.asarray(wins, dtype=float) + alpha * mu0) / (np.asarray(n, dtype=float) + alpha)


# --------------------------------------------------------------------------- #
# Cálculo de tablas as-of (estado acumulado por (team, date))
# --------------------------------------------------------------------------- #
def _cumulative_herfindahl(g: pd.DataFrame) -> pd.DataFrame:
    """
    Herfindahl acumulado de goleadores por (team, date).

    H(d) = sum_s (c_s(d) / N(d))^2, con c_s = goles acumulados del goleador s y
    N = goles acumulados del equipo. Se mantiene sum_s c_s^2 de forma incremental
    (delta al sumar k goles a un goleador con c previos: 2*c*k + k^2), O(goles).
    """
    gg = g[g["scorer"].notna()]
    if gg.empty:
        return pd.DataFrame(columns=["team", "date", "striker_concentration"])

    per = (
        gg.groupby(["team", "date", "scorer"], sort=True)
        .size()
        .reset_index(name="k")
    )

    rows: list[tuple] = []
    for team, grp in per.groupby("team", sort=True):
        counts: dict[str, int] = {}
        sum_sq = 0.0
        total = 0
        for date, dgrp in grp.groupby("date", sort=True):
            for scorer, k in zip(dgrp["scorer"].values, dgrp["k"].values):
                c = counts.get(scorer, 0)
                sum_sq += 2 * c * k + k * k
                counts[scorer] = c + k
                total += int(k)
            H = sum_sq / (total * total) if total > 0 else np.nan
            rows.append((team, date, H))

    return pd.DataFrame(rows, columns=["team", "date", "striker_concentration"])


def compute_team_goal_stats_asof(goalscorers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla as-of de stats de goleo por (team, date), con valores ACUMULADOS al
    cierre de esa fecha. Se excluyen autogoles (`own_goal`) porque acreditan al
    rival, no al perfil ofensivo del equipo.
    Columnas: team, date, penalty_share, striker_concentration.
    """
    g = goalscorers_df.copy()
    g["date"] = pd.to_datetime(g["date"])
    if "own_goal" in g.columns:
        g = g[~g["own_goal"].fillna(False).astype(bool)]
    g = g[g["team"].notna()].copy()

    if "penalty" in g.columns:
        g["is_pen"] = g["penalty"].fillna(False).astype(bool)
    else:
        g["is_pen"] = False

    daily = (
        g.groupby(["team", "date"], sort=True)
        .agg(goals=("team", "size"), pen=("is_pen", "sum"))
        .reset_index()
        .sort_values(["team", "date"])
    )
    daily["cum_goals"] = daily.groupby("team")["goals"].cumsum()
    daily["cum_pen"] = daily.groupby("team")["pen"].cumsum()
    daily["penalty_share"] = daily["cum_pen"] / daily["cum_goals"]

    herf = _cumulative_herfindahl(g)
    out = daily[["team", "date", "penalty_share"]].merge(
        herf, on=["team", "date"], how="left"
    )
    # Herfindahl es acumulativo: si una fecha no tuvo goleador identificado,
    # arrastrar el último valor conocido del equipo.
    out["striker_concentration"] = out.groupby("team")["striker_concentration"].ffill()
    return out.sort_values(["team", "date"]).reset_index(drop=True)


def compute_team_shootout_stats_asof(shootouts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla as-of de tandas de penales por (team, date), con apariciones y
    victorias ACUMULADAS al cierre de esa fecha.
    Columnas: team, date, shootout_n, shootout_wins.
    """
    s = shootouts_df.copy()
    s["date"] = pd.to_datetime(s["date"])

    # Cada tanda genera dos filas-equipo (home y away).
    home = pd.DataFrame({
        "team": s["home_team"], "date": s["date"],
        "won": (s["winner"] == s["home_team"]).astype(int),
    })
    away = pd.DataFrame({
        "team": s["away_team"], "date": s["date"],
        "won": (s["winner"] == s["away_team"]).astype(int),
    })
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["team"])

    daily = (
        long.groupby(["team", "date"], sort=True)
        .agg(apps=("team", "size"), wins=("won", "sum"))
        .reset_index()
        .sort_values(["team", "date"])
    )
    daily["shootout_n"] = daily.groupby("team")["apps"].cumsum()
    daily["shootout_wins"] = daily.groupby("team")["wins"].cumsum()
    return daily[["team", "date", "shootout_n", "shootout_wins"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Lookup as-of (leak-free) y construcción de diffs a nivel de partido
# --------------------------------------------------------------------------- #
def _asof_lookup(
    matches_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    value_col: str,
    team_col: str,
) -> np.ndarray:
    """
    Para cada partido devuelve el valor de `value_col` del equipo en `team_col`
    vigente ESTRICTAMENTE antes de la fecha del partido (merge_asof backward con
    allow_exact_matches=False). Filas sin match -> NaN (se rellenan fuera).
    """
    s = stat_df[["team", "date", value_col]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")

    left = matches_df[["date", team_col]].copy()
    left["date"] = pd.to_datetime(left["date"])
    left["_row"] = np.arange(len(left))
    side = left.rename(columns={team_col: "team"}).sort_values("date")

    merged = pd.merge_asof(
        side, s, on="date", by="team",
        direction="backward", allow_exact_matches=False,
    )
    return (
        merged.set_index("_row")[value_col]
        .reindex(np.arange(len(left)))
        .to_numpy()
    )


def _diff_column(
    matches_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    value_col: str,
    fill: float,
) -> np.ndarray:
    """diff = valor(home) - valor(away), rellenando ausentes con `fill` (neutral:
    el diff de dos fills es 0)."""
    home = _asof_lookup(matches_df, stat_df, value_col, "home_team")
    away = _asof_lookup(matches_df, stat_df, value_col, "away_team")
    home = np.where(np.isnan(home), fill, home)
    away = np.where(np.isnan(away), fill, away)
    return home - away


def _global_fill(stat_df: pd.DataFrame, value_col: str, default: float) -> float:
    """Media global de la métrica (referencia neutral para equipos ausentes)."""
    if stat_df.empty or value_col not in stat_df.columns:
        return default
    m = pd.to_numeric(stat_df[value_col], errors="coerce").mean()
    return float(m) if pd.notna(m) else default


def attach_goal_stat_diffs(
    matches_df: pd.DataFrame, goalscorers_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Devuelve {penalty_share_diff, striker_concentration_diff}."""
    stats = compute_team_goal_stats_asof(goalscorers_df)
    fills = {
        "penalty_share": _global_fill(stats, "penalty_share", 0.07),
        "striker_concentration": _global_fill(stats, "striker_concentration", 0.4),
    }
    return {
        f"{col}_diff": _diff_column(matches_df, stats, col, fills[col])
        for col in GOAL_STAT_COLS
    }


def attach_shootout_stat_diff(
    matches_df: pd.DataFrame, shootouts_df: pd.DataFrame
) -> np.ndarray:
    """Devuelve el array shootout_winrate_diff (home - away), con shrinkage."""
    stats = compute_team_shootout_stats_asof(shootouts_df)
    stats = stats.copy()
    stats["shootout_winrate"] = bayesian_shootout_winrate(
        stats["shootout_wins"], stats["shootout_n"]
    )
    return _diff_column(matches_df, stats, "shootout_winrate", SHOOTOUT_PRIOR)


# --------------------------------------------------------------------------- #
# Estado por equipo a una fecha (para la simulación, as-of SNAPSHOT_DATE)
# --------------------------------------------------------------------------- #
def team_goal_stats_at_date(
    goalscorers_df: pd.DataFrame, as_of: pd.Timestamp
) -> dict[str, dict[str, float]]:
    """{team -> {penalty_share, striker_concentration}} a `as_of`."""
    stats = compute_team_goal_stats_asof(goalscorers_df)
    stats = stats[stats["date"] <= as_of]
    last = stats.sort_values("date").groupby("team").last()
    return last[GOAL_STAT_COLS].to_dict("index")


def team_shootout_winrate_at_date(
    shootouts_df: pd.DataFrame, as_of: pd.Timestamp
) -> dict[str, float]:
    """{team -> shootout_winrate (con shrinkage)} a `as_of`."""
    stats = compute_team_shootout_stats_asof(shootouts_df)
    stats = stats[stats["date"] <= as_of]
    last = stats.sort_values("date").groupby("team").last()
    wr = bayesian_shootout_winrate(last["shootout_wins"], last["shootout_n"])
    return dict(zip(last.index, wr))
