"""
Benchmark externo: compara las P(campeón) del modelo contra las probabilidades
implícitas del mercado de apuestas (snapshot manual con fuente y fecha en
data/raw/market_odds_2026.csv). El mercado es el baseline estándar de la
literatura de predicción futbolística (Dixon & Coles, 1997, se valida contra
el mercado de apuestas).

Las odds americanas se convierten a probabilidad implícita y se les quita el
vigorish con normalización proporcional (las 48 probabilidades suman 1).

Uso: python -m src.analysis.benchmark
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def american_to_prob(odds: float) -> float:
    """Probabilidad implícita (con vigorish) de una odd americana."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def load_market_probs(path: Path = RAW_DIR / "market_odds_2026.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["implied_raw"] = df["american_odds"].apply(american_to_prob)
    overround = df["implied_raw"].sum()
    df["market_pct"] = df["implied_raw"] / overround * 100.0
    print(f"Snapshot: {df['source'].iloc[0]} ({df['snapshot_date'].iloc[0]}) | "
          f"overround = {overround:.3f}")
    return df[["team", "market_pct"]]


def compare(
    sim_path: Path = PROCESSED_DIR / "simulation_results.csv",
    odds_path: Path = RAW_DIR / "market_odds_2026.csv",
) -> pd.DataFrame:
    market = load_market_probs(odds_path)
    sim = pd.read_csv(sim_path)[["team", "champion_pct"]]
    m = sim.merge(market, on="team", how="outer")
    missing = m[m.isna().any(axis=1)]
    if not missing.empty:
        print("AVISO - equipos sin match de nombre:\n", missing.to_string(index=False))
    m = m.dropna().copy()
    m["delta_pp"] = (m["champion_pct"] - m["market_pct"]).round(2)
    m["ratio"] = (m["champion_pct"] / m["market_pct"]).round(2)
    m = m.sort_values("market_pct", ascending=False).reset_index(drop=True)

    # Divergencia agregada: distancia L1/2 y Jensen-Shannon entre distribuciones.
    p = (m["champion_pct"] / m["champion_pct"].sum()).values
    q = (m["market_pct"] / m["market_pct"].sum()).values
    l1 = float(np.abs(p - q).sum()) / 2  # total variation
    mix = (p + q) / 2
    js = float(
        0.5 * np.sum(p * np.log(np.clip(p / mix, 1e-12, None)))
        + 0.5 * np.sum(q * np.log(np.clip(q / mix, 1e-12, None)))
    )
    print(f"\nDivergencia modelo vs mercado: TV = {l1:.4f} | JS = {js:.4f}")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vs mercado")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    result = compare()
    print(f"\n=== Modelo vs mercado (top {args.top} por mercado) ===")
    print(result.head(args.top).to_string(index=False))

    out = PROCESSED_DIR / "benchmark_market.csv"
    result.to_csv(out, index=False)
    print(f"\nGuardado en {out}")
