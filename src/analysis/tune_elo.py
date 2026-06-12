"""
Validación empírica de los parámetros del ELO (HOME_ADVANTAGE y multiplicador
por margen), para fijar los defaults con evidencia en lugar de constantes
ad-hoc.

Protocolo (sin tocar el test del modelo, date>=2022):
  1. ELO sobre la historia completa para cada variante del grid.
  2. elo_diff por partido sobre los torneos relevantes.
  3. Logística multinomial elo_diff -> {H,E,V}: train 2010-2018, val 2019-2020.
  4. Se reporta el log-loss de validación de cada variante.

Uso:  python -m src.analysis.tune_elo
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import load_international_results, filter_relevant_matches
from src.features.elo import calculate_elo_ratings
from src.features.features import _vectorized_elo_diff, encode_target

GRID_HOME_ADV = [0.0, 50.0, 100.0, 150.0]
GRID_MARGIN = [False, True]

TRAIN_START, TRAIN_END = 2010, 2019   # [2010, 2019)
VAL_START, VAL_END = 2019, 2021       # [2019, 2021)


def main() -> pd.DataFrame:
    raw = load_international_results()
    raw = raw.assign(date=pd.to_datetime(raw["date"])).sort_values("date")
    matches = filter_relevant_matches(raw, year_cutoff=1993)
    matches = matches.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    matches["date"] = pd.to_datetime(matches["date"])
    y = encode_target(matches)
    years = matches["date"].dt.year
    tr = (years >= TRAIN_START) & (years < TRAIN_END)
    va = (years >= VAL_START) & (years < VAL_END)
    print(f"train: {tr.sum():,} partidos ({TRAIN_START}-{TRAIN_END - 1}) | "
          f"val: {va.sum():,} ({VAL_START}-{VAL_END - 1})")

    rows = []
    for ha in GRID_HOME_ADV:
        for margin in GRID_MARGIN:
            elo_df = calculate_elo_ratings(raw, home_advantage=ha, use_margin=margin)
            diff = _vectorized_elo_diff(matches, elo_df)
            X = diff.reshape(-1, 1)
            pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
            pipe.fit(X[tr], y[tr])
            ll = log_loss(y[va], pipe.predict_proba(X[va]), labels=[0, 1, 2])
            rows.append({"home_advantage": ha, "use_margin": margin, "val_log_loss": round(ll, 4)})
            print(f"  HA={ha:5.0f} margin={str(margin):5s} -> val LL={ll:.4f}")

    out = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    print("\nMejor configuracion:")
    print(out.head(3).to_string(index=False))
    return out


if __name__ == "__main__":
    result = main()
    from src.features.features import PROCESSED_DIR
    path = PROCESSED_DIR / "tune_elo_results.csv"
    result.to_csv(path, index=False)
    print(f"\nGuardado en {path}")
