import numpy as np
import pandas as pd

DEFAULT_LAMBDA = 0.001
# Fecha de referencia determinista para el decay temporal. Usamos el inicio
# del año del Mundial 2026 para que los pesos sean reproducibles entre
# ejecuciones y consistentes con el horizonte de predicción.
REFERENCE_DATE = pd.Timestamp("2026-01-01")

# Fecha del snapshot de datos más reciente (ranking FIFA `fifa_rankings_2026.csv`
# y as-of para extraer features a nivel de equipo en la simulación). Es distinta
# de REFERENCE_DATE a propósito: REFERENCE_DATE define el horizonte del decay
# temporal (peso de los partidos), mientras que SNAPSHOT_DATE es la fecha real
# en que se observó el último estado del mundo (ranking, plantillas, stats).
# Todos los partidos del Mundial 2026 ocurren después de SNAPSHOT_DATE, por lo
# que usarla como as-of es libre de leakage.
SNAPSHOT_DATE = pd.Timestamp("2026-05-30")


def compute_time_decay_weights(
    dates: pd.Series,
    lambda_: float = DEFAULT_LAMBDA,
    reference_date: pd.Timestamp | None = None,
) -> np.ndarray:
    if reference_date is None:
        reference_date = REFERENCE_DATE

    delta_days = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0).values
    return np.exp(-lambda_ * delta_days)


def lambda_to_halflife_years(lambda_: float) -> float:
    return np.log(2) / (lambda_ * 365)


def halflife_years_to_lambda(years: float) -> float:
    return np.log(2) / (years * 365)
