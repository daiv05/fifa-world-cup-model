"""
Función de decaimiento exponencial para pesar partidos históricos.
W(t) = e^(-λ · Δt)  donde Δt es la diferencia en días desde hoy.
"""

import numpy as np
import pandas as pd

DEFAULT_LAMBDA = 0.002


def compute_time_decay_weights(
    dates: pd.Series,
    lambda_: float = DEFAULT_LAMBDA,
    reference_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """
    Calcula el peso exponencial de cada partido según su antigüedad.

    Parámetros
    ----------
    dates : Series de fechas de los partidos
    lambda_ : tasa de decaimiento (a calibrar con Optuna; default 0.002 ≈ half-life 1 año)
    reference_date : fecha de referencia (hoy si es None)

    Devuelve
    --------
    Array de pesos en (0, 1], donde 1.0 es un partido de hoy.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    delta_days = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0).values
    return np.exp(-lambda_ * delta_days)


def lambda_to_halflife_years(lambda_: float) -> float:
    """Convierte lambda a vida media en años (útil para interpretar el hiperparámetro)."""
    return np.log(2) / (lambda_ * 365)


def halflife_years_to_lambda(years: float) -> float:
    """Convierte una vida media en años a lambda."""
    return np.log(2) / (years * 365)
