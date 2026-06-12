import numpy as np
import pandas as pd

from src.analysis.compare_runs import paired_bootstrap


def _series(vals):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2022-01-01") + pd.Timedelta(days=i), f"H{i}", f"A{i}")
         for i in range(len(vals))],
        names=["date", "home_team", "away_team"],
    )
    return pd.Series(vals, index=idx)


def test_identical_losses_not_significant():
    a = _series(np.random.default_rng(1).uniform(0.5, 1.5, 500))
    r = paired_bootstrap(a, a.copy())
    assert r["delta"] == 0.0
    assert not r["significant"]


def test_clear_improvement_is_significant():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.5, 1.5, 500)
    a = _series(base)
    b = _series(base - 0.2)  # b uniformemente mejor
    r = paired_bootstrap(a, b)
    assert r["delta"] < 0
    assert r["significant"]
    assert r["ci_high"] < 0


def test_noise_not_significant():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.5, 1.5, 200)
    a = _series(base)
    b = _series(base + rng.normal(0, 0.5, 200))  # ruido sin sesgo
    r = paired_bootstrap(a, b)
    assert r["ci_low"] < 0 < r["ci_high"]
    assert not r["significant"]
