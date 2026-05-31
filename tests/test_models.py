import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    temporal_split, compute_combined_weights, train_baseline, FEATURE_COLS,
    TRAIN_END, VAL_END,
)


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.to_datetime(np.random.choice(
        pd.date_range("2018-01-01", "2023-12-31"), size=n, replace=True,
    ))
    df = pd.DataFrame({
        "date": dates,
        "elo_diff": rng.normal(0, 100, n),
        "squad_value_diff": rng.normal(0, 1, n),
        "xg_avg_for": rng.normal(0, 0.5, n),
        "xg_avg_against": rng.normal(0, 0.5, n),
        "travel_distance_diff": rng.uniform(-10000, 10000, n),
        "ranking_diff": rng.normal(0, 50, n),
        "penalty_share_diff": rng.normal(0, 0.05, n),
        "striker_concentration_diff": rng.normal(0, 0.1, n),
        "shootout_winrate_diff": rng.normal(0, 0.1, n),
        "time_weight": rng.uniform(0.5, 1.0, n),
        "target": rng.integers(0, 3, n),
    })
    return df.sort_values("date").reset_index(drop=True)


def test_temporal_split_no_leakage(synthetic_features):
    df = synthetic_features
    train_mask, val_mask, test_mask = temporal_split(df)
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(df)
    if train_mask.any() and test_mask.any():
        assert pd.to_datetime(df["date"][train_mask]).max() < pd.to_datetime(df["date"][test_mask]).min()
    if train_mask.any() and val_mask.any():
        assert pd.to_datetime(df["date"][train_mask]).max() < TRAIN_END
        assert pd.to_datetime(df["date"][val_mask]).min() >= TRAIN_END


def test_combined_weights_mean_is_one(synthetic_features):
    y = synthetic_features["target"].values
    tw = synthetic_features["time_weight"].values.astype(np.float32)
    w = compute_combined_weights(y, tw)
    assert abs(w.mean() - 1.0) < 1e-5


def test_baseline_predict_proba_sums_to_one(synthetic_features):
    df = synthetic_features
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)
    model = train_baseline(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(df), 3)
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
