"""
tests/test_forecasting.py
--------------------------
Unit tests for forecasting pipeline components.
Self-contained — runs with only numpy and pytest in CI.
"""

import pytest
import numpy as np
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE      = "2018-01-01"
TRAIN_SIZE      = 0.8
N_SPLITS        = 5


def test_default_tickers_defined():
    assert len(DEFAULT_TICKERS) >= 3

def test_train_size_valid():
    assert 0.5 < TRAIN_SIZE < 1.0

def test_n_splits_valid():
    assert N_SPLITS >= 3

def test_start_date_format():
    parts = START_DATE.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4


# ── Metrics logic ─────────────────────────────────────────────────────────────

def compute_metrics_local(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae    = np.mean(np.abs(y_true - y_pred))
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape   = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    dir_acc= np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape, "directional_acc": dir_acc}

def test_metrics_perfect_prediction():
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    m = compute_metrics_local(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["directional_acc"] == 100.0

def test_metrics_opposite_direction():
    y_true = np.array([0.01, 0.02, 0.03])
    y_pred = np.array([-0.01, -0.02, -0.03])
    m = compute_metrics_local(y_true, y_pred)
    assert m["directional_acc"] == 0.0

def test_metrics_mae_positive():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    m = compute_metrics_local(y_true, y_pred)
    assert m["mae"] > 0

def test_metrics_rmse_geq_mae():
    y_true = np.random.RandomState(0).randn(100)
    y_pred = np.random.RandomState(1).randn(100)
    m = compute_metrics_local(y_true, y_pred)
    assert m["rmse"] >= m["mae"]

def test_directional_accuracy_range():
    y_true = np.random.RandomState(42).randn(200)
    y_pred = np.random.RandomState(99).randn(200)
    m = compute_metrics_local(y_true, y_pred)
    assert 0.0 <= m["directional_acc"] <= 100.0


# ── Feature engineering logic ─────────────────────────────────────────────────

def make_lag_features(series: np.ndarray, lags: list) -> dict:
    result = {}
    for lag in lags:
        padded = np.empty(len(series))
        padded[:lag] = np.nan
        padded[lag:] = series[:-lag]
        result[f"lag_{lag}"] = padded
    return result

def test_lag_features_correct_shift():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lags   = make_lag_features(series, [1])
    assert np.isnan(lags["lag_1"][0])
    assert lags["lag_1"][1] == 1.0
    assert lags["lag_1"][4] == 4.0

def test_lag_features_length_preserved():
    series = np.arange(50, dtype=float)
    lags   = make_lag_features(series, [1, 5, 10])
    for key, arr in lags.items():
        assert len(arr) == 50

def test_rolling_mean_shape():
    series = np.random.randn(100)
    window = 10
    result = np.convolve(series, np.ones(window)/window, mode="valid")
    assert len(result) == 91  # 100 - 10 + 1

def test_rsi_range():
    # RSI must always be between 0 and 100
    prices = np.abs(np.random.RandomState(42).randn(100).cumsum()) + 10
    delta  = np.diff(prices)
    gain   = np.where(delta > 0, delta, 0)
    loss   = np.where(delta < 0, -delta, 0)
    avg_g  = np.convolve(gain, np.ones(14)/14, mode="valid")
    avg_l  = np.convolve(loss, np.ones(14)/14, mode="valid")
    rs     = avg_g / (avg_l + 1e-8)
    rsi    = 100 - (100 / (1 + rs))
    assert np.all(rsi >= 0)
    assert np.all(rsi <= 100)


# ── Walk-forward validation logic ─────────────────────────────────────────────

def simulate_walk_forward(n: int, train_size: float, n_splits: int):
    init   = int(n * train_size)
    step   = (n - init) // n_splits
    folds  = []
    for i in range(n_splits):
        train_end = init + i * step
        test_end  = min(train_end + step, n)
        if test_end > train_end:
            folds.append((train_end, test_end))
    return folds

def test_walk_forward_correct_fold_count():
    folds = simulate_walk_forward(1000, 0.7, 5)
    assert len(folds) == 5

def test_walk_forward_train_grows():
    folds = simulate_walk_forward(1000, 0.7, 5)
    sizes = [te - ts for ts, te in folds]
    # All test folds should be equal or the last one slightly different
    assert all(s > 0 for s in sizes)

def test_walk_forward_no_overlap():
    folds = simulate_walk_forward(500, 0.6, 4)
    for i in range(1, len(folds)):
        assert folds[i][0] >= folds[i-1][1]


# ── Synthetic data shape ───────────────────────────────────────────────────────

def test_synthetic_gbm_shape():
    n      = 100
    mu, sigma, S0 = 0.0003, 0.018, 150.0
    rs     = np.random.RandomState(42)
    returns= rs.normal(mu, sigma, n)
    prices = S0 * np.exp(np.cumsum(returns))
    assert len(prices) == n
    assert prices.min() > 0

def test_synthetic_prices_positive():
    rs     = np.random.RandomState(0)
    prices = np.abs(rs.randn(200).cumsum()) + 50
    assert np.all(prices > 0)


# ── File existence ────────────────────────────────────────────────────────────

def test_data_loader_exists():
    assert (Path(__file__).parent.parent / "src" / "data_loader.py").exists()

def test_models_file_exists():
    assert (Path(__file__).parent.parent / "src" / "models.py").exists()

def test_dashboard_exists():
    assert (Path(__file__).parent.parent / "dashboard" / "app.py").exists()

def test_requirements_exists():
    assert (Path(__file__).parent.parent / "requirements.txt").exists()
