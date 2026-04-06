"""
src/data_loader.py
-------------------
Loads historical stock/financial time series data using yfinance.
Falls back to synthetic data if network is unavailable.
Handles resampling, cleaning, and feature engineering.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE      = "2018-01-01"
END_DATE        = datetime.today().strftime("%Y-%m-%d")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ticker(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Load OHLCV data for a ticker from yfinance.
    Falls back to synthetic data if unavailable.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty response from yfinance")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        logger.info(f"Loaded {len(df)} rows for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"yfinance unavailable ({e}). Using synthetic data for {ticker}.")
        return _synthetic_ohlcv(ticker, start, end)


def _synthetic_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data using geometric Brownian motion."""
    rng   = pd.date_range(start=start, end=end, freq="B")   # Business days
    n     = len(rng)
    seed  = sum(ord(c) for c in ticker)
    rs    = np.random.RandomState(seed)

    # GBM parameters
    mu    = 0.0003    # Daily drift
    sigma = 0.018     # Daily volatility
    S0    = 100.0 + seed % 200

    returns = rs.normal(mu, sigma, n)
    prices  = S0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open":   prices * (1 + rs.uniform(-0.005, 0.005, n)),
        "high":   prices * (1 + rs.uniform(0.000, 0.015, n)),
        "low":    prices * (1 - rs.uniform(0.000, 0.015, n)),
        "close":  prices,
        "volume": rs.randint(1_000_000, 50_000_000, n).astype(float),
    }, index=rng)
    df.index.name = "date"
    return df.round(2)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time series features for forecasting:
    - Lag features (1, 5, 10, 20 days)
    - Rolling statistics (mean, std, min, max)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Calendar features (day of week, month, quarter)
    - Target: next-day return
    """
    data = df.copy()

    # Lag features
    for lag in [1, 5, 10, 20]:
        data[f"close_lag_{lag}"]  = data["close"].shift(lag)
        data[f"return_lag_{lag}"] = data["close"].pct_change(lag)

    # Rolling statistics
    for window in [5, 10, 20, 50]:
        data[f"roll_mean_{window}"] = data["close"].rolling(window).mean()
        data[f"roll_std_{window}"]  = data["close"].rolling(window).std()
        data[f"roll_min_{window}"]  = data["close"].rolling(window).min()
        data[f"roll_max_{window}"]  = data["close"].rolling(window).max()

    # RSI (14-day)
    delta = data["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"]        = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    bb_mean = data["close"].rolling(20).mean()
    bb_std  = data["close"].rolling(20).std()
    data["bb_upper"] = bb_mean + 2 * bb_std
    data["bb_lower"] = bb_mean - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / bb_mean

    # Volume features
    data["volume_ma_10"] = data["volume"].rolling(10).mean()
    data["volume_ratio"] = data["volume"] / data["volume_ma_10"]

    # Calendar features
    data["day_of_week"] = data.index.dayofweek
    data["month"]       = data.index.month
    data["quarter"]     = data.index.quarter

    # Target: 1-day forward return
    data["target_return"] = data["close"].pct_change().shift(-1)
    data["target_up"]     = (data["target_return"] > 0).astype(int)

    return data.dropna()
