"""
src/models.py
--------------
Time series forecasting models:
  - Baseline: Moving Average, ARIMA (via statsmodels)
  - ML: XGBoost, LightGBM with walk-forward validation
  - Evaluation: MAE, RMSE, MASE, Directional Accuracy
"""

from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MASE, and Directional Accuracy."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MASE instead of MAPE: daily returns sit near zero, so MAPE divides by ~0 and explodes.
    # MASE < 1 means the model beats a naive "repeat yesterday's return" forecast.
    naive_mae = np.mean(np.abs(np.diff(y_true))) if len(y_true) > 1 else np.nan
    mase = mae / naive_mae if naive_mae and naive_mae > 0 else np.nan

    # Directional accuracy: did we predict the right direction?
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    return {
        "mae":               round(float(mae), 6),
        "rmse":              round(float(rmse), 6),
        "mase":              round(float(mase), 4),
        "directional_acc":   round(float(dir_acc), 2),
    }


# ── Baseline: Moving Average ──────────────────────────────────────────────────

class MovingAverageBaseline:
    """Naive baseline: predict next return = average of last N returns."""

    def __init__(self, window: int = 5):
        self.window = window
        self.name   = f"MovingAverage(w={window})"

    def predict(self, series: pd.Series, n_steps: int = 30) -> np.ndarray:
        preds = []
        vals  = list(series.values)
        for _ in range(n_steps):
            pred = np.mean(vals[-self.window:])
            preds.append(pred)
            vals.append(pred)
        return np.array(preds)

    def evaluate(self, y_true: np.ndarray, series: pd.Series) -> dict:
        y_pred = self.predict(series, len(y_true))
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = self.name
        return metrics


# ── Baseline: ARIMA ───────────────────────────────────────────────────────────

class ARIMABaseline:
    """ARIMA(1,0,1) baseline for comparison."""

    def __init__(self, order: tuple = (1, 0, 1)):
        self.order = order
        self.name  = f"ARIMA{order}"
        self.model_ = None

    def fit(self, series: pd.Series):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model_ = ARIMA(series.values, order=self.order).fit()
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}")
        return self

    def predict(self, n_steps: int = 30) -> np.ndarray:
        if self.model_ is None:
            return np.zeros(n_steps)
        try:
            return self.model_.forecast(steps=n_steps)
        except Exception:
            return np.zeros(n_steps)

    def evaluate(self, y_true: np.ndarray, train_series: pd.Series) -> dict:
        self.fit(train_series)
        y_pred  = self.predict(len(y_true))
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = self.name
        return metrics


# ── ML: XGBoost ──────────────────────────────────────────────────────────────

class XGBoostForecaster:
    """XGBoost regression on engineered lag/technical features."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
        )
        self.name    = "XGBoost"
        self.model_  = None
        self.scaler_ = StandardScaler()
        self.feature_cols_: list = []

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        exclude = {"target_return", "target_up", "open", "high", "low",
                   "close", "volume"}
        return [c for c in df.columns if c not in exclude]

    def fit(self, df: pd.DataFrame):
        from xgboost import XGBRegressor
        self.feature_cols_ = self._get_feature_cols(df)
        X = self.scaler_.fit_transform(df[self.feature_cols_])
        y = df["target_return"].values
        self.model_ = XGBRegressor(**self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        X = self.scaler_.transform(df[self.feature_cols_])
        return self.model_.predict(X)

    def evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        self.fit(train_df)
        y_pred  = self.predict(test_df)
        y_true  = test_df["target_return"].values
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = self.name
        return metrics

    def feature_importance(self) -> pd.DataFrame:
        if self.model_ is None:
            return pd.DataFrame()
        imp = self.model_.feature_importances_
        return pd.DataFrame({
            "feature":    self.feature_cols_,
            "importance": imp,
        }).sort_values("importance", ascending=False).head(15)


# ── ML: LightGBM ─────────────────────────────────────────────────────────────

class LightGBMForecaster:
    """LightGBM regression — faster than XGBoost on large datasets."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbose=-1,
        )
        self.name    = "LightGBM"
        self.model_  = None
        self.scaler_ = StandardScaler()
        self.feature_cols_: list = []

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        exclude = {"target_return", "target_up", "open", "high", "low",
                   "close", "volume"}
        return [c for c in df.columns if c not in exclude]

    def fit(self, df: pd.DataFrame):
        from lightgbm import LGBMRegressor
        self.feature_cols_ = self._get_feature_cols(df)
        X = self.scaler_.fit_transform(df[self.feature_cols_])
        y = df["target_return"].values
        self.model_ = LGBMRegressor(**self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        X = self.scaler_.transform(df[self.feature_cols_])
        return self.model_.predict(X)

    def evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        self.fit(train_df)
        y_pred  = self.predict(test_df)
        y_true  = test_df["target_return"].values
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = self.name
        return metrics


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validate(
    df: pd.DataFrame,
    model,
    train_size: float = 0.7,
    n_splits: int = 5,
) -> list[dict]:
    """
    Walk-forward (expanding window) cross-validation.
    Returns list of per-fold metric dicts.
    """
    n       = len(df)
    init    = int(n * train_size)
    step    = (n - init) // n_splits
    results = []

    for i in range(n_splits):
        train_end = init + i * step
        test_end  = min(train_end + step, n)
        if test_end <= train_end:
            break

        train = df.iloc[:train_end]
        test  = df.iloc[train_end:test_end]

        try:
            fold_metrics = model.evaluate(train, test)
            fold_metrics["fold"] = i + 1
            fold_metrics["train_size"] = len(train)
            fold_metrics["test_size"]  = len(test)
            results.append(fold_metrics)
            logger.info(
                f"Fold {i+1}/{n_splits} | {fold_metrics['model']} | "
                f"MAE={fold_metrics['mae']:.5f} | "
                f"DirAcc={fold_metrics['directional_acc']:.1f}%"
            )
        except Exception as e:
            logger.warning(f"Fold {i+1} failed: {e}")

    return results
