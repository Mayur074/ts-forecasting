# 📈 Time Series Forecasting Dashboard

End-to-end **time series forecasting pipeline** comparing ML models (XGBoost, LightGBM) against statistical baselines (ARIMA, Moving Average) on financial data, with walk-forward validation, feature importance, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![CI](https://github.com/Mayur074/ts-forecasting/actions/workflows/ci.yml/badge.svg)

---

## Architecture

```
yfinance API (or synthetic GBM fallback)
        │
        ▼
Feature Engineering
  ├── Lag features (1, 5, 10, 20 days)
  ├── Rolling stats (mean, std, min, max — windows 5/10/20/50)
  ├── Technical indicators (RSI-14, MACD, Bollinger Bands)
  ├── Volume features (MA-10, volume ratio)
  └── Calendar features (day of week, month, quarter)
        │
        ▼
Model Training & Comparison
  ├── Baseline: Moving Average (w=5)
  ├── Baseline: ARIMA(1,0,1)
  ├── ML: XGBoost Regressor
  └── ML: LightGBM Regressor
        │
        ▼
Walk-Forward Validation (5 folds, expanding window)
        │
        ▼
Evaluation: MAE · RMSE · MASE · Directional Accuracy
        │
        ▼
Streamlit Dashboard
  ├── Price chart (train/test split)
  ├── Forecast vs Actual returns
  ├── Model comparison table
  ├── Feature importance (top 15)
  └── Walk-forward validation by fold
```

## Quickstart

```bash
git clone https://github.com/Mayur074/ts-forecasting.git
cd ts-forecasting
pip install -r requirements.txt

streamlit run dashboard/app.py
```

Select a ticker (AAPL, MSFT, GOOGL, AMZN, TSLA), click **Run Forecast**.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error on daily returns |
| RMSE | Root Mean Squared Error |
| MASE | Mean Absolute Scaled Error (below 1 beats a naive forecast) |
| Directional Accuracy | % of correct up/down predictions |

## Project Structure

```
ts-forecasting/
├── src/
│   ├── data_loader.py   # yfinance loader + GBM fallback + feature engineering
│   └── models.py        # XGBoost, LightGBM, ARIMA, MA + walk-forward validation
├── dashboard/
│   └── app.py           # Interactive Streamlit dashboard
├── tests/
│   └── test_forecasting.py  # 22 self-contained unit tests
└── requirements.txt
```

## Author

**Mayur Sangle** — M.S. Data Science, University of Maryland
[LinkedIn](https://linkedin.com/in/mayur-sangle-04m) • [GitHub](https://github.com/Mayur074)
