"""
dashboard/app.py
-----------------
Interactive Streamlit forecasting dashboard.
Select ticker, date range, model — see forecast vs actual,
feature importance, model comparison, and evaluation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_ticker, engineer_features
from src.models import (
    XGBoostForecaster, LightGBMForecaster,
    MovingAverageBaseline, ARIMABaseline,
    walk_forward_validate, compute_metrics,
)

st.set_page_config(
    page_title = "Time Series Forecasting",
    page_icon  = "📈",
    layout     = "wide",
)

st.markdown("""
<style>
.main { background: #0f1117; }
.metric-card { background: #1a1d2e; border-radius: 8px; padding: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 TS Forecasting")
    st.caption("XGBoost · LightGBM · ARIMA · Moving Average")
    st.divider()

    ticker    = st.selectbox("Ticker", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    start     = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
    model_sel = st.selectbox("Forecast Model", ["XGBoost", "LightGBM"])
    train_pct = st.slider("Train / Test Split", 0.6, 0.9, 0.8, 0.05)
    run_btn   = st.button("▶ Run Forecast", type="primary", use_container_width=True)
    st.divider()
    show_fi   = st.checkbox("Show Feature Importance", value=True)
    show_wfv  = st.checkbox("Show Walk-Forward Validation", value=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📈 Time Series Forecasting Dashboard")
st.markdown("Baseline vs ML forecasting on financial time series with walk-forward validation.")

if run_btn:
    with st.spinner(f"Loading {ticker} data and engineering features..."):
        raw  = load_ticker(ticker, start=str(start))
        feat = engineer_features(raw)

    n_train = int(len(feat) * train_pct)
    train   = feat.iloc[:n_train]
    test    = feat.iloc[n_train:]

    st.subheader(f"{ticker} — {len(feat):,} trading days loaded")

    # ── Price chart ───────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train["close"],
        name="Train", line=dict(color="#6c63ff")
    ))
    fig.add_trace(go.Scatter(
        x=test.index, y=test["close"],
        name="Test", line=dict(color="#f59e0b")
    ))
    fig.update_layout(
        title=f"{ticker} Closing Price — Train/Test Split",
        template="plotly_dark", xaxis_title="Date", yaxis_title="Price ($)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model training + evaluation ───────────────────────────────────────────
    st.subheader("Model Comparison")

    models_to_run = {
        "XGBoost":        XGBoostForecaster(),
        "LightGBM":       LightGBMForecaster(),
        "MovingAverage":  MovingAverageBaseline(window=5),
        "ARIMA":          ARIMABaseline(order=(1, 0, 1)),
    }

    all_metrics = []
    for name, m in models_to_run.items():
        with st.spinner(f"Training {name}..."):
            try:
                if hasattr(m, "evaluate") and "train_df" in m.evaluate.__code__.co_varnames:
                    metrics = m.evaluate(train, test)
                else:
                    metrics = m.evaluate(test["target_return"].values, train["target_return"])
                all_metrics.append(metrics)
            except Exception as e:
                st.warning(f"{name} failed: {e}")

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics).set_index("model")
        st.dataframe(
            metrics_df[["mae", "rmse", "mase", "directional_acc"]]
            .style.highlight_min(axis=0, subset=["mae", "rmse", "mase"])
            .highlight_max(axis=0, subset=["directional_acc"]),
            use_container_width=True,
        )

    # ── Forecast vs Actual ────────────────────────────────────────────────────
    st.subheader(f"{model_sel} — Forecast vs Actual Returns")
    sel_model = XGBoostForecaster() if model_sel == "XGBoost" else LightGBMForecaster()

    with st.spinner("Generating forecast..."):
        try:
            sel_model.fit(train)
            y_pred = sel_model.predict(test)
            y_true = test["target_return"].values

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=test.index, y=y_true,
                name="Actual", line=dict(color="#6c63ff", width=1.5)
            ))
            fig2.add_trace(go.Scatter(
                x=test.index, y=y_pred,
                name="Forecast", line=dict(color="#f59e0b", dash="dot", width=1.5)
            ))
            fig2.update_layout(
                title=f"{model_sel} Forecast vs Actual — {ticker}",
                template="plotly_dark",
                xaxis_title="Date", yaxis_title="Daily Return"
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Forecast failed: {e}")

    # ── Feature importance ────────────────────────────────────────────────────
    if show_fi and hasattr(sel_model, "feature_importance"):
        st.subheader("Top 15 Feature Importances")
        fi = sel_model.feature_importance()
        if not fi.empty:
            fig3 = px.bar(fi, x="importance", y="feature", orientation="h",
                          title="Feature Importance", template="plotly_dark")
            fig3.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)

    # ── Walk-forward validation ───────────────────────────────────────────────
    if show_wfv:
        st.subheader("Walk-Forward Validation (5 Folds)")
        with st.spinner("Running walk-forward validation..."):
            try:
                wfv_model   = XGBoostForecaster() if model_sel == "XGBoost" else LightGBMForecaster()
                wfv_results = walk_forward_validate(feat, wfv_model, n_splits=5)
                wfv_df      = pd.DataFrame(wfv_results)
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(
                    x=wfv_df["fold"].astype(str), y=wfv_df["directional_acc"],
                    name="Directional Accuracy (%)", marker_color="#6c63ff"
                ))
                fig4.update_layout(
                    title="Walk-Forward Directional Accuracy by Fold",
                    template="plotly_dark", yaxis_title="Accuracy (%)"
                )
                st.plotly_chart(fig4, use_container_width=True)
                st.dataframe(wfv_df[["fold","train_size","test_size","mae","rmse","directional_acc"]],
                             use_container_width=True)
            except Exception as e:
                st.warning(f"Walk-forward validation failed: {e}")

else:
    st.info("Select a ticker and click **▶ Run Forecast** in the sidebar to begin.")
