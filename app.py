import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Trade Anomaly Detective",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------

st.title("📈 Trade Anomaly Detective")

st.markdown("""
Detect suspicious trading behaviour in NSE stocks using
**Isolation Forest Machine Learning**.
""")

st.divider()

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("Configuration")

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "WIPRO.NS"
]

selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    stocks,
    default=[
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS"
    ]
)

period = st.sidebar.selectbox(
    "Historical Period",
    ["3mo", "6mo", "1y"],
    index=1
)

contamination = st.sidebar.slider(
    "Anomaly Sensitivity",
    0.01,
    0.10,
    0.05,
    0.01
)

run_button = st.sidebar.button(
    "🚀 Detect Anomalies"
)

# -------------------------------
# Waiting Screen
# -------------------------------

if not run_button:

    st.info(
        "Select stocks from the sidebar and click **Detect Anomalies**."
    )

    st.stop()




# --------------------------------------------------
# Fetch Stock Data
# --------------------------------------------------

@st.cache_data
def fetch_stock_data(tickers, period):

    all_data = []

    for ticker in tickers:

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Ticker"] = ticker
        df.reset_index(inplace=True)

        all_data.append(df)

    if len(all_data) == 0:
        return pd.DataFrame()

    return pd.concat(all_data).reset_index(drop=True)


# --------------------------------------------------
# Download Data
# --------------------------------------------------

with st.spinner("Downloading stock data..."):

    raw_data = fetch_stock_data(
        selected_stocks,
        period
    )

if raw_data.empty:

    st.error("No stock data could be downloaded.")

    st.stop()


with st.expander("📄 Raw Stock Data", expanded=False):

    st.success("Stock data downloaded successfully!")

    st.dataframe(
        raw_data.head(),
        use_container_width=True
    )


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def engineer_features(df):

    df = df.copy()

    # 20-day average volume
    df["Volume_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # Volume Spike
    df["Volume_Spike"] = (
        df["Volume"] /
        df["Volume_MA20"]
    )

    # Daily price change %
    df["Price_Change_Pct"] = (
        df["Close"]
        .pct_change()
        * 100
    )

    # Intraday range %
    df["Day_Range_Pct"] = (
        (df["High"] - df["Low"])
        / df["Low"]
    ) * 100

    # 20-Day Moving Average
    df["MA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    # Price Deviation
    df["Price_Deviation"] = (
        (df["Close"] - df["MA20"])
        / df["MA20"]
    ) * 100

    # Volume Price Pressure
    df["Volume_Price_Pressure"] = (
        df["Volume_Spike"]
        * abs(df["Price_Change_Pct"])
    )

    df.dropna(inplace=True)

    return df


# --------------------------------------------------
# Apply Feature Engineering
# --------------------------------------------------

featured_frames = []

for ticker in selected_stocks:

    stock_df = raw_data[
        raw_data["Ticker"] == ticker
    ].copy()

    stock_df = engineer_features(
        stock_df
    )

    featured_frames.append(
        stock_df
    )


final_df = (
    pd.concat(featured_frames)
    .reset_index(drop=True)
)

# Remove zero-volume days
final_df = final_df[
    final_df["Volume"] > 0
].reset_index(drop=True)


with st.expander("⚙️ Engineered Features", expanded=False):

    st.success("Feature engineering completed!")

    st.dataframe(
        final_df[
            [
                "Ticker",
                "Date",
                "Volume_Spike",
                "Price_Change_Pct",
                "Day_Range_Pct",
                "Price_Deviation",
                "Volume_Price_Pressure"
            ]
        ].head(),
        use_container_width=True
    )

# --------------------------------------------------
# Isolation Forest Anomaly Detection
# --------------------------------------------------

st.divider()
st.header("🤖 Isolation Forest Anomaly Detection")

features = [
    "Volume_Spike",
    "Price_Change_Pct",
    "Day_Range_Pct",
    "Price_Deviation",
    "Volume_Price_Pressure"
]

model_frames = []

for ticker in selected_stocks:

    stock = final_df[
        final_df["Ticker"] == ticker
    ].copy()

    if len(stock) < 25:
        continue

    scaler = StandardScaler()

    X = scaler.fit_transform(
        stock[features]
    )

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    stock["Anomaly"] = model.fit_predict(X)

    stock["Anomaly_Score"] = (
        -model.decision_function(X)
    )

    model_frames.append(stock)

results_df = pd.concat(
    model_frames
).reset_index(drop=True)

st.success("Isolation Forest model completed.")

st.subheader("Detected Anomalies")

st.dataframe(
    results_df[
        [
            "Ticker",
            "Date",
            "Close",
            "Anomaly",
            "Anomaly_Score"
        ]
    ],
    use_container_width=True
)

# --------------------------------------------------
# Risk Score & Risk Level
# --------------------------------------------------

score_scaler = MinMaxScaler(feature_range=(0, 100))

results_df["Risk_Score"] = score_scaler.fit_transform(
    results_df[["Anomaly_Score"]]
)

def get_risk_level(score):
    if score >= 80:
        return "🔴 Critical"
    elif score >= 60:
        return "🟠 High"
    elif score >= 40:
        return "🟡 Medium"
    else:
        return "🟢 Low"

results_df["Risk_Level"] = results_df["Risk_Score"].apply(get_risk_level)

st.subheader("📊 Risk Assessment")

st.dataframe(
    results_df[
        [
            "Ticker",
            "Date",
            "Close",
            "Risk_Score",
            "Risk_Level"
        ]
    ].sort_values(
        "Risk_Score",
        ascending=False
    ),
    use_container_width=True
)

# --------------------------------------------------
# Anomaly Explanation
# --------------------------------------------------

def explain_anomaly(row):

    reasons = []

    if row["Volume_Spike"] > 2:
        reasons.append("Unusual volume spike")

    if abs(row["Price_Change_Pct"]) > 3:
        reasons.append("Large price movement")

    if row["Day_Range_Pct"] > 4:
        reasons.append("High intraday volatility")

    if abs(row["Price_Deviation"]) > 5:
        reasons.append("Price far from 20-day average")

    if row["Volume_Price_Pressure"] > 8:
        reasons.append("Strong volume-price pressure")

    if len(reasons) == 0:
        return "Normal market behaviour"

    return ", ".join(reasons)


results_df["Reason"] = results_df.apply(
    explain_anomaly,
    axis=1
)


anomalies = results_df[
    results_df["Anomaly"] == -1
].copy()

# --------------------------------------------------
# Dashboard KPI Cards
# --------------------------------------------------

st.divider()
st.header("📊 Dashboard")

total_stocks = len(selected_stocks)
total_records = len(results_df)
total_anomalies = len(anomalies)

avg_risk = anomalies["Risk_Score"].mean() if total_anomalies > 0 else 0

highest_stock = (
    anomalies.groupby("Ticker")["Risk_Score"]
    .mean()
    .idxmax()
    if total_anomalies > 0
    else "N/A"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stocks", total_stocks)

with col2:
    st.metric("Records", total_records)

with col3:
    st.metric("Anomalies", total_anomalies)

with col4:
    st.metric("Avg Risk", f"{avg_risk:.1f}")

st.success(f"Highest Risk Stock: {highest_stock}")

st.subheader("📝 Anomaly Explanation")

st.write("Total anomalies found:", len(anomalies))
st.subheader("📝 Detected Trade Anomalies")
st.dataframe(
    anomalies[
        [
            "Ticker",
            "Date",
            "Close",
            "Risk_Score",
            "Risk_Level",
            "Reason"
        ]
    ].sort_values(
        "Risk_Score",
        ascending=False
    ),
    use_container_width=True
)



# --------------------------------------------------
# Interactive Stock Price Chart
# --------------------------------------------------

st.divider()
st.header("📈 Stock Price with Detected Anomalies")

selected_chart_stock = st.selectbox(
    "Select Stock to Visualize",
    selected_stocks
)

chart_data = results_df[
    results_df["Ticker"] == selected_chart_stock
].copy()

chart_anomalies = chart_data[
    chart_data["Anomaly"] == -1
].copy()

fig = go.Figure()

# Closing Price Line
fig.add_trace(
    go.Scatter(
        x=chart_data["Date"],
        y=chart_data["Close"],
        mode="lines",
        name="Closing Price",
        line=dict(width=2)
    )
)

# Anomaly Points
fig.add_trace(
    go.Scatter(
        x=chart_anomalies["Date"],
        y=chart_anomalies["Close"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            color="red",
            size=12,
            symbol="x"
        ),
        text=[
            f"""
Risk Score: {r:.2f}<br>
Risk Level: {l}<br>
Reason: {reason}
"""
            for r, l, reason in zip(
                chart_anomalies["Risk_Score"],
                chart_anomalies["Risk_Level"],
                chart_anomalies["Reason"]
            )
        ],
        hovertemplate=
        "<b>%{x}</b><br>"
        "Close: %{y}<br>"
        "%{text}<extra></extra>"
    )
)

fig.update_layout(
    title=f"{selected_chart_stock} Price Trend",
    xaxis_title="Date",
    yaxis_title="Closing Price",
    hovermode="x unified",
    template="plotly_white",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# Stock Risk Ranking
# --------------------------------------------------

st.divider()
st.header("🏆 Stock Risk Ranking")

ranking = (
    anomalies.groupby("Ticker")
    .agg(
        Avg_Risk=("Risk_Score", "mean"),
        Max_Risk=("Risk_Score", "max"),
        Total_Anomalies=("Risk_Score", "count")
    )
    .reset_index()
)

ranking = ranking.sort_values(
    "Avg_Risk",
    ascending=False
)

ranking.insert(
    0,
    "Rank",
    range(1, len(ranking) + 1)
)

st.dataframe(
    ranking.style.format({
        "Avg_Risk": "{:.2f}",
        "Max_Risk": "{:.2f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# Executive Summary
# --------------------------------------------------

st.divider()
st.header("📋 Executive Summary")

highest_risk = ranking.iloc[0]

summary = f"""
Out of **{total_records}** trading records analyzed across **{total_stocks}** stocks,
the Isolation Forest model detected **{total_anomalies}** suspicious trading events.

The stock with the highest average risk is **{highest_risk['Ticker']}**
with an average risk score of **{highest_risk['Avg_Risk']:.2f}**.

These anomalies may indicate unusual trading behaviour such as
volume spikes, abnormal price movement, or increased market volatility.
"""

st.info(summary)

# --------------------------------------------------
# Download Results
# --------------------------------------------------

st.divider()
st.header("📥 Download Anomaly Report")

download_df = anomalies[
    [
        "Ticker",
        "Date",
        "Close",
        "Risk_Score",
        "Risk_Level",
        "Reason"
    ]
].copy()

download_df["Risk_Score"] = download_df["Risk_Score"].round(2)

csv = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download CSV Report",
    data=csv,
    file_name="trade_anomaly_report.csv",
    mime="text/csv"
)