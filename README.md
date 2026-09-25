#  Trade Anomaly Detective

An ML-powered stock trade anomaly detection dashboard built using **Python, Streamlit, Isolation Forest, Plotly, and Yahoo Finance**.

##  Live Demo
https://aditirasal20-trade-anomaly-detective-app-b4oxlf.streamlit.app/


##  Overview

Trade Anomaly Detective analyzes historical NSE stock market data and automatically identifies suspicious trading behavior using the **Isolation Forest** anomaly detection algorithm.

The application calculates multiple trading indicators, assigns a risk score, explains why each trade is considered suspicious, and presents the results through an interactive dashboard.

---

##  Features

*  Interactive Streamlit Dashboard
*  Isolation Forest Machine Learning
*  Interactive Plotly Stock Charts
*  Risk Score (0–100)
*  Risk Levels (Low / Medium / High / Critical)
*  Human-readable anomaly explanations
*  Stock Risk Ranking
*  Download anomaly reports as CSV
*  Raw and engineered data exploration

---

##  Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Yahoo Finance (yfinance)

---

##  Workflow

1. Download historical NSE stock data.
2. Perform feature engineering.
3. Detect anomalies using Isolation Forest.
4. Calculate normalized risk scores.
5. Generate human-readable explanations.
6. Visualize anomalies through interactive dashboards.
7. Export results as CSV.

---

##  Engineered Features

* Volume Spike
* Price Change (%)
* Intraday Range (%)
* 20-Day Moving Average
* Price Deviation
* Volume Price Pressure

---

##  Future Improvements

* Multi-timeframe analysis
* Technical indicators (RSI, MACD, Bollinger Bands)
* Real-time anomaly detection
* Email alerts
* Portfolio monitoring
* Dark mode

---
