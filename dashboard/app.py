"""
Real-time fraud monitoring dashboard.

Reads the SQLite log that the streaming simulator (or the live API, if you wire it up the
same way) writes to, and auto-refreshes -- this is the "SOC analyst screen" a fraud team
would actually watch. Run alongside the API and the simulator:

    uvicorn app.main:app --port 8000 &
    python streaming/simulator.py --api http://127.0.0.1:8000 --rate 5 --duration 600 &
    streamlit run dashboard/app.py
"""
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "streaming" / "live_scores.db"

st.set_page_config(page_title="Fraud & Anomaly Monitor", layout="wide")
st.title("🚨 Real-Time Fraud & Anomaly Detection — Live Monitor")

placeholder = st.empty()


def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scores ORDER BY id DESC LIMIT 2000", conn)
    conn.close()
    return df


refresh = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)
auto = st.sidebar.checkbox("Auto-refresh", value=True)

while True:
    df = load_data()
    with placeholder.container():
        if df.empty:
            st.info("No data yet — start the API and the streaming simulator "
                     "(see docstring at top of dashboard/app.py).")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions scored", len(df))
            c2.metric("Blocked", int((df.decision == "BLOCK").sum()))
            c3.metric("Flagged", int((df.decision == "FLAG").sum()))
            c4.metric("Avg risk score", f"{df.fraud_risk_score.mean():.3f}")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Risk score over time")
                chart_df = df.sort_values("id").tail(300)[["id", "fraud_risk_score"]].set_index("id")
                st.line_chart(chart_df)
            with col2:
                st.subheader("Decisions")
                st.bar_chart(df.decision.value_counts())

            st.subheader("Recent flagged / blocked transactions")
            recent_alerts = df[df.decision != "ALLOW"].head(25)
            st.dataframe(recent_alerts[["ts", "user_id", "amount", "channel", "country",
                                        "fraud_risk_score", "decision", "reasons"]],
                         use_container_width=True)

            st.subheader("All recent transactions")
            st.dataframe(df.head(50)[["ts", "user_id", "amount", "channel", "country",
                                       "fraud_risk_score", "decision"]],
                         use_container_width=True)
    if not auto:
        break
    time.sleep(refresh)
