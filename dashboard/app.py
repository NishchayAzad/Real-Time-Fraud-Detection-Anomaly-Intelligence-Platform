"""
Real-time fraud monitoring dashboard -- self-contained, deployable standalone.

Two modes, auto-detected:

1. LIVE DEMO MODE (default): the dashboard generates its own simulated transaction
   stream in-process and calls a configurable scoring API URL, keeping results in
   Streamlit session state. This is what makes it deployable on Streamlit Community
   Cloud independently of where the API is hosted (e.g. the API on Render, the
   dashboard on Streamlit Cloud) -- no shared filesystem/DB required between them.

2. LOCAL SQLITE MODE: if streaming/live_scores.db exists (i.e. you're running
   streaming/simulator.py locally alongside this), the dashboard reads from it
   instead, which is convenient for local development.

Run locally:
    uvicorn app.main:app --port 8000 &
    streamlit run dashboard/app.py

Run against a deployed API (e.g. after deploying to Render):
    streamlit run dashboard/app.py
    # then set the API URL in the sidebar to your deployed API's URL
"""
import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "streaming" / "live_scores.db"
DEFAULT_API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")

CHANNELS = ["CARD_PRESENT", "CARD_NOT_PRESENT", "UPI_WALLET", "WIRE_TRANSFER"]
COUNTRIES = ["IN", "US", "GB", "AE", "SG", "NG", "RU", "BR"]
USER_POOL = [f"U{100000+i}" for i in range(200)]

st.set_page_config(page_title="Fraud & Anomaly Monitor", layout="wide", page_icon="🚨")
st.title("🚨 Real-Time Fraud & Anomaly Detection — Live Monitor")

if "live_rows" not in st.session_state:
    st.session_state.live_rows = []
if "running" not in st.session_state:
    st.session_state.running = False

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Scoring API URL", value=DEFAULT_API_URL,
                             help="Point this at your deployed FastAPI service, "
                                  "e.g. https://your-app.onrender.com")
    rate = st.slider("Transactions / second", 1, 15, 4)
    fraud_prob = st.slider("Fraud burst probability", 0.0, 0.5, 0.12)
    refresh = st.slider("Chart refresh (seconds)", 1, 5, 2)
    use_sqlite = DB_PATH.exists()
    if use_sqlite:
        st.info("Local streaming/live_scores.db detected — reading from it.")
    col_a, col_b = st.columns(2)
    start = col_a.button("Start", use_container_width=True, disabled=use_sqlite)
    stop = col_b.button("Stop", use_container_width=True, disabled=use_sqlite)
    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False
    if st.button("Clear history", use_container_width=True):
        st.session_state.live_rows = []


def call_api(client, payload):
    try:
        resp = client.post(f"{api_url}/score", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.session_state.running = False
        st.error(f"Could not reach the scoring API at {api_url}: {e}")
        return None


def gen_transaction(fraud_burst: bool):
    if fraud_burst:
        user = random.choice(USER_POOL)
        return {
            "user_id": user, "amount": round(random.uniform(0.5, 3.0), 2),
            "channel": "CARD_NOT_PRESENT", "country": random.choice(["NG", "RU"]),
            "merchant_id": f"M{random.randint(1000,9999)}",
            "device_id": f"D_ANOM_{random.randint(1000,9999)}",
        }
    user = random.choice(USER_POOL)
    return {
        "user_id": user, "amount": round(max(1, random.gauss(60, 30)), 2),
        "channel": random.choice(CHANNELS),
        "country": "IN" if random.random() < 0.9 else random.choice(COUNTRIES),
        "merchant_id": f"M{random.randint(1000,9999)}", "device_id": f"D{hash(user) % 5000}",
    }


def load_sqlite():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scores ORDER BY id DESC LIMIT 2000", conn)
    conn.close()
    return df


def render(df: pd.DataFrame):
    if df.empty:
        st.info("No data yet. Click **Start** in the sidebar to begin simulating "
                 "live traffic, or run `streaming/simulator.py` locally.")
        return
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
    alerts = df[df.decision != "ALLOW"].head(25)
    st.dataframe(alerts[["ts", "user_id", "amount", "channel", "country",
                          "fraud_risk_score", "decision", "reasons"]],
                 use_container_width=True)

    st.subheader("All recent transactions")
    st.dataframe(df.head(50)[["ts", "user_id", "amount", "channel", "country",
                               "fraud_risk_score", "decision"]],
                 use_container_width=True)


placeholder = st.empty()

if use_sqlite:
    while True:
        with placeholder.container():
            render(load_sqlite())
        time.sleep(refresh)
elif st.session_state.running:
    client = httpx.Client()
    with placeholder.container():
        st.caption(f"Streaming live against `{api_url}` ...")
    n_ticks = rate * refresh
    for _ in range(n_ticks):
        if not st.session_state.running:
            break
        burst = random.random() < fraud_prob
        n_send = random.randint(4, 8) if burst else 1
        for _ in range(n_send):
            payload = gen_transaction(burst)
            result = call_api(client, payload)
            if result is None:
                break
            row = {"id": len(st.session_state.live_rows),
                   "ts": datetime.now(timezone.utc).isoformat(),
                   "user_id": payload["user_id"], "amount": payload["amount"],
                   "channel": payload["channel"], "country": payload["country"],
                   "fraud_risk_score": result["fraud_risk_score"],
                   "decision": result["decision"],
                   "reasons": "; ".join(result["reasons"])}
            st.session_state.live_rows.append(row)
        time.sleep(1.0 / rate)
    with placeholder.container():
        render(pd.DataFrame(st.session_state.live_rows[::-1]))
    st.rerun()
else:
    with placeholder.container():
        render(pd.DataFrame(st.session_state.live_rows[::-1]))
