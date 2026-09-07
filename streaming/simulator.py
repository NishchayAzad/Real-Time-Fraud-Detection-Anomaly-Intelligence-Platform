"""
Real-time stream simulator.

This plays the role of a message broker consumer in a production system: in prod you'd
have Kafka/Kinesis pushing transaction events and a consumer group calling the scoring
service (see docker-compose.yml for a Kafka/Redis production topology). Here, an asyncio
producer generates a live mixed stream of normal transactions and randomly-injected fraud
bursts (card testing, account takeover) and posts each one to the FastAPI /score endpoint
in real time, logging every scored result to a local SQLite DB that the Streamlit dashboard
reads from -- so you get a genuine real-time, end-to-end loop without any extra infra to
stand up for a demo.

Run:
    python streaming/simulator.py --api http://127.0.0.1:8000 --rate 5 --duration 120
"""
import argparse
import asyncio
import random
import sqlite3
import time
from datetime import datetime, timezone

import httpx

DB_PATH = "streaming/live_scores.db"
CHANNELS = ["CARD_PRESENT", "CARD_NOT_PRESENT", "UPI_WALLET", "WIRE_TRANSFER"]
COUNTRIES = ["IN", "US", "GB", "AE", "SG", "NG", "RU", "BR"]
USER_POOL = [f"U{100000+i}" for i in range(200)]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, user_id TEXT, amount REAL, channel TEXT, country TEXT,
            fraud_risk_score REAL, decision TEXT, reasons TEXT
        )
    """)
    conn.commit()
    return conn


def log_result(conn, user_id, amount, channel, country, result):
    conn.execute(
        "INSERT INTO scores (ts, user_id, amount, channel, country, fraud_risk_score, decision, reasons)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), user_id, amount, channel, country,
         result["fraud_risk_score"], result["decision"], "; ".join(result["reasons"])),
    )
    conn.commit()


async def send_normal(client: httpx.AsyncClient, api: str, conn):
    user = random.choice(USER_POOL)
    amount = round(max(1, random.gauss(60, 30)), 2)
    channel = random.choice(CHANNELS)
    country = "IN" if random.random() < 0.9 else random.choice(COUNTRIES)
    payload = {
        "user_id": user, "amount": amount, "channel": channel, "country": country,
        "merchant_id": f"M{random.randint(1000,9999)}", "device_id": f"D{hash(user) % 5000}",
    }
    resp = await client.post(f"{api}/score", json=payload, timeout=5)
    result = resp.json()
    log_result(conn, user, amount, channel, country, result)
    return result


async def send_fraud_burst(client: httpx.AsyncClient, api: str, conn):
    user = random.choice(USER_POOL)
    device = f"D_ANOM_{random.randint(1000,9999)}"
    country = random.choice(["NG", "RU"])
    n = random.randint(4, 8)
    for i in range(n):
        amount = round(random.uniform(0.5, 3.0), 2)
        payload = {
            "user_id": user, "amount": amount, "channel": "CARD_NOT_PRESENT",
            "country": country, "merchant_id": f"M{random.randint(1000,9999)}",
            "device_id": device,
        }
        resp = await client.post(f"{api}/score", json=payload, timeout=5)
        result = resp.json()
        log_result(conn, user, amount, "CARD_NOT_PRESENT", country, result)
        await asyncio.sleep(0.15)


async def run(api: str, rate: float, duration: int, fraud_prob: float):
    conn = init_db()
    async with httpx.AsyncClient() as client:
        end_time = time.time() + duration
        n_sent = 0
        while time.time() < end_time:
            if random.random() < fraud_prob:
                await send_fraud_burst(client, api, conn)
            else:
                result = await send_normal(client, api, conn)
                n_sent += 1
                print(f"[{n_sent}] {result['decision']:5s} risk={result['fraud_risk_score']:.3f}")
            await asyncio.sleep(1.0 / rate)
    conn.close()
    print("Simulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=5.0, help="transactions per second")
    parser.add_argument("--duration", type=int, default=60, help="seconds to run")
    parser.add_argument("--fraud-prob", type=float, default=0.08,
                         help="probability of injecting a fraud burst each tick")
    args = parser.parse_args()
    asyncio.run(run(args.api, args.rate, args.duration, args.fraud_prob))
