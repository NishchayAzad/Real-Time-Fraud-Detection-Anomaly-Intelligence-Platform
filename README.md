# 🚨 Real-Time Fraud & Anomaly Detection System

A hybrid **supervised + unsupervised** fraud scoring engine, served in real time via FastAPI,
with stateful causal feature engineering, SHAP explainability, and a live monitoring dashboard.

Built on a **self-generated, multi-channel synthetic transaction dataset** — every column and
every fraud pattern is something I designed and can explain, rather than an anonymized
Kaggle CSV with columns like `V1...V28`.

---

## Why this project is different from a typical "fraud detection" portfolio project

Most fraud detection projects on GitHub are a single classifier (usually XGBoost or a random
forest) trained on a public Kaggle dataset, evaluated offline with a precision/recall table.
That's a modeling exercise, not a fraud **system**. This project is built around the three
things that actually separate a real fraud engine from a notebook:

| Gap in typical projects | What this project does instead |
|---|---|
| Anonymized dataset (`V1..V28`) you can't explain | Self-written synthetic data generator with 4 named fraud typologies you can describe end-to-end |
| One supervised model, blind to novel fraud | **Hybrid** supervised (XGBoost) + unsupervised (Isolation Forest) scoring — the second model exists specifically to catch fraud patterns that don't resemble any label |
| Features computed by scanning a CSV/DB at request time (leaks the future, too slow) | **Causal, O(1) stateful feature engineering** — every feature uses only what happened *before* the current transaction, computed with the exact same code path offline (training) and online (serving) |
| "Here's my F1 score" and nothing else | A hybrid **rule + ML risk engine** with human-readable reason codes and per-transaction **SHAP** explanations — the auditability a real fraud/compliance team needs |
| A trained model in a notebook, no serving story | A running **FastAPI** service, an **async streaming simulator**, and a **live Streamlit dashboard** — the full loop from transaction to alert |

---

## Architecture

```
                     ┌─────────────────────────┐
 Transaction Stream  │   streaming/simulator.py │   (stand-in for a Kafka consumer
 (async, real-time)  │   posts txns in real time│    in production — see below)
                     └────────────┬─────────────┘
                                  │ HTTP POST /score
                                  ▼
                     ┌─────────────────────────┐
                     │   FastAPI  (app/main.py) │
                     │                         │
                     │  1. UserStateStore      │  ← O(1) causal feature lookup/update
                     │     (in-memory, Redis-  │    per user (velocity, recency,
                     │      swappable)         │    deviation, device/geo history)
                     │  2. XGBoost             │  ← known fraud typologies
                     │  3. Isolation Forest    │  ← novel / zero-day anomalies
                     │  4. Rule heuristics     │  ← auditable guardrails
                     │  5. SHAP explainer      │  ← per-transaction reason codes
                     └────────────┬─────────────┘
                                  │ writes
                                  ▼
                     ┌─────────────────────────┐
                     │  SQLite live_scores.db   │
                     └────────────┬─────────────┘
                                  │ reads (auto-refresh)
                                  ▼
                     ┌─────────────────────────┐
                     │  Streamlit dashboard     │  ← live alert feed, risk trend,
                     │  (dashboard/app.py)      │    decision breakdown
                     └─────────────────────────┘
```

---

## The dataset (`data/generate_dataset.py`)

I generate a synthetic, multi-channel transaction dataset rather than using a pre-existing
one, so I can explain every field and design decision:

- **Channels**: `CARD_PRESENT`, `CARD_NOT_PRESENT`, `UPI_WALLET`, `WIRE_TRANSFER` — each has
  a different "normal" profile (amount distribution, preferred channel per user).
- **4,000 synthetic users**, each with a home country, average transaction size, preferred
  channel, and account age — so "normal" is defined per-user, not globally.
- **Four injected fraud typologies**, each with a distinct behavioral signature:
  - `CARD_TESTING` — 6–15 tiny transactions in rapid succession on a new device (fraudsters
    validating a stolen card before a large purchase).
  - `ACCOUNT_TAKEOVER` — a burst of large transactions from a new device *and* a foreign
    country inconsistent with the user's history.
  - `MULE_FANOUT` — one large incoming wire transfer immediately followed by several
    smaller outgoing transfers to previously-unseen beneficiaries.
  - `MERCHANT_COLLUSION` — repeated, suspiciously round-number charges to the same merchant.

```bash
python data/generate_dataset.py --n-users 4000 --n-transactions 260000 --fraud-rate 0.012
```

## Causal feature engineering (`src/features.py`)

The single most important design decision in this project: **the same `UserStateStore`
class is used both offline (replaying history during training) and online (serving live
requests)**. Every feature is computed using only transactions that happened *before* the
current one for that user — no future leakage, and no offline/online feature mismatch,
which is one of the most common real-world bugs in fraud systems.

Features (all O(1) or O(k) with small bounded k):
`amount`, `hour`, `day_of_week`, `is_weekend`, `is_late_night`, `time_since_prev_txn_min`,
`txns_last_10min`, `txns_last_1h`, `txns_last_24h`, `user_avg_amount`,
`amount_over_user_avg`, `amount_zscore`, `is_new_device`, `is_new_country`,
`is_channel_switch`.

## Hybrid risk engine (`src/risk_engine.py`)

```
final_risk = 0.55 × XGBoost_probability   (known fraud patterns)
           + 0.25 × IsolationForest_score (novel/anomalous patterns)
           + 0.20 × rule_heuristic_score  (auditable guardrails)
```

Decision thresholds: `≥0.80 → BLOCK`, `≥0.45 → FLAG`, else `ALLOW`. Every decision comes
with plain-English reasons (e.g. *"New device AND new country observed simultaneously"*)
plus the top SHAP-contributing features for the ML component.

## Results (on the generated dataset, holdout split)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| XGBoost (supervised) | 0.999 | 1.000 |
| Isolation Forest (unsupervised, as anomaly ranker) | 0.881 | — |

The supervised model performs near-perfectly *on the fraud patterns it was trained on* —
expected, since the synthetic typologies have a clear behavioral signature. The honest
interview talking point is the isolation forest number: it recovers ~88% of fraud **using
no labels at all**, which is the actual point of the hybrid design — it's the layer that
would still catch a fifth, never-labeled fraud typology that XGBoost has never seen.

---

## Running it end to end

```bash
pip install -r requirements.txt

# 1. Generate data + train both models
python data/generate_dataset.py
cd src && python train.py && cd ..

# 2. Start the real-time scoring API
uvicorn app.main:app --reload --port 8000
# docs at http://127.0.0.1:8000/docs

# 3. In another terminal, simulate a live transaction stream
python streaming/simulator.py --api http://127.0.0.1:8000 --rate 5 --duration 300

# 4. In another terminal, watch the live dashboard
streamlit run dashboard/app.py
```

Or with Docker (API only):
```bash
docker compose up --build
```

### Example request

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "user_id": "U100042", "amount": 1500, "channel": "UPI_WALLET",
  "country": "IN", "merchant_id": "M4821", "device_id": "D1234"
}'
```

```json
{
  "transaction_id": "REQ-8e3fa49174",
  "fraud_risk_score": 0.83,
  "decision": "BLOCK",
  "ml_supervised_score": 1.0,
  "ml_anomaly_score": 0.99,
  "rule_heuristic_score": 0.35,
  "reasons": [
    "Very high transaction velocity (5+ in last 10 minutes)",
    "ML model strongly matches a known fraud pattern (p=1.00)",
    "Transaction is statistically anomalous vs. population (score=0.99)"
  ],
  "top_shap_features": [
    {"feature": "txns_last_10min", "value": 6, "shap_contribution": 3.21}
  ],
  "latency_ms": 4.2
}
```

---

## Productionizing (what I'd change to run this for real)

- Swap `UserStateStore`'s in-memory dict for **Redis** (interface is already isolated in
  `src/features.py` so this is a drop-in change, not a rewrite).
- Replace `streaming/simulator.py` with an actual **Kafka/Kinesis** consumer group feeding
  the same `/score` endpoint — `docker-compose.yml` documents this topology.
- Add **model monitoring** (e.g. population stability index / Evidently) to catch feature
  drift as fraud patterns evolve — a static model degrades within weeks in production.
- Add **shadow deployment** and champion/challenger evaluation before promoting a retrained
  model, instead of hot-swapping the `.joblib` file.

---

## Project structure

```
fraud-anomaly-detection/
├── data/
│   └── generate_dataset.py     # synthetic multi-channel transaction + fraud generator
├── src/
│   ├── features.py             # causal, stateful feature engineering (shared train/serve)
│   ├── train.py                # trains XGBoost + Isolation Forest
│   └── risk_engine.py          # hybrid scoring + rule heuristics + decisioning
├── app/
│   ├── main.py                 # FastAPI service (/score, /health)
│   └── schemas.py              # request/response models
├── streaming/
│   └── simulator.py            # async real-time transaction stream simulator
├── dashboard/
│   └── app.py                  # Streamlit live monitoring dashboard
├── tests/
│   └── test_api.py
├── models/                     # trained models (checked in, small enough to ship)
├── .github/workflows/ci.yml    # generates data, trains, runs tests on every push
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## Tech stack

FastAPI · XGBoost · scikit-learn (Isolation Forest) · SHAP · Streamlit · Pydantic ·
pandas/numpy · pytest · Docker · GitHub Actions
