"""
FastAPI real-time fraud + anomaly scoring service.

Cold start: models + a SHAP TreeExplainer are loaded ONCE at startup.
Hot path (/score): O(1) feature lookup/update per user (in-memory UserStateStore),
                    one XGBoost inference, one IsolationForest inference, SHAP values
                    for the top contributing features. No disk/DB access on the hot path.

Run:
    uvicorn app.main:app --reload --port 8000
    open http://127.0.0.1:8000/docs
"""
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone

import joblib
import shap
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import UserStateStore, FEATURE_ORDER  # noqa: E402
from risk_engine import score_transaction  # noqa: E402

from app.schemas import TransactionRequest, ScoreResponse  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_state = {"store": None, "xgb_model": None, "iso_model": None, "explainer": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["store"] = UserStateStore()
    _state["xgb_model"] = joblib.load(MODEL_DIR / "xgb_model.joblib")
    _state["iso_model"] = joblib.load(MODEL_DIR / "iso_forest.joblib")
    _state["explainer"] = shap.TreeExplainer(_state["xgb_model"])
    print(f"Models loaded from {MODEL_DIR}. Service ready.")
    yield
    _state.clear()


app = FastAPI(
    title="Real-Time Fraud & Anomaly Detection API",
    description="Hybrid supervised (XGBoost) + unsupervised (Isolation Forest) + "
                "rule-based fraud scoring with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    store = _state["store"]
    return {
        "status": "ok",
        "tracked_users": store.snapshot_size() if store else 0,
        "model_loaded": _state["xgb_model"] is not None,
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: TransactionRequest):
    start = time.perf_counter()
    if _state["xgb_model"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    ts = txn.timestamp or datetime.now(timezone.utc).replace(tzinfo=None)

    store: UserStateStore = _state["store"]
    features = store.compute_features(
        user_id=txn.user_id, amount=txn.amount, ts=ts,
        channel=txn.channel, country=txn.country, device_id=txn.device_id,
    )

    result = score_transaction(
        features, _state["xgb_model"], _state["iso_model"], FEATURE_ORDER
    )

    # SHAP explanation for the supervised component
    X = np.array([[features[f] for f in FEATURE_ORDER]])
    shap_values = _state["explainer"].shap_values(X)[0]
    top_idx = np.argsort(-np.abs(shap_values))[:4]
    top_shap = [
        {"feature": FEATURE_ORDER[i], "value": round(float(features[FEATURE_ORDER[i]]), 3),
         "shap_contribution": round(float(shap_values[i]), 4)}
        for i in top_idx
    ]

    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    return ScoreResponse(
        transaction_id=f"REQ-{uuid.uuid4().hex[:10]}",
        top_shap_features=top_shap,
        latency_ms=latency_ms,
        **result.to_dict(),
    )


@app.get("/")
def root():
    return {
        "service": "Real-Time Fraud & Anomaly Detection API",
        "docs": "/docs",
        "health": "/health",
    }
