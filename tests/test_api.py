"""
Basic API + logic tests. Run with: pytest tests/ -v

These are deliberately simple (this is a portfolio project, not a production test suite)
but cover the things most likely to silently break: health check, a normal transaction
scoring to ALLOW, and a velocity burst escalating to FLAG/BLOCK.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers startup/shutdown lifespan events
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_normal_transaction_allows(client):
    resp = client.post("/score", json={
        "user_id": "TEST_USER_1", "amount": 50.0, "channel": "UPI_WALLET",
        "country": "IN", "merchant_id": "M1", "device_id": "D1",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in {"ALLOW", "FLAG", "BLOCK"}
    assert 0.0 <= body["fraud_risk_score"] <= 1.0
    assert len(body["top_shap_features"]) > 0


def test_velocity_burst_escalates(client):
    decisions = []
    for i in range(6):
        resp = client.post("/score", json={
            "user_id": "TEST_USER_BURST", "amount": 1.5, "channel": "CARD_NOT_PRESENT",
            "country": "NG", "merchant_id": f"M{i}", "device_id": "D_SUSPECT",
        })
        decisions.append(resp.json()["decision"])
    # by the last transaction in a rapid tiny-amount burst, risk should have escalated
    assert decisions[-1] in {"FLAG", "BLOCK"}


def test_rejects_negative_amount(client):
    resp = client.post("/score", json={
        "user_id": "TEST_USER_2", "amount": -10.0, "channel": "UPI_WALLET",
        "country": "IN", "merchant_id": "M1", "device_id": "D1",
    })
    assert resp.status_code == 422
