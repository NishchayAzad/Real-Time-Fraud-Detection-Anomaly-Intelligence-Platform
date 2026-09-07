from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    user_id: str = Field(..., examples=["U100042"])
    amount: float = Field(..., gt=0, examples=[1500.0])
    timestamp: Optional[datetime] = Field(default=None, description="Defaults to now() if omitted")
    channel: str = Field(..., examples=["UPI_WALLET"])
    country: str = Field(..., examples=["IN"])
    merchant_id: str = Field(..., examples=["M4821"])
    device_id: str = Field(..., examples=["D1234"])


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_risk_score: float
    decision: str
    ml_supervised_score: float
    ml_anomaly_score: float
    rule_heuristic_score: float
    reasons: list[str]
    top_shap_features: list[dict]
    latency_ms: float
