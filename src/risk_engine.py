"""
Hybrid risk engine.

Final risk score = weighted blend of THREE independent signals, each catching a
different failure mode:

  1. ML_SUPERVISED   (XGBoost probability)     -> known fraud typologies
  2. ML_ANOMALY      (Isolation Forest score)   -> novel / zero-day patterns
  3. RULE_HEURISTIC  (hand-written business rules) -> instant, auditable guardrails
                                                       that don't depend on a model being
                                                       loaded correctly (defense in depth)

Why not just trust the ML score?
  - A single ML model is a black box a regulator/analyst can't always explain in the moment.
  - Business rules are cheap insurance against model staleness, feature pipeline bugs, or
    adversarial drift, and they give instantly auditable reasons.

This mirrors how real hybrid fraud engines are built: ML for scale and pattern discovery,
rules for guaranteed coverage of known hard constraints, blended into one number with
full reason-code transparency (SHAP for the ML part, plain logic for the rules part).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

WEIGHTS = {
    "ml_supervised": 0.55,
    "ml_anomaly": 0.25,
    "rule_heuristic": 0.20,
}

DECISION_THRESHOLDS = {
    "BLOCK": 0.80,
    "FLAG": 0.45,
    # below FLAG -> ALLOW
}

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "models" / "anomaly_calibration.json"
_calibration_cache = None


def load_anomaly_calibration() -> dict:
    """Loads the Isolation Forest score calibration computed at training time
    (see train.py::calibrate_anomaly_scores). Falls back to a permissive default
    if the file is missing, so this never hard-crashes serving -- but a missing
    calibration file means anomaly scores are uncalibrated and should be treated
    as unreliable until models are retrained."""
    global _calibration_cache
    if _calibration_cache is not None:
        return _calibration_cache
    if _CALIBRATION_PATH.exists():
        with open(_CALIBRATION_PATH) as f:
            _calibration_cache = json.load(f)
    else:
        _calibration_cache = {"p50": -0.5, "p99": -0.2}
    return _calibration_cache


@dataclass
class RiskResult:
    fraud_risk_score: float
    decision: str
    ml_supervised_score: float
    ml_anomaly_score: float
    rule_heuristic_score: float
    reasons: list

    def to_dict(self):
        return asdict(self)


def rule_heuristic_score(features: dict) -> tuple[float, list]:
    """Cheap, explainable, model-independent guardrails."""
    score = 0.0
    reasons = []

    if features["txns_last_10min"] >= 5:
        score += 0.35
        reasons.append("Very high transaction velocity (5+ in last 10 minutes)")
    elif features["txns_last_1h"] >= 8:
        score += 0.20
        reasons.append("High transaction velocity (8+ in last hour)")

    if features["time_since_prev_txn_min"] < 0.5 and features["txns_last_10min"] > 1:
        score += 0.15
        reasons.append("Extremely short time since previous transaction")

    if features["amount_zscore"] > 4:
        score += 0.30
        reasons.append("Transaction amount far above user's historical pattern")
    elif features["amount_over_user_avg"] > 5:
        score += 0.15
        reasons.append("Transaction amount several times user's average")

    if features["is_new_device"] and features["is_new_country"]:
        score += 0.30
        reasons.append("New device AND new country observed simultaneously")
    elif features["is_new_country"]:
        score += 0.15
        reasons.append("Transaction from a country not seen before for this user")
    elif features["is_new_device"]:
        score += 0.10
        reasons.append("Transaction from a previously unseen device")

    if features["is_late_night"] and features["amount_over_user_avg"] > 2:
        score += 0.10
        reasons.append("Large late-night transaction")

    return min(score, 1.0), reasons


def normalize_anomaly_score(raw_score: float, calibration: dict | None = None) -> float:
    """Maps a raw IsolationForest.score_samples value (higher = more normal) to a
    [0, 1] anomaly score, calibrated against the ACTUAL distribution of scores on
    legitimate training data -- not an assumed fixed center.

    calibration['p50'] = median anomaly level ("-score_samples") for legit data
                          -> maps to ~0 (typical legit transaction)
    calibration['p99'] = 99th percentile anomaly level for legit data
                          -> maps to ~1 (as anomalous as the most unusual 1% of
                             legit traffic ever gets; genuine fraud typically
                             sits well beyond this)

    Everything between is linearly interpolated and clipped to [0, 1]. This
    means the score is directly interpretable as "how far into the unusual
    tail of normal behavior is this, relative to what normal actually looks
    like" rather than an arbitrarily-centered sigmoid.
    """
    if calibration is None:
        calibration = load_anomaly_calibration()
    anomaly = -raw_score  # higher = more anomalous
    p50, p99 = calibration["p50"], calibration["p99"]
    span = max(p99 - p50, 1e-6)
    return float(np.clip((anomaly - p50) / span, 0.0, 1.0))


def score_transaction(features: dict, xgb_model, iso_model, feature_order) -> RiskResult:
    X = np.array([[features[f] for f in feature_order]])

    ml_supervised = float(xgb_model.predict_proba(X)[0, 1])
    raw_anomaly = float(iso_model.score_samples(X)[0])
    ml_anomaly = normalize_anomaly_score(raw_anomaly)
    rule_score, rule_reasons = rule_heuristic_score(features)

    final = (
        WEIGHTS["ml_supervised"] * ml_supervised
        + WEIGHTS["ml_anomaly"] * ml_anomaly
        + WEIGHTS["rule_heuristic"] * rule_score
    )
    final = round(min(final, 1.0), 4)

    if final >= DECISION_THRESHOLDS["BLOCK"]:
        decision = "BLOCK"
    elif final >= DECISION_THRESHOLDS["FLAG"]:
        decision = "FLAG"
    else:
        decision = "ALLOW"

    reasons = list(rule_reasons)
    if ml_supervised > 0.7:
        reasons.append(f"ML model strongly matches a known fraud pattern (p={ml_supervised:.2f})")
    if ml_anomaly > 0.7:
        reasons.append(f"Transaction is statistically anomalous vs. population (score={ml_anomaly:.2f})")
    if not reasons:
        reasons.append("No significant risk signals detected")

    return RiskResult(
        fraud_risk_score=final,
        decision=decision,
        ml_supervised_score=round(ml_supervised, 4),
        ml_anomaly_score=round(ml_anomaly, 4),
        rule_heuristic_score=round(rule_score, 4),
        reasons=reasons,
    )
