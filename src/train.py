"""
Training pipeline.

Two model families are trained, on purpose:

1. XGBoost (supervised) — learns the *known* fraud typologies from labels.
   Strength: high precision on patterns seen before (card testing, mule fan-out, etc).
   Weakness: blind to fraud patterns that don't resemble anything in the training labels.

2. Isolation Forest (unsupervised) — trained ONLY on legitimate transactions.
   Strength: flags anything that looks statistically unusual for a user/population,
   even if it doesn't match any known fraud label. This is the "zero-day fraud" net.
   Weakness: noisier, more false positives if used alone.

The hybrid risk engine (src/risk_engine.py) combines both, which is the actual point of
this project -- a single model, however good, is not what production fraud systems run.

Both models consume EXACTLY the same feature vector (FEATURE_ORDER in features.py), computed
by replaying the historical CSV through the same UserStateStore class used online. This
guarantees training/serving parity.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, classification_report
)
import xgboost as xgb

from features import UserStateStore, FEATURE_ORDER


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Replay transactions in timestamp order through the SAME stateful feature
    engineering used online, so training features are byte-for-byte what serving sees."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    store = UserStateStore()
    rows = []
    for row in df.itertuples(index=False):
        feats = store.compute_features(
            user_id=row.user_id, amount=row.amount, ts=row.timestamp,
            channel=row.channel, country=row.country, device_id=row.device_id,
        )
        feats["transaction_id"] = row.transaction_id
        feats["is_fraud"] = row.is_fraud
        feats["fraud_type"] = row.fraud_type
        rows.append(feats)
    return pd.DataFrame(rows)


def train_supervised(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        reg_lambda=1.0, n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_unsupervised(X_train_legit):
    model = IsolationForest(
        n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1,
    )
    model.fit(X_train_legit)
    return model


def evaluate(model, X_val, y_val, name):
    proba = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, proba)
    auc = roc_auc_score(y_val, proba)
    preds = (proba >= 0.5).astype(int)
    print(f"\n=== {name} ===")
    print(f"PR-AUC: {ap:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_val, preds, digits=3))
    return {"pr_auc": ap, "roc_auc": auc}


def main(args):
    df = pd.read_csv(args.transactions)
    print(f"Loaded {len(df):,} raw transactions")

    feat_df = build_feature_table(df)
    feat_df.to_parquet(args.features_out, index=False)
    print(f"Feature table -> {args.features_out} ({feat_df.shape})")

    X = feat_df[FEATURE_ORDER]
    y = feat_df["is_fraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    xgb_model = train_supervised(X_train, y_train, X_val, y_val)
    metrics = evaluate(xgb_model, X_val, y_val, "XGBoost (supervised)")

    # Isolation Forest trained ONLY on legit training transactions
    X_train_legit = X_train[y_train == 0]
    iso_model = train_unsupervised(X_train_legit)

    # quick sanity check: anomaly scores should separate fraud from legit on val set
    anomaly_scores_val = -iso_model.score_samples(X_val)  # higher = more anomalous
    iso_ap = average_precision_score(y_val, anomaly_scores_val)
    print(f"\n=== Isolation Forest (unsupervised) ===\nPR-AUC (as anomaly ranker): {iso_ap:.4f}")

    # Calibrate the raw anomaly score against the ACTUAL legit-data distribution
    # (see src/risk_engine.py::normalize_anomaly_score for why this matters: an
    # assumed fixed center produced a badly miscalibrated score that saturated
    # near 1.0 for almost every transaction, legit or not).
    anomaly_scores_legit_train = -iso_model.score_samples(X_train_legit)
    calibration = {
        "p50": float(np.percentile(anomaly_scores_legit_train, 50)),
        "p99": float(np.percentile(anomaly_scores_legit_train, 99)),
    }
    with open(args.model_dir + "/anomaly_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Anomaly score calibration (from legit training data): {calibration}")

    joblib.dump(xgb_model, args.model_dir + "/xgb_model.joblib")
    joblib.dump(iso_model, args.model_dir + "/iso_forest.joblib")
    with open(args.model_dir + "/feature_order.json", "w") as f:
        json.dump(FEATURE_ORDER, f)
    with open(args.model_dir + "/metrics.json", "w") as f:
        json.dump({"xgboost": metrics, "isolation_forest_pr_auc": iso_ap}, f, indent=2)

    print(f"\nModels saved to {args.model_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="data/transactions.csv")
    parser.add_argument("--features-out", default="data/feature_table.parquet")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    main(args)
