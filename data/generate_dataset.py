"""
Synthetic multi-channel transaction data generator.

Why synthetic (and not a Kaggle CSV)?
--------------------------------------
1. Real fraud datasets (Kaggle's ULB creditcard.csv, IEEE-CIS, PaySim) are PCA-anonymized
   or heavily preprocessed -- you can't explain a single feature's meaning to an interviewer
   because the columns are literally "V1...V28". That's a red flag in an interview, not a
   strength.
2. Generating your own data forces you to *design* the fraud typologies, which is exactly
   what a fraud analytics team does before any modeling happens. You can explain every
   column and every fraud pattern because you wrote the generator.
3. It's fully reproducible (seeded) and channel-realistic: card-present, card-not-present,
   UPI/wallet, and wire-transfer transactions each have their own normal behavior profile
   and their own fraud typology, mirroring how real payment processors segment risk.

Fraud typologies injected (each with a distinct behavioral signature):
  - CARD_TESTING       : many tiny transactions in rapid succession on a new/stolen card,
                          used by fraudsters to validate stolen card numbers before a big hit.
  - ACCOUNT_TAKEOVER   : a burst of large transactions from a login location/device that is
                          geographically inconsistent with the user's recent history.
  - MULE_FANOUT        : a single funding event immediately followed by many small transfers
                          out to different, previously-unseen beneficiaries (money-mule pattern).
  - MERCHANT_COLLUSION : repeated, suspiciously round-number transactions with the same
                          merchant just under manual-review thresholds.

Usage:
    python generate_dataset.py --n-users 4000 --n-transactions 250000 --fraud-rate 0.012
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
CHANNELS = ["CARD_PRESENT", "CARD_NOT_PRESENT", "UPI_WALLET", "WIRE_TRANSFER"]
COUNTRIES = ["IN", "US", "GB", "AE", "SG", "NG", "RU", "BR"]
HOME_COUNTRY_WEIGHTS = [0.55, 0.15, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02]


def make_users(n_users, rng):
    users = pd.DataFrame({
        "user_id": [f"U{100000+i}" for i in range(n_users)],
        "home_country": rng.choice(COUNTRIES, size=n_users, p=HOME_COUNTRY_WEIGHTS),
        "avg_txn_amount": np.round(rng.gamma(shape=2.2, scale=45, size=n_users) + 5, 2),
        "preferred_channel": rng.choice(CHANNELS, size=n_users, p=[0.35, 0.30, 0.25, 0.10]),
        "account_age_days": rng.integers(1, 2500, size=n_users),
    })
    return users


def normal_transaction(user, ts, rng):
    amount = max(1.0, rng.normal(loc=user.avg_txn_amount, scale=user.avg_txn_amount * 0.35))
    channel = user.preferred_channel if rng.random() < 0.8 else rng.choice(CHANNELS)
    country = user.home_country if rng.random() < 0.95 else rng.choice(COUNTRIES)
    return {
        "user_id": user.user_id,
        "timestamp": ts,
        "amount": round(amount, 2),
        "channel": channel,
        "country": country,
        "merchant_id": f"M{rng.integers(1000, 9999)}",
        "device_id": f"D{hash(user.user_id) % 5000}",
        "is_fraud": 0,
        "fraud_type": "NONE",
    }


def inject_card_testing(user, ts, rng, txns):
    n = rng.integers(6, 15)
    device = f"D{rng.integers(90000, 99999)}"
    for i in range(n):
        txns.append({
            "user_id": user.user_id,
            "timestamp": ts + timedelta(seconds=int(rng.integers(1, 25)) * i),
            "amount": round(rng.uniform(0.5, 3.0), 2),
            "channel": "CARD_NOT_PRESENT",
            "country": rng.choice(COUNTRIES),
            "merchant_id": f"M{rng.integers(1000, 9999)}",
            "device_id": device,
            "is_fraud": 1,
            "fraud_type": "CARD_TESTING",
        })


def inject_account_takeover(user, ts, rng, txns):
    foreign_country = rng.choice([c for c in COUNTRIES if c != user.home_country])
    device = f"D{rng.integers(90000, 99999)}"
    n = rng.integers(2, 5)
    for i in range(n):
        txns.append({
            "user_id": user.user_id,
            "timestamp": ts + timedelta(minutes=int(rng.integers(1, 20)) * i),
            "amount": round(user.avg_txn_amount * rng.uniform(4, 12), 2),
            "channel": rng.choice(["CARD_NOT_PRESENT", "UPI_WALLET"]),
            "country": foreign_country,
            "merchant_id": f"M{rng.integers(1000, 9999)}",
            "device_id": device,
            "is_fraud": 1,
            "fraud_type": "ACCOUNT_TAKEOVER",
        })


def inject_mule_fanout(user, ts, rng, txns):
    funding_amt = round(rng.uniform(2000, 9000), 2)
    txns.append({
        "user_id": user.user_id, "timestamp": ts, "amount": funding_amt,
        "channel": "WIRE_TRANSFER", "country": user.home_country,
        "merchant_id": "EXTERNAL", "device_id": f"D{hash(user.user_id) % 5000}",
        "is_fraud": 1, "fraud_type": "MULE_FANOUT",
    })
    remaining = funding_amt
    n_out = rng.integers(4, 9)
    for i in range(n_out):
        out_amt = round(remaining / n_out * rng.uniform(0.7, 1.3), 2)
        txns.append({
            "user_id": user.user_id,
            "timestamp": ts + timedelta(minutes=int(rng.integers(2, 10)) * (i + 1)),
            "amount": out_amt, "channel": "UPI_WALLET", "country": user.home_country,
            "merchant_id": f"BENEFICIARY_{rng.integers(10000,99999)}",
            "device_id": f"D{hash(user.user_id) % 5000}",
            "is_fraud": 1, "fraud_type": "MULE_FANOUT",
        })


def inject_merchant_collusion(user, ts, rng, txns):
    merchant = f"M{rng.integers(1000, 9999)}"
    n = rng.integers(3, 6)
    round_amt = rng.choice([499, 999, 1999, 4999])
    for i in range(n):
        txns.append({
            "user_id": user.user_id,
            "timestamp": ts + timedelta(days=int(rng.integers(1, 10)) * (i + 1)),
            "amount": float(round_amt), "channel": "CARD_NOT_PRESENT",
            "country": user.home_country, "merchant_id": merchant,
            "device_id": f"D{hash(user.user_id) % 5000}",
            "is_fraud": 1, "fraud_type": "MERCHANT_COLLUSION",
        })


FRAUD_INJECTORS = [inject_card_testing, inject_account_takeover, inject_mule_fanout, inject_merchant_collusion]


def generate(n_users, n_transactions, fraud_rate, start_date, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    users = make_users(n_users, rng)
    start = datetime.fromisoformat(start_date)
    txns = []

    n_fraud_events = int(n_transactions * fraud_rate / 4)  # ~4 txns per fraud event on avg
    n_normal = n_transactions - n_fraud_events * 4

    for _ in range(n_normal):
        u = users.iloc[rng.integers(0, n_users)]
        ts = start + timedelta(seconds=int(rng.integers(0, 60 * 60 * 24 * 90)))
        txns.append(normal_transaction(u, ts, rng))

    for _ in range(n_fraud_events):
        u = users.iloc[rng.integers(0, n_users)]
        ts = start + timedelta(seconds=int(rng.integers(0, 60 * 60 * 24 * 90)))
        injector = FRAUD_INJECTORS[rng.integers(0, len(FRAUD_INJECTORS))]
        injector(u, ts, rng, txns)

    df = pd.DataFrame(txns).sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"T{i:08d}" for i in range(len(df))]
    df = df[["transaction_id", "user_id", "timestamp", "amount", "channel",
             "country", "merchant_id", "device_id", "is_fraud", "fraud_type"]]
    return df, users


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=4000)
    parser.add_argument("--n-transactions", type=int, default=250000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    parser.add_argument("--users-out", type=str, default="data/users.csv")
    args = parser.parse_args()

    df, users = generate(args.n_users, args.n_transactions, args.fraud_rate, args.start_date)
    df.to_csv(args.out, index=False)
    users.to_csv(args.users_out, index=False)
    print(f"Generated {len(df):,} transactions ({df.is_fraud.sum():,} fraud, "
          f"{df.is_fraud.mean()*100:.2f}%) for {args.n_users:,} users -> {args.out}")
    print(df.fraud_type.value_counts())
