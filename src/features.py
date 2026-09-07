"""
Stateful, causal feature engineering.

Design principle (the single most important thing to be able to explain in an interview):
--------------------------------------------------------------------------------------------
Every feature for transaction T is computed using ONLY transactions that happened strictly
BEFORE T for that user. This is what "causal" means here, and it's the difference between
a model that works in a notebook and one that would leak the future in production.

The same UserStateStore class is used both:
  (a) offline, replaying the historical CSV in timestamp order during training, and
  (b) online, in the FastAPI service, updated incrementally per request.

This guarantees training/serving feature parity -- a very common real bug in fraud systems
is a mismatch between how a feature was computed offline vs online. Centralizing the logic
here removes that risk by construction.

State kept per user (all O(1) or O(k) with a small bounded k, so scoring stays fast):
  - rolling amount stats (count, mean, M2 for variance) -> Welford's online algorithm
  - timestamps of the last N transactions -> velocity features (txns in last 10min/1h/24h)
  - last known country / device -> geo & device consistency checks
  - timestamp of previous transaction -> recency feature
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

FEATURE_ORDER = [
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_late_night",
    "time_since_prev_txn_min",
    "txns_last_10min",
    "txns_last_1h",
    "txns_last_24h",
    "user_avg_amount",
    "amount_over_user_avg",
    "amount_zscore",
    "is_new_device",
    "is_new_country",
    "is_channel_switch",
]


@dataclass
class UserState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    last_ts: datetime | None = None
    recent_ts: deque = field(default_factory=lambda: deque(maxlen=200))
    known_devices: set = field(default_factory=set)
    known_countries: set = field(default_factory=set)
    last_channel: str | None = None

    def update_amount(self, amount: float):
        self.n += 1
        delta = amount - self.mean
        self.mean += delta / self.n
        delta2 = amount - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


class UserStateStore:
    """In-memory store of per-user behavioral state. Swap-in-able for Redis in production
    (see README 'Productionizing' section) without changing the feature math at all."""

    def __init__(self):
        self._states: dict[str, UserState] = {}

    def get(self, user_id: str) -> UserState:
        if user_id not in self._states:
            self._states[user_id] = UserState()
        return self._states[user_id]

    def compute_features(self, user_id: str, amount: float, ts: datetime,
                          channel: str, country: str, device_id: str) -> dict:
        """Compute features for an incoming transaction using ONLY prior state,
        then update state with this transaction. Order matters: read-then-write."""
        state = self.get(user_id)

        time_since_prev_min = (
            (ts - state.last_ts).total_seconds() / 60.0 if state.last_ts else 1440.0
        )
        txns_last_10min = sum(1 for t in state.recent_ts if ts - t <= timedelta(minutes=10))
        txns_last_1h = sum(1 for t in state.recent_ts if ts - t <= timedelta(hours=1))
        txns_last_24h = sum(1 for t in state.recent_ts if ts - t <= timedelta(hours=24))

        user_avg_amount = state.mean if state.n > 0 else amount
        amount_over_avg = amount / user_avg_amount if user_avg_amount > 0 else 1.0
        amount_zscore = (amount - state.mean) / state.std if state.std > 0 else 0.0

        is_new_device = 1 if device_id not in state.known_devices and state.n > 0 else 0
        is_new_country = 1 if country not in state.known_countries and state.n > 0 else 0
        is_channel_switch = 1 if state.last_channel and state.last_channel != channel else 0

        features = {
            "amount": amount,
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "is_weekend": int(ts.weekday() >= 5),
            "is_late_night": int(ts.hour <= 5 or ts.hour >= 23),
            "time_since_prev_txn_min": min(time_since_prev_min, 1440.0),
            "txns_last_10min": txns_last_10min,
            "txns_last_1h": txns_last_1h,
            "txns_last_24h": txns_last_24h,
            "user_avg_amount": round(user_avg_amount, 2),
            "amount_over_user_avg": round(amount_over_avg, 3),
            "amount_zscore": round(amount_zscore, 3),
            "is_new_device": is_new_device,
            "is_new_country": is_new_country,
            "is_channel_switch": is_channel_switch,
        }

        # --- update state AFTER reading features (causality) ---
        state.update_amount(amount)
        state.recent_ts.append(ts)
        state.known_devices.add(device_id)
        state.known_countries.add(country)
        state.last_channel = channel
        state.last_ts = ts

        return features

    def snapshot_size(self) -> int:
        return len(self._states)
