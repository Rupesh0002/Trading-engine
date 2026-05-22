"""
ML filter — XGBoost NO_MOVE classifier.

Single job: predict whether an incoming signal will exit via NO_MOVE.
If P(NO_MOVE) > 0.55, skip the trade.

Phase 1  (0–39 real trades)  : inactive — log all features + outcomes
Phase 2  (40+ real trades)   : XGBoost binary classifier, retrain every 15 trades
Phase 3  (100+ real trades)  : LSTM on last 8 candles added (0.6 × XGB + 0.4 × LSTM)

Features (12):
  adx_value, adx_slope, rsi_value, rsi_slope,
  body_pct, green_count, ema_spread_pct, vwap_distance_pct,
  hour_of_day, fcr_pct, near_round_number, vix_level
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_PATH = "ml/models/no_move_filter.pkl"
_LSTM_PATH  = "ml/models/no_move_lstm.h5"
_NO_MOVE_THRESHOLD = 0.55   # skip if P(NO_MOVE) > this
_PHASE2_TRADES     = 40
_PHASE3_TRADES     = 100
_RETRAIN_INTERVAL  = 15

FEATURE_COLS = [
    "adx_value", "adx_slope", "rsi_value", "rsi_slope",
    "body_pct", "green_count", "ema_spread_pct", "vwap_distance_pct",
    "hour_of_day", "fcr_pct", "near_round_number", "vix_level",
]


class MLFilter:
    """
    Adaptive NO_MOVE filter.

    Usage in backtest / live:
      ml = MLFilter()
      take, conf = ml.should_take_trade(features, candle_seq)
      ...
      ml.log_outcome(features, exit_reason)
    """

    def __init__(self, model_path: str = _MODEL_PATH) -> None:
        self.model_path    = model_path
        self.model         = None
        self.lstm_model    = None
        self.trade_count   = 0
        self.training_data: List[Dict] = []
        self._load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def should_take_trade(
        self,
        features: Dict[str, float],
        candle_sequence: Optional[List[List[float]]] = None,
    ) -> Tuple[bool, float]:
        """
        Returns (take_trade, confidence).
        confidence = 1 - P(NO_MOVE), so higher = more confident this is a real trade.
        Phase 1: always True, 1.0.
        """
        if self.trade_count < _PHASE2_TRADES or self.model is None:
            return True, 1.0

        prob_no_move = self._xgb_predict(features)

        if (
            self.trade_count >= _PHASE3_TRADES
            and self.lstm_model is not None
            and candle_sequence is not None
        ):
            lstm_prob    = self._lstm_predict(candle_sequence)
            prob_no_move = 0.6 * prob_no_move + 0.4 * lstm_prob

        take_trade = prob_no_move < _NO_MOVE_THRESHOLD
        confidence = 1.0 - prob_no_move
        return take_trade, round(confidence, 3)

    def log_outcome(self, features: Dict[str, float], exit_reason: str) -> None:
        """Call after every completed trade to accumulate training data."""
        label = 1 if exit_reason == "NO_MOVE" else 0
        self.training_data.append({**features, "label": label})
        self.trade_count += 1

        if (
            self.trade_count >= _PHASE2_TRADES
            and self.trade_count % _RETRAIN_INTERVAL == 0
        ):
            self.retrain()

    def retrain(self) -> None:
        """Train / retrain XGBoost on all accumulated trade data."""
        if len(self.training_data) < _PHASE2_TRADES:
            return
        try:
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost / sklearn not installed — ML filter inactive")
            return

        df = pd.DataFrame(self.training_data)
        X  = df[FEATURE_COLS].fillna(0)
        y  = df["label"]

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=2,          # shallow — prevents overfitting on small datasets
            learning_rate=0.10,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        self.model.fit(X, y)

        try:
            cv_scores = cross_val_score(
                self.model, X, y, cv=min(5, len(y) // 5), scoring="roc_auc"
            )
            logger.info(
                "ML retrained | AUC %.3f ±%.3f | n=%d (NO_MOVE=%d)",
                cv_scores.mean(), cv_scores.std(), len(y), n_pos,
            )
            print(
                f"[ML] Retrained | AUC: {cv_scores.mean():.3f} "
                f"±{cv_scores.std():.3f} | "
                f"Samples: {len(df)} (NO_MOVE: {n_pos})"
            )
        except Exception:
            pass

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        imp = dict(zip(FEATURE_COLS, self.model.feature_importances_))
        return (
            pd.DataFrame.from_dict(imp, orient="index", columns=["importance"])
            .sort_values("importance", ascending=False)
        )

    def phase(self) -> int:
        if self.trade_count >= _PHASE3_TRADES and self.lstm_model is not None:
            return 3
        if self.trade_count >= _PHASE2_TRADES and self.model is not None:
            return 2
        return 1

    # ── Internal ──────────────────────────────────────────────────────────────

    def _xgb_predict(self, features: Dict[str, float]) -> float:
        row = pd.DataFrame([[features.get(c, 0.0) for c in FEATURE_COLS]], columns=FEATURE_COLS)
        return float(self.model.predict_proba(row)[0][1])

    def _lstm_predict(self, candle_sequence: List[List[float]]) -> float:
        import numpy as _np
        seq = _np.array(candle_sequence, dtype=np.float32).reshape(1, -1, len(candle_sequence[0]))
        return float(self.lstm_model.predict(seq, verbose=0)[0][0])

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info("ML filter loaded: %s", self.model_path)
            except Exception as exc:
                logger.warning("ML model load failed: %s", exc)


# ── Feature extraction helper ─────────────────────────────────────────────────

def extract_features(
    df: pd.DataFrame,
    candle_idx: int,
    adx_val: float,
    rsi_val: float,
    ema9: float,
    ema21: float,
    ema50: float,
    vix: float,
    gap_pct: float,
    fcr_pct: float,
    index: str = "BANKNIFTY",
) -> Dict[str, float]:
    """
    Extract all 12 ML features from the signal candle.
    df must include the pre-warmed rows so indicator slopes are meaningful.
    candle_idx is absolute position in df.
    """
    n   = len(df)
    idx = candle_idx if candle_idx >= 0 else n + candle_idx

    row  = df.iloc[idx]
    prev = df.iloc[max(0, idx - 1)]

    # ADX slope (normalised)
    from signals.indicators import compute_adx, compute_rsi_wilder, compute_vwap
    adx_s    = compute_adx(df)
    adx_prev = float(adx_s.iloc[max(0, idx - 1)])
    adx_slope = (adx_val - adx_prev) / max(adx_prev, 1.0)

    # RSI slope
    rsi_s     = compute_rsi_wilder(df["close"], 14)
    rsi_prev  = float(rsi_s.iloc[max(0, idx - 1)])
    rsi_slope = rsi_val - rsi_prev

    # Body pct (candle body / full range)
    high = float(row["high"])
    low  = float(row["low"])
    candle_range = high - low
    body  = abs(float(row["close"]) - float(row["open"]))
    body_pct = body / candle_range if candle_range > 0 else 0.5

    # Consecutive green candles before signal
    green_count = 0
    for k in range(idx - 1, max(idx - 6, -1), -1):
        r = df.iloc[k]
        if float(r["close"]) > float(r["open"]):
            green_count += 1
        else:
            break

    # EMA spread (how far apart EMA21 and EMA50 are, relative to price)
    close = float(row["close"])
    ema_spread_pct = abs(ema21 - ema50) / close * 100

    # VWAP distance
    try:
        vwap_s = compute_vwap(df)
        vwap_v = float(vwap_s.iloc[idx])
        vwap_distance_pct = (close - vwap_v) / close * 100
    except Exception:
        vwap_distance_pct = 0.0

    # Hour of day
    ts         = df["timestamp"].iloc[idx]
    hour_frac  = ts.hour + ts.minute / 60.0

    # Round number check
    step  = 1000 if index == "BANKNIFTY" else 500
    nearest_round = round(close / step) * step
    near_round = float(abs(close - nearest_round) / close < 0.002)

    return {
        "adx_value":          round(adx_val, 2),
        "adx_slope":          round(adx_slope, 4),
        "rsi_value":          round(rsi_val, 2),
        "rsi_slope":          round(rsi_slope, 2),
        "body_pct":           round(body_pct, 3),
        "green_count":        float(green_count),
        "ema_spread_pct":     round(ema_spread_pct, 3),
        "vwap_distance_pct":  round(vwap_distance_pct, 3),
        "hour_of_day":        round(hour_frac, 2),
        "fcr_pct":            round(fcr_pct, 4),
        "near_round_number":  near_round,
        "vix_level":          round(vix, 1),
    }


def get_candle_sequence(df: pd.DataFrame, candle_idx: int, lookback: int = 8) -> List[List[float]]:
    """
    Returns last `lookback` 5-min candles as [[o, h, l, c, range_ratio], ...] for LSTM.
    Normalised relative to entry candle close.
    """
    n   = len(df)
    idx = candle_idx if candle_idx >= 0 else n + candle_idx
    start = max(0, idx - lookback + 1)
    rows  = df.iloc[start : idx + 1]

    base = float(df["close"].iloc[idx]) or 1.0
    seq  = []
    for _, r in rows.iterrows():
        o = float(r["open"]) / base
        h = float(r["high"]) / base
        l = float(r["low"])  / base
        c = float(r["close"]) / base
        rng = (h - l)
        seq.append([o, h, l, c, rng])

    # Pad with first row if sequence shorter than lookback
    while len(seq) < lookback:
        seq.insert(0, seq[0] if seq else [1.0, 1.0, 1.0, 1.0, 0.0])
    return seq[-lookback:]
