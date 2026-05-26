"""
Key levels for engine_v3 — computed at market open, updated live.

PDH / PDL / PDC  : prior-day OHLC (from 15-min history)
ORB_HIGH/LOW     : high/low of first 30 min (9:15–9:44 inclusive, 6 five-min candles)
VWAP             : range-weighted (uses H-L as weight; avoids zero-volume index data)
Round numbers    : NIFTY=100pt steps, BANKNIFTY=200pt steps, SENSEX=500pt steps
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import List, Tuple

import pandas as pd

_ROUND_STEP   = {"NIFTY": 100, "BANKNIFTY": 200, "SENSEX": 500}
_ORB_END_TIME = time(9, 45)   # ORB includes candles with timestamp < 9:45


@dataclass
class KeyLevels:
    pdh:         float
    pdl:         float
    pdc:         float
    round_above: float
    round_below: float
    orb_high:    float = 0.0
    orb_low:     float = 0.0
    orb_range:   float = 0.0
    orb_ready:   bool  = False
    vwap:        float = 0.0


def build_prior_levels(
    prior_15:   pd.DataFrame,
    today_open: float,
    index:      str = "NIFTY",
) -> KeyLevels:
    """PDH/PDL/PDC from prior 15-min candles; round numbers around today_open."""
    if prior_15 is None or prior_15.empty:
        pdh, pdl, pdc = today_open * 1.005, today_open * 0.995, today_open
    else:
        ts        = pd.to_datetime(prior_15["timestamp"])
        prev_date = ts.dt.date.max()
        prev_df   = prior_15[ts.dt.date == prev_date]
        pdh = float(prev_df["high"].max())
        pdl = float(prev_df["low"].min())
        pdc = float(prev_df["close"].iloc[-1])

    step = _ROUND_STEP.get(index, 100)
    ra   = math.ceil(today_open / step) * step
    rb   = math.floor(today_open / step) * step
    if ra == today_open: ra += step
    if rb == today_open: rb -= step

    return KeyLevels(
        pdh=round(pdh, 2), pdl=round(pdl, 2), pdc=round(pdc, 2),
        round_above=float(ra), round_below=float(rb),
    )


def compute_orb(today_5: pd.DataFrame) -> Tuple[float, float]:
    """Extract ORB_HIGH / ORB_LOW from the first 30-min candles (9:15–9:44)."""
    mask = pd.to_datetime(today_5["timestamp"]).dt.time < _ORB_END_TIME
    orb  = today_5[mask]
    if orb.empty:
        return 0.0, 0.0
    return float(orb["high"].max()), float(orb["low"].min())


def update_vwap(
    prev_num: float,
    prev_den: float,
    candle:   pd.Series,
) -> Tuple[float, float, float]:
    """
    Incremental VWAP update.
    Weight = candle range (H-L); typical price = (H+L+C)/3.
    Returns (vwap_value, new_num, new_den).
    """
    h   = float(candle["high"])
    l   = float(candle["low"])
    c   = float(candle["close"])
    rng = h - l
    if rng <= 0:
        denom = prev_den if prev_den > 0 else 1.0
        return prev_num / denom, prev_num, prev_den
    typ     = (h + l + c) / 3.0
    new_num = prev_num + typ * rng
    new_den = prev_den + rng
    return new_num / new_den, new_num, new_den


def is_volume_ok(
    candle:       pd.Series,
    recent_5:     pd.DataFrame,
    multiplier:   float = 1.5,
) -> bool:
    """
    Volume proxy (index volume = 0 on Kite, so we use candle geometry):
      Condition 1: |close - open| / close > 0.1%  (meaningful body)
      Condition 2: (H-L) > multiplier × avg(H-L) of last 5 candles
    Both must be true.
    """
    c   = float(candle["close"])
    o   = float(candle["open"])
    h   = float(candle["high"])
    l   = float(candle["low"])
    rng = h - l

    body_ok = abs(c - o) / c > 0.001

    tail = recent_5.tail(5)
    if len(tail) >= 3:
        avg_rng  = float((tail["high"].astype(float) - tail["low"].astype(float)).mean())
        range_ok = (rng > avg_rng * multiplier) if avg_rng > 0 else False
    else:
        range_ok = body_ok

    return body_ok and range_ok


def compute_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains  = [max(d, 0.0) for d in recent]
    losses = [-min(d, 0.0) for d in recent]
    ag     = sum(gains) / period
    al     = sum(losses) / period
    if al == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + ag / al), 1)


def compute_ema(closes: List[float], span: int) -> float:
    if not closes:
        return 0.0
    alpha = 2.0 / (span + 1)
    ema   = closes[0]
    for v in closes[1:]:
        ema = v * alpha + ema * (1 - alpha)
    return ema
