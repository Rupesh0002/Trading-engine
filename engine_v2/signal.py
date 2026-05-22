"""
4-indicator signal engine on 5-min candles.

All 4 must agree for direction to fire:
  1. EMA triple alignment  : close > ema9 > ema21 > ema50  (CALL)
                             close < ema9 < ema21 < ema50  (PUT)
  2. Range ratio           : candle_range > 1.4 × mean(last 10 candle ranges)
                             (proxy for volume — index candles have no volume data)
  3. RSI 14 Wilder         : rsi > 55 AND rising           (CALL)
                             rsi < 45 AND falling          (PUT)
  4. ADX 14                : adx > 20  (requires 3-day pre-warm)

Signal window: 9:30–13:00 IST only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

from signals.indicators import compute_adx, compute_ema, compute_rsi_wilder

# ── Indicator parameters ─────────────────────────────────────────────────────
EMA_FAST         = 9
EMA_MID          = 21
EMA_SLOW         = 50
RANGE_RATIO_MIN  = 1.4   # candle range must be ≥1.4× avg of prior 10 candles
RANGE_LOOKBACK   = 10
RSI_PERIOD_VAL   = 14
RSI_BULL         = 55.0
RSI_BEAR         = 45.0
ADX_MIN          = 20.0
SIGNAL_START     = time(9, 30)
SIGNAL_END       = time(13, 0)


@dataclass
class SignalResult:
    direction:   str    # "CALL", "PUT", or "NONE"
    ema_ok:      bool
    range_ok:    bool   # candle range ratio (volume proxy for indices)
    rsi_ok:      bool
    adx_ok:      bool
    ema9:        float
    ema21:       float
    ema50:       float
    rsi:         float
    adx:         float
    range_ratio: float
    candle_time: time


def compute_signal(df: pd.DataFrame, candle_idx: int = -1) -> SignalResult:
    """
    Evaluate 4-indicator signal at position candle_idx.

    df must contain at least 3 prior trading days of 5-min candles prepended
    (pre-warm) so ADX is correctly seeded. Only call for candles within
    the 9:30–13:00 IST window.

    Returns SignalResult with direction="CALL"/"PUT"/"NONE".
    """
    n = len(df)
    idx = candle_idx if candle_idx >= 0 else n + candle_idx

    # ── EMA triple alignment ─────────────────────────────────────────────────
    ema9_s  = compute_ema(df["close"], EMA_FAST)
    ema21_s = compute_ema(df["close"], EMA_MID)
    ema50_s = compute_ema(df["close"], EMA_SLOW)

    close = float(df["close"].iloc[idx])
    e9    = float(ema9_s.iloc[idx])
    e21   = float(ema21_s.iloc[idx])
    e50   = float(ema50_s.iloc[idx])

    bull_ema = close > e9 > e21 > e50
    bear_ema = close < e9 < e21 < e50
    ema_ok   = bull_ema or bear_ema

    # ── Range ratio (volume proxy for zero-volume index feeds) ───────────────
    curr_range  = float(df["high"].iloc[idx]) - float(df["low"].iloc[idx])
    start       = max(0, idx - RANGE_LOOKBACK)
    prev_ranges = (df["high"].iloc[start:idx] - df["low"].iloc[start:idx])
    avg_range   = float(prev_ranges.mean()) if len(prev_ranges) > 0 else 0.0
    range_ratio = curr_range / avg_range if avg_range > 0 else 0.0
    range_ok    = range_ratio >= RANGE_RATIO_MIN

    # ── RSI 14 with direction ────────────────────────────────────────────────
    rsi_s    = compute_rsi_wilder(df["close"], RSI_PERIOD_VAL)
    rsi_val  = float(rsi_s.iloc[idx])
    rsi_prev = float(rsi_s.iloc[idx - 1]) if idx >= 1 else rsi_val
    rsi_rising  = rsi_val > rsi_prev
    rsi_falling = rsi_val < rsi_prev

    bull_rsi = (rsi_val > RSI_BULL) and rsi_rising
    bear_rsi = (rsi_val < RSI_BEAR) and rsi_falling
    rsi_ok   = bull_rsi or bear_rsi

    # ── ADX 14 (trend strength) ──────────────────────────────────────────────
    adx_s   = compute_adx(df)
    adx_val = float(adx_s.iloc[idx])
    adx_ok  = (not np.isnan(adx_val)) and (adx_val >= ADX_MIN)

    # ── Time window ──────────────────────────────────────────────────────────
    ts     = df["timestamp"].iloc[idx]
    ctime  = ts.time() if hasattr(ts, "time") else time(0, 0)
    in_win = SIGNAL_START <= ctime <= SIGNAL_END

    # ── Direction decision ───────────────────────────────────────────────────
    if in_win and bull_ema and range_ok and bull_rsi and adx_ok:
        direction = "CALL"
    elif in_win and bear_ema and range_ok and bear_rsi and adx_ok:
        direction = "PUT"
    else:
        direction = "NONE"

    return SignalResult(
        direction=direction,
        ema_ok=ema_ok,
        range_ok=range_ok,
        rsi_ok=rsi_ok,
        adx_ok=adx_ok,
        ema9=e9, ema21=e21, ema50=e50,
        rsi=rsi_val, adx=adx_val,
        range_ratio=range_ratio,
        candle_time=ctime,
    )


def check_trend_continuation(df: pd.DataFrame, candle_idx: int = -1) -> bool:
    """
    Used by ExitManager TREND_CHECK at +36% TP decision point.
    Returns True (hold) if all 4 conditions pass, False (exit) otherwise.

    Conditions:
      - EMA9 > EMA21
      - RSI > 55
      - ADX > 20
      - Last 2 candles are green (close > open)
    """
    if len(df) < 3:
        return False

    n   = len(df)
    idx = candle_idx if candle_idx >= 0 else n + candle_idx

    ema9_s  = compute_ema(df["close"], EMA_FAST)
    ema21_s = compute_ema(df["close"], EMA_MID)
    e9  = float(ema9_s.iloc[idx])
    e21 = float(ema21_s.iloc[idx])

    rsi_s  = compute_rsi_wilder(df["close"], RSI_PERIOD_VAL)
    rsi_val = float(rsi_s.iloc[idx])

    adx_s   = compute_adx(df)
    adx_val = float(adx_s.iloc[idx])

    green1 = float(df["close"].iloc[idx])     > float(df["open"].iloc[idx])
    green2 = float(df["close"].iloc[idx - 1]) > float(df["open"].iloc[idx - 1])

    return (
        e9 > e21
        and rsi_val > 55.0
        and (not np.isnan(adx_val)) and adx_val > ADX_MIN
        and green1 and green2
    )
