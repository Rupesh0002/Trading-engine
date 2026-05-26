"""
Setup 3 — VWAP Reclaim / Rejection.

Entry window : 10:30 AM – 1:30 PM IST.

CALL signal (VWAP reclaim from below):
  Last 3 candles all closed BELOW VWAP.
  Current candle closes ABOVE VWAP  (crosses from below).
  Close > EMA21 on 5-min chart.
  5-min RSI crosses above 50 (prev RSI ≤ 50, current > 50).
  Volume proxy passes.

PUT signal (VWAP rejection from above):
  Last 3 candles all closed ABOVE VWAP.
  Current candle closes BELOW VWAP.
  Close < EMA21.
  RSI crosses below 50.
  Volume proxy passes.

Entry : open of NEXT candle.
SL    : low of the crossing candle (CALL) / high (PUT).
TP    : 0 — no fixed TP; exit_manager trails.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import pandas as pd

from engine_v3.levels import is_volume_ok, compute_rsi, compute_ema

_ENTRY_START    = time(10, 30)
_ENTRY_END      = time(13, 30)
_CONSEC_N       = 3     # candles needed on same side before cross
_EMA_SPAN       = 21
_VOL_MULT       = 1.4


@dataclass
class VWAPSignal:
    direction:   str    # CALL | PUT
    signal_time: time
    signal_spot: float
    sl_spot:     float  # low (CALL) or high (PUT) of crossing candle
    tp_spot:     float  # 0.0 = trail only
    rsi:         float
    ema21:       float


def check_vwap(
    candle:       pd.Series,
    prev_candles: pd.DataFrame,
    vwap:         float,
    prev_vwap:    float,
) -> Optional[VWAPSignal]:
    """
    Evaluate one 5-min candle for a VWAP cross.
    prev_candles: all today's candles BEFORE this one.
    vwap:      current VWAP (after this candle).
    prev_vwap: VWAP after the previous candle.
    """
    t5 = pd.Timestamp(candle["timestamp"]).time()
    if t5 < _ENTRY_START or t5 >= _ENTRY_END:
        return None
    if len(prev_candles) < _CONSEC_N + 2:
        return None

    c5 = float(candle["close"])
    o5 = float(candle["open"])
    h5 = float(candle["high"])
    l5 = float(candle["low"])

    prev_closes: List[float] = prev_candles["close"].astype(float).tolist()

    # RSI on recent closes (use enough history for period=14)
    rsi_series = prev_closes[-20:] if len(prev_closes) >= 20 else prev_closes
    rsi_now    = compute_rsi(rsi_series + [c5])
    rsi_prev   = compute_rsi(rsi_series)

    # EMA21 from prior closes
    ema_series = prev_closes[-40:] if len(prev_closes) >= 40 else prev_closes
    ema21      = compute_ema(ema_series, _EMA_SPAN)

    # Last N prev candle closes relative to prev_vwap
    last_n = prev_candles["close"].astype(float).values[-_CONSEC_N:]

    # ── CALL: reclaim VWAP from below ────────────────────────────────
    if (c5 > vwap                                   # current above VWAP
            and o5 < prev_vwap                      # opened below
            and all(v < prev_vwap for v in last_n)  # last N were below
            and c5 > ema21
            and rsi_now > 50 and rsi_prev <= 50
            and is_volume_ok(candle, prev_candles, _VOL_MULT)):
        return VWAPSignal(
            direction="CALL", signal_time=t5, signal_spot=c5,
            sl_spot=round(l5, 2), tp_spot=0.0,
            rsi=rsi_now, ema21=round(ema21, 2),
        )

    # ── PUT: rejection below VWAP ────────────────────────────────────
    if (c5 < vwap
            and o5 > prev_vwap
            and all(v > prev_vwap for v in last_n)
            and c5 < ema21
            and rsi_now < 50 and rsi_prev >= 50
            and is_volume_ok(candle, prev_candles, _VOL_MULT)):
        return VWAPSignal(
            direction="PUT", signal_time=t5, signal_spot=c5,
            sl_spot=round(h5, 2), tp_spot=0.0,
            rsi=rsi_now, ema21=round(ema21, 2),
        )

    return None
