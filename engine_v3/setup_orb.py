"""
Setup 1 — ORB Breakout.

Entry window : 10:00 AM – 12:00 PM IST.
Signal candle: 5-min close above ORB_HIGH (CALL) or below ORB_LOW (PUT)
               with VWAP alignment + volume proxy + RSI filter.
Entry        : open of the NEXT 5-min candle (never chase the signal candle).

SL  CALL: ORB_HIGH - ORB_RANGE × 0.30
SL  PUT:  ORB_LOW  + ORB_RANGE × 0.30
TP  CALL: ORB_HIGH + ORB_RANGE × 1.50
TP  PUT:  ORB_LOW  - ORB_RANGE × 1.50

Structural R:R ≈ 5:1  (0.3 risk vs 1.5 reward)

ORB range guard:
  Skip if range < 0.25% of spot  (choppy open)
  Skip if range > 1.5%  of spot  (gap/volatile open)
  Skip if absolute range is too tight for the index.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import pandas as pd

from engine_v3.levels import KeyLevels, is_volume_ok

_ENTRY_START    = time(10, 0)
_ENTRY_END      = time(12, 0)
_SL_FRAC        = 0.30
_TP_FRAC        = 1.50
_RSI_CALL_MIN   = 55.0
_RSI_PUT_MAX    = 45.0
_VOL_MULT       = 1.5
_MIN_ORB_PCT    = 0.0025
_MAX_ORB_PCT    = 0.015
_MIN_ORB_RANGE  = {"NIFTY": 80.0, "BANKNIFTY": 200.0}
_MIN_BREAKOUT_PCT = 15.0


@dataclass
class ORBSignal:
    direction:   str
    signal_time: time
    signal_spot: float   # close of breakout candle
    sl_spot:     float
    tp_spot:     float
    orb_high:    float
    orb_low:     float
    orb_range:   float
    rsi:         float


def check_orb(
    candle:       pd.Series,
    prev_candles: pd.DataFrame,
    levels:       KeyLevels,
    vwap:         float,
    rsi:          float,
    index:        str = "NIFTY",
) -> Optional[ORBSignal]:
    """
    Evaluate one 5-min candle for an ORB breakout signal.
    Returns ORBSignal on first match; None otherwise.
    """
    if not levels.orb_ready:
        return None

    t5 = pd.Timestamp(candle["timestamp"]).time()
    if t5 < _ENTRY_START or t5 >= _ENTRY_END:
        return None

    mid   = (levels.orb_high + levels.orb_low) / 2.0
    orb_r = levels.orb_range
    if mid <= 0 or orb_r / mid < _MIN_ORB_PCT or orb_r / mid > _MAX_ORB_PCT:
        return None
    if orb_r <= _MIN_ORB_RANGE.get(index.upper(), _MIN_ORB_RANGE["NIFTY"]):
        return None

    spot = float(candle["close"])

    # CALL: close above ORB_HIGH
    call_breakout_pct = ((spot - levels.orb_high) / orb_r * 100.0) if orb_r > 0 else 0.0
    if (spot > levels.orb_high
            and call_breakout_pct > _MIN_BREAKOUT_PCT
            and spot > vwap
            and rsi > _RSI_CALL_MIN
            and is_volume_ok(candle, prev_candles, _VOL_MULT)):
        return ORBSignal(
            direction="CALL", signal_time=t5, signal_spot=spot,
            sl_spot  = levels.orb_high - orb_r * _SL_FRAC,
            tp_spot  = levels.orb_high + orb_r * _TP_FRAC,
            orb_high = levels.orb_high, orb_low = levels.orb_low,
            orb_range= orb_r, rsi=rsi,
        )

    # PUT: close below ORB_LOW
    put_breakout_pct = ((levels.orb_low - spot) / orb_r * 100.0) if orb_r > 0 else 0.0
    if (spot < levels.orb_low
            and put_breakout_pct > _MIN_BREAKOUT_PCT
            and spot < vwap
            and rsi < _RSI_PUT_MAX
            and is_volume_ok(candle, prev_candles, _VOL_MULT)):
        return ORBSignal(
            direction="PUT", signal_time=t5, signal_spot=spot,
            sl_spot  = levels.orb_low  + orb_r * _SL_FRAC,
            tp_spot  = levels.orb_low  - orb_r * _TP_FRAC,
            orb_high = levels.orb_high, orb_low = levels.orb_low,
            orb_range= orb_r, rsi=rsi,
        )

    return None
