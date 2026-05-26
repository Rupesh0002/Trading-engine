"""
Setup 2 — PDH / PDL Retest.

Entry window : 10:00 AM – 2:00 PM IST.

CALL signal (bounce off PDL):
  Price comes within 0.15% of PDL.
  Previous 2 candles are RED (descending toward level).
  Current candle closes GREEN (rejection / bounce).
  Close > VWAP (bullish bias).
  Volume proxy passes.

PUT signal (rejection off PDH):
  Price comes within 0.15% of PDH.
  Previous 2 candles are GREEN (rallying toward level).
  Current candle closes RED (rejection).
  Close < VWAP (bearish bias).
  Volume proxy passes.

Entry : open of NEXT candle.
SL    : 0.10 × (PDH-PDL) beyond the level.
TP    : VWAP (initial hard TP; exit_manager trails after).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from engine_v3.levels import KeyLevels, is_volume_ok

_ENTRY_START = time(10, 0)
_ENTRY_END   = time(14, 0)
_PROX_PCT    = 0.0015   # 0.15%
_VOL_MULT    = 1.3
_SL_FRAC     = 0.10


@dataclass
class PDHSignal:
    direction:   str    # CALL | PUT
    signal_time: time
    signal_spot: float
    level:       float  # PDH or PDL value
    level_kind:  str    # "PDH" | "PDL"
    sl_spot:     float
    tp_spot:     float  # VWAP at signal time


def check_pdh_pdl(
    candle:       pd.Series,
    prev_candles: pd.DataFrame,
    levels:       KeyLevels,
    vwap:         float,
) -> Optional[PDHSignal]:
    """Evaluate one 5-min candle for PDH/PDL retest rejection."""
    t5 = pd.Timestamp(candle["timestamp"]).time()
    if t5 < _ENTRY_START or t5 >= _ENTRY_END:
        return None
    if len(prev_candles) < 2:
        return None

    h5  = float(candle["high"])
    l5  = float(candle["low"])
    c5  = float(candle["close"])
    o5  = float(candle["open"])
    day_range = max(levels.pdh - levels.pdl, 1.0)

    last2 = prev_candles.tail(2)
    p_cls = last2["close"].astype(float).values
    p_opn = last2["open"].astype(float).values
    last2_red   = [p_cls[i] < p_opn[i] for i in range(2)]
    last2_green = [p_cls[i] > p_opn[i] for i in range(2)]

    # ── CALL: bounce off PDL ──────────────────────────────────────────
    if (abs(l5 - levels.pdl) / levels.pdl <= _PROX_PCT
            and all(last2_red)
            and c5 > o5
            and c5 > vwap
            and is_volume_ok(candle, prev_candles, _VOL_MULT)):
        sl = levels.pdl - day_range * _SL_FRAC
        return PDHSignal(
            direction="CALL", signal_time=t5, signal_spot=c5,
            level=levels.pdl, level_kind="PDL",
            sl_spot=round(sl, 2), tp_spot=round(vwap, 2),
        )

    # ── PUT: rejection off PDH ────────────────────────────────────────
    if (abs(h5 - levels.pdh) / levels.pdh <= _PROX_PCT
            and all(last2_green)
            and c5 < o5
            and c5 < vwap
            and is_volume_ok(candle, prev_candles, _VOL_MULT)):
        sl = levels.pdh + day_range * _SL_FRAC
        return PDHSignal(
            direction="PUT", signal_time=t5, signal_spot=c5,
            level=levels.pdh, level_kind="PDH",
            sl_spot=round(sl, 2), tp_spot=round(vwap, 2),
        )

    return None
