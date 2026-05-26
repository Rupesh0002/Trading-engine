"""
Setup 4 — SMC (Smart Money Concepts) Supply/Demand Zone Sweep.

Supply zone : max high of last 20 five-min candles
Demand zone : min low  of last 20 five-min candles

CALL signal (demand sweep — stop-hunt at lows):
  Current low pierces BELOW demand zone AND close recovers ABOVE it.
  Lower wick (candle low → close) ≥ 65% of total candle range.
  Volume proxy passes.

PUT signal (supply sweep — stop-hunt at highs):
  Current high pierces ABOVE supply zone AND close falls BACK BELOW it.
  Volume proxy passes (wide-range candle confirms reaction).

Entry : SIGNAL CANDLE CLOSE  (not next candle — sweep already confirmed)
SL    : wick extreme + index-aware noise buffer
TP    : entry ± 2 × SL_distance  (2:1 R:R in spot)

Lookback: 20 candles (100 min) for zone definition.
Time    : 10:00 – 14:00 IST.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from engine_v3.levels import is_volume_ok

_ENTRY_START  = time(10, 0)
_ENTRY_END    = time(14, 0)
_ZONE_LOOKBACK = 20
_WICK_RATIO    = 0.65   # lower wick ≥ 65% of candle range for CALL
_VOL_MULT      = 1.5

_SMC_BUFFER_RULES = {
    "NIFTY": (10.0, 0.0004),
    "BANKNIFTY": (30.0, 0.0006),
}


@dataclass
class SMCSignal:
    direction:   str    # CALL | PUT
    signal_time: time
    signal_spot: float  # CLOSE of sweep candle (entry price)
    sl_spot:     float  # wick extreme + buffer
    tp_spot:     float  # 2× SL distance
    zone_level:  float  # supply or demand zone price
    wick_pct:    float  # pierce as % of candle range


def check_smc(
    candle:       pd.Series,
    prev_candles: pd.DataFrame,
    index:        str = "NIFTY",
) -> Optional[SMCSignal]:
    """
    Evaluate one 5-min candle for a supply/demand zone sweep.
    prev_candles: all today's 5-min candles before this one.
    Entry at signal candle close (immediate — no pending entry).
    """
    t5 = pd.Timestamp(candle["timestamp"]).time()
    if t5 < _ENTRY_START or t5 >= _ENTRY_END:
        return None
    if len(prev_candles) < _ZONE_LOOKBACK:
        return None

    h5   = float(candle["high"])
    l5   = float(candle["low"])
    c5   = float(candle["close"])
    rng  = h5 - l5
    if rng <= 0:
        return None

    recent = prev_candles.tail(_ZONE_LOOKBACK)
    supply_high = float(recent["high"].max())
    demand_low  = float(recent["low"].min())
    min_buffer, pct_buffer = _SMC_BUFFER_RULES.get(index.upper(), _SMC_BUFFER_RULES["NIFTY"])
    buffer_pts = max(min_buffer, c5 * pct_buffer)

    # ── CALL: wick below demand zone, close recovers above ──────────
    if l5 < demand_low and c5 > demand_low:
        pierce      = demand_low - l5
        lower_wick  = c5 - l5
        if lower_wick / rng >= _WICK_RATIO and is_volume_ok(candle, prev_candles, _VOL_MULT):
            sl    = l5 - buffer_pts
            risk  = c5 - sl
            tp    = c5 + risk * 2.0
            return SMCSignal(
                direction="CALL", signal_time=t5, signal_spot=c5,
                sl_spot=round(sl, 2), tp_spot=round(tp, 2),
                zone_level=round(demand_low, 2),
                wick_pct=round(pierce / rng * 100, 1),
            )

    # ── PUT: wick above supply zone, close falls back below ─────────
    if h5 > supply_high and c5 < supply_high:
        pierce = h5 - supply_high
        if is_volume_ok(candle, prev_candles, _VOL_MULT):
            sl   = h5 + buffer_pts
            risk = sl - c5
            tp   = c5 - risk * 2.0
            return SMCSignal(
                direction="PUT", signal_time=t5, signal_spot=c5,
                sl_spot=round(sl, 2), tp_spot=round(tp, 2),
                zone_level=round(supply_high, 2),
                wick_pct=round(pierce / rng * 100, 1),
            )

    return None
