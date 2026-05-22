"""
Setup Detector — 2 high-conviction intraday setups.

Setup A — Gap & Go (GAP_GO)  ~72% hist. win rate
  Trigger at 9:30 AM:
    gap_pct  = abs(today_open - yesterday_close) / yesterday_close × 100
    gap_pct > 0.35%
    First 15-min candle closes in gap direction
    VIX between 12-22
  Entry: first pullback to 3-period EMA (≈ 9-per EMA on 5-min)
         in window 9:30-10:30 AM (candle indices 1-6)
  SL:  Low of entry candle (CALL) / High (PUT)
  TP:  3× SL distance in premium terms
  Max: 3 fifteen-min candles (45 min)

Setup B — 11 AM Trend Confirmation (TREND_11AM)  ~65% hist. win rate
  Trigger at 10:45 AM (candle index 6):
    ADX > 25 (pre-warmed from previous day)
    VWAP distance > 0.25%
    RSI > 58 (CALL) or < 42 (PUT)
    Previous 3 fifteen-min candles (10:00-10:30) all same direction
  Entry: Open of 11:00 AM candle (index 7)
  SL:  Low of 10:45 candle (CALL) / High (PUT)
  TP:  2.5× SL distance
  Max: 4 fifteen-min candles (60 min)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
GAP_MIN_PCT         = 0.35      # %
GAP_BEST_PCT        = 0.50      # % → max lots
VIX_LOW             = 12.0
VIX_HIGH            = 22.0
VIX_BEST_LOW        = 14.0
VIX_BEST_HIGH       = 18.0
ADX_THRESHOLD       = 25.0
VWAP_DIST_MIN_PCT   = 0.25      # %
RSI_CALL_MIN        = 58.0
RSI_PUT_MAX         = 42.0
GAP_EMA_PERIOD      = 3         # ≈ 9-per 5-min EMA; 3 fifteen-min = 45 min
GAP_ENTRY_WINDOW    = range(1, 7)  # candle indices 1-6 (9:30-10:30 AM)
TREND_CHECK_IDX     = 6         # candle index for 10:45 AM (from 9:15)
TREND_ENTRY_IDX     = 7         # candle index for 11:00 AM
MAX_HOLD_GAP        = 3         # candles
MAX_HOLD_TREND      = 4         # candles
TP_MULT_GAP         = 3.0
TP_MULT_TREND       = 2.5


@dataclass(frozen=True)
class GapSetup:
    direction: str      # "CALL" | "PUT"
    gap_pct: float      # absolute %
    lot_band: str       # "BEST" | "NORMAL"
    entry_candle_idx: int
    sl_spot_level: float  # spot price of entry candle low (CALL) / high (PUT)
    entry_spot: float
    open_price: float


@dataclass(frozen=True)
class TrendSetup:
    direction: str
    vwap_dist_pct: float
    adx_val: float
    rsi_val: float
    sl_spot_level: float
    entry_candle_idx: int   # always TREND_ENTRY_IDX
    entry_spot: float       # open of 11:00 candle


def check_gap_setup(
    day_df: pd.DataFrame,
    prev_close: float,
    vix: float,
    ema9_today: pd.Series,   # pre-warmed 3-period EMA aligned to today's candles
) -> Optional[GapSetup]:
    """
    Detect Gap & Go setup.
    day_df     : today's 15-min candles (index 0 = 9:15 candle)
    prev_close : previous session close
    vix        : India VIX value
    ema9_today : EMA series aligned to day_df (same length)
    """
    if len(day_df) < 2 or prev_close <= 0:
        return None

    first = day_df.iloc[0]
    today_open  = float(first["open"])
    first_close = float(first["close"])

    gap_pct = (today_open - prev_close) / prev_close * 100.0
    abs_gap = abs(gap_pct)

    if abs_gap < GAP_MIN_PCT:
        return None
    if not (VIX_LOW <= vix <= VIX_HIGH):
        return None

    direction = "CALL" if gap_pct > 0 else "PUT"

    # First 15-min candle must close in gap direction
    if direction == "CALL" and first_close < today_open:
        return None
    if direction == "PUT" and first_close > today_open:
        return None

    # Find first pullback to EMA in search window
    entry_idx = _find_ema_pullback(day_df, ema9_today, direction)
    if entry_idx is None:
        return None

    entry_candle = day_df.iloc[entry_idx]
    entry_spot   = float(entry_candle["close"])  # enter on close of pullback candle

    if direction == "CALL":
        sl_level = float(entry_candle["low"])
    else:
        sl_level = float(entry_candle["high"])

    lot_band = "BEST" if (abs_gap >= GAP_BEST_PCT and VIX_BEST_LOW <= vix <= VIX_BEST_HIGH) else "NORMAL"

    logger.info("GapSetup: %s gap=%.2f%% entry_idx=%d lot=%s", direction, abs_gap, entry_idx, lot_band)
    return GapSetup(
        direction=direction,
        gap_pct=abs_gap,
        lot_band=lot_band,
        entry_candle_idx=entry_idx,
        sl_spot_level=sl_level,
        entry_spot=entry_spot,
        open_price=today_open,
    )


def check_11am_trend(
    day_df: pd.DataFrame,
    adx_val: float,
    rsi_val: float,
    vwap_today: pd.Series,   # VWAP aligned to day_df
    vix: float,
) -> Optional[TrendSetup]:
    """
    Detect 11 AM Trend setup.
    Checks at the TREND_CHECK_IDX (10:45) candle.
    Entry is at open of TREND_ENTRY_IDX (11:00) candle.
    """
    if len(day_df) <= TREND_ENTRY_IDX:
        return None
    if not (VIX_LOW <= vix <= VIX_HIGH):
        return None

    c_1045 = day_df.iloc[TREND_CHECK_IDX]
    close_ = float(c_1045["close"])

    # VWAP distance
    if len(vwap_today) > TREND_CHECK_IDX:
        vwap_ = float(vwap_today.iloc[TREND_CHECK_IDX])
    else:
        vwap_ = float(vwap_today.iloc[-1])
    vwap_dist = (close_ - vwap_) / vwap_ * 100.0 if vwap_ > 0 else 0.0

    # ADX and RSI
    if adx_val < ADX_THRESHOLD:
        return None

    # Previous 3 candles all same direction (10:00, 10:15, 10:30 = indices 3,4,5)
    if TREND_CHECK_IDX < 3:
        return None
    prev_3 = [day_df.iloc[k] for k in range(TREND_CHECK_IDX - 3, TREND_CHECK_IDX)]
    all_green = all(float(c["close"]) > float(c["open"]) for c in prev_3)
    all_red   = all(float(c["close"]) < float(c["open"]) for c in prev_3)

    direction = None
    if all_green and rsi_val > RSI_CALL_MIN and vwap_dist > VWAP_DIST_MIN_PCT:
        direction = "CALL"
    elif all_red and rsi_val < RSI_PUT_MAX and vwap_dist < -VWAP_DIST_MIN_PCT:
        direction = "PUT"

    if direction is None:
        return None

    entry_candle = day_df.iloc[TREND_ENTRY_IDX]
    entry_spot   = float(entry_candle["open"])

    if direction == "CALL":
        sl_level = float(c_1045["low"])
    else:
        sl_level = float(c_1045["high"])

    logger.info("TrendSetup: %s adx=%.1f rsi=%.1f vwap_dist=%.2f%%", direction, adx_val, rsi_val, vwap_dist)
    return TrendSetup(
        direction=direction,
        vwap_dist_pct=vwap_dist,
        adx_val=adx_val,
        rsi_val=rsi_val,
        sl_spot_level=sl_level,
        entry_candle_idx=TREND_ENTRY_IDX,
        entry_spot=entry_spot,
    )


def _find_ema_pullback(
    day_df: pd.DataFrame,
    ema_series: pd.Series,
    direction: str,
) -> Optional[int]:
    """
    Find first candle in GAP_ENTRY_WINDOW where price pulls back to EMA.
    Returns candle index, or None if no pullback found.
    """
    for i in GAP_ENTRY_WINDOW:
        if i >= len(day_df) or i >= len(ema_series):
            break
        candle  = day_df.iloc[i]
        ema_val = float(ema_series.iloc[i])

        if ema_val <= 0:
            continue

        if direction == "CALL":
            # Pullback: candle low touches or crosses below EMA
            if float(candle["low"]) <= ema_val:
                return i
        else:
            # Pullback for PUT: candle high touches or crosses above EMA
            if float(candle["high"]) >= ema_val:
                return i

    return None
