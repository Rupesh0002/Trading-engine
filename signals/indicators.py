"""
Technical indicators: VWAP, Fibonacci retracement, RSI (Wilder), Volume spike.

Rules:
  - Always compute on a full DataFrame; callers read df.iloc[-2] (last CLOSED candle).
  - All thresholds come from config/settings.py — zero hardcoded values here.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    FIB_PROXIMITY_PCT,
    FIB_MIN_SWING_PCT,
    RSI_PERIOD,
    VOLUME_SPIKE_MULT,
    VWAP_ZONE_PCT,
)

# ---------------------------------------------------------------------------
# VWAP  — cumulative from the first bar of the session (9:15 AM)
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday cumulative VWAP — always resets at today's first candle.
    df may contain multi-day warmup rows; only today's rows contribute to VWAP.
    Warmup rows get NaN so candle_idx=-2 (last closed bar) always lands in today.
    """
    import pandas as _pd
    result = _pd.Series(np.nan, index=df.index, dtype=float)

    if "timestamp" in df.columns:
        last_date = df["timestamp"].dt.date.iloc[-1]
        mask = df["timestamp"].dt.date == last_date
    else:
        mask = _pd.Series(True, index=df.index)

    today = df[mask]
    if today.empty:
        return result

    typical    = (today["high"] + today["low"] + today["close"]) / 3.0
    cum_vol    = today["volume"].cumsum()
    if cum_vol.iloc[-1] == 0:
        n    = _pd.Series(range(1, len(today) + 1), index=today.index, dtype=float)
        vwap = typical.cumsum() / n
    else:
        vwap = (typical * today["volume"]).cumsum() / cum_vol.replace(0, np.nan)

    result[mask] = vwap.values
    return result


def is_above_vwap(df: pd.DataFrame, candle_idx: int = -2) -> Tuple[bool, float, float]:
    """
    Returns (above, close, vwap_value) for the candle at candle_idx.
    Default candle_idx=-2 → last closed candle.
    """
    vwap = compute_vwap(df)
    close = float(df.iloc[candle_idx]["close"])
    vwap_val = float(vwap.iloc[candle_idx])
    zone_pts = close * (VWAP_ZONE_PCT / 100.0)
    return close > vwap_val + zone_pts, close, vwap_val


def is_below_vwap(df: pd.DataFrame, candle_idx: int = -2) -> Tuple[bool, float, float]:
    vwap = compute_vwap(df)
    close = float(df.iloc[candle_idx]["close"])
    vwap_val = float(vwap.iloc[candle_idx])
    zone_pts = close * (VWAP_ZONE_PCT / 100.0)
    return close < vwap_val - zone_pts, close, vwap_val


# ---------------------------------------------------------------------------
# Fibonacci retracement levels
# ---------------------------------------------------------------------------

# Only the three "golden" levels — 0%, 23.6%, 78.6%, 100% are swing extremes that
# flood the range on small intraday swings and fire on virtually every bar.
FIB_RATIOS: Dict[str, float] = {
    "38.2": 0.382,
    "50.0": 0.500,
    "61.8": 0.618,
}


def compute_fibonacci_levels(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], float, float]:
    """
    Computes Fibonacci retracement levels from session swing high/low.
    Returns (levels_dict, swing_high, swing_low).
    """
    swing_high = float(df["high"].max())
    swing_low = float(df["low"].min())
    diff = swing_high - swing_low

    levels = {
        label: round(swing_high - ratio * diff, 2)
        for label, ratio in FIB_RATIOS.items()
    }
    return levels, swing_high, swing_low


def is_near_fib_level(
    price: float,
    fib_levels: Dict[str, float],
    swing_range: float = 0.0,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Returns (near, label, level) if price is within FIB_PROXIMITY_PCT% of a key level.
    Returns (False, None, None) when swing_range is too small to produce meaningful levels
    (< FIB_MIN_SWING_PCT% of price) — prevents the condition from always firing on
    low-volatility days where all levels are compressed into a tight band.
    """
    min_swing = price * (FIB_MIN_SWING_PCT / 100.0)
    if swing_range < min_swing:
        return False, None, None
    zone_pts = price * (FIB_PROXIMITY_PCT / 100.0)
    for label, level in fib_levels.items():
        if abs(price - level) <= zone_pts:
            return True, label, level
    return False, None, None


# ---------------------------------------------------------------------------
# RSI — Wilder's smoothing method
# ---------------------------------------------------------------------------

def compute_rsi_wilder(
    series: pd.Series,
    period: Optional[int] = None,
) -> pd.Series:
    """
    RSI using Wilder's exponential smoothing (alpha = 1 / period).
    period defaults to RSI_PERIOD from .env.
    """
    n = period if period is not None else RSI_PERIOD
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # ewm with com = n-1 replicates Wilder's alpha = 1/n
    avg_gain = gain.ewm(com=n - 1, min_periods=n).mean()
    avg_loss = loss.ewm(com=n - 1, min_periods=n).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# EMA 20 / EMA 50 — trend direction
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA using span smoothing (alpha = 2 / (span + 1))."""
    return series.ewm(span=period, adjust=False).mean()


def ema_trend(
    df: pd.DataFrame,
    candle_idx: int = -2,
) -> Tuple[bool, bool, float, float]:
    """
    Returns (bullish, bearish, ema_fast, ema_slow) for the candle at candle_idx.
    Periods: EMA_FAST_PERIOD / EMA_SLOW_PERIOD from .env (default 9 / 21).
    Sized for 15-min intraday: EMA9 ≈ 135 min, EMA21 ≈ 315 min (full session).
    Bullish : close > EMA_fast > EMA_slow  — all three stacked up.
    Bearish : close < EMA_fast < EMA_slow  — all three stacked down.
    Neither when EMAs are crossed but price is between them (chop zone).
    """
    from config.settings import EMA_FAST_PERIOD, EMA_SLOW_PERIOD
    ema_fast = compute_ema(df["close"], EMA_FAST_PERIOD)
    ema_slow = compute_ema(df["close"], EMA_SLOW_PERIOD)
    close  = float(df.iloc[candle_idx]["close"])
    e_fast = float(ema_fast.iloc[candle_idx])
    e_slow = float(ema_slow.iloc[candle_idx])
    bullish = close > e_fast and e_fast > e_slow
    bearish = close < e_fast and e_fast < e_slow
    return bullish, bearish, e_fast, e_slow


# ---------------------------------------------------------------------------
# ADX — Average Directional Index (Wilder smoothing)
# ---------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    """
    ADX using Wilder's smoothing.
    Measures trend strength — direction-neutral (works for up AND down trends).
    ADX > 20: trending market.  ADX < 20: choppy/drifting — avoid.
    """
    from config.settings import ADX_PERIOD
    n = period if period is not None else ADX_PERIOD

    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    # True Range
    hl   = high - low
    hpc  = (high - close.shift(1)).abs()
    lpc  = (low  - close.shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

    # Directional Movement
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    # Wilder smoothing (alpha = 1/n)
    def _wilder(s: pd.Series) -> pd.Series:
        return s.ewm(com=n - 1, min_periods=n).mean()

    atr       = _wilder(tr)
    plus_di   = 100 * _wilder(pd.Series(plus_dm,  index=df.index)) / atr.replace(0, np.nan)
    minus_di  = 100 * _wilder(pd.Series(minus_dm, index=df.index)) / atr.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx       = _wilder(dx.fillna(0))
    return adx


def adx_condition(
    df: pd.DataFrame,
    candle_idx: int = -2,
    threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    """
    Returns (is_trending, adx_value) for the candle at candle_idx.
    is_trending = True when ADX > threshold (default ADX_THRESHOLD from .env).
    """
    from config.settings import ADX_THRESHOLD
    thr = threshold if threshold is not None else ADX_THRESHOLD
    adx = compute_adx(df)
    if len(adx) < abs(candle_idx):
        return False, 0.0
    val = float(adx.iloc[candle_idx])
    if np.isnan(val):
        return False, 0.0
    return val >= thr, round(val, 2)


# ---------------------------------------------------------------------------
# Fibonacci from a single daily candle (fixed anchors)
# ---------------------------------------------------------------------------

def get_fib_levels_from_daily(
    daily_candle,
) -> Tuple[Dict[str, float], float, float]:
    """
    Computes Fibonacci retracement levels from yesterday's daily candle high/low.
    daily_candle: dict or Series with 'high' and 'low' keys.
    Returns (levels_dict, swing_high, swing_low) — identical shape to compute_fibonacci_levels.
    Levels are fixed all day, eliminating the intraday shifting-anchor problem.
    """
    swing_high = float(daily_candle["high"])
    swing_low  = float(daily_candle["low"])
    diff = swing_high - swing_low
    if diff <= 0:
        return {}, swing_high, swing_low
    levels = {
        label: round(swing_high - ratio * diff, 2)
        for label, ratio in FIB_RATIOS.items()
    }
    return levels, swing_high, swing_low


# ---------------------------------------------------------------------------
# Volume spike
# ---------------------------------------------------------------------------

_VOLUME_MA_PERIOD = 20  # rolling window for volume baseline (not user-configurable)


def check_volume_spike(df: pd.DataFrame, candle_idx: int = -2) -> Tuple[Optional[bool], float, float]:
    """
    Returns (is_spike, current_volume, volume_ma) for the candle at candle_idx.
    Spike threshold = VOLUME_SPIKE_MULT × 20-period volume MA (from .env).
    Returns (None, 0, 0) when volume data is unavailable (all zeros — index instruments).
    """
    # Index instruments (NIFTY, BANKNIFTY) return volume=0 — treat as data unavailable
    if df["volume"].sum() == 0:
        return None, 0.0, 0.0

    if len(df) < _VOLUME_MA_PERIOD:
        return False, 0.0, 0.0

    vol_ma = df["volume"].rolling(_VOLUME_MA_PERIOD).mean()
    current_vol = float(df.iloc[candle_idx]["volume"])
    ma_val = float(vol_ma.iloc[candle_idx])

    if ma_val == 0 or np.isnan(ma_val):
        return False, current_vol, 0.0

    is_spike = current_vol >= VOLUME_SPIKE_MULT * ma_val
    return is_spike, current_vol, ma_val
