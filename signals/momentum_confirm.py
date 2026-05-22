"""
Momentum Confirmation — 5-min candle filter before entry.

ALL five conditions must be true to confirm momentum:
  1. Green candle  : close > open
  2. Body size     : (close - open) / spot >= BODY_MIN_PCT (0.15%)
  3. Volume        : candle_volume >= VOLUME_MULT × avg_5min_volume (1.3×)
  4. VWAP position : close > vwap (for CALL) | close < vwap (for PUT)
  5. Premium rising: option LTP rising last 2 ticks (checked separately via
                     PremiumMomentum; this module receives a boolean `premium_rising`)

Returns MomentumConfirm dataclass with pass/fail per condition and overall result.
Cancels if not confirmed within CONFIRM_CANDLES (3) five-min candles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BODY_MIN_PCT   = 0.0015   # 0.15% of spot
_VOLUME_MULT    = 1.30     # 1.3× avg 5-min volume
CONFIRM_CANDLES = 3        # max candles to wait for confirmation


@dataclass(frozen=True)
class MomentumConfirm:
    confirmed: bool
    green_candle: bool
    body_ok: bool
    volume_ok: bool
    vwap_ok: bool
    premium_rising: bool
    # diagnostic
    body_pct: float
    volume_ratio: float


def check_momentum_confirm(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    avg_5min_volume: float,
    vwap: float,
    direction: str,           # "CALL" | "PUT"
    premium_rising: bool,
    spot: float,
    *,
    body_min_pct: float  = _BODY_MIN_PCT,
    volume_mult: float   = _VOLUME_MULT,
) -> MomentumConfirm:
    """
    Evaluate all 5 momentum conditions on a single 5-min candle.

    Parameters
    ----------
    open_, close    : 5-min candle OHLCV (low/high not used here but kept for context)
    avg_5min_volume : rolling average of 5-min candle volumes (same session, same index)
    vwap            : running VWAP up to this candle
    direction       : "CALL" (bullish) or "PUT" (bearish)
    premium_rising  : True if option LTP rose over the last 2 ticks
    spot            : underlying spot price (for body_pct denominator)
    """
    green_candle = close > open_

    body_pct = (close - open_) / spot if spot > 0 else 0.0
    body_ok  = body_pct >= body_min_pct       # directional body (always positive for call)

    volume_ratio = volume / avg_5min_volume if avg_5min_volume > 0 else 1.0
    volume_ok    = volume_ratio >= volume_mult

    if direction == "CALL":
        vwap_ok = close > vwap
    else:
        vwap_ok = close < vwap
        # For PUT: green candle and positive body_pct won't apply; re-evaluate
        green_candle = close < open_          # bearish candle for PUT
        body_pct     = (open_ - close) / spot if spot > 0 else 0.0
        body_ok      = body_pct >= body_min_pct

    confirmed = green_candle and body_ok and volume_ok and vwap_ok and premium_rising

    logger.debug(
        "MomentumConfirm[%s] green=%s body=%.3f%% vol=%.2f× vwap=%s prem_rising=%s → %s",
        direction, green_candle, body_pct * 100, volume_ratio,
        vwap_ok, premium_rising, "OK" if confirmed else "NO",
    )
    return MomentumConfirm(
        confirmed=confirmed,
        green_candle=green_candle,
        body_ok=body_ok,
        volume_ok=volume_ok,
        vwap_ok=vwap_ok,
        premium_rising=premium_rising,
        body_pct=body_pct,
        volume_ratio=volume_ratio,
    )
