"""
Morning context engine — runs once at 9:15 AM.

Collects:
  1. Gap analysis          (open vs prev_close)
  2. First Candle Range    (FCR — measures day volatility)
  3. VIX regime            (fetch live from Kite)
  4. Round number check    (NIFTY 500-pt / BANKNIFTY 1000-pt levels)

Returns MorningContext dataclass.
If skip_day=True → engine sends alert and takes no trades.

Skip conditions:
  - VIX > 24   (extreme fear — option premium unpredictable)
  - FCR < 0.2% AND gap neutral  (dead day — premiums won't move)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

import pandas as pd
import pytz

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Thresholds ────────────────────────────────────────────────────────────────
_GAP_BULL_PCT     = 0.30    # >+0.3% → bullish gap
_GAP_BEAR_PCT     = -0.30   # <-0.3% → bearish gap
_VIX_LOW          = 12.0
_VIX_NORMAL_HIGH  = 18.0
_VIX_HIGH_MAX     = 24.0    # above this → skip day
_FCR_DEAD_PCT     = 0.20    # FCR below 0.2% of open = dead day
_FCR_NORMAL_PCT   = 0.50
_FCR_ACTIVE_PCT   = 0.80
_ROUND_NIFTY      = 500
_ROUND_BANKNIFTY  = 1000
_ROUND_PROXIMITY  = 0.002   # within 0.2% of round level


@dataclass
class FCRData:
    range_pts:      float
    range_pct:      float     # range / open × 100
    day_type:       str       # "DEAD", "NORMAL", "ACTIVE", "STRONG"
    lot_multiplier: float     # 0.5, 1.0, or 1.5


@dataclass
class MorningContext:
    gap_pct:          float
    gap_bias:         str      # "BULLISH", "BEARISH", "NEUTRAL"
    fcr:              FCRData
    vix:              float
    vix_regime:       str      # "LOW", "NORMAL", "HIGH", "SKIP"
    near_round:       bool     # spot near round number at open
    nearest_round:    float
    skip_day:         bool
    skip_reason:      str
    open_price:       float
    prev_close:       float


def build_morning_context(
    today_candles: pd.DataFrame,
    prev_close: float,
    vix: float,
    index: str = "BANKNIFTY",
) -> MorningContext:
    """
    Build morning context from today's first 15-min candle + prev_close + live VIX.
    today_candles: DataFrame with at least the 9:15 candle.
    """
    # ── First candle (9:15 open candle) ─────────────────────────────────────
    first = today_candles[today_candles["timestamp"].dt.time == time(9, 15)]
    if first.empty:
        first = today_candles.head(1)
    open_price = float(first["open"].iloc[0])

    # ── Gap ─────────────────────────────────────────────────────────────────
    gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close else 0.0
    if gap_pct > _GAP_BULL_PCT:
        gap_bias = "BULLISH"
    elif gap_pct < _GAP_BEAR_PCT:
        gap_bias = "BEARISH"
    else:
        gap_bias = "NEUTRAL"

    # ── FCR (First Candle Range) ─────────────────────────────────────────────
    fcr_high = float(first["high"].iloc[0])
    fcr_low  = float(first["low"].iloc[0])
    fcr_pts  = fcr_high - fcr_low
    fcr_pct  = fcr_pts / open_price * 100

    if fcr_pct < _FCR_DEAD_PCT:
        day_type    = "DEAD"
        lot_mult    = 0.5
    elif fcr_pct < _FCR_NORMAL_PCT:
        day_type    = "NORMAL"
        lot_mult    = 1.0
    elif fcr_pct < _FCR_ACTIVE_PCT:
        day_type    = "ACTIVE"
        lot_mult    = 1.0
    else:
        day_type    = "STRONG"
        lot_mult    = 1.5

    fcr = FCRData(
        range_pts=round(fcr_pts, 2),
        range_pct=round(fcr_pct, 3),
        day_type=day_type,
        lot_multiplier=lot_mult,
    )

    # ── VIX regime ───────────────────────────────────────────────────────────
    if vix < _VIX_LOW:
        vix_regime = "LOW"
    elif vix <= _VIX_NORMAL_HIGH:
        vix_regime = "NORMAL"
    elif vix <= _VIX_HIGH_MAX:
        vix_regime = "HIGH"
    else:
        vix_regime = "SKIP"

    # ── Round number proximity ───────────────────────────────────────────────
    step    = _ROUND_BANKNIFTY if index == "BANKNIFTY" else _ROUND_NIFTY
    nearest = round(open_price / step) * step
    near_round = abs(open_price - nearest) / open_price < _ROUND_PROXIMITY

    # ── Skip day logic ───────────────────────────────────────────────────────
    skip_day    = False
    skip_reason = ""

    if vix_regime == "SKIP":
        skip_day    = True
        skip_reason = f"VIX {vix:.1f} > 24 — extreme fear"
    elif day_type == "DEAD" and gap_bias == "NEUTRAL":
        skip_day    = True
        skip_reason = f"Dead day — FCR {fcr_pct:.2f}% + neutral gap"

    return MorningContext(
        gap_pct=round(gap_pct, 2),
        gap_bias=gap_bias,
        fcr=fcr,
        vix=round(vix, 1),
        vix_regime=vix_regime,
        near_round=near_round,
        nearest_round=nearest,
        skip_day=skip_day,
        skip_reason=skip_reason,
        open_price=round(open_price, 2),
        prev_close=round(prev_close, 2),
    )


def morning_telegram_text(ctx: MorningContext, capital: float, paper: bool = True) -> str:
    """Format the 9:15 AM Telegram morning message."""
    mode = "PAPER" if paper else "LIVE"
    today = datetime.now(IST).strftime("%d %b %Y")

    if ctx.skip_day:
        return (
            f"📵 <b>Skip Day — {today}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Reason : {ctx.skip_reason}\n"
            f"VIX    : {ctx.vix:.1f}\n"
            f"Gap    : {ctx.gap_pct:+.2f}%\n"
            f"FCR    : {ctx.fcr.range_pct:.2f}% ({ctx.fcr.day_type})"
        )

    rnd_flag = f"  ⚠️ Near round {ctx.nearest_round:,.0f}" if ctx.near_round else ""
    return (
        f"🌅 <b>Engine Started — {today}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Gap      : {ctx.gap_pct:+.2f}% ({ctx.gap_bias}){rnd_flag}\n"
        f"FCR      : {ctx.fcr.range_pts:.0f}pts  {ctx.fcr.range_pct:.2f}% ({ctx.fcr.day_type})\n"
        f"VIX      : {ctx.vix:.1f} ({ctx.vix_regime})\n"
        f"Capital  : ₹{capital:,.0f}\n"
        f"Mode     : {mode}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Signal window : 09:30–13:00"
    )
