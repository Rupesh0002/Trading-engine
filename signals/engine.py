"""
Signal engine — 5-condition AND system.

Per-candle signal rules (evaluated on last CLOSED candle, df.iloc[-2]):
  1. EMA 20/50  — Trend structure. Bullish: close > EMA20 > EMA50.
                  Bearish: close < EMA20 < EMA50. Chop zone = skip.
  2. RSI (14)   — Momentum. Bullish: RSI > 55. Bearish: RSI < 45. Neutral = skip.
  3. VWAP       — Intraday bias (hard gate). Long above, short below.
                  Never trade against VWAP direction — no exceptions.
  4. PDH/PDL    — Entry zone gate. Uses yesterday's daily candle high/low
                  as fixed support/resistance levels (no intraday shifting).
                  CALL: price within 0.20% of PDL (previous day low = support bounce).
                  PUT : price within 0.20% of PDH (previous day high = resistance rejection).
                  Skip if previous-day range < 0.50% of price (levels too dense).
  5. ADX (14)   — Trend-strength filter. ADX > 20 = trending market.
                  ADX < 20 = choppy/drifting — theta eats premium, skip.

PCR morning bias filter (passed as `pcr`):
  PCR > 1.1 → fear → bias long. PCR < 0.7 → greed → bias short.
  PCR is informational — does not block trades when all 5 agree.

Signal fires when core 4 agree (EMA + RSI + VWAP + PDH/PDL). ADX boosts to 5/5 for OTM selection.
R:R ≥ 1:2 by construction (STOP_LOSS=20%, TARGET=50% → 2.5×).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import (
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    PCR_BULL_MIN,
    PCR_BEAR_MAX,
    ADX_THRESHOLD,
    PDH_PROXIMITY_PCT,
    PDH_MIN_RANGE_PCT,
)
from signals.indicators import (
    is_above_vwap,
    is_below_vwap,
    compute_rsi_wilder,
    ema_trend,
    adx_condition,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    direction: str          # "CALL", "PUT", or "WAIT"
    conditions_met: int
    total_conditions: int = 5
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.direction in ("CALL", "PUT")

    def __repr__(self) -> str:
        return (
            f"Signal({self.direction} | {self.conditions_met}/{self.total_conditions} | "
            f"close={self.details.get('close', '?'):.2f})"
        )


class SignalEngine:
    """
    5-condition AND signal engine.
    Core 3 (EMA + RSI + VWAP) + Fibonacci entry zone + ADX trend filter.
    PCR morning bias is tracked in details — informational, does not block signals.
    Fibonacci anchored to yesterday's daily candle when daily_candle is provided.
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        pcr: Optional[float] = None,
        daily_candle=None,
        adx_threshold: Optional[float] = None,
    ) -> SignalResult:
        """
        df           : intraday OHLCV DataFrame (columns: timestamp, open, high, low, close, volume).
        pcr          : morning Put-Call Ratio (None → PCR bias is neutral).
        daily_candle : yesterday's daily OHLC dict/Series for PDH/PDL levels.
                       When None, falls back to intraday session high/low.
        Returns SignalResult with direction CALL / PUT / WAIT.
        """
        if len(df) < 2:
            return SignalResult("WAIT", 0, details={"reason": "Insufficient bars"})

        # ── Reference candle: iloc[-2] = last CLOSED candle ──────────────
        candle = df.iloc[-2]
        close  = float(candle["close"])

        # ── Condition 1: EMA 20/50 — Trend structure ─────────────────────
        ema_bull, ema_bear, ema20_val, ema50_val = ema_trend(df, candle_idx=-2)

        # ── Condition 2: RSI (14) — Momentum confirmation ─────────────────
        rsi_series  = compute_rsi_wilder(df["close"])
        rsi_val     = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else 50.0
        rsi_is_nan  = pd.isna(rsi_val)
        rsi_bullish = (not rsi_is_nan) and rsi_val > RSI_OVERSOLD    # RSI > 55
        rsi_bearish = (not rsi_is_nan) and rsi_val < RSI_OVERBOUGHT  # RSI < 45

        # ── Condition 3: VWAP — Intraday bias + hard gate ─────────────────
        above, _, vwap_val = is_above_vwap(df, candle_idx=-2)
        below, _, _        = is_below_vwap(df, candle_idx=-2)

        # ── Condition 4: PDH/PDL — Previous Day Support/Resistance ───────────
        # CALL: price within 0.20% of PDL (yesterday's low = support bounce).
        # PUT : price within 0.20% of PDH (yesterday's high = resistance rejection).
        # Skip if previous-day range < 0.50% of price — levels too dense to be useful.
        _PDH_PROXIMITY = PDH_PROXIMITY_PCT   # configurable via .env (default 0.50%)
        _PDH_MIN_RANGE = PDH_MIN_RANGE_PCT   # minimum daily range to use levels

        if daily_candle is not None:
            pdh = float(daily_candle["high"])
            pdl = float(daily_candle["low"])
        else:
            pdh = float(df["high"].max())
            pdl = float(df["low"].min())

        _range_pct = (pdh - pdl) / close if close > 0 else 0
        if _range_pct >= _PDH_MIN_RANGE:
            _threshold = close * _PDH_PROXIMITY
            near_pdh = abs(close - pdh) <= _threshold
            near_pdl = abs(close - pdl) <= _threshold
        else:
            near_pdh = near_pdl = False
        pdh_pdl_label = "near_PDH" if near_pdh else ("near_PDL" if near_pdl else "")

        # ── Condition 5: ADX — Trend-strength gate ────────────────────────
        # ADX > threshold: real trend underway → 5/5 signal (OTM eligible).
        # threshold/2 to threshold: borderline — 4/5 signal (ATM only).
        # ADX < threshold/2: truly flat/sideways — hard-block entirely.
        # Per-index threshold overrides the global ADX_THRESHOLD from .env.
        # e.g. NIFTY uses 25 (stricter), BANKNIFTY/SENSEX use 20.
        _adx_thr = adx_threshold if adx_threshold is not None else ADX_THRESHOLD
        adx_ok, adx_val = adx_condition(df, candle_idx=-2, threshold=_adx_thr)
        adx_flat_blocked = adx_val < (_adx_thr / 2)

        if adx_flat_blocked:
            return SignalResult("WAIT", 0, total_conditions=5, details={
                "close": close, "adx": adx_val, "adx_ok": False,
                "reason": f"ADX flat-blocked ({adx_val:.1f} < {_adx_thr/2:.0f})",
            })

        # ── PCR morning bias filter ───────────────────────────────────────
        pcr_bias_long  = (pcr is not None) and (pcr > PCR_BULL_MIN)
        pcr_bias_short = (pcr is not None) and (pcr < PCR_BEAR_MAX)

        # ── Core 3: EMA + RSI + VWAP (all must agree) ────────────────────
        call_core = ema_bull and rsi_bullish and above
        put_core  = ema_bear and rsi_bearish and below

        total_conds = 5
        call_score  = sum([
            int(ema_bull),
            int(rsi_bullish),
            int(above),
            int(near_pdl),
            int(adx_ok),
        ])
        put_score   = sum([
            int(ema_bear),
            int(rsi_bearish),
            int(below),
            int(near_pdh),
            int(adx_ok),
        ])

        # backward-compat aliases — backtest CSV writer and ML both read these keys
        near_fib  = near_pdh or near_pdl
        fib_label = pdh_pdl_label
        fib_level = pdh if near_pdh else (pdl if near_pdl else 0.0)

        details = {
            "close":         close,
            "ema_fast":      round(ema20_val, 2),
            "ema_slow":      round(ema50_val, 2),
            "ema_bull":      ema_bull,
            "ema_bear":      ema_bear,
            "rsi":           round(rsi_val, 2) if not rsi_is_nan else None,
            "rsi_bullish":   rsi_bullish,
            "rsi_bearish":   rsi_bearish,
            "adx":           adx_val,
            "adx_ok":        adx_ok,
            "near_fib":      near_fib,       # backward compat for ML / CSV
            "near_fib_call": near_pdl,       # CALL fires near PDL
            "near_fib_put":  near_pdh,       # PUT fires near PDH
            "near_pdh":      near_pdh,
            "near_pdl":      near_pdl,
            "pdh":           round(pdh, 2),
            "pdl":           round(pdl, 2),
            "pdh_pdl_label": pdh_pdl_label,
            "fib_label":     fib_label,      # backward compat = pdh_pdl_label
            "fib_level":     round(fib_level, 2),
            "swing_high":    round(pdh, 2),  # repurposed: holds PDH
            "swing_low":     round(pdl, 2),  # repurposed: holds PDL
            "fib_anchored":  daily_candle is not None,
            "vwap":          vwap_val,
            "above_vwap":    above,
            "below_vwap":    below,
            "pcr":           pcr,
            "pcr_bias_long": pcr_bias_long,
            "pcr_bias_short":pcr_bias_short,
            "pcr_bull":      pcr_bias_long,   # backward-compat alias
            "pcr_bear":      pcr_bias_short,  # backward-compat alias
            "call_score":    call_score,
            "put_score":     put_score,
            "pcr_against":   False,
        }

        logger.debug(
            "Signal eval: close=%.2f EMA20=%.2f EMA50=%.2f RSI=%.1f ADX=%.1f VWAP=%.2f PCR=%s "
            "core_call=%s near_PDL=%s core_put=%s near_PDH=%s adx_ok=%s",
            close, ema20_val, ema50_val,
            rsi_val if not rsi_is_nan else -1,
            adx_val, vwap_val, pcr,
            call_core, near_pdl, put_core, near_pdh, adx_ok,
        )

        # ── Signal: core 4 must agree; ADX upgrades to 5/5 for OTM selection ──
        # EMA + RSI + VWAP + Fib = required gate (4/5 minimum).
        # ADX > threshold = bonus — signals fire without it, but 5/5 gets OTM treatment.
        # This preserves ADX as a quality signal without hard-blocking non-trending days.
        if call_core and near_pdl:
            pcr_against = pcr_bias_short
            details["pcr_against"] = pcr_against
            if pcr_against:
                logger.warning(
                    "CALL signal against PCR morning bias (PCR=%.2f, greed). "
                    "Firing — core 4 conditions agree.",
                    pcr,
                )
            logger.info(
                "CALL signal: %d/5 | close=%.2f EMA20=%.2f EMA50=%.2f RSI=%.1f ADX=%.1f Level=%s",
                call_score, close, ema20_val, ema50_val,
                rsi_val if not rsi_is_nan else -1,
                adx_val, pdh_pdl_label,
            )
            return SignalResult("CALL", call_score, total_conditions=total_conds, details=details)

        if put_core and near_pdh:
            pcr_against = pcr_bias_long
            details["pcr_against"] = pcr_against
            if pcr_against:
                logger.warning(
                    "PUT signal against PCR morning bias (PCR=%.2f, fear). "
                    "Firing — core 4 conditions agree.",
                    pcr,
                )
            logger.info(
                "PUT signal: %d/5 | close=%.2f EMA20=%.2f EMA50=%.2f RSI=%.1f ADX=%.1f Level=%s",
                put_score, close, ema20_val, ema50_val,
                rsi_val if not rsi_is_nan else -1,
                adx_val, pdh_pdl_label,
            )
            return SignalResult("PUT", put_score, total_conditions=total_conds, details=details)

        return SignalResult("WAIT", max(call_score, put_score), total_conditions=total_conds, details=details)
