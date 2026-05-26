"""
Liquidity Sweep (Stop Hunt) signal detector.

Concept: key S/R levels (PDH/PDL, weekly H/L, swing pivots, round numbers) attract
clustered SL orders from retail traders. Smart money pushes through those levels to
trigger the stops, then reverses sharply — creating a high-probability entry in the
opposite direction with a very tight structural SL (just beyond the sweep extreme).

Two-candle confirmation (v2):
  Step 1 — Sweep candle: wick pierces THROUGH a level, candle body closes BACK inside.
  Step 2 — Confirming candle: NEXT 5-min candle closes in the reversal direction
            AND is still on the correct side of the swept level.
  Entry  = confirming candle close (not the sweep candle close).

This extra candle filters fake sweeps that immediately continue beyond the level.

Quality filters applied to the SWEEP candle:
  - Wick through level ≥ 30% of candle range (meaningful pierce)
  - Wick must travel ≥ 0.05% beyond the level in absolute terms
  - Only consider levels within 1.5% of current price
  - Trend check: skip counter-trend sweeps when trend is strong (ADX > 28)
  - R:R always ≥ 1.5:1 enforced; fallback = 2:1 in spot terms if no structural TP

Confirming candle filter:
  - CALL: confirming candle close > swept support level  (still holding above)
          AND confirming candle close > confirming candle open  (green / bullish)
  - PUT:  confirming candle close < swept resistance level (still holding below)
          AND confirming candle close < confirming candle open  (red / bearish)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import pandas as pd

from signals.levels import SRLevel, nearest_resistance

logger = logging.getLogger(__name__)

# Level classification
_SUPPORT_KINDS    = {"PDL", "WEEK_L", "SWING_L", "OR_L"}
_RESISTANCE_KINDS = {"PDH", "WEEK_H", "SWING_H", "OR_H"}
_NEUTRAL_KINDS    = {"ROUND"}

# Signal thresholds
_MIN_WICK_PCT    = 0.30    # wick through level ≥ 30% of candle H-L range
_MIN_PIERCE_PCT  = 0.0005  # wick must pierce ≥ 0.05% beyond level
_SL_BUFFER_PCT   = 0.0003  # SL placed 0.03% beyond sweep extreme
_MIN_RR          = 1.5     # minimum structural R:R; else fallback to 2:1
_LEVEL_PROX_PCT  = 0.015   # only sweep levels within 1.5% of mid-price


@dataclass
class SweepSignal:
    direction:    str    # "CALL" | "PUT"
    sweep_time:   time   # timestamp of the CONFIRMING candle (entry candle)
    entry_price:  float  # confirming candle close = entry spot reference
    sl_spot:      float  # just beyond sweep candle extreme (very tight)
    tp_spot:      float  # next structural level in reversal direction
    level_price:  float  # the swept S/R level
    level_kind:   str    # PDH / PDL / SWING_H / OR_L / ROUND …
    wick_pts:     float  # points the wick pierced through the level
    risk_pts:     float  # entry → SL distance
    reward_pts:   float  # entry → TP distance
    rr:           float  # reward / risk ratio


class LiquiditySweepEngine:

    def scan_5min(
        self,
        df_5:          pd.DataFrame,
        levels:        List[SRLevel],
        context_adx:   float = 0.0,
        context_trend: str   = "NEUTRAL",
        signal_start:  time  = time(9, 30),
        signal_end:    time  = time(14, 0),
    ) -> Optional[SweepSignal]:
        """
        Scan 5-min candles for a qualifying liquidity sweep WITH a confirming candle.

        Candle N   : sweep candle  — wick through level, close back inside.
        Candle N+1 : confirming    — closes in reversal direction, still beyond level.

        Entry is at the close of candle N+1 (not the sweep candle).
        Returns the FIRST qualifying signal.
        """
        if df_5 is None or df_5.empty or not levels:
            return None

        rows = list(df_5.iterrows())

        for i, (_, row) in enumerate(rows):
            t5 = pd.Timestamp(row["timestamp"]).time()
            if t5 < signal_start or t5 >= signal_end:
                continue

            h5  = float(row["high"])
            l5  = float(row["low"])
            c5  = float(row["close"])
            rng = h5 - l5
            if rng <= 0:
                continue

            mid = (h5 + l5) / 2.0

            # ── Check sweep candle against each nearby level ──────────────
            sweep_result = None
            for lv in levels:
                lp = lv.price
                if abs(lp - mid) / mid > _LEVEL_PROX_PCT:
                    continue

                is_sup = lv.kind in _SUPPORT_KINDS or (
                    lv.kind in _NEUTRAL_KINDS and lp <= mid
                )
                is_res = lv.kind in _RESISTANCE_KINDS or (
                    lv.kind in _NEUTRAL_KINDS and lp > mid
                )

                # Bullish sweep (wick below support, close above)
                if is_sup:
                    pierce = lp - l5
                    if pierce < lp * _MIN_PIERCE_PCT:
                        continue
                    if c5 <= lp:
                        continue
                    if pierce / rng < _MIN_WICK_PCT:
                        continue
                    if context_trend == "BEAR" and context_adx > 28:
                        continue
                    sweep_result = ("CALL", lv, pierce, l5, h5)
                    break

                # Bearish sweep (wick above resistance, close below)
                if is_res:
                    pierce = h5 - lp
                    if pierce < lp * _MIN_PIERCE_PCT:
                        continue
                    if c5 >= lp:
                        continue
                    if pierce / rng < _MIN_WICK_PCT:
                        continue
                    if context_trend == "BULL" and context_adx > 28:
                        continue
                    sweep_result = ("PUT", lv, pierce, l5, h5)
                    break

            if sweep_result is None:
                continue

            # ── Check confirming candle (N+1) ─────────────────────────────
            if i + 1 >= len(rows):
                continue

            _, conf_row = rows[i + 1]
            t_conf = pd.Timestamp(conf_row["timestamp"]).time()
            if t_conf >= signal_end:
                continue

            direction, lv, pierce, sw_low, sw_high = sweep_result
            lp           = lv.price
            conf_close   = float(conf_row["close"])
            conf_open    = float(conf_row["open"])

            if direction == "CALL":
                # Confirming: price still above swept support AND green candle
                if conf_close <= lp:
                    continue   # fell back below support → fake sweep
                if conf_close < conf_open:
                    continue   # bearish confirming candle → momentum not bullish

                entry_price = conf_close
                sl_spot     = sw_low * (1 - _SL_BUFFER_PCT)
                risk_pts    = entry_price - sl_spot
                if risk_pts <= 0:
                    continue

                tp_spot    = nearest_resistance(levels, entry_price, "CALL")
                reward_pts = tp_spot - entry_price
                if reward_pts / risk_pts < _MIN_RR:
                    tp_spot    = entry_price + risk_pts * 2.0
                    reward_pts = tp_spot - entry_price

                rr = reward_pts / risk_pts
                logger.info(
                    "[SWEEP] CALL | %s lv=%.2f low=%.2f entry=%.2f "
                    "wick=%.1fpts(%.0f%%) SL=%.2f TP=%.2f R:R=%.2f",
                    lv.kind, lp, sw_low, entry_price,
                    pierce, pierce / rng * 100, sl_spot, tp_spot, rr,
                )
                return SweepSignal(
                    direction="CALL", sweep_time=t_conf,
                    entry_price=entry_price, sl_spot=sl_spot, tp_spot=tp_spot,
                    level_price=lp, level_kind=lv.kind,
                    wick_pts=pierce, risk_pts=risk_pts,
                    reward_pts=reward_pts, rr=rr,
                )

            else:  # PUT
                # Confirming: price still below swept resistance AND red candle
                if conf_close >= lp:
                    continue   # went back above resistance → fake sweep
                if conf_close > conf_open:
                    continue   # bullish confirming candle → momentum not bearish

                entry_price = conf_close
                sl_spot     = sw_high * (1 + _SL_BUFFER_PCT)
                risk_pts    = sl_spot - entry_price
                if risk_pts <= 0:
                    continue

                tp_spot    = nearest_resistance(levels, entry_price, "PUT")
                reward_pts = entry_price - tp_spot
                if reward_pts / risk_pts < _MIN_RR:
                    tp_spot    = entry_price - risk_pts * 2.0
                    reward_pts = entry_price - tp_spot

                rr = reward_pts / risk_pts
                logger.info(
                    "[SWEEP] PUT  | %s lv=%.2f high=%.2f entry=%.2f "
                    "wick=%.1fpts(%.0f%%) SL=%.2f TP=%.2f R:R=%.2f",
                    lv.kind, lp, sw_high, entry_price,
                    pierce, pierce / rng * 100, sl_spot, tp_spot, rr,
                )
                return SweepSignal(
                    direction="PUT", sweep_time=t_conf,
                    entry_price=entry_price, sl_spot=sl_spot, tp_spot=tp_spot,
                    level_price=lp, level_kind=lv.kind,
                    wick_pts=pierce, risk_pts=risk_pts,
                    reward_pts=reward_pts, rr=rr,
                )

        return None
