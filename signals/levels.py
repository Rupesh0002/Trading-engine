"""
Auto Support/Resistance detection from price-action on 15-min candles.
No volume used — NIFTY/BANKNIFTY/SENSEX always report volume=0 on Kite.

Levels detected:
  PDH / PDL     — previous day high/low (strength 3)
  WEEK_H/WEEK_L — high/low of last 5 trading days (strength 2)
  SWING_H/L     — n-candle swing pivot (strength 1, boosted by touch count)
  ROUND         — round-number steps (100 pts NIFTY, 500 pts BN/SENSEX, strength 1)

Levels within 0.15% of each other are merged, keeping the strongest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_
from typing import List

import pandas as pd

_INDEX_ROUND_STEP: dict[str, int] = {
    "NIFTY":     100,
    "BANKNIFTY": 500,
    "SENSEX":    500,
}
_SWING_N     = 5      # candles each side to qualify as a swing pivot
_WEEK_DAYS   = 5      # how many prior trading days for weekly range
_MERGE_TOL   = 0.0015 # merge levels within 0.15% of spot


@dataclass
class SRLevel:
    price:    float
    kind:     str   # PDH / PDL / WEEK_H / WEEK_L / SWING_H / SWING_L / ROUND
    strength: int   # 1–3; higher = more significant


def detect_levels(
    df_prior: pd.DataFrame,
    spot: float,
    index: str = "NIFTY",
) -> List[SRLevel]:
    """
    Build key S/R levels from 15-min candles that precede today.
    df_prior must contain columns: timestamp, high, low, close.
    spot  : today's reference price (for round-number anchoring).
    """
    if df_prior is None or df_prior.empty:
        return []

    df = df_prior.copy()
    df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
    dates = sorted(df["_date"].unique())

    levels: List[SRLevel] = []

    # ── PDH / PDL ──────────────────────────────────────────────────────────────
    if dates:
        prev = dates[-1]
        pd_df = df[df["_date"] == prev]
        if not pd_df.empty:
            levels.append(SRLevel(float(pd_df["high"].max()), "PDH", 3))
            levels.append(SRLevel(float(pd_df["low"].min()),  "PDL", 3))

    # ── Weekly high / low ─────────────────────────────────────────────────────
    week_dates = dates[-_WEEK_DAYS:]
    wk_df = df[df["_date"].isin(week_dates)]
    if not wk_df.empty:
        wh = float(wk_df["high"].max())
        wl = float(wk_df["low"].min())
        levels.append(SRLevel(wh, "WEEK_H", 2))
        levels.append(SRLevel(wl, "WEEK_L", 2))

    # ── Swing highs / lows (n-candle pivot) ───────────────────────────────────
    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    n = _SWING_N
    for i in range(n, len(highs) - n):
        if highs[i] == max(highs[i - n: i + n + 1]):
            levels.append(SRLevel(float(highs[i]), "SWING_H", 1))
        if lows[i]  == min(lows[i - n: i + n + 1]):
            levels.append(SRLevel(float(lows[i]),  "SWING_L", 1))

    # ── Round numbers ─────────────────────────────────────────────────────────
    step = _INDEX_ROUND_STEP.get(index, 100)
    lo   = int(spot * 0.97 / step) * step
    hi   = int(spot * 1.03 / step) * step + step
    for p in range(lo, hi + step, step):
        levels.append(SRLevel(float(p), "ROUND", 1))

    return _merge(levels, spot)


def nearest_resistance(levels: List[SRLevel], price: float, direction: str) -> float:
    """Nearest level above (CALL) or below (PUT). Returns a 5% fallback if none."""
    if direction == "CALL":
        above = [lv.price for lv in levels if lv.price > price]
        return min(above) if above else price * 1.05
    else:
        below = [lv.price for lv in levels if lv.price < price]
        return max(below) if below else price * 0.95


# ── Internal ──────────────────────────────────────────────────────────────────

def _merge(levels: List[SRLevel], ref: float) -> List[SRLevel]:
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: x.price)
    merged: List[SRLevel] = []
    cluster: List[SRLevel] = [levels[0]]
    for lv in levels[1:]:
        if abs(lv.price - cluster[0].price) / ref <= _MERGE_TOL:
            cluster.append(lv)
        else:
            merged.append(max(cluster, key=lambda x: x.strength))
            cluster = [lv]
    merged.append(max(cluster, key=lambda x: x.strength))
    return merged
