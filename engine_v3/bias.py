"""
Intraday bias — computed once at 9:45 AM after ORB is built.

BULLISH  : gap up > 0.3% AND open > prior close
BEARISH  : gap down > 0.3% AND open < prior close
NEUTRAL  : everything else

Lot multiplier:
  Trade with bias  → 1.0× (full size)
  Trade vs bias    → 0.5× (half size)
  Neutral day      → 1.0× for all directions
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DayBias:
    bias:    str    # BULLISH | BEARISH | NEUTRAL
    gap_pct: float


def compute_bias(today_open: float, pdc: float) -> DayBias:
    gap_pct = (today_open - pdc) / pdc * 100.0
    if gap_pct > 0.3:
        bias = "BULLISH"
    elif gap_pct < -0.3:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    return DayBias(bias=bias, gap_pct=round(gap_pct, 3))


def lots_multiplier(bias: str, direction: str) -> float:
    if bias == "NEUTRAL":
        return 1.0
    with_bias = (bias == "BULLISH" and direction == "CALL") or \
                (bias == "BEARISH" and direction == "PUT")
    return 1.0 if with_bias else 0.5
