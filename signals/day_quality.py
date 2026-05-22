"""
Day Quality Filter — runs at 9:15-9:30 AM before any trades.

4 binary checks:
  1. Gap %          : abs(open - prev_close) / prev_close >= GAP_MIN_PCT
  2. First candle   : range (high-low) >= FIRST_CANDLE_RANGE_MIN_PCT × spot
  3. VIX range      : 13 <= vix <= 22
  4. Volume vs avg  : first_candle_volume >= VOLUME_VS_AVG × 20d_avg_volume

Score → Day Quality:
  4/4 → PREMIUM   (trade at 1.5× base lots)
  3/4 → NORMAL    (trade at 1.0× base lots)
  2/4 → CAUTIOUS  (trade at 0.5× base lots)
  0-1 → SKIP      (no trades today)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Thresholds (all tunable via caller, not from .env — keep this module param-free)
_GAP_MIN_PCT         = 0.0015   # 0.15% gap
_RANGE_MIN_PCT       = 0.0020   # 0.20% of spot as first-candle range
_VIX_LOW             = 13.0
_VIX_HIGH            = 22.0
_VOLUME_MULTIPLIER   = 1.3      # first candle volume ≥ 1.3× 20d avg


@dataclass(frozen=True)
class DayQuality:
    label: str            # "PREMIUM" | "NORMAL" | "CAUTIOUS" | "SKIP"
    score: int            # 0-4
    gap_ok: bool
    range_ok: bool
    vix_ok: bool
    volume_ok: bool
    lot_multiplier: float # 1.5 / 1.0 / 0.5 / 0.0

    @property
    def tradeable(self) -> bool:
        return self.lot_multiplier > 0.0


_LABEL_MAP = {4: "PREMIUM", 3: "NORMAL", 2: "CAUTIOUS"}
_MULT_MAP  = {4: 1.5,       3: 1.0,      2: 0.5}


def assess_day_quality(
    prev_close: float,
    open_price: float,
    first_high: float,
    first_low: float,
    first_volume: float,
    avg_volume_20d: float,
    vix: Optional[float],
    *,
    gap_min_pct: float    = _GAP_MIN_PCT,
    range_min_pct: float  = _RANGE_MIN_PCT,
    vix_low: float        = _VIX_LOW,
    vix_high: float       = _VIX_HIGH,
    volume_mult: float    = _VOLUME_MULTIPLIER,
) -> DayQuality:
    """
    Compute day quality from the 9:15 candle data.

    Parameters
    ----------
    prev_close      : previous day's closing spot price
    open_price      : today's opening spot price (9:15 candle open)
    first_high/low  : 9:15 candle high/low
    first_volume    : 9:15 candle volume
    avg_volume_20d  : 20-day average daily volume (or avg 9:15-candle volume)
    vix             : current India VIX (None → check skipped, assumed OK)
    """
    gap_pct  = abs(open_price - prev_close) / prev_close if prev_close > 0 else 0.0
    gap_ok   = gap_pct >= gap_min_pct

    spot         = open_price
    candle_range = (first_high - first_low) / spot if spot > 0 else 0.0
    range_ok     = candle_range >= range_min_pct

    vix_ok = (vix is None) or (vix_low <= vix <= vix_high)

    volume_ok = (
        avg_volume_20d <= 0
        or first_volume >= avg_volume_20d * volume_mult
    )

    checks = [gap_ok, range_ok, vix_ok, volume_ok]
    score  = sum(checks)

    if score >= 2:
        label = _LABEL_MAP.get(score, "CAUTIOUS")
        mult  = _MULT_MAP.get(score, 0.5)
    else:
        label = "SKIP"
        mult  = 0.0

    logger.debug(
        "DayQuality score=%d/4 [gap=%s range=%s vix=%s vol=%s] → %s",
        score, gap_ok, range_ok, vix_ok, volume_ok, label,
    )
    return DayQuality(
        label=label, score=score,
        gap_ok=gap_ok, range_ok=range_ok, vix_ok=vix_ok, volume_ok=volume_ok,
        lot_multiplier=mult,
    )
