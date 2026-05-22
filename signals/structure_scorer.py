"""
Price structure scorer — PDH/PDL breakout, weekly H/L, round numbers.
Maximum 25 points:  PDH/PDL (0-10) + Weekly H/L (0-8) + Round numbers (0-7).

No hard blocks — structure is additive context, not a gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StructureScore:
    pdh_pdl_score: int            # 0-10
    weekly_score:  int            # 0-8
    round_score:   int            # 0-7
    total:         int            # 0-25
    pdh:           float
    pdl:           float
    weekly_high:   float
    weekly_low:    float
    near_round:    Optional[float]
    direction:     str


# Round number step sizes per index (major psychological levels)
_ROUND_STEPS: dict[str, int] = {
    "NIFTY":     500,    # 24000, 24500, 25000 ...
    "BANKNIFTY": 1000,   # 50000, 51000, 52000 ...
    "SENSEX":    1000,   # 79000, 80000, 81000 ...
}


def score_structure(
    close: float,
    direction: str,
    pdh: float,
    pdl: float,
    weekly_high: float,
    weekly_low: float,
    index: str = "NIFTY",
) -> StructureScore:
    """
    Score price structure for the given direction.

    close       : current close price
    pdh / pdl   : previous day high / low
    weekly_high : current week's high so far (Mon → today)
    weekly_low  : current week's low so far
    index       : used to pick the round-number step size
    """
    _PROX   = 0.002   # 0.20% — PDH/PDL proximity threshold
    _BOUNCE = 0.005   # 0.50% — PDL bounce zone (above PDL)
    _WPROX  = 0.003   # 0.30% — weekly H/L proximity

    # ── PDH/PDL ───────────────────────────────────────────────────────────────
    # Require a meaningful previous-day range (≥ 0.5% of price) before using levels.
    daily_range_ok = pdh > 0 and pdl > 0 and (pdh - pdl) / close >= 0.005

    if daily_range_ok:
        if direction == "CALL":
            if close > pdh:
                pdh_pdl_score = 10   # clear breakout above PDH
            elif abs(close - pdh) / close < _PROX:
                pdh_pdl_score = 7    # testing PDH (breakout attempt)
            elif close > pdl and abs(close - pdl) / close < _BOUNCE:
                pdh_pdl_score = 4    # price bounced from PDL support
            else:
                pdh_pdl_score = 0
        else:  # PUT
            if close < pdl:
                pdh_pdl_score = 10   # clear breakdown below PDL
            elif abs(close - pdl) / close < _PROX:
                pdh_pdl_score = 7    # testing PDL (breakdown attempt)
            elif close < pdh and abs(close - pdh) / close < _BOUNCE:
                pdh_pdl_score = 4    # price rejected from PDH resistance
            else:
                pdh_pdl_score = 0
    else:
        pdh_pdl_score = 0

    # ── Weekly H/L ────────────────────────────────────────────────────────────
    weekly_mid = (weekly_high + weekly_low) / 2.0 if weekly_high > 0 and weekly_low > 0 else 0.0

    if weekly_high > 0 and weekly_low > 0:
        if direction == "CALL":
            if close > weekly_high:
                weekly_score = 8     # breaking above week's high — very bullish
            elif abs(close - weekly_high) / close < _WPROX:
                weekly_score = 5     # testing weekly high
            elif close > weekly_mid:
                weekly_score = 2     # above weekly midpoint (mild bullish)
            else:
                weekly_score = 0
        else:  # PUT
            if close < weekly_low:
                weekly_score = 8     # breaking below week's low — very bearish
            elif abs(close - weekly_low) / close < _WPROX:
                weekly_score = 5     # testing weekly low
            elif close < weekly_mid:
                weekly_score = 2     # below weekly midpoint (mild bearish)
            else:
                weekly_score = 0
    else:
        weekly_score = 2   # neutral default when weekly data unavailable

    # ── Round numbers ─────────────────────────────────────────────────────────
    # Price near a major round number adds psychological support/resistance context.
    step    = _ROUND_STEPS.get(index, 500)
    rounded = round(close / step) * step
    dist_pct = abs(close - rounded) / close * 100.0

    if dist_pct < 0.1:
        round_score = 7
        near_round: Optional[float] = float(rounded)
    elif dist_pct < 0.2:
        round_score = 4
        near_round = float(rounded)
    else:
        round_score = 0
        near_round = None

    return StructureScore(
        pdh_pdl_score=pdh_pdl_score,
        weekly_score=weekly_score,
        round_score=round_score,
        total=pdh_pdl_score + weekly_score + round_score,
        pdh=round(pdh, 2),
        pdl=round(pdl, 2),
        weekly_high=round(weekly_high, 2),
        weekly_low=round(weekly_low, 2),
        near_round=near_round,
        direction=direction,
    )
