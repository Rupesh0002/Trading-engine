"""
Option Premium Momentum — checks if option LTP is accelerating before entry.

Live mode  : collects 3 real LTP ticks spaced 10 s apart via Kite quote.
Backtest   : accepts pre-computed tick series from the simulated premium array.

Decision rule:
  premium_momentum = (tick3 - tick1) / tick1 × 100
  > +0.5% → ENTER  (premium rising)
  < 0.0%  → CANCEL (premium falling)
  else    → WAIT   (neutral, try again next candle)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_ENTER_THRESHOLD  =  0.5   # % change → enter
_CANCEL_THRESHOLD =  0.0   # % change → cancel
_TICK_INTERVAL_S  = 10     # seconds between ticks (live only)
_NUM_TICKS        =  3


@dataclass(frozen=True)
class PremiumMomentumResult:
    decision: str            # "ENTER" | "CANCEL" | "WAIT"
    momentum_pct: float      # (tick3 - tick1) / tick1 × 100
    ticks: List[float]       # raw tick prices


def check_premium_momentum(
    ticks: List[float],
    *,
    enter_threshold: float  = _ENTER_THRESHOLD,
    cancel_threshold: float = _CANCEL_THRESHOLD,
) -> PremiumMomentumResult:
    """
    Evaluate 3-tick premium momentum.

    Parameters
    ----------
    ticks : list of 3 option LTP prices (oldest → newest)
    """
    if len(ticks) < 2:
        return PremiumMomentumResult(decision="WAIT", momentum_pct=0.0, ticks=ticks)

    t1, t3 = ticks[0], ticks[-1]
    momentum_pct = (t3 - t1) / t1 * 100.0 if t1 > 0 else 0.0

    if momentum_pct > enter_threshold:
        decision = "ENTER"
    elif momentum_pct < cancel_threshold:
        decision = "CANCEL"
    else:
        decision = "WAIT"

    logger.debug(
        "PremiumMomentum ticks=%s → %.2f%% → %s",
        [round(t, 2) for t in ticks], momentum_pct, decision,
    )
    return PremiumMomentumResult(decision=decision, momentum_pct=momentum_pct, ticks=list(ticks))


def collect_ticks_live(
    kite,
    symbol_key: str,
    num_ticks: int = _NUM_TICKS,
    interval_s: int = _TICK_INTERVAL_S,
) -> List[float]:
    """
    Fetch `num_ticks` option LTP ticks from Kite, spaced `interval_s` seconds apart.
    Returns list of floats (may be shorter than num_ticks on error).
    """
    ticks: List[float] = []
    for i in range(num_ticks):
        if i > 0:
            time.sleep(interval_s)
        try:
            quote = kite.quote([symbol_key])
            ltp   = float((quote.get(symbol_key) or {}).get("last_price", 0))
            if ltp > 0:
                ticks.append(ltp)
        except Exception as exc:
            logger.debug("Tick fetch failed [%s]: %s", symbol_key, exc)
    return ticks


def is_premium_rising(ticks: List[float]) -> bool:
    """Convenience: True if the last tick is above the first tick."""
    if len(ticks) < 2:
        return True   # assume OK when no data (backtest)
    return ticks[-1] > ticks[0]
