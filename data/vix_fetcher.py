"""
Live VIX fetcher — thin wrapper over data/feed.py + data/iv_fetcher.py.
Returns current India VIX and IV Rank in one call.
Used by signals/day_quality.py for the VIX 13-22 range check.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_VIX_LOW  = 13.0
_VIX_HIGH = 22.0


def get_vix(kite=None) -> Optional[float]:
    """Fetch current India VIX. Returns None on failure."""
    if kite is None:
        return None
    try:
        quote = kite.quote("NSE:INDIA VIX")
        return float(quote["NSE:INDIA VIX"]["last_price"])
    except Exception as exc:
        logger.debug("VIX fetch failed: %s", exc)
        return None


def is_vix_in_range(vix: Optional[float]) -> bool:
    """True if VIX is in the 13-22 tradeable range."""
    if vix is None:
        return True  # assume tradeable if data unavailable
    return _VIX_LOW <= vix <= _VIX_HIGH


def get_vix_and_rank(kite=None) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (vix, iv_rank).
    iv_rank uses IVFetcher's rolling 52-week range (or synthetic in backtest).
    """
    from data.iv_fetcher import IVFetcher
    fetcher  = IVFetcher(kite)
    vix      = get_vix(kite)
    iv_rank  = fetcher.get_iv_rank() if kite else None
    return vix, iv_rank
