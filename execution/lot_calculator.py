"""
Lot size calculator — maps conviction score to number of lots.
Also enforces the daily risk cap (max 10% of capital per day across 2 trades).
"""
from __future__ import annotations

from config.settings import TRADING_CAPITAL

_LOT_BANDS = [
    (93, 5),
    (87, 4),
    (80, 3),
    (72, 2),
    (65, 1),
    (0,  0),
]

# Max daily loss as fraction of capital (2 trades × 5% = 10% hard cap)
_MAX_DAILY_RISK_PCT = 0.10


def conviction_to_lots(score: int) -> int:
    """Map conviction score (0-100) to number of lots (0-5)."""
    for threshold, lots in _LOT_BANDS:
        if score >= threshold:
            return lots
    return 0


def max_risk_for_lots(
    lots: int,
    lot_size: int,
    entry_premium: float,
    sl_pct: float,
) -> float:
    """Max rupee risk for a given lot count: lots × lot_size × premium × sl_pct."""
    return lots * lot_size * entry_premium * sl_pct


def adjust_lots_for_daily_risk(
    requested_lots: int,
    lot_size: int,
    entry_premium: float,
    sl_pct: float,
    daily_pnl: float,
    max_daily_risk_pct: float = _MAX_DAILY_RISK_PCT,
) -> int:
    """
    Reduce lots if adding this trade would breach the daily risk cap.
    daily_pnl     : realized P&L so far today (negative = loss)
    Returns adjusted lot count (0 if risk cap is already exhausted).
    """
    cap        = TRADING_CAPITAL * max_daily_risk_pct
    used       = max(0.0, -daily_pnl)
    remaining  = cap - used

    if remaining <= 0:
        return 0

    for n in range(requested_lots, 0, -1):
        if max_risk_for_lots(n, lot_size, entry_premium, sl_pct) <= remaining:
            return n

    return 0
