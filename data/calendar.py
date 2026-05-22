"""
Unified calendar interface for day-quality checks and setup detection.
Thin wrapper over config/events_calendar.py with F&O expiry helpers.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from config.events_calendar import (
    ALL_HOLIDAYS,
    HIGH_IMPACT_EVENTS,
    RBI_MPC_MONTHS,
    get_event_name,
    get_next_expiry,
    is_event_day,
    is_expiry_day,
    is_market_holiday,
)

__all__ = [
    "is_high_impact_event",
    "get_event_label",
    "is_rbi_month",
    "is_expiry_week",
    "is_trading_day",
    "days_to_expiry",
    # re-exports
    "is_event_day",
    "is_expiry_day",
    "is_market_holiday",
    "get_next_expiry",
]


def is_high_impact_event(dt: Optional[date] = None) -> bool:
    """True if dt is a pre-listed high-impact event (Budget, Election results, etc.)."""
    return is_event_day(dt)


def get_event_label(dt: Optional[date] = None) -> Optional[str]:
    """Human-readable event name or None."""
    return get_event_name(dt)


def is_rbi_month(dt: Optional[date] = None) -> bool:
    """True if RBI MPC typically meets this month (Feb/Apr/Jun/Aug/Oct/Dec)."""
    dt = dt or date.today()
    return dt.month in RBI_MPC_MONTHS


def is_expiry_week(index: str = "NIFTY", dt: Optional[date] = None) -> bool:
    """True if the weekly expiry falls within the same Mon-Fri week as dt."""
    dt = dt or date.today()
    expiry = get_next_expiry(index, dt, days_buffer=0)
    # Same ISO week → same Mon-Sun block
    return expiry.isocalendar()[1] == dt.isocalendar()[1]


def is_trading_day(dt: Optional[date] = None) -> bool:
    """True if the market is open (not a weekend or NSE holiday)."""
    return not is_market_holiday(dt)


def days_to_expiry(index: str = "NIFTY", dt: Optional[date] = None) -> int:
    """Calendar days from dt to the next suitable expiry (post buffer)."""
    dt = dt or date.today()
    expiry = get_next_expiry(index, dt)
    return (expiry - dt).days
