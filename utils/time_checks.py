"""
IST time helpers for trading session management.
All session boundary times come from config/settings.py → .env.
Never hardcode times in this file.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pytz

from config.settings import (
    CANDLE_MINUTES,
    MARKET_CLOSE,
    MARKET_OPEN,
    SIGNAL_START,
    TRADE_END,
    TRADE_START,
)

IST = pytz.timezone("Asia/Kolkata")


def _parse(t_str: str) -> time:
    h, m = t_str.split(":")
    return time(int(h), int(m))


# Parse once at import time from .env values — not hardcoded
MARKET_OPEN_TIME   = _parse(MARKET_OPEN)
TRADE_START_TIME   = _parse(TRADE_START)
SIGNAL_START_TIME  = _parse(SIGNAL_START)
TRADE_END_TIME     = _parse(TRADE_END)
MARKET_CLOSE_TIME  = _parse(MARKET_CLOSE)


def now_ist() -> datetime:
    return datetime.now(IST)


def current_time_ist() -> time:
    return now_ist().time().replace(tzinfo=None)


def is_market_hours() -> bool:
    t = current_time_ist()
    return MARKET_OPEN_TIME <= t <= MARKET_CLOSE_TIME


def is_trade_time() -> bool:
    """Within TRADE_START → TRADE_END (the active trading window)."""
    t = current_time_ist()
    return TRADE_START_TIME <= t < TRADE_END_TIME


def is_signal_time() -> bool:
    """Within SIGNAL_START → TRADE_END (after warm-up, before hard close)."""
    t = current_time_ist()
    return SIGNAL_START_TIME <= t < TRADE_END_TIME


def is_hard_close_time() -> bool:
    """At or past TRADE_END — close all positions immediately."""
    return current_time_ist() >= TRADE_END_TIME


def seconds_to_next_candle() -> int:
    """
    Returns seconds remaining until the next 15-min candle opens.
    Uses CANDLE_MINUTES from .env.
    """
    now = now_ist()
    minutes_past = now.minute % CANDLE_MINUTES
    seconds_past = minutes_past * 60 + now.second + now.microsecond / 1_000_000
    remaining = CANDLE_MINUTES * 60 - seconds_past
    return max(1, int(remaining))


def ist_now_str() -> str:
    return now_ist().strftime("%Y-%m-%d %H:%M:%S IST")
