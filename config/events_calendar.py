"""
NSE market holidays, F&O expiry dates, and high-impact event days.
Used to pause or reduce trading on volatile event days.
No hardcoded parameters — only dates and event metadata here.
"""
from datetime import date, timedelta
from typing import Optional


# NSE market holidays (update each year)
MARKET_HOLIDAYS_2024 = {
    date(2024, 1, 22),   # Ram Mandir Pran Pratishtha (special)
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 14),   # Dr. Ambedkar Jayanti
    date(2024, 4, 17),   # Ram Navami
    date(2024, 4, 21),   # Mahavir Jayanti
    date(2024, 5, 23),   # Buddha Purnima
    date(2024, 6, 17),   # Eid ul-Adha (Bakri Id)
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Mahatma Gandhi Jayanti
    date(2024, 11, 1),   # Diwali Laxmi Pujan
    date(2024, 11, 15),  # Gurunanak Jayanti
    date(2024, 12, 25),  # Christmas
}

MARKET_HOLIDAYS_2025 = {
    date(2025, 2, 26),   # Maha Shivaratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramzan Id)
    date(2025, 4, 10),   # Shri Ram Navami
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 20),  # Diwali Laxmi Pujan
    date(2025, 10, 21),  # Diwali-Balipratipada
    date(2025, 11, 5),   # Prakash Gurpurb Sri Guru Nanak Dev Ji
    date(2025, 12, 25),  # Christmas
}

ALL_HOLIDAYS = MARKET_HOLIDAYS_2024 | MARKET_HOLIDAYS_2025

# High-impact event dates — strategy reduces position size or skips
HIGH_IMPACT_EVENTS = {
    date(2024, 2, 1):  "Union Budget 2024",
    date(2024, 6, 4):  "Lok Sabha Election Results",
    date(2025, 2, 1):  "Union Budget 2025",
}

# RBI MPC meeting months (policy decision ~last week of Feb/Apr/Jun/Aug/Oct/Dec)
RBI_MPC_MONTHS = {2, 4, 6, 8, 10, 12}


def is_market_holiday(dt: Optional[date] = None) -> bool:
    dt = dt or date.today()
    return dt in ALL_HOLIDAYS or dt.weekday() >= 5


def is_event_day(dt: Optional[date] = None) -> bool:
    dt = dt or date.today()
    return dt in HIGH_IMPACT_EVENTS


def get_event_name(dt: Optional[date] = None) -> Optional[str]:
    dt = dt or date.today()
    return HIGH_IMPACT_EVENTS.get(dt)


def get_nifty_weekly_expiry(dt: Optional[date] = None) -> date:
    """Returns the nearest Thursday (Nifty weekly expiry) on or after dt."""
    dt = dt or date.today()
    days_ahead = (3 - dt.weekday()) % 7  # 3 = Thursday
    expiry = dt + timedelta(days=days_ahead)
    while expiry in ALL_HOLIDAYS:
        expiry -= timedelta(days=1)  # move to Wednesday if Thursday is holiday
    return expiry


def get_banknifty_weekly_expiry(dt: Optional[date] = None) -> date:
    """Returns the nearest Wednesday (BankNifty weekly expiry) on or after dt."""
    dt = dt or date.today()
    days_ahead = (2 - dt.weekday()) % 7  # 2 = Wednesday
    expiry = dt + timedelta(days=days_ahead)
    while expiry in ALL_HOLIDAYS:
        expiry -= timedelta(days=1)
    return expiry


def get_sensex_weekly_expiry(dt: Optional[date] = None) -> date:
    """Returns the nearest Friday (BSE Sensex weekly expiry) on or after dt."""
    dt = dt or date.today()
    days_ahead = (4 - dt.weekday()) % 7  # 4 = Friday
    if days_ahead == 0:
        days_ahead = 7  # if today is Friday, move to next Friday
    expiry = dt + timedelta(days=days_ahead)
    while expiry in ALL_HOLIDAYS:
        expiry -= timedelta(days=1)  # move to Thursday if Friday is holiday
    return expiry


def get_next_expiry(
    index: str = "NIFTY",
    dt: Optional[date] = None,
    days_buffer: int = 2,
) -> date:
    """
    Returns the safest expiry for options entry — avoids theta-decay risk.

    Logic:
      - Finds the nearest weekly expiry for the index.
      - If that expiry is within `days_buffer` calendar days (default 2),
        rolls forward to the FOLLOWING week's expiry instead.
      - This prevents entering short-DTE options where theta decay rapidly
        erodes premium and a 20% SL can trigger on time value alone.

    NIFTY    → Thursday expiry
    BANKNIFTY → Wednesday expiry
    SENSEX   → Friday expiry
    """
    dt = dt or date.today()

    def _nearest(d: date) -> date:
        if index == "BANKNIFTY":
            return get_banknifty_weekly_expiry(d)
        if index == "SENSEX":
            return get_sensex_weekly_expiry(d)
        return get_nifty_weekly_expiry(d)

    nearest = _nearest(dt)
    dte = (nearest - dt).days

    if dte <= days_buffer:
        # Roll past this expiry to next week's
        roll_from = nearest + timedelta(days=1)
        return _nearest(roll_from)

    return nearest


def is_expiry_day(index: str = "NIFTY", dt: Optional[date] = None) -> bool:
    dt = dt or date.today()
    if index == "BANKNIFTY":
        return dt == get_banknifty_weekly_expiry(dt)
    if index == "SENSEX":
        return dt == get_sensex_weekly_expiry(dt)
    return dt == get_nifty_weekly_expiry(dt)  # NIFTY default
