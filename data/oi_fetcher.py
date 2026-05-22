"""
OI change fetcher — tracks ATM CE and PE open interest per candle.
Computes percentage OI change since the previous call to detect buildup / unwinding.

Used by options_scorer.score_options() for the OI buildup component (0-10 pts).
Returns (None, None) if Kite data is unavailable (backtest always returns None).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Tuple

from config.settings import INDEX_CONFIG

logger = logging.getLogger(__name__)


class OIFetcher:
    """
    Fetches ATM CE and PE OI from Kite quotes each candle; computes candle-over-candle change.

    Usage (live mode):
        fetcher = OIFetcher(kite)
        ce_chg, pe_chg = fetcher.get_oi_changes(spot=24500, index="NIFTY", expiry=expiry_date)
    """

    def __init__(self, kite=None) -> None:
        self.kite        = kite
        self._prev_ce_oi: dict[str, float] = {}
        self._prev_pe_oi: dict[str, float] = {}

    def get_oi_changes(
        self,
        spot: float,
        index: str = "NIFTY",
        expiry: Optional[date] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (ce_oi_change_pct, pe_oi_change_pct) vs previous call.
        Positive = OI increasing; negative = OI decreasing.
        Returns (None, None) when data is unavailable (backtest / auth error).
        """
        if self.kite is None:
            return None, None

        cfg        = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])
        fno_ex     = cfg["fno_exchange"]
        step       = cfg["strike_step"]
        atm_strike = round(spot / step) * step

        if expiry is None:
            from config.events_calendar import get_next_expiry
            expiry = get_next_expiry(index, date.today())

        yy  = expiry.strftime("%y")
        mon = expiry.strftime("%b").upper()
        day = expiry.strftime("%d")
        ce_sym = f"{index}{yy}{mon}{day}{atm_strike}CE"
        pe_sym = f"{index}{yy}{mon}{day}{atm_strike}PE"
        ce_key = f"{fno_ex}:{ce_sym}"
        pe_key = f"{fno_ex}:{pe_sym}"

        try:
            quotes = self.kite.quote([ce_key, pe_key])
            ce_oi  = float((quotes.get(ce_key) or {}).get("oi", 0))
            pe_oi  = float((quotes.get(pe_key) or {}).get("oi", 0))
        except Exception as exc:
            logger.debug("OI fetch failed [%s]: %s", index, exc)
            return None, None

        prev_ce = self._prev_ce_oi.get(index, 0)
        prev_pe = self._prev_pe_oi.get(index, 0)

        ce_change: Optional[float] = (ce_oi - prev_ce) / prev_ce * 100.0 if prev_ce > 0 else None
        pe_change: Optional[float] = (pe_oi - prev_pe) / prev_pe * 100.0 if prev_pe > 0 else None

        self._prev_ce_oi[index] = ce_oi
        self._prev_pe_oi[index] = pe_oi

        return ce_change, pe_change
