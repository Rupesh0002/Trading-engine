"""
Option chain data via Zerodha Kite Connect.
Supports NIFTY (NFO), BANKNIFTY (NFO), and SENSEX (BFO — BSE F&O).
All parameters from config/settings.py → .env.

Exchange mapping:
  NIFTY / BANKNIFTY → F&O on NFO (NSE Futures & Options)
  SENSEX            → F&O on BFO (BSE Futures & Options)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from config.settings import (
    INDEX_CONFIG,
    MAX_PREMIUM,
    MIN_PREMIUM,
    PCR_MAX,
    PCR_MIN,
)
from config.events_calendar import (
    get_banknifty_weekly_expiry,
    get_nifty_weekly_expiry,
    get_sensex_weekly_expiry,
)

logger = logging.getLogger(__name__)


class OptionChain:
    """Fetches option data and computes PCR for NIFTY, BANKNIFTY, and SENSEX."""

    def __init__(self, kite) -> None:
        self.kite = kite

    # ------------------------------------------------------------------
    # Strike helpers
    # ------------------------------------------------------------------

    def get_atm_strike(self, spot_price: float, index: str = "NIFTY") -> int:
        """Round spot to nearest valid strike step from .env via INDEX_CONFIG."""
        step = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["strike_step"]
        return round(spot_price / step) * step

    def get_otm_strike(
        self,
        spot_price: float,
        option_type: str,
        index: str = "NIFTY",
        strikes_away: int = 1,
    ) -> int:
        """
        Returns OTM strike strikes_away steps from ATM.
        CE: OTM is higher (e.g. ATM+50 for NIFTY).
        PE: OTM is lower  (e.g. ATM-50 for NIFTY).
        Works for both buy (buy OTM for leverage) and sell (sell OTM for safety margin).
        """
        step = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["strike_step"]
        atm  = self.get_atm_strike(spot_price, index)
        if option_type.upper() == "CE":
            return atm + strikes_away * step
        else:
            return atm - strikes_away * step

    def get_option_symbol(
        self,
        strike: int,
        option_type: str,
        index: str = "NIFTY",
        expiry: Optional[date] = None,
    ) -> str:
        """
        Builds the Kite tradingsymbol.
        NSE format: NIFTY26MAY2420000CE
        BSE format: SENSEX26MAY2482000CE  (same pattern, different exchange)
        """
        if expiry is None:
            expiry = self._nearest_expiry(index)
        yy  = expiry.strftime("%y")
        mon = expiry.strftime("%b").upper()
        day = expiry.strftime("%d")
        return f"{index}{yy}{mon}{day}{strike}{option_type.upper()}"

    # ------------------------------------------------------------------
    # LTP and quote — uses the correct F&O exchange per index
    # ------------------------------------------------------------------

    def get_ltp(self, tradingsymbol: str, index: str = "NIFTY") -> Optional[float]:
        fno_exchange = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["fno_exchange"]
        try:
            key   = f"{fno_exchange}:{tradingsymbol}"
            quote = self.kite.quote(key)
            return float(quote[key]["last_price"])
        except Exception as exc:
            logger.warning("LTP error [%s/%s]: %s", index, tradingsymbol, exc)
            return None

    def get_full_quote(self, tradingsymbol: str, index: str = "NIFTY") -> Optional[dict]:
        fno_exchange = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["fno_exchange"]
        try:
            key = f"{fno_exchange}:{tradingsymbol}"
            return self.kite.quote(key)[key]
        except Exception as exc:
            logger.warning("Quote error [%s/%s]: %s", index, tradingsymbol, exc)
            return None

    # ------------------------------------------------------------------
    # PCR — uses the correct F&O exchange per index
    # ------------------------------------------------------------------

    def get_pcr(
        self,
        spot_price: float,
        index: str = "NIFTY",
        expiry: Optional[date] = None,
        strikes_range: int = 5,
    ) -> Optional[float]:
        """PCR = total put OI / total call OI for ATM ± strikes_range strikes."""
        if expiry is None:
            expiry = self._nearest_expiry(index)

        cfg     = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])
        step    = cfg["strike_step"]
        fno_ex  = cfg["fno_exchange"]
        atm     = self.get_atm_strike(spot_price, index)
        strikes = [atm + i * step for i in range(-strikes_range, strikes_range + 1)]

        keys: dict[str, str] = {}
        for s in strikes:
            ce_sym = self.get_option_symbol(s, "CE", index, expiry)
            pe_sym = self.get_option_symbol(s, "PE", index, expiry)
            keys[f"{fno_ex}:{ce_sym}"] = "CE"
            keys[f"{fno_ex}:{pe_sym}"] = "PE"

        try:
            quotes = self.kite.quote(list(keys.keys()))
        except Exception as exc:
            logger.warning("[%s] PCR fetch error: %s", index, exc)
            return None

        call_oi = sum(
            (quotes[k].get("oi") or 0)
            for k, t in keys.items() if t == "CE" and k in quotes
        )
        put_oi = sum(
            (quotes[k].get("oi") or 0)
            for k, t in keys.items() if t == "PE" and k in quotes
        )

        if call_oi == 0:
            return None

        pcr = put_oi / call_oi
        logger.debug("[%s] PCR=%.2f (P=%d C=%d) on %s", index, pcr, put_oi, call_oi, fno_ex)
        return pcr

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_pcr_valid(self, pcr: Optional[float]) -> bool:
        return pcr is not None and PCR_MIN <= pcr <= PCR_MAX

    def is_premium_valid(self, premium: float) -> bool:
        return MIN_PREMIUM <= premium <= MAX_PREMIUM

    # ------------------------------------------------------------------
    # Expiry — per index
    # ------------------------------------------------------------------

    def _nearest_expiry(self, index: str) -> date:
        expiry_day = INDEX_CONFIG.get(index, {}).get("expiry_day", "thursday")
        if expiry_day == "wednesday":
            return get_banknifty_weekly_expiry()
        elif expiry_day == "friday":
            return get_sensex_weekly_expiry()
        else:
            return get_nifty_weekly_expiry()   # Thursday default
