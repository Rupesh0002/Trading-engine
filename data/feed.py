"""
Zerodha Kite Connect data feed — supports NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY.
All instrument tokens and parameters come from config/settings.py → .env.
Instrument tokens must be verified against Kite instruments CSV each expiry series.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import pytz

from config.settings import (
    ACTIVE_INDICES,
    BSE_EXCHANGE,
    CANDLE_INTERVAL,
    INDEX_CONFIG,
    MARKET_OPEN,
    NSE_EXCHANGE,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# India VIX token (NSE-defined, not user-configurable)
_INDIA_VIX_TOKEN = 264969
_INDIA_VIX_KEY   = f"{NSE_EXCHANGE}:INDIA VIX"


class DataFeed:
    """
    Fetches intraday and historical OHLCV candles from Kite for all active indices.
    Call get_today_candles(index) per candle cycle.
    """

    def __init__(self, kite) -> None:
        self.kite = kite

    # ------------------------------------------------------------------
    # Today's candles — from MARKET_OPEN IST to now
    # ------------------------------------------------------------------

    def get_today_candles(self, index: str = "NIFTY") -> Optional[pd.DataFrame]:
        """
        Returns 15-min bars with 5-day warmup so ADX/EMA/RSI indicators are
        properly initialised from the first candle of the day.
        VWAP is computed separately only on today's candles by compute_vwap().
        """
        cfg = INDEX_CONFIG.get(index)
        if cfg is None:
            logger.error("Unknown index: %s. Valid: %s", index, list(INDEX_CONFIG.keys()))
            return None

        now_ist = datetime.now(IST)
        from_dt = (now_ist - timedelta(days=5)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        return self._fetch(cfg["token"], from_dt, now_ist, label=index)

    def get_all_indices_candles(self) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetches today's candles for every index in ACTIVE_INDICES."""
        return {idx: self.get_today_candles(idx) for idx in ACTIVE_INDICES}

    # ------------------------------------------------------------------
    # Historical candles — for backtesting
    # ------------------------------------------------------------------

    def get_historical_candles(
        self,
        from_date: str,
        to_date: str,
        index: str = "NIFTY",
    ) -> Optional[pd.DataFrame]:
        """
        Returns 15-min bars between from_date and to_date.
        Kite limits one request to ~60 days; this method auto-chunks.
        """
        cfg = INDEX_CONFIG.get(index)
        if cfg is None:
            logger.error("Unknown index: %s", index)
            return None

        from_dt = IST.localize(datetime.strptime(from_date, "%Y-%m-%d"))
        to_dt   = IST.localize(datetime.strptime(to_date, "%Y-%m-%d"))

        chunks: list[pd.DataFrame] = []
        cursor = from_dt
        while cursor < to_dt:
            end   = min(cursor + timedelta(days=59), to_dt)
            chunk = self._fetch(cfg["token"], cursor, end, label=index)
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
            cursor = end + timedelta(days=1)

        if not chunks:
            return None

        df = pd.concat(chunks).drop_duplicates("timestamp").sort_values("timestamp")
        logger.info("Historical: %d bars for %s (%s → %s)", len(df), index, from_date, to_date)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Spot price
    # ------------------------------------------------------------------

    def get_spot_price(self, index: str = "NIFTY") -> Optional[float]:
        """
        Returns current spot price.
        Uses BSE exchange for SENSEX, NSE for NIFTY/BANKNIFTY.
        """
        cfg = INDEX_CONFIG.get(index)
        if cfg is None:
            return None
        try:
            exchange = cfg["spot_exchange"]   # NSE or BSE
            key      = f"{exchange}:{index}"
            quote    = self.kite.quote(key)
            return float(quote[key]["last_price"])
        except Exception as exc:
            logger.warning("Spot price error [%s]: %s", index, exc)
            return None

    # ------------------------------------------------------------------
    # India VIX
    # ------------------------------------------------------------------

    def get_vix(self) -> Optional[float]:
        try:
            quote = self.kite.quote(_INDIA_VIX_KEY)
            return float(quote[_INDIA_VIX_KEY]["last_price"])
        except Exception as exc:
            msg = str(exc)
            # Token errors are expected outside market hours — log once at DEBUG, not WARNING
            if "api_key" in msg or "access_token" in msg or "token" in msg.lower():
                logger.debug("VIX fetch skipped (token issue): %s", exc)
            else:
                logger.warning("VIX fetch error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(
        self,
        token: int,
        from_dt: datetime,
        to_dt: datetime,
        label: str = "",
    ) -> Optional[pd.DataFrame]:
        try:
            records = self.kite.historical_data(
                instrument_token=token,
                from_date=from_dt,
                to_date=to_dt,
                interval=CANDLE_INTERVAL,
            )
        except Exception as exc:
            logger.error("Kite historical_data [%s]: %s", label, exc)
            return None

        if not records:
            return None

        df = pd.DataFrame(records)
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(IST)

        return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
