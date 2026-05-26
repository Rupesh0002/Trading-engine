"""
Opening Range Breakout (ORB) strategy.

Opening range = high/low of the first 15-min candle (9:15–9:30 IST).
A breakout is confirmed when a subsequent candle CLOSES beyond the OR
boundary with a small buffer. Fires at most once per index per day,
only between SIGNAL_START and ORB_TRADE_END (default 13:00).

Complements the main 5-condition engine by firing 3–5× more often on
trending days where the main signal never gets all 5 conditions aligned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ORBResult:
    direction: str        # "CALL", "PUT", or "WAIT"
    or_high: float
    or_low: float
    or_range_pct: float
    breakout_price: float
    conditions_met: int = 3   # fixed — maps to ATM strike in _enter_trade

    def __bool__(self) -> bool:
        return self.direction in ("CALL", "PUT")


_WAIT = ORBResult("WAIT", 0.0, 0.0, 0.0, 0.0)


class ORBEngine:
    """
    Stateful per-day ORB engine.
    Instantiate once; call reset_day() at the start of each new trading day.
    """

    def __init__(
        self,
        min_range_pct: float = 0.003,
        max_range_pct: float = 0.020,
        buffer_pct: float    = 0.001,
        trade_end: str       = "13:00",
    ) -> None:
        self.min_range_pct = min_range_pct
        self.max_range_pct = max_range_pct
        self.buffer_pct    = buffer_pct
        _h, _m             = trade_end.split(":")
        self._end_minutes  = int(_h) * 60 + int(_m)

        self._ranges: Dict[str, Dict] = {}   # index → {high, low, range_pct}
        self._traded: set             = set()  # indices that already fired ORB today

    # ── Public API ─────────────────────────────────────────────────────────

    def compute_opening_range(self, index: str, df: pd.DataFrame) -> bool:
        """
        Extract OR from the 9:15 candle in df. Idempotent — safe to call every cycle.
        Returns True if the range is available (either just computed or already stored).
        """
        if index in self._ranges:
            return True
        if df is None or df.empty:
            return False

        for _, row in df.iterrows():
            t = pd.Timestamp(row["date"])
            if t.hour == 9 and t.minute == 15:
                high      = float(row["high"])
                low       = float(row["low"])
                rng_pct   = (high - low) / low if low > 0 else 0.0
                self._ranges[index] = {"high": high, "low": low, "range_pct": rng_pct}
                logger.info(
                    "[ORB] [%s] Opening range: H=%.2f  L=%.2f  range=%.2f%%",
                    index, high, low, rng_pct * 100,
                )
                return True

        return False  # 9:15 candle not yet in df (engine started before 9:30)

    def is_range_computed(self, index: str) -> bool:
        return index in self._ranges

    def evaluate(self, index: str, df: pd.DataFrame, now_ist) -> ORBResult:
        """
        Check whether the last CLOSED candle breaks out of the opening range.
        Returns ORBResult with direction="CALL"/"PUT" on breakout, else "WAIT".
        """
        if index in self._traded:
            return _WAIT
        if now_ist.hour * 60 + now_ist.minute >= self._end_minutes:
            return _WAIT
        if index not in self._ranges:
            return _WAIT

        orng      = self._ranges[index]
        range_pct = orng["range_pct"]

        if range_pct < self.min_range_pct:
            logger.debug("[ORB] [%s] Range %.2f%% too tight — skip.", index, range_pct * 100)
            return _WAIT
        if range_pct > self.max_range_pct:
            logger.debug("[ORB] [%s] Range %.2f%% too wide (gap day) — skip.", index, range_pct * 100)
            return _WAIT

        if df is None or len(df) < 2:
            return _WAIT

        close    = float(df.iloc[-2]["close"])
        or_high  = orng["high"]
        or_low   = orng["low"]

        if close > or_high * (1 + self.buffer_pct):
            logger.info(
                "[ORB] [%s] CALL breakout | close=%.2f > OR_H=%.2f (buf=%.2f)",
                index, close, or_high, or_high * (1 + self.buffer_pct),
            )
            return ORBResult("CALL", or_high, or_low, range_pct, close)

        if close < or_low * (1 - self.buffer_pct):
            logger.info(
                "[ORB] [%s] PUT breakout  | close=%.2f < OR_L=%.2f (buf=%.2f)",
                index, close, or_low, or_low * (1 - self.buffer_pct),
            )
            return ORBResult("PUT", or_high, or_low, range_pct, close)

        return _WAIT

    def mark_traded(self, index: str) -> None:
        self._traded.add(index)

    def reset_day(self) -> None:
        self._ranges.clear()
        self._traded.clear()

    # ── State persistence ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {"ranges": self._ranges, "traded": list(self._traded)}

    def load_state(self, d: dict) -> None:
        self._ranges = d.get("ranges", {})
        self._traded = set(d.get("traded", []))
