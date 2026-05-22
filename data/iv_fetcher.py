"""
IV Rank fetcher — computes India VIX IV Rank from a 52-week range.

IV Rank = (vix_current - vix_52w_low) / (vix_52w_high - vix_52w_low) × 100

Live mode  : fetches current VIX from Kite; persists 52w range in state_vix.json.
Backtest   : uses BACKTEST_VIX with a synthetic ±40% range → IV Rank ≈ 37–50.

Hard block : IV Rank > IV_RANK_MAX (default 80 from .env).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from config.settings import BACKTEST_VIX, IV_RANK_MAX

logger = logging.getLogger(__name__)

_VIX_RANGE_FILE = "state_vix.json"   # persists 52-week range across sessions


def compute_iv_rank(vix: float, vix_52w_high: float, vix_52w_low: float) -> float:
    """
    Returns IV Rank 0-100.  Returns 30.0 (neutral mid-range) on invalid inputs.
    """
    spread = vix_52w_high - vix_52w_low
    if spread <= 0 or vix_52w_high <= 0:
        return 30.0
    rank = (vix - vix_52w_low) / spread * 100.0
    return max(0.0, min(100.0, round(rank, 1)))


def get_backtest_iv_rank(vix: Optional[float] = None) -> float:
    """
    Estimate IV Rank for backtest use.
    Assumes ±40% range around BACKTEST_VIX:
      e.g. VIX=15 → low=9, high=21 → IV Rank = (15-9)/(21-9) × 100 = 50
    Returns a value typically in the 40-55% range — sweet spot for buying options.
    """
    v            = vix if vix is not None else BACKTEST_VIX
    vix_52w_low  = v * 0.60
    vix_52w_high = v * 1.40
    return compute_iv_rank(v, vix_52w_high, vix_52w_low)


class IVFetcher:
    """Fetches India VIX via Kite and maintains a rolling 52-week range."""

    def __init__(self, kite=None) -> None:
        self.kite         = kite
        self._vix_52w_high = 0.0
        self._vix_52w_low  = 0.0
        self._load_range()

    def _load_range(self) -> None:
        if os.path.exists(_VIX_RANGE_FILE):
            try:
                data = json.loads(open(_VIX_RANGE_FILE).read())
                self._vix_52w_high = float(data.get("vix_52w_high", 0))
                self._vix_52w_low  = float(data.get("vix_52w_low",  0))
            except Exception as exc:
                logger.debug("VIX range file unreadable: %s", exc)

    def _save_range(self, vix: float) -> None:
        if vix <= 0:
            return
        if self._vix_52w_high == 0 or vix > self._vix_52w_high:
            self._vix_52w_high = vix
        if self._vix_52w_low == 0 or vix < self._vix_52w_low:
            self._vix_52w_low = vix
        try:
            with open(_VIX_RANGE_FILE, "w") as f:
                json.dump({"vix_52w_high": self._vix_52w_high,
                           "vix_52w_low":  self._vix_52w_low}, f)
        except Exception as exc:
            logger.debug("Could not persist VIX range: %s", exc)

    def get_iv_rank(self) -> Optional[float]:
        """
        Fetch India VIX and return IV Rank.
        Returns None on fetch failure (caller treats as neutral = 30 pts).
        """
        if self.kite is None:
            return None
        try:
            quote = self.kite.quote("NSE:INDIA VIX")
            vix   = float(quote["NSE:INDIA VIX"]["last_price"])
        except Exception as exc:
            logger.debug("VIX fetch failed: %s", exc)
            return None

        self._save_range(vix)

        if self._vix_52w_high > self._vix_52w_low > 0:
            return compute_iv_rank(vix, self._vix_52w_high, self._vix_52w_low)

        # Insufficient history — use synthetic range until we accumulate data
        return get_backtest_iv_rank(vix)

    def is_hard_blocked(self, iv_rank: float) -> bool:
        return iv_rank > IV_RANK_MAX
