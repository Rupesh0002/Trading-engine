"""
Scale-In Entry Manager — partial entries at confirmation / +3% / +6%.

Entry plan (3 tranches):
  Entry 1: 40% of planned lots at momentum confirmation
  Entry 2: 40% of planned lots at +3% premium gain (price follow-through)
  Entry 3: 20% of planned lots at +6% premium gain (trend continuation)

Rule: If premium falls after Entry 1 (no follow-through), stay at 40% only.
      Entry 2 and 3 require premium to be ABOVE Entry 1 premium.

Usage:
    scaler = ScaleInManager(total_planned_lots=4, entry1_premium=150.0)
    lots, avg_cost = scaler.next_entry(current_premium=155.0)  # Entry 2?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_E1_FRAC     = 0.40   # 40% at confirmation
_E2_FRAC     = 0.40   # 40% at +3%
_E3_FRAC     = 0.20   # 20% at +6%

_E2_GAIN_PCT = 0.03   # +3% above Entry 1 premium
_E3_GAIN_PCT = 0.06   # +6% above Entry 1 premium


@dataclass
class ScaleInEntry:
    lots: int
    tranche: int           # 1, 2, or 3
    premium: float         # premium at which this entry was filled
    avg_entry_premium: float  # blended average after this fill


@dataclass
class ScaleInManager:
    total_planned_lots: int
    entry1_premium: float

    # internal
    e1_done: bool  = False
    e2_done: bool  = False
    e3_done: bool  = False
    lots_entered: int         = 0
    total_cost: float         = 0.0   # sum of (lots × premium)

    @property
    def avg_entry_premium(self) -> float:
        return self.total_cost / self.lots_entered if self.lots_entered > 0 else 0.0

    def _lots_for(self, frac: float) -> int:
        raw = round(self.total_planned_lots * frac)
        return max(1, min(raw, self.total_planned_lots - self.lots_entered))

    def next_entry(self, current_premium: float) -> Optional[ScaleInEntry]:
        """
        Call each candle after Entry 1 to check if a new tranche should be added.
        Returns ScaleInEntry if a new tranche fires, else None.

        Entry 1 should be triggered by the caller directly (not this method).
        Call `record_entry1()` first, then use `next_entry()` for subsequent candles.
        """
        if not self.e1_done:
            return None   # Entry 1 not yet recorded — nothing to scale into

        gain_vs_e1 = (current_premium - self.entry1_premium) / self.entry1_premium

        # If premium fell below Entry 1 → hold 40% only, no further entries
        if current_premium < self.entry1_premium:
            return None

        # Entry 2
        if not self.e2_done and gain_vs_e1 >= _E2_GAIN_PCT:
            self.e2_done = True
            lots = self._lots_for(_E2_FRAC)
            self.lots_entered += lots
            self.total_cost   += lots * current_premium
            logger.info("Scale-In E2: +%d lots @ %.2f (avg %.2f)", lots, current_premium, self.avg_entry_premium)
            return ScaleInEntry(lots=lots, tranche=2, premium=current_premium, avg_entry_premium=self.avg_entry_premium)

        # Entry 3
        if not self.e3_done and self.e2_done and gain_vs_e1 >= _E3_GAIN_PCT:
            self.e3_done = True
            lots = self._lots_for(_E3_FRAC)
            self.lots_entered += lots
            self.total_cost   += lots * current_premium
            logger.info("Scale-In E3: +%d lots @ %.2f (avg %.2f)", lots, current_premium, self.avg_entry_premium)
            return ScaleInEntry(lots=lots, tranche=3, premium=current_premium, avg_entry_premium=self.avg_entry_premium)

        return None

    def record_entry1(self, lots: int, premium: float) -> ScaleInEntry:
        """Record the initial 40% Entry 1 fill."""
        self.e1_done       = True
        self.lots_entered  = lots
        self.total_cost    = lots * premium
        logger.info("Scale-In E1: %d lots @ %.2f", lots, premium)
        return ScaleInEntry(lots=lots, tranche=1, premium=premium, avg_entry_premium=premium)

    @property
    def is_complete(self) -> bool:
        """True when all 3 tranches have been entered."""
        return self.e1_done and self.e2_done and self.e3_done

    @property
    def active_lots(self) -> int:
        return self.lots_entered
