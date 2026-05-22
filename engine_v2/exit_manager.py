"""
Dynamic exit manager — 6-state priority system on 5-min candles.

Exit priority (checked each 5-min candle):
  1. HARD_SL        → -12% from entry
  2. NO_MOVE        → < 6% move after 2 candles  (suspended once >8% seen)
  3. TRAIL_SL       → 15% below highest premium  (activates at +15%)
  4. TREND_CHECK    → at +36% TP, exit unless 4 trend conditions pass
  5. PARTIAL_1      → +50% gain → close 40% of lots
     PARTIAL_2      → +80% gain → close 30% of original lots
  6. LOSING_CLOSE   → 13:30 IST if in loss
     HARD_CLOSE     → 14:15 IST absolute cutoff
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    exit_all: bool
    lots:     int
    reason:   str
    premium:  float


class ExitManager:
    HARD_SL               = 0.12   # -12% from entry
    NO_MOVE_CANDLES       = 2      # check after 2 candles (10 min)
    NO_MOVE_THRESHOLD     = 0.06   # < 6% move → stale
    NO_MOVE_SUSPEND_PCT   = 0.08   # suspend NO_MOVE once >8% gain seen
    TRAIL_ACTIVATION      = 0.15   # trail arm at +15%
    TRAIL_DISTANCE        = 0.15   # trail sits 15% below highest premium
    TREND_CHECK_PCT       = 0.36   # at +36% — hold or exit decision
    PARTIAL1_PCT          = 0.50   # first partial at +50%
    PARTIAL1_LOTS_PCT     = 0.40   # close 40% of original lots
    PARTIAL2_PCT          = 0.80   # second partial at +80%
    PARTIAL2_LOTS_PCT     = 0.30   # close 30% of original lots
    LOSING_CLOSE_TIME     = time(13, 30)
    HARD_CLOSE_TIME       = time(14, 15)

    def __init__(self, entry_premium: float, lots: int, entry_time: time) -> None:
        self.entry          = entry_premium
        self.lots           = lots
        self.remaining_lots = lots
        self.entry_time     = entry_time

        self.highest         = entry_premium
        self.trailing_sl: Optional[float] = None
        self.trailing_active = False

        self.max_gain_seen  = 0.0   # highest gain % seen at any point
        self.candles_held   = 0

        self.trend_check_done = False   # True once +36% TP check has been evaluated
        self.partial1_done    = False
        self.partial2_done    = False

    def check(
        self,
        current_premium: float,
        current_time: time,
        df_recent: Optional[pd.DataFrame] = None,
    ) -> Optional[ExitSignal]:
        """
        Call every 5-min candle. Returns ExitSignal to act on, or None to hold.
        df_recent: last ~20 rows of 5-min spot data (for TREND_CHECK only).
        """
        self.candles_held += 1
        change_pct = (current_premium - self.entry) / self.entry

        if current_premium > self.highest:
            self.highest = current_premium
        if change_pct > self.max_gain_seen:
            self.max_gain_seen = change_pct

        # ── 1. HARD STOP LOSS ────────────────────────────────────────────────
        sl_floor = self.entry * (1 - self.HARD_SL)
        if current_premium <= sl_floor:
            return ExitSignal(
                exit_all=True, lots=self.remaining_lots,
                reason="HARD_SL", premium=sl_floor,
            )

        # ── 2. NO MOVEMENT ───────────────────────────────────────────────────
        # Suspended once the trade has seen >8% move at any point
        no_move_suspended = self.max_gain_seen >= self.NO_MOVE_SUSPEND_PCT
        if (
            self.candles_held >= self.NO_MOVE_CANDLES
            and not self.trailing_active
            and not no_move_suspended
        ):
            if abs(change_pct) < self.NO_MOVE_THRESHOLD:
                return ExitSignal(
                    exit_all=True, lots=self.remaining_lots,
                    reason="NO_MOVE", premium=current_premium,
                )

        # ── 3. TRAILING STOP ─────────────────────────────────────────────────
        if change_pct >= self.TRAIL_ACTIVATION:
            self.trailing_active = True

        if self.trailing_active:
            self.trailing_sl = self.highest * (1 - self.TRAIL_DISTANCE)
            if current_premium <= self.trailing_sl:
                return ExitSignal(
                    exit_all=True, lots=self.remaining_lots,
                    reason="TRAIL_SL", premium=self.trailing_sl,
                )

        # ── 4. TREND CHECK at +36% ───────────────────────────────────────────
        if not self.trend_check_done and change_pct >= self.TREND_CHECK_PCT:
            self.trend_check_done = True
            trend_ok = False
            if df_recent is not None and len(df_recent) >= 3:
                try:
                    from engine_v2.signal import check_trend_continuation
                    trend_ok = check_trend_continuation(df_recent)
                except Exception:
                    trend_ok = False
            if not trend_ok:
                # Exit at current premium — trend fading, lock in 3× gain
                return ExitSignal(
                    exit_all=True, lots=self.remaining_lots,
                    reason="TREND_CHECK_EXIT", premium=current_premium,
                )
            # Trend is strong — let trail handle the rest
            logger.debug("TREND_CHECK_HOLD: prem=%.2f change=%.1f%%",
                         current_premium, change_pct * 100)

        # ── 5a. PARTIAL EXIT 1 at +50% ───────────────────────────────────────
        if not self.partial1_done and change_pct >= self.PARTIAL1_PCT:
            partial_lots = max(1, int(self.lots * self.PARTIAL1_LOTS_PCT))
            partial_lots = min(partial_lots, self.remaining_lots - 1)
            if partial_lots > 0:
                self.remaining_lots -= partial_lots
                self.partial1_done = True
                return ExitSignal(
                    exit_all=False, lots=partial_lots,
                    reason="PARTIAL_1", premium=current_premium,
                )

        # ── 5b. PARTIAL EXIT 2 at +80% ───────────────────────────────────────
        if not self.partial2_done and change_pct >= self.PARTIAL2_PCT:
            partial_lots = max(1, int(self.lots * self.PARTIAL2_LOTS_PCT))
            partial_lots = min(partial_lots, self.remaining_lots - 1)
            if partial_lots > 0:
                self.remaining_lots -= partial_lots
                self.partial2_done = True
                return ExitSignal(
                    exit_all=False, lots=partial_lots,
                    reason="PARTIAL_2", premium=current_premium,
                )

        # ── 6. TIME-BASED CLOSES ─────────────────────────────────────────────
        if current_time >= self.LOSING_CLOSE_TIME and change_pct < 0:
            return ExitSignal(
                exit_all=True, lots=self.remaining_lots,
                reason="LOSING_CLOSE", premium=current_premium,
            )

        if current_time >= self.HARD_CLOSE_TIME:
            return ExitSignal(
                exit_all=True, lots=self.remaining_lots,
                reason="HARD_CLOSE", premium=current_premium,
            )

        return None
