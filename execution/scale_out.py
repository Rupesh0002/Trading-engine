"""
Scale-Out Exit Manager — partial exits at +20% / +40% / +60%.

Exit plan (3 tranches):
  Tranche 1: 40% of lots at +20% gain  → covers risk on remaining lots
  Tranche 2: 40% of lots at +40% gain  OR red/bearish candle (momentum dying)
  Tranche 3: 20% of lots at +60% gain  OR hard exit after 3 five-min candles

Hard rules (always apply):
  - Hard SL: -15% on ALL lots simultaneously
  - NO-MOVE: flat ±3% for 2 candles → exit ALL
  - Trailing SL once Tranche 1 is taken: move SL to breakeven

Usage (backtest + live):
    pos = ScaleOutPosition(...)
    decision = pos.check_exit(current_premium, candle_close, direction)
    if decision.exit_tranche:
        lots_to_exit = decision.lots_to_close
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Thresholds
_T1_GAIN_PCT    =  0.30   # +30% → close 40%  (was +20%)
_T2_GAIN_PCT    =  0.55   # +55% → close 40%  (was +40%)
_T3_GAIN_PCT    =  0.75   # +75% → close 20%  (was +60%)
_HARD_SL_PCT    = -0.15   # -15% hard stop on ALL
_NO_MOVE_PCT    =  0.03   # ±3% flat = no-move
_NO_MOVE_CANDLES = 2      # 2 flat 15-min candles → exit (30 min of no movement)
_MAX_HOLD_5MIN   = 4      # 4 × 15-min candles = 60 min max hold (engine uses 15-min bars)

# Tranche weights (must sum to 1.0)
_T1_FRAC = 0.40
_T2_FRAC = 0.40
_T3_FRAC = 0.20


@dataclass
class ExitDecision:
    should_exit: bool           # close ANY lots now
    exit_all: bool              # close ALL remaining lots
    lots_to_close: int          # exact lot count (0 if no exit)
    exit_reason: str            # human-readable
    exit_price_override: float = 0.0  # non-zero → use this price instead of current_premium


@dataclass
class ScaleOutPosition:
    total_lots: int
    entry_premium: float
    direction: str          # "CALL" | "PUT"

    # Internal tracking
    t1_done: bool  = False
    t2_done: bool  = False
    t3_done: bool  = False
    lots_remaining: int = field(init=False)
    sl_price: float     = field(init=False)   # starts at hard SL; moves to BE after T1
    peak_premium: float = field(init=False)
    candles_held: int   = 0
    flat_candles: int   = 0
    prev_premium: float = field(init=False)

    def __post_init__(self) -> None:
        self.lots_remaining = self.total_lots
        self.sl_price       = self.entry_premium * (1 + _HARD_SL_PCT)
        self.peak_premium   = self.entry_premium
        self.prev_premium   = self.entry_premium

    def _lots_for(self, frac: float) -> int:
        raw = max(1, round(self.total_lots * frac))
        return min(raw, self.lots_remaining)

    def check_exit(self, current_premium: float, is_red_candle: bool = False) -> ExitDecision:
        """
        Evaluate exit conditions and return what to do this candle.

        Parameters
        ----------
        current_premium : latest option LTP
        is_red_candle   : True if 5-min candle closed bearish (for T2 momentum check)
        """
        self.candles_held   += 1
        self.peak_premium    = max(self.peak_premium, current_premium)

        gain_pct = (current_premium - self.entry_premium) / self.entry_premium

        # ── Hard SL ──────────────────────────────────────────────────────────
        if current_premium <= self.sl_price:
            return ExitDecision(
                should_exit=True, exit_all=True,
                lots_to_close=self.lots_remaining,
                exit_reason=f"HARD_SL ({gain_pct*100:+.1f}%)",
                exit_price_override=self.sl_price,  # exit AT SL, not gap-through price
            )

        # ── No-Move detector ─────────────────────────────────────────────────
        flat = abs(current_premium - self.prev_premium) / self.entry_premium <= _NO_MOVE_PCT
        self.flat_candles = self.flat_candles + 1 if flat else 0
        self.prev_premium = current_premium

        if self.flat_candles >= _NO_MOVE_CANDLES:
            return ExitDecision(
                should_exit=True, exit_all=True,
                lots_to_close=self.lots_remaining,
                exit_reason="NO_MOVE",
            )

        # ── Max hold ─────────────────────────────────────────────────────────
        if self.candles_held >= _MAX_HOLD_5MIN:
            return ExitDecision(
                should_exit=True, exit_all=True,
                lots_to_close=self.lots_remaining,
                exit_reason="MAX_HOLD",
            )

        # ── Tranche 3: +60% or end of 3-candle window ────────────────────────
        if not self.t3_done and self.t2_done:
            t3_hit    = gain_pct >= _T3_GAIN_PCT
            time_exit = self.candles_held >= 3    # 3 five-min = 15 min
            if t3_hit or time_exit:
                self.t3_done = True
                lots = self._lots_for(_T3_FRAC)
                self.lots_remaining -= lots
                reason = f"T3_TARGET (+{gain_pct*100:.0f}%)" if t3_hit else "T3_TIME"
                return ExitDecision(
                    should_exit=True, exit_all=(self.lots_remaining == 0),
                    lots_to_close=lots, exit_reason=reason,
                )

        # ── Tranche 2: +40% or red candle ────────────────────────────────────
        if not self.t2_done and self.t1_done:
            if gain_pct >= _T2_GAIN_PCT or is_red_candle:
                self.t2_done = True
                lots = self._lots_for(_T2_FRAC)
                self.lots_remaining -= lots
                reason = f"T2_TARGET (+{gain_pct*100:.0f}%)" if gain_pct >= _T2_GAIN_PCT else "T2_RED_CANDLE"
                return ExitDecision(
                    should_exit=True, exit_all=(self.lots_remaining == 0),
                    lots_to_close=lots, exit_reason=reason,
                )

        # ── Tranche 1: +20% ──────────────────────────────────────────────────
        if not self.t1_done and gain_pct >= _T1_GAIN_PCT:
            self.t1_done = True
            lots = self._lots_for(_T1_FRAC)
            self.lots_remaining -= lots
            # Move SL to breakeven after T1
            self.sl_price = self.entry_premium
            return ExitDecision(
                should_exit=True, exit_all=(self.lots_remaining == 0),
                lots_to_close=lots,
                exit_reason=f"T1_TARGET (+{gain_pct*100:.0f}%)",
            )

        return ExitDecision(should_exit=False, exit_all=False, lots_to_close=0, exit_reason="HOLD")
