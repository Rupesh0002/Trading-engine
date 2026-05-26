"""
Dynamic exit manager for engine_v3.

Exit hierarchy (checked in this order each 5-min candle):
  1. Hard close  — 15:15 IST
  2. Hard SL     — premium drops 15% from entry
  3. No-move     — setup-specific spot move timeout with one EMA-based extension
  4. Breakeven   — at +15%, move SL to entry premium
  5. Partial 1   — at +30%, book 40% of original lots
  6. Partial 2   — at +50%, book 30% of original lots
  7. Trail SL    — activates at +20%, trails 10% below highest premium
  8. Fixed TP    — for ORB setup only (checked in backtest by spot comparison)

Partial booking creates a sub-exit record; the trade stays OPEN for remaining lots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import List, Optional

from engine_v3.levels import compute_ema

_HARD_SL_PCT     = 0.15
_BREAKEVEN_PCT   = 0.15
_TRAIL_ACTIVATE  = 0.20
_TRAIL_OFFSET    = 0.10
_PARTIAL1_PCT    = 0.30
_PARTIAL1_FRAC   = 0.40
_PARTIAL2_PCT    = 0.50
_PARTIAL2_FRAC   = 0.30
_HARD_CLOSE_TIME = time(15, 15)
_NO_MOVE_EXTENSION_CANDLES = 6

NO_MOVE_CANDLES = {
    "ORB": 12,
    "PDH_PDL": 16,
    "VWAP": 10,
    "SMC": 8,
}

NO_MOVE_THRESHOLD = {
    "ORB": 0.0020,
    "PDH_PDL": 0.0015,
    "VWAP": 0.0012,
    "SMC": 0.0012,
}


@dataclass
class TradeState:
    entry_premium:   float
    entry_spot:      float
    setup:           str
    total_lots:      int
    remaining_lots:  int  = field(init=False)
    highest_premium: float = field(init=False)
    trail_sl:        float = field(init=False)
    breakeven_set:   bool  = False
    partial1_done:   bool  = False
    partial1_lots:   int   = 0
    partial1_px:     float = 0.0
    partial2_done:   bool  = False
    partial2_lots:   int   = 0
    partial2_px:     float = 0.0
    candles_held:    int   = 0
    no_move_deadline: int  = field(init=False)
    no_move_extended: bool = False

    def __post_init__(self):
        self.remaining_lots  = self.total_lots
        self.highest_premium = self.entry_premium
        self.trail_sl        = self.entry_premium * (1 - _HARD_SL_PCT)
        self.no_move_deadline = NO_MOVE_CANDLES.get(self.setup, NO_MOVE_CANDLES["VWAP"])


@dataclass
class ExitResult:
    reason:       str
    exit_px:      float
    lots_exited:  int
    is_full:      bool   # True = trade fully closed after this exit


def update_and_check(
    state:        TradeState,
    current_prem: float,
    t5:           time,
    direction:    str,
    spot_close:   float,
    recent_closes: List[float],
) -> Optional[ExitResult]:
    """
    Update trade state for one 5-min candle and check all exit conditions.
    Mutates state in-place.
    Returns ExitResult when an exit fires (partial or full); None = hold.
    """
    # ── Hard close at 15:15 ───────────────────────────────────────────
    if t5 >= _HARD_CLOSE_TIME:
        return ExitResult("Hard close (15:15)", current_prem,
                          state.remaining_lots, is_full=True)

    state.candles_held    += 1
    state.highest_premium  = max(state.highest_premium, current_prem)
    pnl_pct = (current_prem - state.entry_premium) / state.entry_premium

    # ── Hard SL ───────────────────────────────────────────────────────
    if current_prem <= state.entry_premium * (1 - _HARD_SL_PCT):
        return ExitResult("Hard SL", current_prem, state.remaining_lots, is_full=True)

    # ── No-move timeout (one EMA-based extension max) ────────────────
    if (not state.partial1_done
            and state.no_move_deadline > 0
            and state.candles_held == state.no_move_deadline):
        move_pct = abs(spot_close - state.entry_spot) / state.entry_spot if state.entry_spot > 0 else 0.0
        band = NO_MOVE_THRESHOLD.get(state.setup, NO_MOVE_THRESHOLD["VWAP"])
        if move_pct < band:
            ema21 = compute_ema(recent_closes[-40:], 21) if recent_closes else spot_close
            trend_ok = spot_close > ema21 if direction == "CALL" else spot_close < ema21
            if trend_ok and not state.no_move_extended:
                state.no_move_extended = True
                state.no_move_deadline += _NO_MOVE_EXTENSION_CANDLES
            else:
                reason = "NO_MOVE_EXTENDED" if state.no_move_extended else "NO_MOVE_TREND_LOST"
                return ExitResult(reason, current_prem, state.remaining_lots, is_full=True)
        else:
            state.no_move_deadline = 0

    # ── Breakeven stop ────────────────────────────────────────────────
    if not state.breakeven_set and pnl_pct >= _BREAKEVEN_PCT:
        state.trail_sl      = state.entry_premium
        state.breakeven_set = True

    # ── Update trail SL (activates at +20%) ──────────────────────────
    if pnl_pct >= _TRAIL_ACTIVATE:
        new_trail = state.highest_premium * (1 - _TRAIL_OFFSET)
        state.trail_sl = max(state.trail_sl, new_trail)

    if current_prem < state.trail_sl and state.trail_sl > state.entry_premium * (1 - _HARD_SL_PCT):
        return ExitResult("Trail SL", current_prem, state.remaining_lots, is_full=True)

    # ── Partial 1: +30% → book 40% ───────────────────────────────────
    if not state.partial1_done and pnl_pct >= _PARTIAL1_PCT:
        p1 = max(1, round(state.total_lots * _PARTIAL1_FRAC))
        state.partial1_done = True
        state.partial1_lots = p1
        state.partial1_px   = current_prem
        state.remaining_lots -= p1
        if state.remaining_lots <= 0:
            state.remaining_lots = 0
            return ExitResult("Partial 1 + close", current_prem, p1, is_full=True)
        return ExitResult("Partial 1", current_prem, p1, is_full=False)

    # ── Partial 2: +50% → book 30% ───────────────────────────────────
    if state.partial1_done and not state.partial2_done and pnl_pct >= _PARTIAL2_PCT:
        p2 = max(1, round(state.total_lots * _PARTIAL2_FRAC))
        state.partial2_done = True
        state.partial2_lots = p2
        state.partial2_px   = current_prem
        state.remaining_lots -= p2
        if state.remaining_lots <= 0:
            state.remaining_lots = 0
            return ExitResult("Partial 2 + close", current_prem, p2, is_full=True)
        return ExitResult("Partial 2", current_prem, p2, is_full=False)

    return None
