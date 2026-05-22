"""
Dynamic trade manager — 4-mode exit system for open options positions.

Exit modes (classified after FAST_MOVE_CANDLES held):
  FAST     : premium gained >= FAST_MOVE_PCT  → trail at (1 - FAST_TRAIL_PCT) × peak
  SLOW     : premium gained >= SLOW_MOVE_PCT  → exit at fixed SLOW_MOVE_TARGET_PCT gain
  NO_MOVE  : premium within ±NO_MOVE_PCT of entry → exit immediately (avoids theta drag)
  NORMAL   : modest gain — hold toward primary target or max hold

Hard exits (always checked first, regardless of mode):
  Hard SL  : premium <= entry × (1 - STOP_LOSS_PCT)
  Max hold : candles_held >= MAX_CANDLES_HELD

All thresholds from config/settings.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import (
    FAST_MOVE_CANDLES,
    FAST_MOVE_PCT,
    FAST_TRAIL_PCT,
    MAX_CANDLES_HELD,
    NO_MOVE_PCT,
    SLOW_MOVE_PCT,
    SLOW_MOVE_TARGET_PCT,
    STOP_LOSS_PCT,
)

_THETA_PER_CANDLE = 0.012  # 1.2% per 15-min candle
_ATM_DELTA        = 0.5


@dataclass
class Position:
    direction:        str
    entry_premium:    float
    entry_spot:       float
    quantity:         int
    conviction_score: int
    stop_loss:        float          # current SL price (hard stop)
    target:           float          # primary target (for NORMAL mode)
    current_premium:  float
    peak_premium:     float
    prev_spot:        float
    candles_held:     int   = 0
    total_theta:      float = 0.0
    mode:             str   = "NORMAL"   # FAST / SLOW / NO_MOVE / NORMAL
    fast_trail_sl:    float = 0.0
    trade_id:         str   = ""


@dataclass
class ExitDecision:
    should_exit: bool
    exit_price:  float
    exit_reason: str


def step_premium(position: Position, candle_close: float) -> None:
    """Advance position's simulated premium by one 15-min candle (delta + theta)."""
    spot_move = candle_close - position.prev_spot
    if position.direction == "PUT":
        spot_move = -spot_move
    delta_impact          = spot_move * _ATM_DELTA
    theta_impact          = -(position.current_premium * _THETA_PER_CANDLE)
    position.total_theta += abs(theta_impact)
    new_px                = position.current_premium + delta_impact + theta_impact
    position.current_premium = max(new_px, 1.0)
    if position.current_premium > position.peak_premium:
        position.peak_premium = position.current_premium
    position.prev_spot = candle_close


def check_exit(position: Position) -> ExitDecision:
    """
    Evaluate exit conditions after step_premium() has been called for this candle.
    Returns ExitDecision — caller closes the position if should_exit is True.
    """
    px    = position.current_premium
    entry = position.entry_premium
    held  = position.candles_held
    peak  = position.peak_premium

    # 1. Hard SL — always first priority
    if px <= position.stop_loss:
        return ExitDecision(True, position.stop_loss, "Hard SL")

    # 2. Classify mode at FAST_MOVE_CANDLES (default 2 candles = 30 min)
    if held == FAST_MOVE_CANDLES and position.mode == "NORMAL":
        gain_pct = (px - entry) / entry
        if gain_pct >= FAST_MOVE_PCT:
            position.mode          = "FAST"
            position.fast_trail_sl = peak * (1.0 - FAST_TRAIL_PCT)
        elif gain_pct >= SLOW_MOVE_PCT:
            position.mode = "SLOW"
        elif abs(gain_pct) <= NO_MOVE_PCT:
            position.mode = "NO_MOVE"

    # 3. Mode-specific exits
    if position.mode == "FAST":
        # Update trailing stop upward as price moves higher
        new_trail = peak * (1.0 - FAST_TRAIL_PCT)
        if new_trail > position.fast_trail_sl:
            position.fast_trail_sl = new_trail
        if px <= position.fast_trail_sl:
            gain_pct = (peak / entry - 1) * 100
            return ExitDecision(True, px, f"Fast trail (peak +{gain_pct:.0f}%)")

    elif position.mode == "SLOW":
        slow_target = entry * (1.0 + SLOW_MOVE_TARGET_PCT)
        if px >= slow_target:
            return ExitDecision(True, px, f"Slow target (+{SLOW_MOVE_TARGET_PCT*100:.0f}%)")

    elif position.mode == "NO_MOVE":
        return ExitDecision(True, px, "No-move exit (flat 30 min)")

    # 4. Primary target (NORMAL mode or fast/slow not triggered yet)
    if px >= position.target:
        gain_pct = (px / entry - 1) * 100
        return ExitDecision(True, px, f"Target (+{gain_pct:.0f}%)")

    # 5. Max hold — forced exit regardless of P&L
    if held >= MAX_CANDLES_HELD:
        return ExitDecision(True, px, "Max hold (60 min)")

    return ExitDecision(False, px, "")
