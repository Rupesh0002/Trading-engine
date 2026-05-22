"""
Risk manager: position sizing, daily loss guard, VIX filter, flexible exit levels.
All limits and ratios from config/settings.py → .env.

Flexible exit:
  soft target = entry × (1 + STOP_LOSS_PCT × MIN_PROFIT_RATIO)   e.g. +7.5% at 2.5×
  hard target = entry × (1 + STOP_LOSS_PCT × MAX_PROFIT_RATIO)   e.g. +9.0% at 3.0×
  Engine exits at soft target — locks in a GOOD trade.
  Hard target is recorded in the document for reference.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from config.settings import (
    ACTIVE_INDICES,
    DAILY_LOSS_LIMIT_PCT,
    INDEX_CONFIG,
    LOT_SIZING_SL_PCT,
    MAX_LOTS_CAP,
    MAX_OPEN_POSITIONS,
    MAX_PREMIUM,
    MIN_PREMIUM,
    MIN_PROFIT_RATIO,
    MAX_PROFIT_RATIO,
    STRONG_SIGNAL_THRESHOLD,
    WEAK_SIGNAL_TARGET_RATIO,
    RISK_PER_TRADE_PCT,
    SOFT_TARGET_PCT,
    HARD_TARGET_PCT,
    STOP_LOSS_PCT,
    TARGET_PCT,
    TRADING_CAPITAL,
    TRAILING_SL_PCT,
    VIX_MAX,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    All risk rules enforced here.
    No values are hardcoded — everything flows from .env.
    """

    def __init__(self) -> None:
        self.daily_pnl: float = 0.0
        self.open_positions: int = 0
        self._daily_loss_limit = TRADING_CAPITAL * DAILY_LOSS_LIMIT_PCT

        # Exposed for main.py without re-importing settings
        self.stop_loss_pct   = STOP_LOSS_PCT
        self.target_pct      = SOFT_TARGET_PCT      # engine exits at soft target
        self.hard_target_pct = HARD_TARGET_PCT
        self.trailing_sl_pct = TRAILING_SL_PCT

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def can_trade(
        self,
        vix: Optional[float] = None,
        open_positions: Optional[int] = None,
    ) -> bool:
        if open_positions is not None:
            self.open_positions = open_positions

        if self.daily_pnl <= -self._daily_loss_limit:
            logger.warning(
                "DAILY LOSS LIMIT: PnL=₹%.2f  limit=₹%.2f — trading halted.",
                self.daily_pnl, self._daily_loss_limit,
            )
            return False

        if self.open_positions >= MAX_OPEN_POSITIONS:
            logger.info("Max open positions (%d). Waiting.", MAX_OPEN_POSITIONS)
            return False

        if vix is not None and vix > VIX_MAX:
            logger.warning("India VIX=%.2f > max %.2f — paused.", vix, VIX_MAX)
            return False

        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size(
        self,
        entry_premium: float,
        index: str,
        adx: float = 0.0,
    ) -> tuple[int, int]:
        """
        Returns (quantity, lots) using ADX-quality-scaled risk budget.
        Formula (all from .env):
          risk_budget = TRADING_CAPITAL × RISK_PER_TRADE_PCT
          sl_per_lot  = entry_premium × LOT_SIZING_SL_PCT × lot_size
          base_lots   = floor(risk_budget / sl_per_lot)
          adx_mult    = 1.0 (ADX>30) | 0.75 (ADX>25) | 0.5 (else)
          lots        = round(base_lots × adx_mult), capped at MAX_LOTS_CAP
        """
        if entry_premium <= 0:
            return 0, 0

        lot_size    = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])["lot_size"]
        risk_budget = TRADING_CAPITAL * RISK_PER_TRADE_PCT
        sl_per_lot  = entry_premium * LOT_SIZING_SL_PCT * lot_size

        if sl_per_lot <= 0:
            return 0, 0

        base_lots = max(1, int(risk_budget / sl_per_lot))

        if adx > 30:
            adx_mult = 1.0
        elif adx > 25:
            adx_mult = 0.75
        else:
            adx_mult = 0.5

        lots     = max(1, min(round(base_lots * adx_mult), MAX_LOTS_CAP))
        quantity = lots * lot_size

        if quantity == 0:
            logger.warning(
                "[%s] Position size=0 for premium=%.2f. "
                "Increase TRADING_CAPITAL or RISK_PER_TRADE_PCT in .env.",
                index, entry_premium,
            )

        logger.debug(
            "[%s] Sizing: risk=₹%.0f  sl_per_lot=%.2f  base=%d  adx=%.1f  lots=%d",
            index, risk_budget, sl_per_lot, base_lots, adx, lots,
        )
        return quantity, lots

    # ------------------------------------------------------------------
    # Exit levels — flexible 2.5× to 3×
    # ------------------------------------------------------------------

    def compute_exit_levels(self, entry_premium: float, conditions_met: int = 4) -> dict:
        """
        Returns stop_loss and tiered target based on signal strength.
          5/5 signal → 2.5× target (50%) — strong trend, let it run
          4/5 signal → 1.5× target (30%) — take the achievable move
        """
        ratio = MIN_PROFIT_RATIO if conditions_met >= STRONG_SIGNAL_THRESHOLD else WEAK_SIGNAL_TARGET_RATIO
        soft_pct = STOP_LOSS_PCT * ratio
        hard_pct = STOP_LOSS_PCT * MAX_PROFIT_RATIO
        return {
            "stop_loss":    round(entry_premium * (1.0 - STOP_LOSS_PCT), 2),
            "target":       round(entry_premium * (1.0 + soft_pct), 2),
            "target_soft":  round(entry_premium * (1.0 + soft_pct), 2),
            "target_hard":  round(entry_premium * (1.0 + hard_pct), 2),
            "target_ratio": ratio,
        }

    def compute_exit_levels_sell(self, entry_premium: float) -> dict:
        """
        Exit levels for short (sold) option positions — logic is inverted vs buy.
        entry_premium = premium collected when selling the option.
        stop_loss  = premium rises by STOP_LOSS_PCT  → buy back to cut loss.
        target     = premium falls by SOFT_TARGET_PCT → buy back to take profit.
        """
        return {
            "stop_loss":   round(entry_premium * (1.0 + STOP_LOSS_PCT),  2),
            "target":      round(entry_premium * (1.0 - SOFT_TARGET_PCT), 2),
            "target_soft": round(entry_premium * (1.0 - SOFT_TARGET_PCT), 2),
            "target_hard": round(entry_premium * (1.0 - HARD_TARGET_PCT), 2),
        }

    def update_trailing_sl(self, current_premium: float, current_sl: float) -> float:
        """Ratchet stop-loss up as premium moves in our favour (long option)."""
        new_sl = round(current_premium * (1.0 - TRAILING_SL_PCT), 2)
        return max(new_sl, current_sl)

    def update_trailing_sl_sell(self, current_premium: float, current_sl: float) -> float:
        """Ratchet stop-loss down as premium falls (short option — ratchet down)."""
        new_sl = round(current_premium * (1.0 + TRAILING_SL_PCT), 2)
        return min(new_sl, current_sl)

    # ------------------------------------------------------------------
    # Premium validation
    # ------------------------------------------------------------------

    def is_premium_valid(self, premium: float) -> bool:
        if not (MIN_PREMIUM <= premium <= MAX_PREMIUM):
            logger.info(
                "Premium ₹%.2f outside range [%.0f, %.0f]. Skip.",
                premium, MIN_PREMIUM, MAX_PREMIUM,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Trade quality (for reports)
    # ------------------------------------------------------------------

    @staticmethod
    def trade_quality(rr_achieved: float, pnl: float) -> str:
        if pnl <= 0:
            return "LOSS"
        if rr_achieved >= MAX_PROFIT_RATIO:
            return "EXCELLENT"
        if rr_achieved >= MIN_PROFIT_RATIO:
            return "GOOD"
        if rr_achieved >= 1.5:
            return "PARTIAL"
        return "BREAKEVEN"

    # ------------------------------------------------------------------
    # P&L tracking
    # ------------------------------------------------------------------

    def record_trade_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl
        logger.info(
            "Trade PnL: ₹%.2f | Daily: ₹%.2f / limit ₹%.2f",
            pnl, self.daily_pnl, -self._daily_loss_limit,
        )

    def reset_daily(self) -> None:
        self.daily_pnl     = 0.0
        self.open_positions = 0
        logger.info("Risk: daily counters reset.")
