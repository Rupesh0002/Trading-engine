"""
Dynamic lot calculator — Phase 1 uses 1 lot, Phase 2+ uses ADX + ML + context.

Lot sizing inputs:
  - ADX strength      → base lots (1 / 2 / 3)
  - ML confidence     → multiplier or pass-through
  - FCR multiplier    → from MorningContext.fcr.lot_multiplier
  - Round number flag → halve if near major level
  - VIX              → reduce if 18–24
  - Capital risk cap  → never risk > 3% on single trade

Max loss per trade: SL = 12% of option premium.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_RISK_PER_TRADE_PCT = 0.03   # 3% of capital max risk
_HARD_SL_PCT        = 0.12   # option premium drawdown
_ADX_HIGH           = 35.0
_ADX_MID            = 28.0
_ML_CONF_HIGH       = 0.75
_ML_CONF_MID        = 0.60
_VIX_HIGH_THRESHOLD = 18.0


def calculate_lots(
    adx: float,
    entry_premium: float,
    lot_size: int,
    capital: float,
    ml_confidence: float = 1.0,
    fcr_multiplier: float = 1.0,
    near_round_number: bool = False,
    vix: float = 15.0,
    ml_active: bool = False,
) -> int:
    """
    Returns number of lots to trade.

    Parameters
    ----------
    adx              : ADX at signal time
    entry_premium    : simulated option premium at entry
    lot_size         : index lot size (BANKNIFTY=15, NIFTY=50)
    capital          : current trading capital
    ml_confidence    : 1 - P(NO_MOVE), range 0–1.  1.0 = ML inactive
    fcr_multiplier   : from MorningContext.fcr.lot_multiplier
    near_round_number: True → halve lots
    vix              : India VIX
    ml_active        : True when ML Phase 2+ is active
    """
    # ── Base from ADX strength ────────────────────────────────────────────────
    if adx > _ADX_HIGH:
        base = 3
    elif adx > _ADX_MID:
        base = 2
    else:
        base = 1

    # ── ML confidence multiplier ─────────────────────────────────────────────
    if ml_active and ml_confidence < 1.0:
        if ml_confidence > _ML_CONF_HIGH:
            ml_mult = 1.0
        elif ml_confidence > _ML_CONF_MID:
            ml_mult = 0.67
        else:
            ml_mult = 0.0  # edge case: below 0.60 should have been filtered
        base = max(1, int(round(base * ml_mult)))

    # ── FCR multiplier ────────────────────────────────────────────────────────
    if fcr_multiplier != 1.0:
        base = max(1, int(round(base * fcr_multiplier)))

    # ── Round number penalty ──────────────────────────────────────────────────
    if near_round_number:
        base = max(1, base // 2)

    # ── VIX reduction ─────────────────────────────────────────────────────────
    if vix > _VIX_HIGH_THRESHOLD:
        base = max(1, base - 1)

    # ── Capital risk cap ──────────────────────────────────────────────────────
    max_risk     = capital * _RISK_PER_TRADE_PCT
    sl_per_lot   = entry_premium * _HARD_SL_PCT * lot_size
    if sl_per_lot > 0:
        max_lots = max(1, int(max_risk / sl_per_lot))
        base     = min(base, max_lots)

    logger.debug(
        "Lots: adx=%.1f base→%d ml=%.2f fcr=%.1f rnd=%s vix=%.1f → final=%d",
        adx, base, ml_confidence, fcr_multiplier, near_round_number, vix, base,
    )
    return base
