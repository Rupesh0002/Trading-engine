"""
Trade document schema.
Every completed trade is represented as a TradeDocument dict.
This single schema is used for both MongoDB (as-is) and MySQL (flattened).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")


def new_trade_id(index: str) -> str:
    """e.g. TRD-20260516-NIFTY-A3F2"""
    date_str = datetime.now(IST).strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:4].upper()
    return f"TRD-{date_str}-{index}-{suffix}"


def build_trade_document(
    *,
    index: str,
    direction: str,
    entry_type: str = "BUY",
    symbol: str,
    strike: int,
    option_type: str,
    expiry: str,
    lot_size: int,
    lots: int,
    quantity: int,
    entry_premium: float,
    exit_premium: Optional[float],
    spot_at_entry: float,
    spot_at_exit: Optional[float],
    stop_loss: float,
    target_soft: float,
    target_hard: float,
    entry_time: datetime,
    exit_time: Optional[datetime],
    exit_reason: str,
    conditions_met: int,
    conditions_detail: Dict[str, Any],
    india_vix: Optional[float],
    pcr: Optional[float],
    paper_mode: bool,
    trade_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the complete trade document.
    pnl_amount and trade_quality are computed here from the inputs.
    Pass trade_id to reuse a pre-generated ID; otherwise one is created.
    """
    if trade_id is None:
        trade_id = new_trade_id(index)

    pnl_per_unit = (exit_premium - entry_premium) if exit_premium is not None else 0.0
    pnl_amount   = round(pnl_per_unit * quantity, 2)
    pnl_pct      = round(pnl_per_unit / entry_premium * 100, 4) if entry_premium else 0.0
    rr_achieved  = round(pnl_pct / (stop_loss / entry_premium * 100), 4) if exit_premium and entry_premium else 0.0
    capital_deployed = round(entry_premium * quantity, 2)
    risk_amount      = round(entry_premium * quantity * (stop_loss / entry_premium if entry_premium else 0), 2)

    # Trade quality rating
    if exit_premium is None:
        quality = "OPEN"
    elif pnl_amount > 0 and rr_achieved >= 3.0:
        quality = "EXCELLENT"
    elif pnl_amount > 0 and rr_achieved >= 2.5:
        quality = "GOOD"
    elif pnl_amount > 0 and rr_achieved >= 1.5:
        quality = "PARTIAL"
    elif abs(pnl_amount) < (entry_premium * quantity * 0.005):
        quality = "BREAKEVEN"
    else:
        quality = "LOSS"

    return {
        "trade_id":          trade_id,
        "index":             index,
        "entry_type":        entry_type,
        "date":              entry_time.strftime("%Y-%m-%d"),
        "entry_time":        entry_time.isoformat(),
        "exit_time":         exit_time.isoformat() if exit_time else None,
        "direction":         direction,
        "symbol":            symbol,
        "strike":            strike,
        "option_type":       option_type,
        "expiry":            expiry,
        "lot_size":          lot_size,
        "lots":              lots,
        "quantity":          quantity,
        "entry_premium":     round(entry_premium, 2),
        "exit_premium":      round(exit_premium, 2) if exit_premium else None,
        "spot_at_entry":     round(spot_at_entry, 2),
        "spot_at_exit":      round(spot_at_exit, 2) if spot_at_exit else None,
        "stop_loss":         round(stop_loss, 2),
        "target_soft":       round(target_soft, 2),
        "target_hard":       round(target_hard, 2),
        "pnl_per_unit":      round(pnl_per_unit, 2),
        "pnl_amount":        pnl_amount,
        "pnl_pct":           pnl_pct,
        "risk_reward_achieved": rr_achieved,
        "exit_reason":       exit_reason,
        "trade_quality":     quality,
        "capital_deployed":  capital_deployed,
        "risk_amount":       risk_amount,
        "strategy": {
            "name":            "VWAP + Fibonacci + RSI(Wilder) + Volume + PCR",
            "conditions_met":  conditions_met,
            "total":           5,
            "conditions":      conditions_detail,
        },
        "market_context": {
            "india_vix": india_vix,
            "pcr":       pcr,
        },
        "paper_mode":  paper_mode,
        "created_at":  datetime.now(IST).isoformat(),
    }
