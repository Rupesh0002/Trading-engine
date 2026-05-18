"""
Trade document generator.
Produces a human-readable, printable report for every completed trade.
Also saves a JSON file to reports/YYYY-MM-DD/<trade_id>.json

Sample output:
═══════════════════════════════════════════════════════════════
  TRADE REPORT — TRD-20260516-NIFTY-A3F2
═══════════════════════════════════════════════════════════════
  Index      : NIFTY 50
  Direction  : CALL
  Symbol     : NIFTY26MAY2420000CE
  Date       : 16-May-2026
...
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

import pytz

from config.settings import REPORTS_DIR, SOFT_TARGET_PCT, STOP_LOSS_PCT, INDEX_CONFIG

IST = pytz.timezone("Asia/Kolkata")

_QUALITY_EMOJI = {
    "EXCELLENT": "★★★  EXCELLENT",
    "GOOD":      "★★☆  GOOD",
    "PARTIAL":   "★☆☆  PARTIAL",
    "BREAKEVEN": "──   BREAKEVEN",
    "LOSS":      "✗    LOSS",
    "OPEN":      "◌    OPEN",
}


def generate_trade_report(trade: Dict[str, Any]) -> str:
    """Returns a formatted multi-line string report for a trade document."""
    sep_heavy = "═" * 63
    sep_light = "─" * 63

    index     = trade.get("index", "?")
    label     = INDEX_CONFIG.get(index, {}).get("label", index)
    direction = trade.get("direction", "?")
    symbol    = trade.get("symbol", "?")
    date_str  = trade.get("date", "?")

    entry_px  = trade.get("entry_premium", 0)
    exit_px   = trade.get("exit_premium")
    spot_in   = trade.get("spot_at_entry", 0)
    spot_out  = trade.get("spot_at_exit")
    sl        = trade.get("stop_loss", 0)
    tgt_soft  = trade.get("target_soft", 0)
    tgt_hard  = trade.get("target_hard", 0)
    qty       = trade.get("quantity", 0)
    lots      = trade.get("lots", 0)
    cap       = trade.get("capital_deployed", 0)
    risk_amt  = trade.get("risk_amount", 0)

    pnl_amt   = trade.get("pnl_amount", 0) or 0
    pnl_pct   = trade.get("pnl_pct", 0) or 0
    rr        = trade.get("risk_reward_achieved", 0) or 0
    quality   = trade.get("trade_quality", "OPEN")
    reason    = trade.get("exit_reason", "—")

    entry_t   = _fmt_time(trade.get("entry_time"))
    exit_t    = _fmt_time(trade.get("exit_time"))

    strat     = trade.get("strategy", {})
    conds     = strat.get("conditions", {})
    cond_met  = strat.get("conditions_met", 0)

    ctx       = trade.get("market_context", {})
    vix       = ctx.get("india_vix")
    pcr       = ctx.get("pcr")

    paper_tag = "[PAPER]" if trade.get("paper_mode") else "[LIVE]"
    trade_id  = trade.get("trade_id", "?")

    # ── Build report ─────────────────────────────────────────────
    lines = [
        sep_heavy,
        f"  TRADE REPORT  {paper_tag}",
        f"  {trade_id}",
        sep_heavy,
        f"  Index      : {label}  ({index})",
        f"  Direction  : {direction}",
        f"  Symbol     : {symbol}",
        f"  Date       : {_fmt_date(date_str)}",
        sep_light,
        "  ENTRY",
        sep_light,
        f"  Time       : {entry_t} IST",
        f"  Premium    : ₹{entry_px:.2f}",
        f"  Spot       : ₹{spot_in:,.2f}",
        f"  Quantity   : {qty} ({lots} lot{'s' if lots != 1 else ''})",
        f"  Capital    : ₹{cap:,.2f}",
        f"  Max Risk   : ₹{risk_amt:,.2f}",
        sep_light,
        "  EXIT",
        sep_light,
        f"  Time       : {exit_t} IST" if exit_t else "  Time       : OPEN",
        f"  Premium    : ₹{exit_px:.2f}" if exit_px else "  Premium    : —",
        f"  Spot       : ₹{spot_out:,.2f}" if spot_out else "  Spot       : —",
        f"  Reason     : {reason}",
        sep_light,
        "  RISK LEVELS",
        sep_light,
        f"  Stop Loss  : ₹{sl:.2f}  (−{STOP_LOSS_PCT*100:.1f}%)",
        f"  Target 2.5×: ₹{tgt_soft:.2f}  (+{SOFT_TARGET_PCT*100:.1f}%)  ← exit here",
        f"  Target 3.0×: ₹{tgt_hard:.2f}  (+{STOP_LOSS_PCT*3*100:.1f}%)  ← ideal",
        sep_light,
        "  RESULT",
        sep_light,
    ]

    if exit_px is not None:
        pnl_sign = "+" if pnl_amt >= 0 else ""
        lines += [
            f"  P/L Points : {pnl_sign}{entry_px - exit_px if exit_px < entry_px else exit_px - entry_px:.2f} pts  ({'loss' if pnl_amt < 0 else 'gain'})",
            f"  P/L Amount : ₹{pnl_sign}{pnl_amt:,.2f}",
            f"  P/L %      : {pnl_sign}{pnl_pct:.2f}%  (on premium)",
            f"  R:R        : 1:{abs(rr):.2f}  achieved",
            f"  Quality    : {_QUALITY_EMOJI.get(quality, quality)}",
        ]
    else:
        lines.append("  Trade is still OPEN.")

    lines += [
        sep_light,
        f"  STRATEGY CONDITIONS ({cond_met}/5)",
        sep_light,
    ]

    # Each condition
    _cond_display = [
        ("vwap",   "VWAP",   _cond_vwap(conds.get("vwap", {}))),
        ("fibonacci", "FIB",    _cond_fib(conds.get("fibonacci", {}))),
        ("rsi",    "RSI",    _cond_rsi(conds.get("rsi", {}))),
        ("volume", "VOLUME", _cond_vol(conds.get("volume", {}))),
        ("pcr",    "PCR",    _cond_pcr(conds.get("pcr", {}))),
    ]
    for _, name, (triggered, detail) in _cond_display:
        tick = "✓" if triggered else "✗"
        lines.append(f"  {tick} {name:<8}: {detail}")

    lines += [
        sep_light,
        "  MARKET CONTEXT",
        sep_light,
        f"  India VIX  : {vix:.2f}" if vix else "  India VIX  : N/A",
        f"  PCR        : {pcr:.2f}" if pcr else "  PCR        : N/A",
        f"  Mode       : {'PAPER TRADE' if trade.get('paper_mode') else 'LIVE TRADE'}",
        sep_heavy,
    ]

    return "\n".join(lines)


def save_trade_report(trade: Dict[str, Any]) -> str:
    """
    Saves the trade as a JSON file in reports/YYYY-MM-DD/<trade_id>.json
    Returns the file path.
    """
    date_str = trade.get("date", datetime.now(IST).strftime("%Y-%m-%d"))
    folder   = os.path.join(REPORTS_DIR, date_str)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{trade['trade_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trade, f, indent=2, ensure_ascii=False, default=str)
    return path


# ------------------------------------------------------------------
# Condition description helpers
# ------------------------------------------------------------------

def _cond_vwap(c: dict) -> tuple[bool, str]:
    if not c:
        return False, "N/A"
    triggered = c.get("triggered", False)
    close     = c.get("close", 0)
    vwap      = c.get("vwap", 0)
    diff      = round(close - vwap, 1)
    sign      = "above" if diff >= 0 else "below"
    return triggered, f"Close {close:.0f} is {abs(diff):.1f} pts {sign} VWAP {vwap:.0f}"

def _cond_fib(c: dict) -> tuple[bool, str]:
    if not c:
        return False, "N/A"
    triggered = c.get("triggered", False)
    level     = c.get("fib_label", "?")
    price     = c.get("fib_level", 0)
    dist      = c.get("distance", 0)
    return triggered, f"Near {level}% level ({price:.0f}), distance {dist:.1f} pts"

def _cond_rsi(c: dict) -> tuple[bool, str]:
    if not c:
        return False, "N/A"
    triggered = c.get("triggered", False)
    val       = c.get("value", 0)
    thr       = c.get("threshold", 0)
    side      = "oversold" if val < 50 else "overbought"
    return triggered, f"RSI {val:.1f} ({side}, threshold {thr:.0f})"

def _cond_vol(c: dict) -> tuple[bool, str]:
    if not c:
        return False, "N/A"
    triggered = c.get("triggered", False)
    ratio     = c.get("ratio", 0)
    mult      = c.get("multiplier", 1.5)
    return triggered, f"{ratio:.2f}× MA (spike threshold {mult}×)"

def _cond_pcr(c: dict) -> tuple[bool, str]:
    if not c:
        return False, "N/A"
    triggered = c.get("triggered", False)
    val       = c.get("value")
    if val is None:
        return False, "Not fetched"
    return triggered, f"PCR {val:.2f}"

# ------------------------------------------------------------------
# Time / date formatting helpers
# ------------------------------------------------------------------

def _fmt_time(iso: Any) -> str:
    if not iso:
        return "—"
    try:
        if isinstance(iso, str):
            dt = datetime.fromisoformat(iso)
        else:
            dt = iso
        if dt.tzinfo is None:
            dt = IST.localize(dt)
        return dt.astimezone(IST).strftime("%H:%M:%S")
    except Exception:
        return str(iso)

def _fmt_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%b-%Y")
    except Exception:
        return date_str
