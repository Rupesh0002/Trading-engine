"""
Telegram alert sender for the trading engine.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env.
If either is missing or blank, all functions are silent no-ops (no crash).

Events sent:
  - Morning bias (PCR + direction)
  - Signal fired (all 5-condition details)
  - Trade entered (strike, expiry, lots, premium, SL, TP)
  - SL hit (symbol, loss, capital remaining)
  - Target hit (symbol, profit, capital)
  - Hard close (symbol, P&L, reason)
  - Day summary (trades, P&L, capital)
  - Token expiry warning (on startup)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Credentials (from .env) ───────────────────────────────────────────────────
_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
_API_URL   = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
_ENABLED   = bool(_BOT_TOKEN and _CHAT_ID)

if not _ENABLED:
    logger.info("Telegram alerts disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set).")


# ── Internal sender ────────────────────────────────────────────────────────────

def _send(text: str) -> None:
    """Post a message to Telegram. Silently logs on failure — never crashes the engine."""
    if not _ENABLED:
        return
    try:
        resp = requests.post(
            _API_URL,
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
        if not resp.ok:
            logger.warning("Telegram send failed: %s — %s", resp.status_code, resp.text[:200])
    except requests.exceptions.RequestException as exc:
        logger.warning("Telegram network error: %s", exc)


# ── Public alert functions ─────────────────────────────────────────────────────

def send_morning_bias(index: str, pcr: float, bias: str) -> None:
    """09:00 AM — PCR and directional bias for the day."""
    mode = "📄 PAPER" if _is_paper() else "💰 LIVE"
    _send(
        f"☀️ <b>Morning Bias — {index}</b>  [{mode}]\n"
        f"PCR  : <b>{pcr:.2f}</b>\n"
        f"Bias : {bias}\n"
        f"──────────────────"
    )


def send_signal_fired(
    index: str,
    direction: str,
    conditions: int,
    adx: float,
    rsi: float,
    fib_level: str,
    paper: bool = True,
) -> None:
    """Signal evaluated — whether trade entered or in paper mode."""
    arrow  = "🟢" if direction == "CALL" else "🔴"
    mode   = "📄 PAPER" if paper else "💰 LIVE"
    _send(
        f"{arrow} <b>Signal: {direction}  {conditions}/5</b>  [{mode}]\n"
        f"Index  : {index}\n"
        f"ADX    : {adx:.1f}\n"
        f"RSI    : {rsi:.1f}\n"
        f"Fib    : {fib_level}\n"
        f"──────────────────"
    )


def send_trade_entered(
    index: str,
    direction: str,
    strike: int,
    expiry: str,
    lots: int,
    premium: float,
    sl: float,
    target: float,
    paper: bool = True,
    ml_tier: str = "normal",
    ml_conf: float = None,
) -> None:
    """Trade entry confirmation."""
    arrow  = "🟢" if direction == "CALL" else "🔴"
    mode   = "📄 PAPER" if paper else "💰 LIVE"
    opt    = "CE" if direction == "CALL" else "PE"
    tier_labels = {
        "strong":          "★ Strong",
        "normal":          "✓ Normal",
        "weak":            "~ Cautious (half size)",
        "indicators_only": "○ Inactive — training",
    }
    tier_tag = tier_labels.get(ml_tier, "")
    conf_str = f" ML={ml_conf:.2f}" if (ml_conf and ml_tier != "indicators_only") else ""
    _send(
        f"{arrow} <b>Trade Entered — {index} {direction}</b>  [{mode}]\n"
        f"Symbol  : {index}{expiry}{strike}{opt}\n"
        f"Lots    : {lots}\n"
        f"Premium : ₹{premium:.2f}\n"
        f"SL      : ₹{sl:.2f}\n"
        f"Target  : ₹{target:.2f}\n"
        f"ML      : {tier_tag}{conf_str}\n"
        f"──────────────────"
    )


def send_sl_hit(
    index: str,
    symbol: str,
    pnl: float,
    capital_after: float,
) -> None:
    """Stop-loss triggered."""
    _send(
        f"🛑 <b>Stop-Loss Hit — {index}</b>\n"
        f"Symbol  : {symbol}\n"
        f"Loss    : ₹{pnl:+,.2f}\n"
        f"Capital : ₹{capital_after:,.0f}\n"
        f"──────────────────"
    )


def send_target_hit(
    index: str,
    symbol: str,
    pnl: float,
    capital_after: float,
) -> None:
    """Target achieved."""
    _send(
        f"✅ <b>Target Hit — {index}</b>\n"
        f"Symbol  : {symbol}\n"
        f"Profit  : ₹{pnl:+,.2f}\n"
        f"Capital : ₹{capital_after:,.0f}\n"
        f"──────────────────"
    )


def send_hard_close(
    index: str,
    symbol: str,
    pnl: float,
    reason: str = "Hard close (15:00)",
) -> None:
    """Position force-closed at 15:00."""
    sign = "✅" if pnl >= 0 else "🟡"
    _send(
        f"{sign} <b>Hard Close — {index}</b>\n"
        f"Symbol  : {symbol}\n"
        f"P&L     : ₹{pnl:+,.2f}\n"
        f"Reason  : {reason}\n"
        f"──────────────────"
    )


def send_day_summary(
    trades: int,
    pnl: float,
    capital: float,
    paper: bool = True,
    ml_auc: float = None,
    shadow_count: int = 0,
    drift_line: str = "",
) -> None:
    """15:30 — end-of-day summary."""
    sign = "📈" if pnl >= 0 else "📉"
    mode = "📄 PAPER" if paper else "💰 LIVE"
    auc_line = f"ML AUC  : {ml_auc:.3f}" if ml_auc else "ML AUC  : n/a"
    body = (
        f"{sign} <b>Day Summary</b>  [{mode}]\n"
        f"Trades  : {trades}\n"
        f"P&L     : ₹{pnl:+,.2f}\n"
        f"Capital : ₹{capital:,.0f}\n"
        f"{auc_line}\n"
        f"Shadows : {shadow_count}\n"
    )
    if drift_line:
        body += f"{drift_line}\n"
    body += "──────────────────"
    _send(body)


def send_ml_conflict(index: str, xgb: float, mem: float) -> None:
    """XGBoost and adaptive memory scores conflict — engine deferred to memory."""
    _send(
        f"⚠️ <b>ML Conflict — {index}</b>\n"
        f"XGB     : {xgb:.2f}\n"
        f"Memory  : {mem:.2f}\n"
        f"Action  : Deferring to memory · half size\n"
        f"──────────────────"
    )


def send_token_expiry_warning() -> None:
    """Sent at startup if access token is expired."""
    _send(
        "⚠️ <b>ACCESS_TOKEN Expired</b>\n"
        "Generate a new token via Zerodha Kite login\n"
        "and update config/access_token.txt\n"
        "Engine will not start until token is valid.\n"
        "──────────────────"
    )


def send_startup(indices: list, paper: bool) -> None:
    """Engine startup notification."""
    mode = "📄 PAPER MODE" if paper else "💰 LIVE MODE"
    _send(
        f"🚀 <b>Engine Started</b>  [{mode}]\n"
        f"Indices : {', '.join(indices)}\n"
        f"──────────────────"
    )


# ── Internal helper ────────────────────────────────────────────────────────────

def _is_paper() -> bool:
    try:
        from config.settings import PAPER_MODE
        return PAPER_MODE
    except ImportError:
        return True
