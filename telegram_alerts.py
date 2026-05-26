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

import datetime as _dt
import logging
import os
from typing import Optional

import pytz as _pytz
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

def send_engine_start(
    today: str,
    current_time: str,
    indices: str,
    capital: float,
    auc: Optional[float],
    pattern_buckets: int,
    pcr: Optional[float],
    paper: bool = True,
) -> None:
    """09:15 IST — fired once on the first candle run of each trading day."""
    mode    = "PAPER" if paper else "LIVE"
    pcr_str = f"{pcr:.2f}" if pcr is not None else "N/A"
    auc_str = f"{auc:.3f}" if auc else "N/A"
    _send(
        f"🟢 <b>Trading Engine Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Date     : {today}\n"
        f"⏰ Time     : {current_time} IST\n"
        f"📊 Index    : {indices}\n"
        f"💰 Capital  : ₹{capital:,.0f}\n"
        f"📈 Mode     : {mode}\n"
        f"🤖 ML Model : AUC {auc_str} | {pattern_buckets} buckets\n"
        f"🌡 PCR      : {pcr_str}\n"
        f"⚡ Status   : All systems ready\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Signal window: 10:00 – 15:00 IST"
    )


def send_engine_end(
    today: str,
    trades_today: int,
    wins: int,
    losses: int,
    pnl_today: float,
    total_trades: int,
    total_pnl: float,
    capital: float,
    auc: Optional[float],
    ml_mode: str,
    signals_today: int,
    ml_skipped: int,
) -> None:
    """15:15 IST — fired once after the trading session closes."""
    win_rate = wins / trades_today * 100 if trades_today else 0.0
    auc_str  = f"{auc:.3f}" if auc else "N/A"
    _send(
        f"🔴 <b>Trading Engine Stopped</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Date        : {today}\n"
        f"⏰ Stopped at  : 15:15 IST\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>TODAY'S SUMMARY</b>\n"
        f"  Trades placed  : {trades_today}\n"
        f"  Wins / Losses  : {wins} / {losses}\n"
        f"  Win rate       : {win_rate:.0f}%\n"
        f"  P&L today      : ₹{pnl_today:+,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>OVERALL</b>\n"
        f"  Total trades   : {total_trades}\n"
        f"  Total P&L      : ₹{total_pnl:+,.0f}\n"
        f"  Capital        : ₹{capital:,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 <b>ML STATUS</b>\n"
        f"  AUC            : {auc_str}\n"
        f"  Mode           : {ml_mode}\n"
        f"  Signals today  : {signals_today}\n"
        f"  ML skipped     : {ml_skipped}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Next session: Tomorrow 09:15 IST"
    )


def send_hourly_status(
    time_str: str,
    open_positions: int,
    daily_pnl: float,
    trades_today: int,
    capital: float,
    paper: bool = True,
) -> None:
    """Sent at the top of each market hour (10:00, 11:00, …, 14:00) as a heartbeat."""
    mode  = "📄 PAPER" if paper else "💰 LIVE"
    pos_str = f"{open_positions} open" if open_positions else "No open positions"
    _send(
        f"🔁 <b>Engine Running</b>  [{mode}]\n"
        f"⏰ {time_str} IST\n"
        f"📊 Positions  : {pos_str}\n"
        f"📈 Trades     : {trades_today} today\n"
        f"💰 P&L today  : ₹{daily_pnl:+,.0f}\n"
        f"🏦 Capital    : ₹{capital:,.0f}\n"
        f"──────────────────"
    )


def send_morning_bias(index: str, pcr: float, bias: str) -> None:
    """09:00 AM — PCR and directional bias for the day."""
    mode = "📄 PAPER" if _is_paper() else "💰 LIVE"
    _send(
        f"☀️ <b>Morning Bias — {index}</b>  [{mode}]\n"
        f"PCR  : <b>{pcr:.2f}</b>\n"
        f"Bias : {bias}\n"
        f"──────────────────"
    )


def send_orb_signal(
    index: str,
    direction: str,
    or_high: float,
    or_low: float,
    breakout_price: float,
    paper: bool = True,
) -> None:
    arrow  = "🟢" if direction == "CALL" else "🔴"
    mode   = "📄 PAPER" if paper else "💰 LIVE"
    side   = "above OR High" if direction == "CALL" else "below OR Low"
    ref    = or_high if direction == "CALL" else or_low
    _send(
        f"{arrow} <b>ORB Breakout: {direction}  [{mode}]</b>\n"
        f"Index  : {index}\n"
        f"Closed {side}\n"
        f"OR High : {or_high:.2f}  |  OR Low : {or_low:.2f}\n"
        f"Break @ : {breakout_price:.2f}  (ref {ref:.2f})\n"
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
    ml_conf: Optional[float] = None,
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


def send_vix_blocked(vix: float, vix_max: float) -> None:
    """One-time alert when India VIX exceeds the configured maximum."""
    now = _dt.datetime.now(_pytz.timezone("Asia/Kolkata")).strftime("%H:%M")
    _send(
        f"⚠️ <b>Trading Paused — VIX Too High</b>\n"
        f"⏰ {now} IST\n"
        f"India VIX : <b>{vix:.2f}</b>  (limit {vix_max:.1f})\n"
        f"Engine will resume scanning once VIX drops below {vix_max:.1f}.\n"
        f"──────────────────"
    )


def send_loss_limit_hit(daily_pnl: float, limit: float) -> None:
    """One-time alert when the daily loss limit is breached."""
    now = _dt.datetime.now(_pytz.timezone("Asia/Kolkata")).strftime("%H:%M")
    _send(
        f"🚨 <b>Daily Loss Limit Hit — Trading Halted</b>\n"
        f"⏰ {now} IST\n"
        f"P&L today : ₹{daily_pnl:+,.0f}\n"
        f"Limit     : ₹{-limit:+,.0f}\n"
        f"No more trades will be placed today.\n"
        f"──────────────────"
    )


def send_day_summary(
    trades: int,
    pnl: float,
    capital: float,
    paper: bool = True,
    ml_auc: Optional[float] = None,
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
        "⚠️ <b>Kite Token Expired</b>\n"
        "Engine did not start — access token is invalid.\n"
        "👉 GitHub → Repo → Settings → Secrets → Actions\n"
        "   Update <code>KITE_ACCESS_TOKEN</code> with today's token.\n"
        "──────────────────"
    )


def send_auth_failure(error: str) -> None:
    """Sent from _run_candle() when Kite auth or engine init fails."""
    now = _dt.datetime.now(_pytz.timezone("Asia/Kolkata")).strftime("%H:%M")
    is_token = "expired" in error.lower() or "invalid" in error.lower() or "access_token" in error.lower()
    if is_token:
        _send(
            f"🔑 <b>Engine Failed — Token Expired</b>\n"
            f"⏰ {now} IST\n"
            f"Kite access token is expired or invalid.\n"
            f"👉 Go to: <b>GitHub → Repo → Settings →\n"
            f"   Secrets → Actions → KITE_ACCESS_TOKEN</b>\n"
            f"   Paste today's fresh token and re-run.\n"
            f"──────────────────"
        )
    else:
        _send(
            f"❌ <b>Engine Failed to Start</b>\n"
            f"⏰ {now} IST\n"
            f"Error: <code>{error[:200]}</code>\n"
            f"──────────────────"
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
