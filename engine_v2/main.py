"""
Engine V3 — live trading entry point.

Triggered by GitHub Actions scheduler every 5 minutes, 09:15–14:20 IST.
Pass --candle to process one candle cycle.

Flow per candle:
  1. Load state + validate token
  2. Morning setup (once per day at 09:15)
  3. Skip day check
  4. Manage open position if any
  5. Check signal window + daily trade limit
  6. Evaluate 4-indicator signal
  7. ML filter (Phase 2+)
  8. Calculate lots
  9. Enter trade + send Telegram alert
  10. Save state

State is persisted in state_v3.json between candle runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import pytz

from config.auth import get_kite_client
from config.settings import (
    ACTIVE_INDEX,
    BACKTEST_VIX,
    INDEX_CONFIG,
    PAPER_MODE,
    TRADING_CAPITAL,
)
from data.feed import DataFeed
from engine_v2.exit_manager import ExitManager
from engine_v2.lot_calculator import calculate_lots
from engine_v2.ml_filter import MLFilter, extract_features, get_candle_sequence
from engine_v2.sentiment import build_morning_context, morning_telegram_text
from engine_v2.signal import SignalResult, compute_signal
from signals.indicators import compute_adx, compute_ema, compute_rsi_wilder
from telegram_alerts import _send as tg_send
from utils.logger import get_logger

logger = get_logger("engine_v3")
IST    = pytz.timezone("Asia/Kolkata")

_STATE_FILE     = "state_v3.json"
_DAILY_MAX      = 2     # max trades per day
_PREWARM_DAYS   = 3     # days of 5-min data for indicator warm-up
_SIGNAL_START   = time(9, 30)
_SIGNAL_END     = time(13, 0)


# ── State helpers ─────────────────────────────────────────────────────────────

def load_state() -> Dict[str, Any]:
    if os.path.exists(_STATE_FILE):
        with open(_STATE_FILE) as f:
            return json.load(f)
    return _default_state()


def save_state(state: Dict[str, Any]) -> None:
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _default_state() -> Dict[str, Any]:
    return {
        "date":           None,
        "index":          ACTIVE_INDEX,
        "skip_day":       False,
        "skip_reason":    "",
        "daily_trades":   0,
        "open_position":  None,
        "capital":        TRADING_CAPITAL,
        "daily_pnl":      0.0,
        "ml_trade_count": 0,
        "morning_done":   False,
        "vix":            BACKTEST_VIX,
        "gap_pct":        0.0,
        "gap_bias":       "NEUTRAL",
        "fcr_pct":        0.0,
        "fcr_type":       "NORMAL",
        "fcr_lot_mult":   1.0,
        "near_round":     False,
        "nearest_round":  0.0,
        "prev_close":     0.0,
    }


def _reset_daily(state: Dict[str, Any]) -> None:
    state["date"]         = str(date.today())
    state["skip_day"]     = False
    state["skip_reason"]  = ""
    state["daily_trades"] = 0
    state["open_position"] = None
    state["daily_pnl"]    = 0.0
    state["morning_done"] = False


# ── ML singleton (persists across candle runs in-process) ─────────────────────
_ml = MLFilter()


# ── Main candle loop ──────────────────────────────────────────────────────────

def run_candle(index: str = ACTIVE_INDEX) -> None:
    ist_now = datetime.now(IST)
    today   = ist_now.date()
    state   = load_state()

    # Daily reset
    if state.get("date") != str(today):
        _reset_daily(state)
        state["index"] = index

    kite = get_kite_client()
    feed = DataFeed(kite)

    # ── Morning setup (once at 09:15) ─────────────────────────────────────────
    if not state["morning_done"] and ist_now.time() >= time(9, 15):
        _run_morning_setup(feed, state, index)
        save_state(state)
        if state["skip_day"]:
            return

    # ── Skip day guard ────────────────────────────────────────────────────────
    if state["skip_day"]:
        logger.info("Skip day: %s", state["skip_reason"])
        return

    # ── Manage open position ──────────────────────────────────────────────────
    if state.get("open_position"):
        _manage_position(feed, kite, state, index)
        save_state(state)
        return

    # ── Signal window check ───────────────────────────────────────────────────
    cur_time = ist_now.time()
    if not (_SIGNAL_START <= cur_time <= _SIGNAL_END):
        logger.debug("Outside signal window (%s)", cur_time)
        return

    # ── Daily trade limit ─────────────────────────────────────────────────────
    if state.get("daily_trades", 0) >= _DAILY_MAX:
        logger.info("Daily trade limit reached (%d)", _DAILY_MAX)
        return

    # ── Fetch 5-min candles with 3-day pre-warm ───────────────────────────────
    prewarm_start = (today - timedelta(days=_PREWARM_DAYS + 4)).strftime("%Y-%m-%d")
    today_str     = today.strftime("%Y-%m-%d")
    df5 = feed.get_historical_candles(prewarm_start, today_str, index=index, interval="5minute")
    if df5 is None or df5.empty:
        logger.warning("No 5-min candles for %s", index)
        return

    # ── Signal check on last closed candle ───────────────────────────────────
    sr = compute_signal(df5, candle_idx=-2)
    if sr.direction == "NONE":
        logger.debug("No signal: EMA=%s rng=%s RSI=%s ADX=%s",
                     sr.ema_ok, sr.range_ok, sr.rsi_ok, sr.adx_ok)
        save_state(state)
        return

    # Must agree with gap direction
    if state["gap_bias"] != "NEUTRAL" and sr.direction != state["gap_bias"][:4]:
        logger.debug("Signal direction %s disagrees with gap %s", sr.direction, state["gap_bias"])
        save_state(state)
        return

    # ── ML filter ────────────────────────────────────────────────────────────
    cfg      = INDEX_CONFIG[index]
    fcr_pct  = state.get("fcr_pct", 0.0)
    feats    = extract_features(
        df5, candle_idx=-2,
        adx_val=sr.adx, rsi_val=sr.rsi, ema9=sr.ema9, ema21=sr.ema21, ema50=sr.ema50,
        vix=state["vix"], gap_pct=state["gap_pct"], fcr_pct=fcr_pct, index=index,
    )
    seq      = get_candle_sequence(df5, candle_idx=-2)
    take, ml_conf = _ml.should_take_trade(feats, seq)

    if not take:
        logger.info("ML filtered signal (conf=%.2f)", ml_conf)
        save_state(state)
        return

    # ── Lot sizing ────────────────────────────────────────────────────────────
    spot          = float(df5["close"].iloc[-2])
    import math
    from config.settings import BACKTEST_VIX
    dte           = _days_to_expiry(today, index)
    entry_premium = 0.4 * spot * (state["vix"] / 100) * math.sqrt(max(dte, 1) / 365)

    lots = calculate_lots(
        adx=sr.adx,
        entry_premium=entry_premium,
        lot_size=cfg["lot_size"],
        capital=state["capital"],
        ml_confidence=ml_conf,
        fcr_multiplier=state.get("fcr_lot_mult", 1.0),
        near_round_number=state.get("near_round", False),
        vix=state["vix"],
        ml_active=(_ml.phase() >= 2),
    )

    # ── Record trade in state ─────────────────────────────────────────────────
    entry_time = ist_now.time()
    position   = {
        "direction":      sr.direction,
        "entry_premium":  round(entry_premium + 1.5, 2),  # +slippage
        "entry_spot":     spot,
        "entry_time":     str(entry_time),
        "lots":           lots,
        "features":       feats,
    }
    state["open_position"]   = position
    state["daily_trades"]    = state.get("daily_trades", 0) + 1
    state["ml_trade_count"]  = state.get("ml_trade_count", 0) + 1

    # ── Entry alert ───────────────────────────────────────────────────────────
    tg_send(
        f"📈 <b>{'BUY CALL' if sr.direction == 'CALL' else 'BUY PUT'} — {index}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Premium  : ₹{position['entry_premium']:.2f}\n"
        f"Lots     : {lots}  ({lots * cfg['lot_size']} qty)\n"
        f"ADX      : {sr.adx:.1f}   RSI : {sr.rsi:.1f}\n"
        f"ML conf  : {ml_conf:.2f}   Phase {_ml.phase()}\n"
        f"Gap      : {state['gap_pct']:+.2f}%   VIX : {state['vix']:.1f}\n"
        f"SL       : ₹{position['entry_premium'] * 0.88:.2f}  (-12%)\n"
        f"Mode     : {'PAPER' if PAPER_MODE else 'LIVE'}"
    )

    logger.info(
        "ENTRY %s %s | prem=%.2f lots=%d ml_conf=%.2f phase=%d",
        index, sr.direction, position["entry_premium"], lots, ml_conf, _ml.phase(),
    )
    save_state(state)


# ── Position management ───────────────────────────────────────────────────────

def _manage_position(
    feed: DataFeed,
    kite,
    state: Dict[str, Any],
    index: str,
) -> None:
    pos = state["open_position"]
    if pos is None:
        return

    today_str     = date.today().strftime("%Y-%m-%d")
    prewarm_start = (date.today() - timedelta(days=_PREWARM_DAYS + 4)).strftime("%Y-%m-%d")
    df5 = feed.get_historical_candles(prewarm_start, today_str, index=index, interval="5minute")
    if df5 is None or df5.empty:
        return

    from config.settings import BACKTEST_VIX
    import math
    entry_prem = pos["entry_premium"]
    entry_spot = pos["entry_spot"]
    entry_time = time.fromisoformat(pos["entry_time"]) if isinstance(pos["entry_time"], str) else pos["entry_time"]
    direction  = pos["direction"]

    em = ExitManager(entry_prem, pos["lots"], entry_time)

    # Reconstruct exit state from current candle
    spot_now   = float(df5["close"].iloc[-2])
    dte        = _days_to_expiry(date.today(), index)
    candles_held = max(1, len(df5[df5["timestamp"].dt.date == date.today()]) - 1)

    spot_move  = spot_now - entry_spot
    if direction == "PUT":
        spot_move = -spot_move
    current_prem = max(
        entry_prem + 0.5 * spot_move - candles_held * 0.004 * entry_prem,
        entry_prem * 0.05,
    )
    em.candles_held = candles_held
    em.highest      = max(entry_prem, current_prem)

    cur_time = datetime.now(IST).time()
    signal   = em.check(current_prem, cur_time, df_recent=df5.tail(20))

    if signal is not None and signal.exit_all:
        pnl_per_lot = (signal.premium - 1.5 - entry_prem) * INDEX_CONFIG[index]["lot_size"]
        net_pnl = pnl_per_lot * signal.lots - 40.0
        state["daily_pnl"] = state.get("daily_pnl", 0.0) + net_pnl
        state["open_position"] = None
        state["capital"]   = state.get("capital", TRADING_CAPITAL) + net_pnl

        # Log outcome to ML
        feats = pos.get("features", {})
        if feats:
            _ml.log_outcome(feats, signal.reason)

        emoji   = "✅" if net_pnl > 0 else "❌"
        tg_send(
            f"{emoji} <b>EXIT {index} {direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Reason   : {signal.reason}\n"
            f"Exit px  : ₹{signal.premium:.2f}\n"
            f"Net P&L  : ₹{net_pnl:+,.0f}\n"
            f"Capital  : ₹{state['capital']:,.0f}"
        )
        logger.info(
            "EXIT %s | %s | pnl=%.0f | capital=%.0f",
            index, signal.reason, net_pnl, state["capital"],
        )


# ── Morning setup ──────────────────────────────────────────────────────────────

def _run_morning_setup(feed: DataFeed, state: Dict[str, Any], index: str) -> None:
    from config.settings import BACKTEST_VIX

    today     = date.today()
    today_str = today.strftime("%Y-%m-%d")
    yesterday = (today - timedelta(days=5)).strftime("%Y-%m-%d")

    df15 = feed.get_historical_candles(yesterday, today_str, index=index, interval="15minute")
    if df15 is None or df15.empty:
        logger.warning("No 15-min candles for morning setup")
        state["morning_done"] = True
        return

    # prev_close: last 15-min candle before today
    df15["date"] = df15["timestamp"].dt.date
    prev_rows   = df15[df15["date"] < today]
    if prev_rows.empty:
        state["morning_done"] = True
        return
    prev_close  = float(prev_rows["close"].iloc[-1])
    today_rows  = df15[df15["date"] == today]

    vix = feed.get_vix() or BACKTEST_VIX

    ctx = build_morning_context(today_rows, prev_close, vix, index)

    state["skip_day"]    = ctx.skip_day
    state["skip_reason"] = ctx.skip_reason
    state["vix"]         = ctx.vix
    state["gap_pct"]     = ctx.gap_pct
    state["gap_bias"]    = ctx.gap_bias
    state["fcr_pct"]     = ctx.fcr.range_pct
    state["fcr_type"]    = ctx.fcr.day_type
    state["fcr_lot_mult"] = ctx.fcr.lot_multiplier
    state["near_round"]  = ctx.near_round
    state["nearest_round"] = ctx.nearest_round
    state["prev_close"]  = prev_close
    state["morning_done"] = True

    msg = morning_telegram_text(ctx, state.get("capital", TRADING_CAPITAL), PAPER_MODE)
    tg_send(msg)
    logger.info("Morning context: gap=%.2f%% VIX=%.1f FCR=%s skip=%s",
                ctx.gap_pct, ctx.vix, ctx.fcr.day_type, ctx.skip_day)


# ── Utility ───────────────────────────────────────────────────────────────────

def _days_to_expiry(today: date, index: str) -> int:
    from datetime import timedelta
    cfg     = INDEX_CONFIG[index]
    exp_day = cfg.get("expiry_day", "thursday")
    day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
    target  = day_map.get(exp_day.lower(), 3)
    d = today
    for _ in range(8):
        if d.weekday() == target:
            return max((d - today).days, 1)
        d += timedelta(days=1)
    return 3


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Engine V3 — live candle runner")
    parser.add_argument("--candle", action="store_true", help="Process one candle cycle")
    parser.add_argument("--index",  default=ACTIVE_INDEX, help="Index to trade")
    args = parser.parse_args()

    if args.candle:
        run_candle(index=args.index.upper())
    else:
        print("Usage: python3 -m engine_v2.main --candle [--index BANKNIFTY]")


if __name__ == "__main__":
    main()
