#!/usr/bin/env python3
"""
Trading Engine — entry point.

Usage:
  python main.py                → run live scheduler (local / paper mode)
  python main.py --candle       → run one candle check and exit (GitHub Actions)
  python main.py --backtest     → run 4-month backtest using backtest/engine.py
  python main.py --status       → print today's trades from logs/trade_log.csv
  python main.py --summary      → print full trade stats from logs/trade_log.csv

To switch to live trading: set PAPER_MODE=False in .env directly.
All parameters come from .env — never edit Python files for config.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import pytz

from config.settings import (
    ACTIVE_INDEX,
    ACTIVE_INDICES,
    BACKTEST_END,
    BACKTEST_START,
    PAPER_MODE,
    TRADE_LOG_FILE,
    TRADING_CAPITAL,
    validate_settings,
)
from utils.logger import get_logger

logger = get_logger("main")
IST    = pytz.timezone("Asia/Kolkata")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE F&O intraday options trading engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest using BACKTEST_START/END from .env (or --from/--to)",
    )
    parser.add_argument("--from",  dest="from_date", default=None, help="Backtest start YYYY-MM-DD")
    parser.add_argument("--to",    dest="to_date",   default=None, help="Backtest end   YYYY-MM-DD")
    parser.add_argument("--index", dest="index",     default=ACTIVE_INDEX, help="Index to backtest")
    parser.add_argument(
        "--candle", action="store_true",
        help="Run one candle check and exit (GitHub Actions per-candle mode)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print today's trades from trade log",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print full trade statistics from trade log",
    )
    args = parser.parse_args()

    if args.candle:
        _run_candle()
    elif args.backtest:
        _run_backtest(
            from_date=args.from_date or BACKTEST_START,
            to_date=args.to_date     or BACKTEST_END,
            index=args.index.upper(),
        )
    elif args.status:
        _show_status()
    elif args.summary:
        _show_summary()
    else:
        _run_scheduler()


# ── Candle mode (GitHub Actions) ──────────────────────────────────────────────

def _run_candle() -> None:
    """
    Single-candle execution for GitHub Actions per-candle mode.
    Loads state.json → checks one candle → saves state.json → exits.
    state.json is always written (even on auth failure) so GitHub Actions
    can always commit it to the repo.
    """
    validate_settings()

    # Ensure Neon DB tables exist before anything else runs
    from database.connection import init_db
    init_db()

    try:
        from config.auth import get_kite_client
        kite = get_kite_client()
        from scheduler import TradingScheduler
        TradingScheduler(kite).run_once()
        # run_once() writes state.json via its own finally block
    except Exception as exc:
        logger.error("Candle init failed: %s", exc, exc_info=True)
        try:
            from telegram_alerts import send_auth_failure
            send_auth_failure(str(exc))
        except Exception:
            pass
        _write_fallback_state(error=str(exc))


def _write_fallback_state(error: str = "") -> None:
    """Write a minimal state.json when the scheduler cannot be initialised."""
    import json as _json
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
    # Don't overwrite a valid existing state
    if os.path.exists(state_path):
        try:
            with open(state_path) as _f:
                existing = _json.load(_f)
            if existing.get("date"):
                logger.info("[STATE] Existing state.json kept (init failed: %s)", error)
                return
        except Exception:
            pass
    now = datetime.now(IST)
    state = {
        "date":            now.strftime("%Y-%m-%d"),
        "last_run_time":   now.strftime("%H:%M"),
        "running_capital": TRADING_CAPITAL,
        "daily_pnl":       0.0,
        "paper_mode":      PAPER_MODE,
        "open_positions":  [],
        "daily_trades":    {},
        "error":           error,
    }
    try:
        with open(state_path, "w") as _f:
            _json.dump(state, _f, indent=2, default=str)
        logger.info("[STATE] Fallback state.json written to %s", state_path)
    except Exception as write_exc:
        logger.error("[STATE] Could not write fallback state.json: %s", write_exc)


# ── Scheduler (default mode) ───────────────────────────────────────────────────

def _run_scheduler() -> None:
    validate_settings()

    mode = "PAPER" if PAPER_MODE else "LIVE ← REAL MONEY"
    logger.info("Starting engine in %s mode.", mode)

    from config.auth import get_kite_client
    kite = get_kite_client()

    # Send startup Telegram notification
    try:
        from telegram_alerts import send_startup
        send_startup(ACTIVE_INDICES, PAPER_MODE)
    except Exception:
        pass

    from scheduler import TradingScheduler
    TradingScheduler(kite).start()


# ── Backtest ───────────────────────────────────────────────────────────────────

def _run_backtest(from_date: str, to_date: str, index: str) -> None:
    from config.settings import (
        MIN_CONDITIONS, RISK_REWARD_RATIO,
        SOFT_TARGET_PCT, HARD_TARGET_PCT,
        STOP_LOSS_PCT, BACKTEST_VIX,
    )
    validate_settings()

    print()
    print("─" * 60)
    print("  BACKTEST")
    print(f"  Index   : {index}")
    print(f"  Period  : {from_date} → {to_date}")
    print(f"  Capital : ₹{TRADING_CAPITAL:,.0f}")
    print(f"  Signal  : min {MIN_CONDITIONS}/5 conditions")
    print(f"  Exit    : {SOFT_TARGET_PCT*100:.1f}% (2.5×) – {HARD_TARGET_PCT*100:.1f}% (3×)")
    print(f"  SL      : {STOP_LOSS_PCT*100:.0f}%   R:R 1:{RISK_REWARD_RATIO:.1f}")
    print(f"  VIX     : {BACKTEST_VIX:.1f} (assumed for premium simulation)")
    print("─" * 60)
    print()

    from config.auth import get_kite_client
    kite = get_kite_client()

    from backtest.engine import BacktestEngine
    engine  = BacktestEngine(kite, index=index)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary())
    print()

    if results.trades:
        print("  TRADE LOG")
        print("  " + "─" * 106)
        print(
            f"  {'#':>3}  {'Date':<12} {'Dir':<5} "
            f"{'EntrySpot':>10} {'ExitSpot':>10} "
            f"{'EntryPrem':>10} {'ExitPrem':>10} "
            f"{'Qty':>5} {'PnL':>10} {'%':>7}  Cond  Reason"
        )
        print("  " + "─" * 106)
        for i, t in enumerate(results.trades, 1):
            print(
                f"  {i:>3}  {t.date:<12} {t.direction:<5} "
                f"{t.entry_spot:>10.0f} {t.exit_spot:>10.0f} "
                f"{t.entry_premium:>10.2f} {t.exit_premium:>10.2f} "
                f"{t.quantity:>5} {t.pnl:>+10.2f} {t.pnl_pct:>+6.1f}%  "
                f"{t.conditions_met}/5   {t.exit_reason}"
            )
        print("  " + "─" * 106)
        print(
            f"\n  Total: {results.total_trades} trades | "
            f"Win rate: {results.win_rate:.1f}% | "
            f"PnL: ₹{results.total_pnl:+,.2f} ({results.total_return_pct:+.2f}%)"
        )
    else:
        print("  No trades generated.")
        print("  Try: lowering MIN_CONDITIONS in .env, or widening the date range.")
    print()


# ── Status — today's trades ────────────────────────────────────────────────────

def _show_status() -> None:
    today = datetime.now(IST).strftime("%Y-%m-%d")
    rows  = _read_trade_log()
    today_rows = [r for r in rows if r.get("date") == today]

    print()
    print(f"  TODAY'S TRADES  ({today})")
    print("  " + "─" * 90)
    if not today_rows:
        print("  No trades today.")
    else:
        for i, r in enumerate(today_rows, 1):
            pnl = r.get("pnl", "")
            print(
                f"  {i:>2}. [{r.get('time','')}]  "
                f"{r.get('index',''):<10} {r.get('direction',''):<5} "
                f"Strike {r.get('strike','')}  "
                f"Entry ₹{r.get('entry_premium','')}  "
                f"Exit ₹{r.get('exit_premium','')}  "
                f"PnL ₹{pnl}  "
                f"{r.get('result','')}  "
                f"{r.get('exit_reason','')}"
            )
    print("  " + "─" * 90)
    print()


# ── Summary — full log statistics ─────────────────────────────────────────────

def _show_summary() -> None:
    rows = _read_trade_log()
    closed = [r for r in rows if r.get("exit_reason") not in ("", None, "OPEN")]

    if not closed:
        print("\n  No completed trades in log.\n")
        return

    total  = len(closed)
    wins   = sum(1 for r in closed if r.get("result") == "WIN")
    losses = sum(1 for r in closed if r.get("result") == "LOSS")
    pnl_list = []
    for r in closed:
        try:
            pnl_list.append(float(r.get("pnl") or 0))
        except (ValueError, TypeError):
            pass

    total_pnl  = sum(pnl_list)
    win_rate   = wins / total * 100 if total else 0
    gross_win  = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print()
    print("─" * 60)
    print("  TRADE SUMMARY — all time")
    print("─" * 60)
    print(f"  Total trades  : {total}")
    print(f"  Wins / Losses : {wins} / {losses}")
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Total P&L     : ₹{total_pnl:+,.2f}")
    print(f"  Return        : {total_pnl/TRADING_CAPITAL*100:+.2f}%")
    print(f"  Profit factor : {pf:.2f}")
    print("─" * 60)
    print()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _read_trade_log() -> list:
    if not os.path.exists(TRADE_LOG_FILE):
        return []
    try:
        with open(TRADE_LOG_FILE, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        logger.error("Could not read trade log: %s", exc)
        return []


if __name__ == "__main__":
    main()
