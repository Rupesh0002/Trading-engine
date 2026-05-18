#!/usr/bin/env python3
"""
Backtest runner — NSE F&O options strategy.

Usage:
  python3 run_backtest.py                            # uses BACKTEST_START/END from .env
  python3 run_backtest.py --from 2026-05-10 --to 2026-05-17
  python3 run_backtest.py --from 2026-05-10 --to 2026-05-17 --index BANKNIFTY

All other parameters (capital, R:R, min conditions) always come from .env.
"""
from __future__ import annotations

import argparse
import sys

from config.settings import (
    ACTIVE_INDEX,
    BACKTEST_END,
    BACKTEST_START,
    BACKTEST_VIX,
    MIN_CONDITIONS,
    RISK_REWARD_RATIO,
    SOFT_TARGET_PCT,
    HARD_TARGET_PCT,
    STOP_LOSS_PCT,
    TRADING_CAPITAL,
    validate_settings,
)
from utils.logger import get_logger

logger = get_logger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the trading engine on historical data.")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (overrides .env BACKTEST_START)")
    parser.add_argument("--to",   dest="to_date",   default=None,
                        help="End date YYYY-MM-DD   (overrides .env BACKTEST_END)")
    parser.add_argument("--index", dest="index",    default=ACTIVE_INDEX,
                        help="Index to backtest: NIFTY / BANKNIFTY / SENSEX")
    args = parser.parse_args()

    from_date = args.from_date or BACKTEST_START
    to_date   = args.to_date   or BACKTEST_END
    index     = args.index.upper()

    validate_settings()

    print()
    print("─" * 60)
    print(f"  BACKTEST")
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
        print(f"\n  Total: {results.total_trades} trades | "
              f"Win rate: {results.win_rate:.1f}% | "
              f"PnL: ₹{results.total_pnl:+,.2f} ({results.total_return_pct:+.2f}%)")
    else:
        print("  No trades were generated.")
        print("  Try: lowering MIN_CONDITIONS in .env, or widening the date range.")
    print()


if __name__ == "__main__":
    main()
