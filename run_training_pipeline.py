#!/usr/bin/env python3
"""
One-command ML training pipeline.

Runs a 1-year backtest for every active index, combines all signal CSVs,
and trains the XGBoost model. Run this whenever you want to refresh the model.

Usage:
  python3 run_training_pipeline.py                          # past 1 year, all indices
  python3 run_training_pipeline.py --from 2025-01-01        # custom start, all indices
  python3 run_training_pipeline.py --index NIFTY BANKNIFTY  # specific indices only
  python3 run_training_pipeline.py --skip-backtest          # train on existing CSVs

Output:
  reports/signals_<INDEX>_<START>_<END>.csv  (one per index)
  ml/models/xgboost_model.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import date, timedelta

from config.settings import ACTIVE_INDICES, BACKTEST_VIX, validate_settings
from utils.logger import get_logger

logger = get_logger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest all indices + train XGBoost model in one step."
    )
    parser.add_argument(
        "--from", dest="from_date", default=None,
        help="Backtest start YYYY-MM-DD (default: 1 year ago)",
    )
    parser.add_argument(
        "--to", dest="to_date", default=None,
        help="Backtest end YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--index", dest="indices", nargs="+", default=None,
        help="Indices to backtest (default: all from ACTIVE_INDICES in .env)",
    )
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip running backtests — use existing signal CSVs in reports/",
    )
    args = parser.parse_args()

    to_date   = args.to_date   or str(date.today())
    from_date = args.from_date or str(date.today() - timedelta(days=365))
    indices   = [i.upper() for i in args.indices] if args.indices else list(ACTIVE_INDICES)

    validate_settings()

    print()
    print("═" * 60)
    print("  ML TRAINING PIPELINE")
    print("═" * 60)
    print(f"  Period  : {from_date} → {to_date}")
    print(f"  Indices : {', '.join(indices)}")
    print(f"  VIX     : {BACKTEST_VIX:.1f} (assumed for premium simulation)")
    print("═" * 60)

    signal_csvs: list[str] = []

    # ── Step 1: Run backtests ──────────────────────────────────────────────
    if not args.skip_backtest:
        from config.auth import get_kite_client
        from backtest.engine import BacktestEngine

        kite = get_kite_client()

        for index in indices:
            print(f"\n  Running backtest: {index} ({from_date} → {to_date}) ...")
            engine  = BacktestEngine(kite, index=index)
            results = engine.run(from_date=from_date, to_date=to_date)

            print(results.summary())

            if results.signal_csv and os.path.exists(results.signal_csv):
                signal_csvs.append(results.signal_csv)
                print(f"  Signal CSV → {results.signal_csv}")
            else:
                print(f"  [WARN] No signal CSV generated for {index}.")
    else:
        # ── Step 1b: Collect existing CSVs ───────────────────────────────
        print("\n  --skip-backtest: collecting existing signal CSVs ...")
        for index in indices:
            pattern = os.path.join("reports", f"signals_{index}_*.csv")
            found   = sorted(glob.glob(pattern))
            if found:
                # Use the most recent one per index
                signal_csvs.append(found[-1])
                print(f"  Using existing CSV: {found[-1]}")
            else:
                print(f"  [WARN] No existing signal CSV found for {index} — skipping.")

    if not signal_csvs:
        print("\n  No signal CSVs available. Run without --skip-backtest first.")
        sys.exit(1)

    # ── Step 2: Train the model ────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  TRAINING XGBoost model ...")
    print("─" * 60)

    from ml.train import train
    train(signal_csvs)

    print()
    print("═" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Model saved → ml/models/xgboost_model.pkl")
    print()
    print("  Next steps:")
    print("  1. Run: python main.py  (ML filter is active automatically)")
    from config.settings import ML_MIN_CONFIDENCE
    print(f"  2. Tune ML_MIN_CONFIDENCE in .env (current: {ML_MIN_CONFIDENCE})")
    print("  3. Re-run this pipeline monthly to refresh the model")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
