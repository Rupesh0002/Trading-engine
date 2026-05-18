#!/usr/bin/env python3
"""
Seed adaptive pattern memory from all backtest signal CSVs.

Replays every closed trade through AdaptiveMemory.record_outcome() so
the adaptive layer starts with pre-populated buckets on day 1 of paper trading
instead of needing weeks of live data to activate.

Deduplicates by signal_id across all CSVs to avoid double-counting overlapping runs.

Usage:
  python3 seed_adaptive_memory.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ml.adaptive import AdaptiveMemory

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def main() -> None:
    csvs = sorted(glob.glob(os.path.join(REPORTS_DIR, "signals_*.csv")))
    if not csvs:
        print("No signal CSVs found in reports/. Run a backtest first.")
        sys.exit(1)

    adaptive = AdaptiveMemory()
    seen_ids: set = set()
    loaded = skipped_dup = skipped_bad = 0

    for csv_path in csvs:
        file_trades = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("trade_taken", "0") != "1":
                    continue

                sig_id = row.get("signal_id", "")
                if sig_id and sig_id in seen_ids:
                    skipped_dup += 1
                    continue
                if sig_id:
                    seen_ids.add(sig_id)

                try:
                    index     = row["index"]
                    direction = row["direction"]
                    pnl       = float(row["pnl"])
                    rsi       = float(row.get("rsi") or 50)
                    adx       = float(row.get("adx") or 0)
                    dte       = int(float(row.get("dte") or 1))
                    time_str  = row.get("time", "10:00:00")
                    hour      = int(time_str.split(":")[0])

                    signal_details = {"adx": adx, "rsi": rsi}
                    adaptive.memory.record(
                        key=__import__("ml.adaptive", fromlist=["_make_key"])._make_key(
                            index, direction, adx, rsi, dte, hour
                        ),
                        pnl=pnl,
                    )
                    file_trades += 1
                    loaded += 1
                except (KeyError, ValueError) as e:
                    skipped_bad += 1

        print(f"  {file_trades:>3} trades  ←  {os.path.basename(csv_path)}")

    print()
    print("─" * 50)
    print(f"  Seeded  : {loaded} trades")
    print(f"  Dupes   : {skipped_dup} skipped (same signal_id)")
    print(f"  Bad rows: {skipped_bad} skipped (parse error)")

    buckets = adaptive.memory.all_buckets()
    active  = sum(1 for b in buckets.values() if b["count"] >= 5)
    blocked = sum(1 for b in buckets.values() if b["count"] >= 5 and b["win_rate"] < 0.30)
    strong  = sum(1 for b in buckets.values() if b["count"] >= 5 and b["win_rate"] >= 0.65)

    print()
    print(f"  Pattern memory: {len(buckets)} buckets total")
    print(f"    Active (≥5 trades) : {active}")
    print(f"    Hard-blocked (<30%): {blocked}")
    print(f"    Strong (>65%)      : {strong}")
    print("─" * 50)
    print()
    print(adaptive.summary())


if __name__ == "__main__":
    main()
