"""
ML data preparation — reads the signal CSV produced by the backtest
and outputs a clean feature matrix ready for training.

Features used (all numeric, no lookahead):
  conditions_met, vwap_distance, near_fib, fib_distance,
  rsi, vol_spike, vol_ratio, pcr, dte,
  hour, minute, day_of_week

Label:
  outcome  →  1 = profitable trade, 0 = loss
  (rows with outcome = -1 are skipped — signal fired but trade not taken)
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "conditions_met",
    "vwap_distance",
    "near_fib",
    "fib_distance",
    "rsi",
    "adx",
    "ema_bull",
    "ema_bear",
    "direction_int",
    "vol_spike",
    "vol_ratio",
    "pcr",
    "dte",
    "hour",
    "minute",
    "day_of_week",
]
LABEL_COL = "outcome"


def load_and_prepare(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the signal CSV, engineers features, returns (X, y).
    Only rows where a trade was actually taken (trade_taken=1) are used.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Keep only rows with actual trade outcomes
    df = df[df["trade_taken"] == 1].copy()
    logger.info("%d rows after filtering to trade_taken=1", len(df))

    if df.empty:
        raise ValueError(
            f"No tradeable signals found in {csv_path}. "
            "Run the backtest over a longer date range or lower MIN_CONDITIONS."
        )

    # Time features
    df["hour"]        = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
    df["minute"]      = pd.to_datetime(df["time"], format="%H:%M:%S").dt.minute
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek  # 0=Mon, 4=Fri

    # Fill missing numeric columns with median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)

    wins   = int(y.sum())
    losses = len(y) - wins
    logger.info(
        "Dataset ready: %d samples | %d wins (%.0f%%) | %d losses",
        len(y), wins, wins / len(y) * 100, losses,
    )
    return X, y


def load_multiple(csv_paths: list[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine signal CSVs from multiple indices or date ranges."""
    frames_x, frames_y = [], []
    for path in csv_paths:
        X, y = load_and_prepare(path)
        frames_x.append(X)
        frames_y.append(y)
    X_all = pd.concat(frames_x, ignore_index=True)
    y_all = pd.concat(frames_y, ignore_index=True)
    logger.info("Combined dataset: %d total samples", len(y_all))
    return X_all, y_all
