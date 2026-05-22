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
    Includes actual trades (trade_taken=1) AND shadow signals (trade_taken=0, outcome != -1).
    Handles both old CSV format (conditions_met) and new format (conviction_score).
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Use both actual trades (trade_taken=1) AND shadow signals (trade_taken=0 with outcome != -1).
    # Shadow signals are capacity-blocked or near-miss signals with hypothetical outcomes,
    # giving the ML counterfactual data about patterns that existed but weren't traded.
    df = df[df["outcome"] != -1].copy()
    n_actual = int((df["trade_taken"] == 1).sum())
    n_shadow = int((df["trade_taken"] == 0).sum())
    logger.info("%d rows after filtering (outcome != -1)", n_actual + n_shadow)
    logger.info("  (%d actual trades + %d shadow/counterfactual signals)", n_actual, n_shadow)

    # Normalise column names: old CSV uses 'conditions_met' (3-5 scale),
    # new format has both 'conditions_met' (3-5) and 'conviction_score' (0-100).
    # If conditions_met is missing or is on the 0-100 scale, remap it.
    if "conditions_met" not in df.columns or (df["conditions_met"].fillna(0) == 0).all():
        if "conviction_score" in df.columns:
            # Remap 0-100 → 3-5 to match old format
            df["conditions_met"] = df["conviction_score"].apply(
                lambda s: 5 if s >= 93 else (4 if s >= 80 else 3)
            )
            logger.info("  Mapped conviction_score (0-100) → conditions_met (3-5)")
    elif "conditions_met" in df.columns:
        # If values are on 0-100 scale (not 0-5), normalise them
        cm_max = df["conditions_met"].max()
        if cm_max > 10:
            df["conditions_met"] = df["conditions_met"].apply(
                lambda s: 5 if s >= 93 else (4 if s >= 80 else 3)
            )
            logger.info("  Normalised conditions_met from 0-100 → 3-5 scale")

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
