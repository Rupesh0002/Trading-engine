"""
Live ML confidence scoring for the trading engine.
Loads the trained XGBoost model and scores incoming signals.
Returns a confidence probability (0.0 – 1.0).
Threshold is ML_MIN_CONFIDENCE from .env (default 0.65).
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import pandas as pd

from config.settings import ML_MIN_CONFIDENCE, ML_MODEL_PATH
from ml.data_prep import FEATURE_COLS

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Wraps the XGBoost model for live signal scoring.

    Usage in main.py:
        predictor = MLPredictor()
        if predictor.is_ready():
            conf = predictor.confidence(signal, spot, vix, pcr, dte)
            if conf < ML_MIN_CONFIDENCE:
                continue   # skip this trade
    """

    def __init__(self) -> None:
        self._model = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(ML_MODEL_PATH):
            logger.info(
                "ML model not found at %s. "
                "Run: python3 ml/train.py --csv reports/signals_*.csv",
                ML_MODEL_PATH,
            )
            return
        try:
            with open(ML_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            logger.info("ML model loaded from %s", ML_MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load ML model: %s", exc)

    def is_ready(self) -> bool:
        return self._model is not None

    def confidence(
        self,
        signal_details: dict,
        spot: float,
        vix: Optional[float],
        pcr: Optional[float],
        dte: int,
        entry_time_str: str = "10:15:00",
    ) -> float:
        """
        Returns win-probability for a given signal (0.0 – 1.0).
        If model is not loaded, returns 1.0 (no filter applied).
        """
        if not self.is_ready():
            return 1.0

        try:
            from datetime import datetime
            t          = datetime.strptime(entry_time_str, "%H:%M:%S")
            vwap_val   = float(signal_details.get("vwap", spot))
            fib_level  = signal_details.get("fib_level") or spot
            vol_ma     = float(signal_details.get("vol_ma", 1) or 1)
            vol_cur    = float(signal_details.get("current_vol", 0))

            row = {
                "conditions_met": signal_details.get("conditions_met", 4),
                "vwap_distance":  round(spot - vwap_val, 2),
                "near_fib":       int(bool(signal_details.get("near_fib"))),
                "fib_distance":   round(abs(spot - float(fib_level)), 2),
                "rsi":            float(signal_details.get("rsi") or 50),
                "vol_spike":      int(bool(signal_details.get("vol_spike"))),
                "vol_ratio":      round(vol_cur / vol_ma, 2) if vol_ma else 1.0,
                "pcr":            float(pcr) if pcr is not None else 1.0,
                "dte":            dte,
                "hour":           t.hour,
                "minute":         t.minute,
                "day_of_week":    datetime.now().weekday(),
            }

            X   = pd.DataFrame([row])[FEATURE_COLS]
            prob = float(self._model.predict_proba(X)[0][1])
            logger.debug("ML confidence: %.3f (threshold=%.2f)", prob, ML_MIN_CONFIDENCE)
            return prob

        except Exception as exc:
            logger.warning("ML predict error: %s — defaulting to 1.0", exc)
            return 1.0

    def should_trade(
        self,
        signal_details: dict,
        spot: float,
        vix: Optional[float],
        pcr: Optional[float],
        dte: int,
        entry_time_str: str = "10:15:00",
    ) -> tuple[bool, float]:
        """
        Returns (trade_allowed, confidence).
        Convenience wrapper for main.py.
        """
        conf = self.confidence(signal_details, spot, vix, pcr, dte, entry_time_str)
        return conf >= ML_MIN_CONFIDENCE, conf
