"""
CSV trade and signal logger.
File paths come from config/settings.py → .env (TRADE_LOG_FILE, SIGNAL_LOG_FILE).

trades_log.csv  — one row per trade (entry + exit in same row on close)
signals_log.csv — one row per signal evaluation (every candle, fired or not)
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pytz

from config.settings import SIGNAL_LOG_FILE, TRADE_LOG_FILE, TRADING_CAPITAL

IST = pytz.timezone("Asia/Kolkata")

# ── trades_log.csv columns ────────────────────────────────────────────────────
_TRADE_HEADERS = [
    "date", "time", "index", "direction", "signal_strength",
    "strike", "option_type", "expiry",
    "entry_premium", "sl_premium", "tp_premium",
    "lots", "qty", "actual_risk",
    "exit_premium", "exit_reason",
    "pnl", "result",
    "adx", "rsi", "fib_level", "pcr_bias",
    "capital_after",
    # internal tracking (not in spec display but useful for analysis)
    "paper",
]

# ── signals_log.csv columns ───────────────────────────────────────────────────
_SIGNAL_HEADERS = [
    "date", "time", "index", "direction",
    "ema_ok", "rsi_ok", "vwap_ok", "fib_ok",
    "adx_value", "adx_tier",
    "fired", "skip_reason",
]


class TradeLogger:
    """Appends rows to CSV logs for trades and signals."""

    def __init__(self) -> None:
        self._ensure_file(TRADE_LOG_FILE, _TRADE_HEADERS)
        self._ensure_file(SIGNAL_LOG_FILE, _SIGNAL_HEADERS)
        # Running capital estimate — updated after each closed trade
        self._capital = TRADING_CAPITAL

    # ──────────────────────────────────────────────────────────────────────────
    # Trade log  (call once per trade at entry; call again with exit fields at close)
    # ──────────────────────────────────────────────────────────────────────────

    def log_trade_entry(self, trade: Dict[str, Any]) -> None:
        """
        Log an open trade entry.  Exit fields are empty at this point.
        trade dict keys expected:
          index, direction, signal_strength (conditions_met),
          strike, option_type, expiry,
          entry_premium, sl_premium, tp_premium,
          lots, qty, actual_risk,
          adx, rsi, fib_level, pcr_bias, paper
        """
        now = datetime.now(IST)
        row = {h: "" for h in _TRADE_HEADERS}
        row.update({
            "date":            now.strftime("%Y-%m-%d"),
            "time":            now.strftime("%H:%M"),
            "index":           trade.get("index", ""),
            "direction":       trade.get("direction", ""),
            "signal_strength": trade.get("signal_strength", trade.get("conditions_met", "")),
            "strike":          trade.get("strike", ""),
            "option_type":     trade.get("option_type", ""),
            "expiry":          trade.get("expiry", ""),
            "entry_premium":   _r(trade.get("entry_premium")),
            "sl_premium":      _r(trade.get("sl_premium", trade.get("stop_loss", ""))),
            "tp_premium":      _r(trade.get("tp_premium", trade.get("target", ""))),
            "lots":            trade.get("lots", ""),
            "qty":             trade.get("qty", trade.get("quantity", "")),
            "actual_risk":     _r(trade.get("actual_risk")),
            "adx":             _r(trade.get("adx")),
            "rsi":             _r(trade.get("rsi")),
            "fib_level":       _r(trade.get("fib_level")),
            "pcr_bias":        trade.get("pcr_bias", ""),
            "paper":           trade.get("paper", True),
        })
        self._append(TRADE_LOG_FILE, _TRADE_HEADERS, row)

    def log_trade_exit(self, trade: Dict[str, Any]) -> None:
        """
        Append a complete closed-trade row (entry + exit fields together).
        Use this instead of log_trade_entry when you want a single row per trade.
        """
        pnl    = trade.get("pnl", 0.0) or 0.0
        result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
        self._capital += pnl

        now = datetime.now(IST)
        row = {h: "" for h in _TRADE_HEADERS}
        row.update({
            "date":            trade.get("date", now.strftime("%Y-%m-%d")),
            "time":            trade.get("time", now.strftime("%H:%M")),
            "index":           trade.get("index", ""),
            "direction":       trade.get("direction", ""),
            "signal_strength": trade.get("signal_strength", trade.get("conditions_met", "")),
            "strike":          trade.get("strike", ""),
            "option_type":     trade.get("option_type", ""),
            "expiry":          trade.get("expiry", ""),
            "entry_premium":   _r(trade.get("entry_premium")),
            "sl_premium":      _r(trade.get("sl_premium", trade.get("stop_loss", ""))),
            "tp_premium":      _r(trade.get("tp_premium", trade.get("target", ""))),
            "lots":            trade.get("lots", ""),
            "qty":             trade.get("qty", trade.get("quantity", "")),
            "actual_risk":     _r(trade.get("actual_risk")),
            "exit_premium":    _r(trade.get("exit_premium")),
            "exit_reason":     trade.get("exit_reason", ""),
            "pnl":             _r(pnl),
            "result":          result,
            "adx":             _r(trade.get("adx")),
            "rsi":             _r(trade.get("rsi")),
            "fib_level":       _r(trade.get("fib_level")),
            "pcr_bias":        trade.get("pcr_bias", ""),
            "capital_after":   _r(self._capital),
            "paper":           trade.get("paper", True),
        })
        self._append(TRADE_LOG_FILE, _TRADE_HEADERS, row)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal log  (every candle, fired=yes or no)
    # ──────────────────────────────────────────────────────────────────────────

    def log_signal(self, signal: Dict[str, Any]) -> None:
        """
        Log one signal evaluation row.
        signal dict keys expected:
          index, direction,
          details (SignalResult.details dict),
          fired (bool), skip_reason (str)
        """
        now     = datetime.now(IST)
        details = signal.get("details", {})
        adx_val = details.get("adx", 0.0) or 0.0

        # ADX tier: flat / borderline / trending
        adx_threshold = 20.0
        try:
            from config.settings import ADX_THRESHOLD
            adx_threshold = ADX_THRESHOLD
        except ImportError:
            pass
        if adx_val < adx_threshold / 2:
            adx_tier = "flat"
        elif adx_val < adx_threshold:
            adx_tier = "borderline"
        else:
            adx_tier = "trending"

        row = {
            "date":        now.strftime("%Y-%m-%d"),
            "time":        now.strftime("%H:%M"),
            "index":       signal.get("index", details.get("index", "")),
            "direction":   signal.get("direction", "WAIT"),
            "ema_ok":      _bool(details.get("ema_bull") or details.get("ema_bear")),
            "rsi_ok":      _bool(details.get("rsi_bullish") or details.get("rsi_bearish")),
            "vwap_ok":     _bool(details.get("above_vwap") or details.get("below_vwap")),
            "fib_ok":      _bool(details.get("near_fib")),
            "adx_value":   _r(adx_val),
            "adx_tier":    adx_tier,
            "fired":       "yes" if signal.get("fired", signal.get("direction") in ("CALL", "PUT")) else "no",
            "skip_reason": signal.get("skip_reason", ""),
        }
        self._append(SIGNAL_LOG_FILE, _SIGNAL_HEADERS, row)

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy compatibility — old main.py called log_trade() and log_signal()
    # Keep these so existing code doesn't break
    # ──────────────────────────────────────────────────────────────────────────

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Legacy shim — routes to log_trade_entry or log_trade_exit based on action."""
        action = trade.get("action", "EXIT")
        if action == "ENTRY":
            self.log_trade_entry(trade)
        else:
            self.log_trade_exit(trade)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_file(path: str, headers: list) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=headers).writeheader()

    @staticmethod
    def _append(path: str, headers: list, row: Dict[str, Any]) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writerow(row)


def _r(val: Any, decimals: int = 2) -> str:
    """Round a numeric value to string; return '' for None."""
    if val is None or val == "":
        return ""
    try:
        return str(round(float(val), decimals))
    except (TypeError, ValueError):
        return str(val)


def _bool(val: Any) -> str:
    return "yes" if val else "no"
