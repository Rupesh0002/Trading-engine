"""
engine_v3 — Live Short Options Paper Trader.

Runs one candle at a time (called by GitHub Actions every 5 min).
Sells the opposite-side ATM option when a signal fires:
  CALL signal → Sell ATM PUT
  PUT  signal → Sell ATM CALL

State persists in state_short.json between candles.
Trades logged to logs/short_paper_trades.csv.
Telegram alerts on every entry, exit, and day summary.

Capital   : ₹1,00,000 (NIFTY, 1 lot)
Margin    : ₹65,000/lot
SL        : premium rises to 150% of entry  (50% of credit as max loss)
TP        : premium falls to 30% of entry   (captured 70% of credit)
Hard close: 14:30 IST
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

import pytz

from config.settings import INDEX_CONFIG, REPORTS_DIR, TRADING_CAPITAL

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Constants ─────────────────────────────────────────────────────────────────
_INDEX         = "NIFTY"
_LOT_SIZE      = INDEX_CONFIG.get("NIFTY", {}).get("lot_size", 50)
_MARGIN        = 65_000
_CAPITAL       = TRADING_CAPITAL            # ₹1,00,000
_MAX_LOTS      = 1
_RISK_PCT      = 0.015
_CAPITAL_USE   = 0.85

_ATM_DELTA     = 0.50
_THETA_5M      = 0.004
_SHORT_SL_PCT  = 1.50     # exit if premium reaches 150% of entry
_SHORT_TP_PCT  = 0.30     # exit if premium falls to 30% of entry
_HARD_CLOSE    = time(14, 30)
_SIGNAL_START  = time(10,  0)
_ORB_END       = time( 9, 45)
_SLIPPAGE_PTS  = 1.0
_BROKERAGE     = 40.0
_STT_PCT       = 0.0003

_STATE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "state_short.json")
_LOG_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "logs", "short_paper_trades.csv")
_LOG_FIELDS    = [
    "date", "index", "setup", "direction", "signal_time", "entry_time",
    "entry_spot", "entry_premium", "lots", "quantity",
    "exit_time", "exit_premium", "gross_pnl", "net_pnl", "pnl_pct",
    "exit_reason", "running_capital", "dte",
]


class ShortLiveEngine:
    """Per-candle short options paper trader."""

    def __init__(self, kite) -> None:
        self.kite = kite
        self._state: Dict[str, Any] = {}

    # ── Public entry point ────────────────────────────────────────────

    def run_candle(self) -> None:
        """
        Called once per 5-min candle by GitHub Actions.
        Load state → process → save state → alert.
        """
        now = datetime.now(IST)
        t5  = now.time().replace(second=0, microsecond=0)

        # Only run during signal window
        if t5 < time(9, 15) or t5 > time(15, 30):
            return
        if now.weekday() >= 5:
            return

        self._load_state(now.date())

        try:
            from data.feed import DataFeed
            feed   = DataFeed(self.kite)
            df5    = feed.get_historical_candles(
                now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"),
                index=_INDEX, interval="5minute",
            )
            if df5 is None or df5.empty:
                logger.warning("[SHORT-LIVE] No 5-min data for today.")
                return
            if "date" in df5.columns and "timestamp" not in df5.columns:
                df5.rename(columns={"date": "timestamp"}, inplace=True)
            import pandas as pd
            df5["timestamp"] = pd.to_datetime(df5["timestamp"])
            df5 = df5.sort_values("timestamp").reset_index(drop=True)

            # Current candle = last complete candle
            if len(df5) < 1:
                return
            row   = df5.iloc[-1]
            c5    = float(row["close"])
            ct5   = pd.Timestamp(row["timestamp"]).time()

            # Get 15-min data for levels
            from datetime import timedelta
            warmup = (now.date() - timedelta(days=65)).strftime("%Y-%m-%d")
            df15   = feed.get_historical_candles(
                warmup, now.strftime("%Y-%m-%d"),
                index=_INDEX, interval="15minute",
            )
            if df15 is not None and "date" in df15.columns:
                df15.rename(columns={"date": "timestamp"}, inplace=True)
            if df15 is not None:
                df15["timestamp"] = pd.to_datetime(df15["timestamp"])

            # Build levels from prior days
            from engine_v3.levels import (
                build_prior_levels, compute_orb, update_vwap, compute_rsi,
            )
            prior_15 = df15[df15["timestamp"].dt.date < now.date()].copy() \
                       if df15 is not None else pd.DataFrame()
            today_open = float(df5.iloc[0]["open"])
            levels     = build_prior_levels(prior_15, today_open, index=_INDEX)

            # Build VWAP over today's candles
            vwap_num, vwap_den = 0.0, 0.0
            prev_vwap = today_open
            for _, r in df5.iloc[:-1].iterrows():
                vwap_val, vwap_num, vwap_den = update_vwap(vwap_num, vwap_den, r)
            vwap_val_prev = vwap_val if df5.shape[0] > 1 else today_open
            vwap_val, vwap_num, vwap_den = update_vwap(vwap_num, vwap_den, row)
            levels.vwap = vwap_val

            # Build ORB
            if ct5 >= _ORB_END and not self._state.get("orb_ready"):
                oh, ol = compute_orb(df5)
                if oh > 0:
                    levels.orb_high  = oh
                    levels.orb_low   = ol
                    levels.orb_range = oh - ol
                    levels.orb_ready = True
                    self._state["orb_ready"]  = True
                    self._state["orb_high"]   = oh
                    self._state["orb_low"]    = ol
                    self._state["orb_range"]  = oh - ol
            elif self._state.get("orb_ready"):
                levels.orb_high  = self._state["orb_high"]
                levels.orb_low   = self._state["orb_low"]
                levels.orb_range = self._state["orb_range"]
                levels.orb_ready = True

            # Get VIX (use real VIX for live premium sizing)
            from data.vix_fetcher import get_vix
            vix = get_vix(self.kite) or 15.0
            self._state["last_vix"] = vix

            # DTE
            from config.events_calendar import get_next_expiry
            expiry = get_next_expiry(_INDEX, now.date(), days_buffer=0)
            dte    = max((expiry - now.date()).days, 1)

            prev_df = df5.iloc[:-1] if len(df5) > 1 else df5.iloc[:0]

            # ── Step 1: manage open trade ─────────────────────────────
            if self._state.get("open_trade"):
                self._manage_trade(row, ct5, vix, dte, now)

            # ── Step 2: look for new signal ───────────────────────────
            if (not self._state.get("open_trade")
                    and ct5 >= _SIGNAL_START
                    and ct5 < _HARD_CLOSE
                    and self._state.get("trades_today", 0) < 2
                    and levels.orb_ready):

                closes = [float(r["close"]) for _, r in prev_df.iterrows()]
                rsi    = compute_rsi(closes[-20:]) if len(closes) >= 5 else 50.0

                sig = None
                setup = None
                from engine_v3.setup_orb  import check_orb
                from engine_v3.setup_pdh  import check_pdh_pdl
                from engine_v3.setup_vwap import check_vwap
                from engine_v3.setup_smc  import check_smc

                sig_orb = check_orb(row, prev_df, levels, vwap_val, rsi, index=_INDEX)
                if sig_orb:
                    sig, setup = sig_orb, "ORB"
                else:
                    sig_pdh = check_pdh_pdl(row, prev_df, levels, vwap_val)
                    if sig_pdh:
                        sig, setup = sig_pdh, "PDH_PDL"
                    else:
                        sig_vwap = check_vwap(row, prev_df, vwap_val, vwap_val_prev)
                        if sig_vwap:
                            sig, setup = sig_vwap, "VWAP"
                        else:
                            sig_smc = check_smc(row, prev_df)
                            if sig_smc:
                                sig, setup = sig_smc, "SMC"

                if sig:
                    self._enter_trade(sig, setup, c5, ct5, vix, dte, now)

        except Exception as exc:
            logger.error("[SHORT-LIVE] Candle error: %s", exc, exc_info=True)
        finally:
            self._save_state()

        # Day summary at 14:35
        if t5 >= time(14, 35) and not self._state.get("eod_sent"):
            self._send_day_summary()
            self._state["eod_sent"] = True
            self._save_state()

    # ── Trade management ──────────────────────────────────────────────

    def _manage_trade(self, row, ct5: time, vix: float, dte: int,
                      now: datetime) -> None:
        tr     = self._state["open_trade"]
        c5     = float(row["close"])
        prev_c = self._state.get("prev_close", tr["entry_spot"])
        ep     = tr["entry_premium"]
        direction = tr["direction"]

        # Update current premium (sold option model)
        cp = tr.get("current_premium", ep)
        cp = max(cp * (1 - _THETA_5M), 0.1)
        if direction == "CALL":      # sold PUT: spot up → PE falls
            px_d = (prev_c - c5) * _ATM_DELTA
        else:                        # sold CALL: spot down → CE falls
            px_d = (c5 - prev_c) * _ATM_DELTA
        cp = max(cp + px_d, 0.1)
        tr["current_premium"] = round(cp, 2)
        self._state["prev_close"] = c5

        # Hard close
        if ct5 >= _HARD_CLOSE:
            self._exit_trade(cp, "Hard close (14:30)", ct5, now)
            return

        # SL: premium rose to 150% of entry
        if cp >= ep * _SHORT_SL_PCT:
            self._exit_trade(cp, "Short SL", ct5, now)
            return

        # TP: premium fell to 30% of entry
        if cp <= ep * _SHORT_TP_PCT:
            self._exit_trade(cp, "Short TP", ct5, now)
            return

    def _enter_trade(self, sig, setup: str, spot: float, ct5: time,
                     vix: float, dte: int, now: datetime) -> None:
        ep  = _sim_premium_vix(spot, dte, vix)
        qty = _MAX_LOTS * _LOT_SIZE

        # Margin check
        margin_lots = int(_CAPITAL * _CAPITAL_USE / _MARGIN)
        lots        = max(1, min(_MAX_LOTS, margin_lots))
        qty         = lots * _LOT_SIZE

        direction = sig.direction
        self._state["open_trade"] = {
            "setup":           setup,
            "direction":       direction,
            "signal_time":     str(sig.signal_time),
            "entry_time":      str(ct5),
            "entry_spot":      round(spot, 2),
            "entry_premium":   round(ep, 2),
            "current_premium": round(ep, 2),
            "lots":            lots,
            "quantity":        qty,
            "dte":             dte,
        }
        self._state["prev_close"]     = spot
        self._state["trades_today"]   = self._state.get("trades_today", 0) + 1

        sold_side = "PUT" if direction == "CALL" else "CALL"
        msg = (
            f"📋 <b>SHORT {sold_side} — PAPER</b>\n"
            f"Index : NIFTY\n"
            f"Setup : {setup}  ({direction} signal)\n"
            f"Spot  : ₹{spot:,.2f}\n"
            f"Sold  : ATM {sold_side} @ ₹{ep:.2f}  (DTE {dte})\n"
            f"Lots  : {lots}  ×{_LOT_SIZE} = {qty} qty\n"
            f"SL    : ₹{ep * _SHORT_SL_PCT:.2f}  (+50%)\n"
            f"TP    : ₹{ep * _SHORT_TP_PCT:.2f}  (−70%)\n"
            f"Hard  : 14:30\n"
            f"VIX   : {vix:.1f}\n"
            f"🕐 {str(ct5)[:5]}"
        )
        _telegram(msg)
        logger.info("[SHORT-LIVE] ENTER %s/%s | spot=%.2f prem=₹%.2f lots=%d",
                    setup, direction, spot, ep, lots)

    def _exit_trade(self, exit_prem: float, reason: str, ct5: time,
                    now: datetime) -> None:
        tr  = self._state.pop("open_trade", {})
        if not tr:
            return

        ep  = tr["entry_premium"]
        qty = tr["quantity"]

        entry_recv = (ep        - _SLIPPAGE_PTS) * qty
        exit_pay   = (exit_prem + _SLIPPAGE_PTS) * qty
        gross      = entry_recv - exit_pay
        stt        = exit_pay * _STT_PCT
        net        = gross - _BROKERAGE - stt
        pct        = net / entry_recv * 100 if entry_recv > 0 else 0.0

        cap = self._state.get("running_capital", _CAPITAL)
        cap += net
        self._state["running_capital"] = round(cap, 2)
        self._state["daily_pnl"] = self._state.get("daily_pnl", 0.0) + net
        self._state.pop("prev_close", None)

        # Log to CSV
        self._log_trade({
            "date":             str(now.date()),
            "index":            _INDEX,
            "setup":            tr["setup"],
            "direction":        tr["direction"],
            "signal_time":      tr["signal_time"],
            "entry_time":       tr["entry_time"],
            "entry_spot":       tr["entry_spot"],
            "entry_premium":    ep,
            "lots":             tr["lots"],
            "quantity":         qty,
            "exit_time":        str(ct5),
            "exit_premium":     round(exit_prem, 2),
            "gross_pnl":        round(gross, 2),
            "net_pnl":          round(net, 2),
            "pnl_pct":          round(pct, 2),
            "exit_reason":      reason,
            "running_capital":  round(cap, 2),
            "dte":              tr["dte"],
        })

        emoji  = "✅" if net > 0 else "❌"
        sold   = "PUT" if tr["direction"] == "CALL" else "CALL"
        msg = (
            f"{emoji} <b>EXIT {sold} — PAPER</b>\n"
            f"Setup  : {tr['setup']}  ({tr['direction']} signal)\n"
            f"Sold @ : ₹{ep:.2f}  →  Closed @ ₹{exit_prem:.2f}\n"
            f"Net P&L: ₹{net:+,.0f}  ({pct:+.1f}%)\n"
            f"Reason : {reason}\n"
            f"Capital: ₹{cap:,.0f}\n"
            f"🕐 {str(ct5)[:5]}"
        )
        _telegram(msg)
        logger.info("[SHORT-LIVE] EXIT %s/%s | prem ₹%.2f→₹%.2f | net ₹%.0f | %s",
                    tr["setup"], tr["direction"], ep, exit_prem, net, reason)

    # ── Day summary ───────────────────────────────────────────────────

    def _send_day_summary(self) -> None:
        pnl    = self._state.get("daily_pnl", 0.0)
        trades = self._state.get("trades_today", 0)
        cap    = self._state.get("running_capital", _CAPITAL)
        vix    = self._state.get("last_vix", 0.0)
        emoji  = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>SHORT OPTIONS — PAPER DAY SUMMARY</b>\n"
            f"Index   : NIFTY\n"
            f"Trades  : {trades}\n"
            f"Day P&L : ₹{pnl:+,.0f}\n"
            f"Capital : ₹{cap:,.0f}\n"
            f"VIX     : {vix:.1f}"
        )
        _telegram(msg)

    # ── State persistence ─────────────────────────────────────────────

    def _load_state(self, today: date) -> None:
        path = _state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {}
        else:
            self._state = {}

        if self._state.get("date") != str(today):
            # New day — reset daily state, keep running capital
            cap = self._state.get("running_capital", _CAPITAL)
            self._state = {
                "date":            str(today),
                "running_capital": cap,
                "daily_pnl":       0.0,
                "trades_today":    0,
                "open_trade":      None,
                "orb_ready":       False,
                "eod_sent":        False,
            }

    def _save_state(self) -> None:
        path = _state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as exc:
            logger.error("[SHORT-LIVE] Could not save state: %s", exc)

    def _log_trade(self, row: dict) -> None:
        os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
        exists = os.path.exists(_LOG_FILE)
        try:
            with open(_LOG_FILE, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
                if not exists:
                    w.writeheader()
                w.writerow(row)
        except Exception as exc:
            logger.error("[SHORT-LIVE] Could not log trade: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "state_short.json"
    )


def _sim_premium_vix(spot: float, dte_days: int, vix: float) -> float:
    sigma   = max(vix, 10.0) / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), 30.0)


def _telegram(text: str) -> None:
    try:
        from telegram_alerts import _send
        _send(text)
    except Exception:
        pass
