"""
Trading scheduler — runs the engine every 15 minutes during market hours.

Schedule:
  09:00 IST  → morning_setup  (fetch PCR, set daily bias, Telegram summary)
  10:00–15:00 → candle_cycle  (signal check + position management, every 15 min)
  15:00 IST  → hard_close     (force-close all open positions)
  15:30 IST  → day_summary    (P&L report + Telegram)

Rules:
  - Only runs on weekdays (Mon–Fri)
  - Each job must complete in < 30 seconds
  - All state lives in TradingScheduler instance (no globals)
  - PAPER_MODE=True → no real orders placed
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import schedule
import pytz

import types as _types

from config.settings import (
    ACTIVE_INDICES,
    INDEX_CONFIG,
    MAX_DAILY_TRADES,
    MAX_DAILY_TRADES_OTM,
    MAX_OPEN_POSITIONS,
    ML_MIN_CONFIDENCE,
    MIN_CONDITIONS,
    OTM_STRIKES_AWAY,
    PAPER_MODE,
    RISK_PER_TRADE_PCT,
    STRONG_SIGNAL_THRESHOLD,
    TRADE_LOG_FILE,
    TRADING_CAPITAL,
)
from utils.logger import get_logger
from utils.time_checks import (
    is_signal_time,
    is_trade_time,
    ist_now_str,
    seconds_to_next_candle,
    MARKET_OPEN_TIME,
    TRADE_END_TIME,
    MARKET_CLOSE_TIME,
    SIGNAL_START_TIME,
)

logger = get_logger("scheduler")
IST    = pytz.timezone("Asia/Kolkata")


class TradingScheduler:
    """
    Wraps the full per-candle trading loop in a schedule-aware class.
    Instantiate once and call start() to begin.
    """

    def __init__(self, kite) -> None:
        # ── Broker / data components ──────────────────────────────────────
        from data.feed import DataFeed
        from data.option_chain import OptionChain
        from signals.engine import SignalEngine
        from risk.manager import RiskManager
        from execution.order import OrderExecutor
        from utils.trade_logger import TradeLogger

        from ml.predictor import MLPredictor

        self.kite          = kite
        self.feed          = DataFeed(kite)
        self.option_chain  = OptionChain(kite)
        self.signal_engine = SignalEngine()
        self.risk_manager  = RiskManager()
        self.executor      = OrderExecutor(kite if not PAPER_MODE else None)
        self.trade_logger  = TradeLogger()
        self.ml_predictor  = MLPredictor()
        from ml.adaptive import AdaptiveMemory
        self.adaptive      = AdaptiveMemory()

        from signals.orb import ORBEngine
        from config.settings import ORB_MIN_RANGE_PCT, ORB_MAX_RANGE_PCT, ORB_BUFFER_PCT, ORB_TRADE_END
        self.orb_engine    = ORBEngine(
            min_range_pct = ORB_MIN_RANGE_PCT,
            max_range_pct = ORB_MAX_RANGE_PCT,
            buffer_pct    = ORB_BUFFER_PCT,
            trade_end     = ORB_TRADE_END,
        )

        # ── Session state ─────────────────────────────────────────────────
        self.open_positions: List[Dict[str, Any]] = []
        self.morning_pcr:    Dict[str, Optional[float]] = {}
        self.daily_trades:   Dict[str, int] = {idx: 0 for idx in ACTIVE_INDICES}
        self.directions_traded_today: set = set()   # global CALL/PUT cap across all indices
        self.profit_lock_done: bool = False          # 2 PM profit lock fires once per day
        self.shadow_signals_today: int = 0
        self.ml_override_trades_today: int = 0
        self._pending_shadows: List[Dict[str, Any]] = []   # signals not traded, pending outcome
        self._pending_signal: Optional[Dict[str, Any]] = None  # signal awaiting retest confirmation
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.ml_skipped_today: int = 0
        self.eod_sent: bool = False
        self.vix_alert_sent: bool = False           # sent once when VIX blocks trading
        self.loss_limit_alert_sent: bool = False    # sent once when daily loss limit hit
        self.last_heartbeat_hour: int = -1          # hour (IST) of last heartbeat sent
        self._is_new_day: bool = True               # set False by _load_state when same day
        self.running_capital = TRADING_CAPITAL
        self._today: Optional[date] = None

    # ──────────────────────────────────────────────────────────────────────
    # Scheduler entry point
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the blocking scheduler loop.
        Registers named daily jobs and loops until interrupted.
        """
        logger.info("=" * 60)
        logger.info("  TradingScheduler starting  |  %s  |  mode=%s",
                    ist_now_str(), "PAPER" if PAPER_MODE else "LIVE ← REAL MONEY")
        logger.info("  Indices : %s", ", ".join(ACTIVE_INDICES))
        logger.info("=" * 60)

        # ── Register named daily jobs ──────────────────────────────────────
        schedule.every().day.at("09:00").do(self._morning_setup)
        schedule.every().day.at("15:00").do(self._hard_close)
        schedule.every().day.at("15:30").do(self._day_summary)

        # ── Main loop ─────────────────────────────────────────────────────
        while True:
            try:
                now = datetime.now(IST)

                # Skip weekends entirely
                if now.weekday() >= 5:
                    logger.info("Weekend — sleeping 1h. %s", ist_now_str())
                    time.sleep(3600)
                    continue

                # New trading day — reset daily state
                if self._today != now.date():
                    self._reset_day()
                    self._today = now.date()

                # Run any pending schedule jobs (morning_setup, hard_close, day_summary)
                schedule.run_pending()

                t = now.time().replace(tzinfo=None)

                if is_signal_time():
                    # ── Core candle cycle ──────────────────────────────────
                    self._candle_cycle()
                    wait = seconds_to_next_candle()
                    logger.info(
                        "Cycle done | open=%d | daily_pnl=₹%.0f | next in %ds | %s",
                        len(self.open_positions), self.risk_manager.daily_pnl,
                        wait, ist_now_str(),
                    )
                    time.sleep(wait)

                elif t < SIGNAL_START_TIME:
                    # Between 09:00 and 10:00 — warming up, wait 30s
                    time.sleep(30)

                elif t >= MARKET_CLOSE_TIME:
                    # After 15:30 — sleep until next day
                    time.sleep(1800)

                else:
                    # Between 15:00 and 15:30 — hard close already ran
                    time.sleep(30)

            except KeyboardInterrupt:
                logger.info("Interrupted. Hard-closing all positions.")
                self._hard_close()
                break
            except Exception as exc:
                logger.error("Scheduler error: %s", exc, exc_info=True)
                time.sleep(60)

    # ──────────────────────────────────────────────────────────────────────
    # Named daily jobs
    # ──────────────────────────────────────────────────────────────────────

    def _morning_setup(self) -> None:
        """
        09:00 IST — fetch PCR for all indices, log morning bias.
        Called once per day before signal window opens.
        """
        try:
            from telegram_alerts import send_morning_bias
        except ImportError:
            send_morning_bias = None

        logger.info("── Morning Setup ─────────────────── %s", ist_now_str())
        for index in ACTIVE_INDICES:
            spot = self.feed.get_spot_price(index)
            if spot is None:
                logger.warning("Could not fetch spot for %s at morning setup.", index)
                self.morning_pcr[index] = None
                continue

            pcr = self.option_chain.get_pcr(spot_price=spot, index=index)
            self.morning_pcr[index] = pcr

            if pcr is not None:
                from config.settings import PCR_BULL_MIN, PCR_BEAR_MAX
                if pcr > PCR_BULL_MIN:
                    bias = "BULLISH (fear, lean CALL)"
                elif pcr < PCR_BEAR_MAX:
                    bias = "BEARISH (greed, lean PUT)"
                else:
                    bias = "NEUTRAL"
                logger.info("Morning PCR [%s] = %.2f → %s", index, pcr, bias)
                if send_morning_bias:
                    send_morning_bias(index, pcr, bias)
            else:
                logger.info("Morning PCR [%s] = unavailable", index)

    def _hard_close(self) -> None:
        """
        15:00 IST — force-close all open positions at market.
        Also called on KeyboardInterrupt.
        """
        try:
            from telegram_alerts import send_hard_close
        except ImportError:
            send_hard_close = None

        # Resolve pending signals / shadows before we lose LTP access
        self._flush_pending_shadows()
        self._pending_signal = None   # clear any unconfirmed retest

        if not self.open_positions:
            logger.info("Hard close: no open positions. %s", ist_now_str())
            return

        logger.info(
            "HARD CLOSE — %d position(s) | %s",
            len(self.open_positions), ist_now_str(),
        )
        for pos in list(self.open_positions):
            pnl = self._close_position(pos, reason="Hard close (15:00)")
            if send_hard_close:
                send_hard_close(pos["index"], pos["symbol"], pnl, "Hard close (15:00)")
        self.open_positions.clear()
        self.risk_manager.open_positions = 0

    def _day_summary(self, sync_retrain: bool = False) -> None:
        """15:30 IST — print and send Telegram day summary."""
        try:
            from telegram_alerts import send_day_summary
        except ImportError:
            send_day_summary = None

        trades_today = sum(self.daily_trades.values())
        pnl          = self.risk_manager.daily_pnl
        capital_now  = self.running_capital + pnl

        # Read ML AUC from metadata written by the last retrain
        ml_auc: Optional[float] = None
        try:
            import json as _json
            _meta_path = os.path.join("ml", "models", "model_metadata.json")
            if os.path.exists(_meta_path):
                with open(_meta_path) as _f:
                    ml_auc = float(_json.load(_f).get("cv_auc", 0)) or None
        except Exception:
            pass

        # ── Drift detection: compare live 7-day win rate vs model expected ──────
        drift_line = ""
        try:
            import pandas as _pd
            from datetime import timedelta as _td
            _trades_csv = TRADE_LOG_FILE
            if os.path.exists(_trades_csv):
                _df = _pd.read_csv(_trades_csv)
                if "date" in _df.columns and "result" in _df.columns:
                    _df["_dt"]  = _pd.to_datetime(_df["date"], errors="coerce")
                    _cutoff     = _pd.Timestamp.now() - _td(days=7)
                    _recent     = _df[_df["_dt"] >= _cutoff]
                    _closed     = _recent[_recent["result"].isin(["WIN", "LOSS"])]
                    if len(_closed) >= 3:
                        _live_wr = float((_closed["result"] == "WIN").sum()) / len(_closed)
                        _exp_wr: Optional[float] = None
                        try:
                            _meta_path = os.path.join("ml", "models", "model_metadata.json")
                            if os.path.exists(_meta_path):
                                with open(_meta_path) as _mf:
                                    _exp_wr = float(
                                        _json.load(_mf).get("train_win_rate", 0)
                                    ) or None
                        except Exception:
                            pass
                        if _exp_wr:
                            if _live_wr < _exp_wr - 0.15:
                                drift_line = (
                                    f"🚨 Model drift: live {_live_wr:.0%} "
                                    f"vs expected {_exp_wr:.0%} — retrain recommended"
                                )
                            else:
                                drift_line = (
                                    f"✓ Model tracking: live {_live_wr:.0%} "
                                    f"vs expected {_exp_wr:.0%}"
                                )
        except Exception as _drift_exc:
            logger.debug("Drift detection error: %s", _drift_exc)

        lines = [
            "─" * 50,
            f"  DAY SUMMARY  {ist_now_str()}",
            f"  Trades today : {trades_today}",
            f"  Daily P&L    : ₹{pnl:+,.2f}",
            f"  Capital      : ₹{capital_now:,.0f}",
            f"  Mode         : {'PAPER' if PAPER_MODE else 'LIVE'}",
            f"  ML AUC       : {ml_auc:.3f}" if ml_auc else "  ML AUC       : n/a",
            f"  Shadow sigs  : {self.shadow_signals_today}",
        ]
        if drift_line:
            lines.append(f"  {drift_line}")
        lines.append("─" * 50)
        for line in lines:
            logger.info(line)

        if send_day_summary:
            send_day_summary(
                trades=trades_today,
                pnl=pnl,
                capital=capital_now,
                paper=PAPER_MODE,
                ml_auc=ml_auc,
                shadow_count=self.shadow_signals_today,
                drift_line=drift_line,
            )

        # Engine-end Telegram notification (once per day, guarded by eod_sent)
        if not self.eod_sent:
            self._notify_engine_end(trades_today, pnl, capital_now, ml_auc)
            self.eod_sent = True

        # Retrain ML model (sync in candle/Actions mode, background in scheduler mode)
        self._retrain_eod(trades_today, sync=sync_retrain)

    # ──────────────────────────────────────────────────────────────────────
    # Core per-candle cycle
    # ──────────────────────────────────────────────────────────────────────

    def _candle_cycle(self) -> None:
        """
        Runs on every 15-min candle between 10:00 and 15:00.
        1. Monitor open positions (SL / target check via LTP)
        2. Scan indices for new signals
        3. Enter best qualifying trade
        """
        try:
            from telegram_alerts import (
                send_signal_fired, send_trade_entered,
                send_sl_hit, send_target_hit,
            )
        except ImportError:
            send_signal_fired = send_trade_entered = send_sl_hit = send_target_hit = None

        # ── Resolve pending shadow signals from previous candles ───────────
        self._resolve_shadow_signals()

        # ── Fetch VIX ──────────────────────────────────────────────────────
        vix = self.feed.get_vix()
        now_ist = datetime.now(IST)

        # ── Retest entry: check pending signal from previous candle ────────
        if self._pending_signal is not None:
            ps    = self._pending_signal
            index = ps["index"]
            df_r  = self.feed.get_today_candles(index)
            if df_r is not None and len(df_r) >= 2:
                last_candle = df_r.iloc[-2]   # last fully closed candle
                decision    = self._check_retest_live(last_candle)
                if decision == "ENTER":
                    self._pending_signal = None
                    spot_r = float(last_candle["close"])
                    best_r = {
                        "index":          index,
                        "result":         _types.SimpleNamespace(
                            direction     = ps["direction"],
                            conditions_met= ps["conditions_met"],
                            details       = {**ps.get("signal_details", {}), "retest_entry": True},
                        ),
                        "df":             df_r,
                        "spot":           spot_r,
                        "pcr":            ps.get("pcr"),
                        "conditions_met": ps["conditions_met"],
                        "ml_override":    False,
                    }
                    logger.info(
                        "[%s] %s RETEST ENTER | score=%d",
                        index, ps["direction"], ps["conditions_met"],
                    )
                    self._enter_trade(best_r, vix=vix,
                                      tg_entry=None)
                elif decision == "CANCEL":
                    logger.info("[%s] %s RETEST CANCELLED", index, ps["direction"])
                    self._pending_signal = None
                # else "WAIT" — carry forward to next candle
            else:
                # Can't get candles — cancel to avoid stale pending
                self._pending_signal = None

            if self._pending_signal is not None:
                return   # still waiting — don't scan for new signals

        # ── Hourly heartbeat (fires once per hour: 10, 11, 12, 13, 14 IST) ──
        # Uses last_heartbeat_hour so a late-running cron still fires exactly once per hour.
        if 10 <= now_ist.hour <= 14 and now_ist.hour != self.last_heartbeat_hour:
            self.last_heartbeat_hour = now_ist.hour
            try:
                from telegram_alerts import send_hourly_status
                send_hourly_status(
                    time_str       = now_ist.strftime("%H:%M"),
                    open_positions = len(self.open_positions),
                    daily_pnl      = self.risk_manager.daily_pnl,
                    trades_today   = sum(self.daily_trades.values()),
                    capital        = self.running_capital,
                    paper          = PAPER_MODE,
                )
            except Exception as _he:
                logger.debug("Hourly status send failed: %s", _he)

        # ── 2 PM profit lock — close profitable positions at first cycle ≥ 14:00
        from config.settings import PROFIT_LOCK_TIME as _PLT
        _plt_h, _plt_m = int(_PLT.split(":")[0]), int(_PLT.split(":")[1])
        at_profit_lock = (
            not self.profit_lock_done
            and now_ist.hour * 60 + now_ist.minute >= _plt_h * 60 + _plt_m
        )
        if at_profit_lock:
            self.profit_lock_done = True
            for pos in list(self.open_positions):
                ltp = self.option_chain.get_ltp(pos["symbol"], pos["index"])
                if ltp is not None and ltp > pos["entry_premium"]:
                    pnl = self._close_position(pos, reason="Profit lock (14:00)", exit_premium=ltp)
                    self.open_positions.remove(pos)
                    self.risk_manager.open_positions -= 1
                    if send_target_hit:
                        send_target_hit(pos["index"], pos["symbol"], pnl,
                                        self.running_capital + self.risk_manager.daily_pnl)
                    logger.info("[%s] Profit lock: closed at ₹%.2f (entry ₹%.2f)",
                                pos["index"], ltp, pos["entry_premium"])

        # ── Monitor open positions — ScaleOutPosition (identical to backtest) ─
        from execution.scale_out import ScaleOutPosition

        for pos in list(self.open_positions):
            ltp = self.option_chain.get_ltp(pos["symbol"], pos["index"])
            if ltp is None:
                continue

            so: ScaleOutPosition = pos["scale_out"]
            # Detect red candle (needed for T2 momentum check)
            df_so   = self.feed.get_today_candles(pos["index"])
            is_red  = False
            if df_so is not None and len(df_so) >= 2:
                last = df_so.iloc[-2]
                is_red = float(last["close"]) < float(last["open"])

            decision = so.check_exit(ltp, is_red)

            if not decision.should_exit:
                continue

            exit_px    = decision.exit_price_override if decision.exit_price_override > 0 else ltp
            qty_closed = decision.lots_to_close * pos["lot_size"]

            if decision.exit_all:
                # Full close — log, record ML row
                pnl = self._close_position(pos, reason=decision.exit_reason, exit_premium=exit_px)
                self.open_positions.remove(pos)
                self.risk_manager.open_positions -= 1
                is_sl = "SL" in decision.exit_reason or "HARD" in decision.exit_reason
                if is_sl and send_sl_hit:
                    send_sl_hit(pos["index"], pos["symbol"], pnl,
                                self.running_capital + self.risk_manager.daily_pnl)
                elif send_target_hit:
                    send_target_hit(pos["index"], pos["symbol"], pnl,
                                    self.running_capital + self.risk_manager.daily_pnl)
            else:
                # Partial exit — sell only qty_closed, keep position open
                partial_pnl = (exit_px - pos["entry_premium"]) * qty_closed
                self.risk_manager.record_trade_pnl(partial_pnl)
                self.executor.place_sell_order(
                    pos["symbol"], qty_closed, index=pos["index"], price=exit_px,
                )
                pos["quantity"] = so.lots_remaining * pos["lot_size"]
                pos["lots"]     = so.lots_remaining
                logger.info(
                    "[%s] %s partial exit %d lots | ₹%.2f | reason=%s",
                    pos["index"], pos["direction"], decision.lots_to_close,
                    exit_px, decision.exit_reason,
                )
                if send_target_hit:
                    send_target_hit(pos["index"], pos["symbol"], partial_pnl,
                                    self.running_capital + self.risk_manager.daily_pnl)

        # ── Risk gate — with one-time Telegram alerts ───────────────────────
        from config.settings import VIX_MAX as _VIX_MAX, DAILY_LOSS_LIMIT_PCT, TRADING_CAPITAL as _TC
        _loss_limit = _TC * DAILY_LOSS_LIMIT_PCT
        if vix is not None and vix > _VIX_MAX and not self.vix_alert_sent:
            self.vix_alert_sent = True
            try:
                from telegram_alerts import send_vix_blocked
                send_vix_blocked(vix, _VIX_MAX)
            except Exception:
                pass
        if self.risk_manager.daily_pnl <= -_loss_limit and not self.loss_limit_alert_sent:
            self.loss_limit_alert_sent = True
            try:
                from telegram_alerts import send_loss_limit_hit
                send_loss_limit_hit(self.risk_manager.daily_pnl, _loss_limit)
            except Exception:
                pass
        if not self.risk_manager.can_trade(vix=vix, open_positions=len(self.open_positions)):
            return

        # ── Scan indices for best signal ────────────────────────────────────
        best: Optional[Dict[str, Any]] = None

        for index in ACTIVE_INDICES:
            if any(p["index"] == index for p in self.open_positions):
                continue
            if self.daily_trades.get(index, 0) >= MAX_DAILY_TRADES_OTM:
                continue  # hard cap regardless of signal strength

            df = self.feed.get_today_candles(index)
            if df is None or len(df) < 2:
                continue

            spot = float(df.iloc[-2]["close"])
            pcr  = self.morning_pcr.get(index)

            # Fetch daily candle for Fibonacci anchors
            from datetime import date as _date, timedelta
            yesterday = _date.today() - timedelta(days=1)
            daily_candle = self._get_daily_candle(index, yesterday)

            adx_thr = INDEX_CONFIG[index].get("adx_threshold", 20.0)
            result  = self.signal_engine.evaluate(
                df, pcr=pcr, daily_candle=daily_candle, adx_threshold=adx_thr,
            )

            # Build signal log row
            fired = result.direction in ("CALL", "PUT")
            skip_reason = "" if fired else (
                result.details.get("reason", "conditions not met")
            )
            self.trade_logger.log_signal({
                "index":       index,
                "direction":   result.direction,
                "details":     result.details,
                "fired":       fired,
                "skip_reason": skip_reason,
            })

            # Console candle summary
            self._print_candle_summary(index, result, spot)

            if send_signal_fired and fired:
                send_signal_fired(
                    index=index,
                    direction=result.direction,
                    conditions=result.conditions_met,
                    adx=result.details.get("adx", 0),
                    rsi=result.details.get("rsi", 0),
                    fib_level=result.details.get("fib_label", ""),
                    paper=PAPER_MODE,
                )

            if not fired:
                # Record near-miss for ML shadow training even when indicators don't fire
                if result.conditions_met >= MIN_CONDITIONS - 1:
                    self.shadow_signals_today += 1  # near-miss WAIT
                    self._record_shadow_signal(index, result, spot, pcr, vix)
                try:
                    _exp_nm = self.option_chain._nearest_expiry(index)
                    from datetime import date as _date_nm
                    _dte_nm = max((_exp_nm - _date_nm.today()).days, 1)
                    self.adaptive.check_near_miss(index, result.details, _dte_nm, now_ist.hour)
                except Exception as _e:
                    logger.debug("Near-miss check [%s]: %s", index, _e)

                # ML override: AUC ≥ 0.72 may fire on exactly 3/5 signals in 10:00–12:30

                _override_dir = self._get_ml_override_direction(index, result, now_ist)
                if _override_dir is None:
                    continue
                logger.info(
                    "[%s] ML OVERRIDE: 3/5 conditions → direction=%s (AUC gate passed)",
                    index, _override_dir,
                )
                result = _types.SimpleNamespace(
                    direction=_override_dir,
                    conditions_met=result.conditions_met,
                    details={**result.details, "ml_override": True},
                )
                if send_signal_fired:
                    send_signal_fired(
                        index=index,
                        direction=_override_dir,
                        conditions=result.conditions_met,
                        adx=result.details.get("adx", 0),
                        rsi=result.details.get("rsi", 0),
                        fib_level=result.details.get("fib_label", ""),
                        paper=PAPER_MODE,
                    )

            # Soft cap: 2nd trade only for 5/5 signals (ML override has own counter — bypasses)
            if (self.daily_trades.get(index, 0) >= MAX_DAILY_TRADES
                    and result.conditions_met < STRONG_SIGNAL_THRESHOLD
                    and not result.details.get("ml_override", False)):
                logger.info(
                    "[%s] Daily trade limit (%d). 2nd trade requires 5/5 signal (got %d/5).",
                    index, MAX_DAILY_TRADES, result.conditions_met,
                )
                self.shadow_signals_today += 1
                self._record_shadow_signal(index, result, spot, pcr, vix)
                continue

            # Global same-direction cap: skip if this direction already traded today
            if result.direction in self.directions_traded_today:
                logger.info(
                    "[%s] %s already traded today — skipping correlated signal.",
                    index, result.direction,
                )
                self.shadow_signals_today += 1
                self._record_shadow_signal(index, result, spot, pcr, vix)
                continue

            # Keep best signal (highest conditions_met)
            if best is None or result.conditions_met > best["conditions_met"]:
                best = {
                    "index":          index,
                    "result":         result,
                    "df":             df,
                    "spot":           spot,
                    "pcr":            pcr,
                    "conditions_met": result.conditions_met,
                    "ml_override":    result.details.get("ml_override", False),
                }

        # ── Park best signal as pending — retest confirms on next candle ──
        if best and self._pending_signal is None:
            self._pending_signal = {
                "index":          best["index"],
                "direction":      best["result"].direction,
                "conditions_met": best["conditions_met"],
                "signal_details": best["result"].details,
                "pcr":            best.get("pcr"),
                "candles_waited": 0,
                "saw_pullback":   False,
            }
            logger.info(
                "[%s] %s PENDING retest | score=%d",
                best["index"], best["result"].direction, best["conditions_met"],
            )

        # ── ORB scan — fires only when no main signal is pending and no position open ──
        from config.settings import ORB_ENABLED as _ORB_ENABLED
        if _ORB_ENABLED and self._pending_signal is None and not self.open_positions:
            for _orb_index in ACTIVE_INDICES:
                _df_orb = self.feed.get_today_candles(_orb_index)
                if _df_orb is None or len(_df_orb) < 2:
                    continue

                self.orb_engine.compute_opening_range(_orb_index, _df_orb)
                _orb = self.orb_engine.evaluate(_orb_index, _df_orb, now_ist)
                if not _orb:
                    continue

                logger.info("[ORB] [%s] %s breakout confirmed — entering trade", _orb_index, _orb.direction)
                try:
                    from telegram_alerts import send_orb_signal
                    send_orb_signal(
                        index          = _orb_index,
                        direction      = _orb.direction,
                        or_high        = _orb.or_high,
                        or_low         = _orb.or_low,
                        breakout_price = _orb.breakout_price,
                        paper          = PAPER_MODE,
                    )
                except Exception as _oe:
                    logger.debug("ORB Telegram alert failed: %s", _oe)

                _orb_result = _types.SimpleNamespace(
                    direction      = _orb.direction,
                    conditions_met = _orb.conditions_met,
                    details        = {
                        "orb":          True,
                        "or_high":      _orb.or_high,
                        "or_low":       _orb.or_low,
                        "or_range_pct": _orb.or_range_pct,
                        "close":        _orb.breakout_price,
                    },
                )
                self._enter_trade(
                    {
                        "index":          _orb_index,
                        "result":         _orb_result,
                        "df":             _df_orb,
                        "spot":           _orb.breakout_price,
                        "pcr":            self.morning_pcr.get(_orb_index),
                        "conditions_met": _orb.conditions_met,
                        "ml_override":    False,
                    },
                    vix=vix,
                    tg_entry=None,
                )
                self.orb_engine.mark_traded(_orb_index)
                break  # one ORB entry per cycle

    # ──────────────────────────────────────────────────────────────────────
    # Trade entry
    # ──────────────────────────────────────────────────────────────────────

    def _enter_trade(
        self,
        best: Dict[str, Any],
        vix: Optional[float],
        tg_entry=None,
    ) -> None:
        from execution.scale_out import ScaleOutPosition
        index  = best["index"]
        result = best["result"]
        spot   = best["spot"]
        score  = best["conditions_met"]

        opt_type = "CE" if result.direction == "CALL" else "PE"
        expiry   = self.option_chain._nearest_expiry(index)

        # Strike selection: OTM on 5/5, ATM otherwise
        if score >= STRONG_SIGNAL_THRESHOLD:
            strike      = self.option_chain.get_otm_strike(spot, opt_type, index, OTM_STRIKES_AWAY)
            strike_label = f"OTM+{OTM_STRIKES_AWAY}"
        else:
            strike      = self.option_chain.get_atm_strike(spot, index)
            strike_label = "ATM"

        symbol  = self.option_chain.get_option_symbol(strike, opt_type, index, expiry)
        premium = self.option_chain.get_ltp(symbol, index)

        if premium is None or not self.risk_manager.is_premium_valid(premium):
            logger.info("[%s] Premium unavailable or out of range — skip.", index)
            return

        quantity, lots = self.risk_manager.position_size(
            premium, index, adx=float(result.details.get("adx") or 0),
        )
        if quantity == 0:
            logger.info("[%s] Position size 0 for premium=%.2f — skip.", index, premium)
            return

        # ── ML gate (indicators-only until model is trained enough) ──────────
        # ML activates automatically when CV AUC >= ML_ACTIVATE_AUC (0.65).
        # Until then: compute scores for logging/Telegram only — zero effect on trade.
        _ML_ACTIVATE_AUC     = 0.65   # minimum AUC before ML influences any decision
        _ML_REDUCE_THRESHOLD = 0.50

        from datetime import date as _date2
        dte = max((expiry - _date2.today()).days, 1)
        signal_details = {**result.details, "conditions_met": result.conditions_met}
        entry_time_str = datetime.now(IST).strftime("%H:%M:%S")
        xgb_score = self.ml_predictor.confidence(
            signal_details=signal_details,
            spot=spot,
            vix=vix,
            pcr=best["pcr"],
            dte=dte,
            entry_time_str=entry_time_str,
        )
        blended_conf, raw_mem_score, adaptive_reason = self.adaptive.score(
            index, result.direction, signal_details, dte,
            datetime.now(IST).hour, xgb_score,
        )

        # ── Conflict detection: XGBoost vs pattern memory ────────────────────
        # When the two sources disagree by > 0.35, defer to the memory-based
        # score (it has seen real outcomes) and reduce to half position.
        _reduce_size = False
        if raw_mem_score is not None and abs(xgb_score - raw_mem_score) > 0.35:
            logger.warning(
                "[%s] ML CONFLICT — XGB=%.2f MEM=%.2f (diff=%.2f) → deferring to memory",
                index, xgb_score, raw_mem_score, abs(xgb_score - raw_mem_score),
            )
            blended_conf = raw_mem_score
            _reduce_size = True
            self.trade_logger.log_signal({
                "index":       index,
                "direction":   result.direction,
                "details":     result.details,
                "fired":       True,
                "skip_reason": f"CONFLICT XGB={xgb_score:.2f} MEM={raw_mem_score:.2f}",
            })
            try:
                from telegram_alerts import send_ml_conflict as _send_conflict
                _send_conflict(index, xgb_score, raw_mem_score)
            except Exception:
                pass

        # Check if model has been trained enough to influence decisions
        _current_auc = 0.0
        try:
            import json as _j
            _meta_path = os.path.join("ml", "models", "model_metadata.json")
            if os.path.exists(_meta_path):
                with open(_meta_path) as _f:
                    _current_auc = float(_j.load(_f).get("cv_auc", 0))
        except Exception:
            pass

        _ml_active = _current_auc >= _ML_ACTIVATE_AUC

        if not _ml_active:
            # Model not trained enough — indicators decide, ML is observer only
            ml_tier = "indicators_only"
            logger.info(
                "[%s] ML inactive (AUC %.3f < %.2f) — trading on indicators only.",
                index, _current_auc, _ML_ACTIVATE_AUC,
            )
        else:
            # Model trained — apply tier logic
            if blended_conf < ML_MIN_CONFIDENCE:
                logger.info(
                    "[%s] ML+Adaptive BLOCK: confidence=%.2f — proven loser pattern | %s",
                    index, blended_conf, adaptive_reason,
                )
                self.shadow_signals_today += 1
                self.ml_skipped_today += 1
                return

            ml_tier = "strong" if blended_conf >= 0.60 else (
                      "normal" if blended_conf >= _ML_REDUCE_THRESHOLD else "weak")

            if ml_tier == "weak":
                _reduce_size = True
                logger.info(
                    "[%s] ML uncertain (%.2f) — half position | %s",
                    index, blended_conf, adaptive_reason,
                )

        # Apply size reduction (conflict OR weak ML tier — only once even if both)
        if _reduce_size:
            lot_size = INDEX_CONFIG[index]["lot_size"]
            lots     = max(lots // 2, 1)
            quantity = lots * lot_size

        ml_conf = blended_conf

        # ── ML override counter ───────────────────────────────────────────────
        is_ml_override = best.get("ml_override", False)
        if is_ml_override:
            self.ml_override_trades_today += 1
            trade_type = "ML_OVERRIDE"
            logger.info(
                "[%s] ML override trade #%d placed today.",
                index, self.ml_override_trades_today,
            )
        else:
            trade_type = "NORMAL"

        levels     = self.risk_manager.compute_exit_levels(premium, conditions_met=score)
        _BROKERAGE = 40.0
        _SLP_PTS   = 2
        gross_risk = self.running_capital * RISK_PER_TRADE_PCT
        actual_risk = round(gross_risk - _BROKERAGE + (_SLP_PTS * 2 * quantity), 2)

        # Place order (paper or live)
        from database.models import new_trade_id
        trade_id = new_trade_id(index)
        self.executor.authorize_trade(trade_id, index)
        self.executor.place_buy_order(symbol, quantity, index=index, price=premium)

        d = result.details
        pos = {
            "trade_id":      trade_id,
            "index":         index,
            "direction":     result.direction,
            "symbol":        symbol,
            "strike":        strike,
            "option_type":   opt_type,
            "expiry":        str(expiry),
            "lot_size":      INDEX_CONFIG[index]["lot_size"],
            "lots":          lots,
            "quantity":      quantity,
            "entry_premium": premium,
            "spot_at_entry": spot,
            "stop_loss":     levels["stop_loss"],
            "target":        levels["target"],
            "target_soft":   levels["target_soft"],
            "target_hard":   levels["target_hard"],
            "entry_time":    datetime.now(IST),
            "conditions_met": score,
            "pcr":           best["pcr"],
            "vix":           vix,
            "adx":           d.get("adx"),
            "rsi":           d.get("rsi"),
            "fib_level":     d.get("fib_level"),
            "pcr_bias":      self._pcr_bias_label(best["pcr"]),
            "ml_tier":       ml_tier,
            "ml_conf":       round(blended_conf, 3),
            "trade_type":    trade_type,
            # stored for ML CSV logging on close
            "signal_details": d,
            "actual_risk":   actual_risk,
            "dte":           dte,
            "capital_before": round(self.running_capital, 2),
            # ScaleOutPosition — same class used by backtest
            "scale_out": ScaleOutPosition(
                total_lots=lots,
                entry_premium=premium,
                direction=result.direction,
            ),
        }
        self.open_positions.append(pos)
        self.risk_manager.open_positions += 1
        self.daily_trades[index] = self.daily_trades.get(index, 0) + 1
        self.directions_traded_today.add(result.direction)

        logger.info(
            "ENTRY [%s] %s %s | %s | qty=%d | ₹%.2f | SL=₹%.2f | TGT=₹%.2f | ML=%.2f",
            "PAPER" if PAPER_MODE else "LIVE",
            result.direction, symbol, strike_label,
            quantity, premium, levels["stop_loss"], levels["target"], ml_conf,
        )

        self.trade_logger.log_trade_entry({
            "index":          index,
            "direction":      result.direction,
            "signal_strength": score,
            "strike":         strike,
            "option_type":    opt_type,
            "expiry":         str(expiry),
            "entry_premium":  premium,
            "sl_premium":     levels["stop_loss"],
            "tp_premium":     levels["target"],
            "lots":           lots,
            "quantity":       quantity,
            "actual_risk":    actual_risk,
            "adx":            d.get("adx"),
            "rsi":            d.get("rsi"),
            "fib_level":      d.get("fib_level"),
            "pcr_bias":       self._pcr_bias_label(best["pcr"]),
            "trade_type":     trade_type,
            "paper":          PAPER_MODE,
        })

        if tg_entry:
            tg_entry(
                index=index,
                direction=result.direction,
                strike=strike,
                expiry=str(expiry),
                lots=lots,
                premium=premium,
                sl=levels["stop_loss"],
                target=levels["target"],
                paper=PAPER_MODE,
                ml_tier=ml_tier,
                ml_conf=round(blended_conf, 2),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Position close
    # ──────────────────────────────────────────────────────────────────────

    def _close_position(
        self,
        pos: Dict[str, Any],
        reason: str,
        exit_premium: Optional[float] = None,
    ) -> float:
        if exit_premium is None:
            exit_premium = (
                self.option_chain.get_ltp(pos["symbol"], pos["index"])
                or pos["entry_premium"]
            )

        self.executor.authorize_trade(pos["trade_id"], pos["index"])
        self.executor.place_sell_order(
            pos["symbol"], pos["quantity"],
            index=pos["index"], price=exit_premium,
        )

        pnl     = (exit_premium - pos["entry_premium"]) * pos["quantity"]
        pnl_pct = (exit_premium - pos["entry_premium"]) / pos["entry_premium"] * 100
        self.risk_manager.record_trade_pnl(pnl)
        if pnl > 0:
            self.wins_today += 1
        else:
            self.losses_today += 1
        self._write_ml_signal_row(pos, exit_premium, pnl, pnl_pct, reason)
        self.adaptive.record_outcome(
            pos["index"],
            pos["direction"],
            pos.get("signal_details", {}),
            pos.get("dte", 1),
            pos["entry_time"].hour,
            pnl,
        )

        logger.info(
            "CLOSED %s %s | %s | PnL=₹%.2f (%.1f%%)",
            pos["direction"], pos["symbol"], reason, pnl, pnl_pct,
        )

        self.trade_logger.log_trade_exit({
            "index":          pos["index"],
            "direction":      pos["direction"],
            "signal_strength": pos["conditions_met"],
            "strike":         pos["strike"],
            "option_type":    pos["option_type"],
            "expiry":         pos["expiry"],
            "entry_premium":  pos["entry_premium"],
            "sl_premium":     pos["stop_loss"],
            "tp_premium":     pos["target"],
            "lots":           pos["lots"],
            "quantity":       pos["quantity"],
            "exit_premium":   exit_premium,
            "exit_reason":    reason,
            "pnl":            pnl,
            "adx":            pos.get("adx"),
            "rsi":            pos.get("rsi"),
            "fib_level":      pos.get("fib_level"),
            "pcr_bias":       pos.get("pcr_bias", ""),
            "paper":          PAPER_MODE,
        })
        return pnl

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Live ML signal logging + EOD retraining
    # ──────────────────────────────────────────────────────────────────────

    _LIVE_ML_CSV = os.path.join("reports", "live_signals_ml.csv")

    def _write_ml_signal_row(
        self,
        pos: Dict[str, Any],
        exit_premium: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Append one closed-trade row to live_signals_ml.csv in backtest signal format."""
        try:
            from backtest.engine import SIGNAL_CSV_COLUMNS
            d        = pos.get("signal_details", {})
            close    = float(d.get("close", pos.get("spot_at_entry", 0)))
            vwap     = float(d.get("vwap", 0))
            fib_lvl  = float(d.get("fib_level") or 0)

            cond_met = pos.get("conditions_met", 0)
            row = {
                "signal_id":      uuid.uuid4().hex[:8].upper(),
                "date":           pos["entry_time"].strftime("%Y-%m-%d"),
                "time":           pos["entry_time"].strftime("%H:%M:%S"),
                "index":          pos.get("index", ""),
                "direction":      pos.get("direction", ""),
                "direction_int":  1 if pos.get("direction") == "CALL" else 0,
                # Both score fields — live engine uses 0-5 conditions count
                "conviction_score": cond_met,
                "conditions_met": cond_met,
                "lot_size":       pos.get("lots", 1),
                "day_quality":    "",   # not computed in live engine
                "close":          round(close, 2),
                "vwap":           round(vwap, 2),
                "vwap_distance":  round(close - vwap, 2),
                "ema_bull":       int(bool(d.get("ema_bull"))),
                "ema_bear":       int(bool(d.get("ema_bear"))),
                "ema_fast":       round(float(d.get("ema_fast") or 0), 2),
                "ema_slow":       round(float(d.get("ema_slow") or 0), 2),
                "near_fib":       int(bool(d.get("near_fib"))),
                "fib_distance":   round(abs(close - fib_lvl), 4) / max(close, 1) if fib_lvl else 0.0,
                "vol_spike":      int(bool(d.get("vol_spike"))),
                "vol_ratio":      round(float(d.get("vol_ratio") or 1.0), 3),
                "rsi":            round(float(d.get("rsi") or 50), 2),
                "adx":            round(float(d.get("adx") or 0), 2),
                "pcr":            d.get("pcr", ""),
                "iv_rank":        "",
                "pdh":            round(float(d.get("swing_high", 0)), 2),
                "pdl":            round(float(d.get("swing_low", 0)), 2),
                "weekly_high":    "",
                "weekly_low":     "",
                "dte":            pos.get("dte", 1),
                "capital_before": pos.get("capital_before", round(self.running_capital, 2)),
                "capital_used":   round(pos.get("entry_premium", 0) * pos.get("quantity", 0), 2),
                "risk_amount":    round(pos.get("actual_risk", 0), 2),
                "entry_premium":  round(pos.get("entry_premium", 0), 2),
                "stop_loss":      round(pos.get("stop_loss", 0), 2),
                "target":         round(pos.get("target", 0), 2),
                "exit_premium":   round(exit_premium, 2),
                "pnl":            round(pnl, 2),
                "pnl_pct":        round(pnl_pct, 2),
                "exit_reason":    reason,
                "trade_taken":    1,
                "outcome":        1 if pnl > 0 else 0,
            }

            write_header = (
                not os.path.exists(self._LIVE_ML_CSV)
                or os.path.getsize(self._LIVE_ML_CSV) == 0
            )
            with open(self._LIVE_ML_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=SIGNAL_CSV_COLUMNS, extrasaction="ignore"
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            logger.info("ML signal row written → %s", self._LIVE_ML_CSV)
        except Exception as exc:
            logger.warning("Could not write ML signal row: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Retest entry (live — mirrors backtest/_check_retest_entry)
    # ──────────────────────────────────────────────────────────────────────

    def _check_retest_live(self, candle) -> str:
        """
        Returns "ENTER", "WAIT", or "CANCEL".
        CALL: straight green → ENTER; red pullback then green → ENTER; 2nd red → CANCEL.
        PUT:  mirror.  Cancel after 3 candles total.
        """
        pending   = self._pending_signal
        is_green  = float(candle["close"]) >= float(candle["open"])
        is_red    = not is_green
        pending["candles_waited"] = pending.get("candles_waited", 0) + 1
        direction = pending["direction"]

        if pending["candles_waited"] > 3:
            return "CANCEL"

        if direction == "CALL":
            if not pending.get("saw_pullback"):
                if is_red:
                    pending["saw_pullback"] = True
                    return "WAIT"
                else:
                    return "ENTER"
            else:
                return "ENTER" if is_green else "CANCEL"
        else:  # PUT
            if not pending.get("saw_pullback"):
                if is_green:
                    pending["saw_pullback"] = True
                    return "WAIT"
                else:
                    return "ENTER"
            else:
                return "ENTER" if is_red else "CANCEL"

    # ──────────────────────────────────────────────────────────────────────
    # Shadow signal ML logging (trains ML on ALL signals, not just traded ones)
    # ──────────────────────────────────────────────────────────────────────

    def _record_shadow_signal(
        self,
        index: str,
        result,
        spot: float,
        pcr: Optional[float],
        vix: Optional[float],
    ) -> None:
        """Record a signal that was evaluated but not traded, for future outcome resolution."""
        try:
            opt_type = "CE" if result.direction == "CALL" else "PE"
            from data.option_chain import OptionChain as _OC
            expiry  = self.option_chain._nearest_expiry(index)
            strike  = self.option_chain.get_atm_strike(spot, index)
            symbol  = self.option_chain.get_option_symbol(strike, opt_type, index, expiry)
            premium = self.option_chain.get_ltp(symbol, index)
            if premium is None or premium <= 0:
                return

            from datetime import date as _date
            dte    = max((expiry - _date.today()).days, 1)
            levels = self.risk_manager.compute_exit_levels(premium, conditions_met=result.conditions_met)

            self._pending_shadows.append({
                "signal_id":      uuid.uuid4().hex[:8].upper(),
                "index":          index,
                "direction":      result.direction,
                "symbol":         symbol,
                "entry_premium":  premium,
                "stop_loss":      levels["stop_loss"],
                "target":         levels["target"],
                "candles_held":   0,
                "entry_time":     datetime.now(IST),
                "dte":            dte,
                "conditions_met": result.conditions_met,
                "signal_details": result.details,
                "pcr":            pcr,
                "vix":            vix,
                "spot_at_entry":  spot,
            })
            logger.debug("[%s] Shadow signal recorded: %s prem=₹%.2f", index, result.direction, premium)
        except Exception as exc:
            logger.debug("Shadow signal record failed [%s]: %s", index, exc)

    def _resolve_shadow_signals(self) -> None:
        """Check pending shadow signals against current option LTP; write resolved ones to ML CSV."""
        if not self._pending_shadows:
            return
        still_pending = []
        for shadow in self._pending_shadows:
            shadow["candles_held"] += 1
            ltp = None
            try:
                ltp = self.option_chain.get_ltp(shadow["symbol"], shadow["index"])
            except Exception:
                pass

            if ltp is None:
                if shadow["candles_held"] < 4:
                    still_pending.append(shadow)
                # else drop — can't determine outcome (LTP unavailable for >4 candles)
                continue

            if ltp <= shadow["stop_loss"]:
                self._write_shadow_ml_row(shadow, ltp, outcome=0, reason="shadow_SL")
            elif ltp >= shadow["target"]:
                self._write_shadow_ml_row(shadow, ltp, outcome=1, reason="shadow_TP")
            elif shadow["candles_held"] >= 4:
                outcome = 1 if ltp > shadow["entry_premium"] else 0
                self._write_shadow_ml_row(shadow, ltp, outcome=outcome, reason="shadow_MAX_HOLD")
            else:
                still_pending.append(shadow)
        self._pending_shadows = still_pending

    def _write_shadow_ml_row(
        self,
        shadow: Dict[str, Any],
        exit_premium: float,
        outcome: int,
        reason: str,
    ) -> None:
        """Write a resolved shadow signal to the ML CSV with trade_taken=0."""
        try:
            from backtest.engine import SIGNAL_CSV_COLUMNS
            d     = shadow.get("signal_details", {})
            close = float(d.get("close", shadow.get("spot_at_entry", 0)))
            vwap  = float(d.get("vwap", 0))
            pnl   = (exit_premium - shadow["entry_premium"]) * INDEX_CONFIG.get(
                shadow["index"], INDEX_CONFIG["NIFTY"]
            )["lot_size"]
            pnl_pct = (exit_premium - shadow["entry_premium"]) / shadow["entry_premium"] * 100

            row = {
                "signal_id":      shadow["signal_id"],
                "date":           shadow["entry_time"].strftime("%Y-%m-%d"),
                "time":           shadow["entry_time"].strftime("%H:%M:%S"),
                "index":          shadow["index"],
                "direction":      shadow["direction"],
                "direction_int":  1 if shadow["direction"] == "CALL" else 0,
                "conviction_score": shadow["conditions_met"],
                "conditions_met": shadow["conditions_met"],
                "lot_size":       1,
                "day_quality":    "",
                "close":          round(close, 2),
                "vwap":           round(vwap, 2),
                "vwap_distance":  round(close - vwap, 2),
                "ema_bull":       int(bool(d.get("ema_bull"))),
                "ema_bear":       int(bool(d.get("ema_bear"))),
                "ema_fast":       round(float(d.get("ema_fast") or 0), 2),
                "ema_slow":       round(float(d.get("ema_slow") or 0), 2),
                "near_fib":       int(bool(d.get("near_fib"))),
                "fib_distance":   round(float(d.get("fib_distance") or 0), 4),
                "vol_spike":      int(bool(d.get("vol_spike"))),
                "vol_ratio":      round(float(d.get("vol_ratio") or 1), 3),
                "rsi":            round(float(d.get("rsi") or 50), 2),
                "adx":            round(float(d.get("adx") or 0), 2),
                "pcr":            shadow.get("pcr", ""),
                "iv_rank":        "",
                "pdh":            "", "pdl":          "",
                "weekly_high":    "", "weekly_low":   "",
                "dte":            shadow["dte"],
                "capital_before": round(self.running_capital, 2),
                "capital_used":   0.0,
                "risk_amount":    0.0,
                "entry_premium":  round(shadow["entry_premium"], 2),
                "stop_loss":      round(shadow["stop_loss"], 2),
                "target":         round(shadow["target"], 2),
                "exit_premium":   round(exit_premium, 2),
                "pnl":            round(pnl, 2),
                "pnl_pct":        round(pnl_pct, 2),
                "exit_reason":    reason,
                "trade_taken":    0,
                "outcome":        outcome,
            }
            write_header = (
                not os.path.exists(self._LIVE_ML_CSV)
                or os.path.getsize(self._LIVE_ML_CSV) == 0
            )
            with open(self._LIVE_ML_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SIGNAL_CSV_COLUMNS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            logger.debug("Shadow ML row written (outcome=%d, reason=%s)", outcome, reason)
        except Exception as exc:
            logger.warning("Could not write shadow ML row: %s", exc)

    def _flush_pending_shadows(self) -> None:
        """Force-resolve all pending shadows at EOD (use current LTP or drop)."""
        for shadow in self._pending_shadows:
            try:
                ltp = self.option_chain.get_ltp(shadow["symbol"], shadow["index"])
                if ltp is not None:
                    outcome = 1 if ltp > shadow["entry_premium"] else 0
                    self._write_shadow_ml_row(shadow, ltp, outcome=outcome, reason="shadow_EOD")
            except Exception:
                pass
        self._pending_shadows.clear()

    def _retrain_eod(self, trades_today: int, sync: bool = False) -> None:
        """Retrain the XGBoost model after market close.

        sync=True  → runs in the calling thread (GitHub Actions candle mode,
                      where the process exits immediately after run_once() returns).
        sync=False → runs in a background thread (local scheduler mode).
        """
        # Flush any unresolved shadow signals before retraining
        self._flush_pending_shadows()

        shadow_rows = 0
        try:
            if os.path.exists(self._LIVE_ML_CSV):
                import pandas as _pd
                _df = _pd.read_csv(self._LIVE_ML_CSV)
                shadow_rows = int((_df["trade_taken"] == 0).sum())
        except Exception:
            pass

        if trades_today == 0 and shadow_rows == 0:
            logger.info("No trades or shadow signals today — skipping EOD ML retraining.")
            return

        def _run() -> None:
            try:
                logger.info("EOD ML retraining started ...")
                from ml.train import train

                csvs = sorted(glob.glob(os.path.join("reports", "signals_*.csv")))
                if os.path.exists(self._LIVE_ML_CSV):
                    csvs.append(self._LIVE_ML_CSV)

                if not csvs:
                    logger.warning("EOD retrain: no CSVs found.")
                    return

                train(csvs)
                logger.info(
                    "EOD ML model retrained on %d CSVs. New model active from next trade.",
                    len(csvs),
                )
            except Exception as exc:
                logger.error("EOD ML retraining failed: %s", exc)

        if sync:
            _run()
        else:
            t = threading.Thread(target=_run, daemon=True, name="ml-retrain-eod")
            t.start()
            logger.info("EOD ML retraining scheduled in background (trades today: %d).", trades_today)

    # ──────────────────────────────────────────────────────────────────────
    # Single-candle mode (GitHub Actions)
    # ──────────────────────────────────────────────────────────────────────

    _STATE_FILE = "state.json"

    def run_once(self) -> None:
        """
        Load state → run one candle action → save state → exit.
        Called by: python main.py --candle  (GitHub Actions per-candle mode)
        """
        now = datetime.now(IST)

        if now.weekday() >= 5:
            logger.info("Weekend — no action.")
            return

        today = now.date()
        self._load_state(today)

        hour_min = now.hour * 60 + now.minute

        # On first run of the day, always do morning setup + notify regardless of
        # how late GH Actions starts (cron can lag 30+ min past the 9:15 trigger).
        if self._is_new_day:
            logger.info("── Morning Setup (candle mode) ── %s", ist_now_str())
            self._morning_setup()
            self._notify_engine_start(today, now)

        try:
            if 10 * 60 <= hour_min < 15 * 60:
                self._candle_cycle()

            elif 15 * 60 <= hour_min < 15 * 60 + 30:
                self._hard_close()
                self.open_positions.clear()
                self.risk_manager.open_positions = 0
                self._day_summary(sync_retrain=True)

            else:
                logger.info(
                    "Outside trading window (%02d:%02d IST) — no action.",
                    now.hour, now.minute,
                )
        except Exception as exc:
            logger.error("run_once error: %s", exc, exc_info=True)
        finally:
            # Always persist state so GitHub Actions can always commit state.json
            self._save_state(today)

    def _load_state(self, today) -> None:
        """Load session state from state.json. Resets automatically on a new day."""
        if not os.path.exists(self._STATE_FILE):
            logger.info("No state file — starting fresh.")
            self._is_new_day = True
            self._reset_day()
            return

        try:
            with open(self._STATE_FILE) as f:
                s = json.load(f)

            if s.get("date") != str(today):
                logger.info("New trading day — resetting state.")
                self._is_new_day = True
                self._reset_day()
                return

            self._is_new_day = False

            self.running_capital          = float(s.get("running_capital", TRADING_CAPITAL))
            self.daily_trades             = s.get("daily_trades", {idx: 0 for idx in ACTIVE_INDICES})
            self.directions_traded_today  = set(s.get("directions_traded_today", []))
            self.profit_lock_done         = bool(s.get("profit_lock_done", False))
            self.morning_pcr              = s.get("morning_pcr", {})
            self.shadow_signals_today     = int(s.get("shadow_signals_today", 0))
            self.ml_override_trades_today = int(s.get("ml_override_trades_today", 0))
            self.wins_today               = int(s.get("wins_today", 0))
            self.losses_today             = int(s.get("losses_today", 0))
            self.ml_skipped_today         = int(s.get("ml_skipped_today", 0))
            self.eod_sent                 = bool(s.get("eod_sent", False))
            self.vix_alert_sent           = bool(s.get("vix_alert_sent", False))
            self.loss_limit_alert_sent    = bool(s.get("loss_limit_alert_sent", False))
            self.last_heartbeat_hour      = int(s.get("last_heartbeat_hour", -1))

            self.risk_manager.daily_pnl = float(s.get("daily_pnl", 0.0))
            if "orb_state" in s:
                self.orb_engine.load_state(s["orb_state"])

            # Restore pending retest signal
            self._pending_signal = s.get("pending_signal", None)

            # Restore pending shadow signals (ISO string → datetime)
            restored_shadows = []
            for sh in s.get("pending_shadows", []):
                sh = dict(sh)
                if isinstance(sh.get("entry_time"), str):
                    try:
                        sh["entry_time"] = datetime.fromisoformat(sh["entry_time"])
                        if sh["entry_time"].tzinfo is None:
                            sh["entry_time"] = IST.localize(sh["entry_time"])
                    except Exception:
                        sh["entry_time"] = datetime.now(IST)
                restored_shadows.append(sh)
            self._pending_shadows = restored_shadows

            positions = []
            for p in s.get("open_positions", []):
                if isinstance(p.get("entry_time"), str):
                    try:
                        p["entry_time"] = datetime.fromisoformat(p["entry_time"])
                        if p["entry_time"].tzinfo is None:
                            p["entry_time"] = IST.localize(p["entry_time"])
                    except Exception:
                        p["entry_time"] = datetime.now(IST)
                # Restore ScaleOutPosition from saved dict
                if isinstance(p.get("scale_out"), dict):
                    try:
                        p["scale_out"] = self._deserialize_scale_out(p["scale_out"])
                    except Exception as _se:
                        logger.warning("Could not restore scale_out: %s", _se)
                        from execution.scale_out import ScaleOutPosition
                        p["scale_out"] = ScaleOutPosition(
                            total_lots=p.get("lots", 1),
                            entry_premium=p.get("entry_premium", 100),
                            direction=p.get("direction", "CALL"),
                        )
                positions.append(p)
            self.open_positions = positions
            self.risk_manager.open_positions = len(positions)

            logger.info(
                "State loaded: date=%s | open=%d | daily_pnl=₹%.2f | capital=₹%.0f",
                today, len(self.open_positions),
                self.risk_manager.daily_pnl, self.running_capital,
            )

        except Exception as exc:
            logger.warning("State load failed: %s — starting fresh.", exc)
            self._reset_day()

    @staticmethod
    def _serialize_scale_out(so) -> dict:
        """Convert ScaleOutPosition to a JSON-safe dict."""
        return {
            "total_lots":    so.total_lots,
            "entry_premium": so.entry_premium,
            "direction":     so.direction,
            "t1_done":       so.t1_done,
            "t2_done":       so.t2_done,
            "t3_done":       so.t3_done,
            "lots_remaining":so.lots_remaining,
            "sl_price":      so.sl_price,
            "peak_premium":  so.peak_premium,
            "candles_held":  so.candles_held,
            "flat_candles":  so.flat_candles,
            "prev_premium":  so.prev_premium,
        }

    @staticmethod
    def _deserialize_scale_out(d: dict):
        """Reconstruct ScaleOutPosition from a saved dict."""
        from execution.scale_out import ScaleOutPosition
        so = ScaleOutPosition(
            total_lots=d["total_lots"],
            entry_premium=d["entry_premium"],
            direction=d["direction"],
        )
        so.t1_done        = d.get("t1_done", False)
        so.t2_done        = d.get("t2_done", False)
        so.t3_done        = d.get("t3_done", False)
        so.lots_remaining = d.get("lots_remaining", d["total_lots"])
        so.sl_price       = d.get("sl_price", so.sl_price)
        so.peak_premium   = d.get("peak_premium", d["entry_premium"])
        so.candles_held   = d.get("candles_held", 0)
        so.flat_candles   = d.get("flat_candles", 0)
        so.prev_premium   = d.get("prev_premium", d["entry_premium"])
        return so

    def _save_state(self, today) -> None:
        """Persist session state to state.json for the next candle run."""
        try:
            positions = []
            for pos in self.open_positions:
                p = dict(pos)
                if isinstance(p.get("entry_time"), datetime):
                    p["entry_time"] = p["entry_time"].isoformat()
                # Serialize ScaleOutPosition dataclass → plain dict
                if "scale_out" in p and hasattr(p["scale_out"], "total_lots"):
                    p["scale_out"] = self._serialize_scale_out(p["scale_out"])
                positions.append(p)

            # Serialize _pending_shadows (datetime → ISO string)
            shadows_serial = []
            for sh in self._pending_shadows:
                s = dict(sh)
                if isinstance(s.get("entry_time"), datetime):
                    s["entry_time"] = s["entry_time"].isoformat()
                shadows_serial.append(s)

            state = {
                "date":                    str(today),
                "last_run_time":           datetime.now(IST).strftime("%H:%M"),
                "running_capital":         round(self.running_capital, 2),
                "daily_pnl":               round(self.risk_manager.daily_pnl, 2),
                "paper_mode":              PAPER_MODE,
                "daily_trades":            self.daily_trades,
                "directions_traded_today": list(self.directions_traded_today),
                "profit_lock_done":        self.profit_lock_done,
                "morning_pcr":             self.morning_pcr,
                "open_positions":          positions,
                "pending_signal":          self._pending_signal,
                "pending_shadows":         shadows_serial,
                "shadow_signals_today":      self.shadow_signals_today,
                "ml_override_trades_today": self.ml_override_trades_today,
                "wins_today":               self.wins_today,
                "losses_today":             self.losses_today,
                "ml_skipped_today":         self.ml_skipped_today,
                "eod_sent":                 self.eod_sent,
                "vix_alert_sent":           self.vix_alert_sent,
                "loss_limit_alert_sent":    self.loss_limit_alert_sent,
                "last_heartbeat_hour":      self.last_heartbeat_hour,
                "orb_state":               self.orb_engine.state_dict(),
            }
            with open(self._STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("State saved → %s", self._STATE_FILE)

        except Exception as exc:
            logger.warning("State save failed: %s", exc)

    def _reset_day(self) -> None:
        self.risk_manager.reset_daily()
        self.orb_engine.reset_day()
        self.open_positions.clear()
        self._pending_shadows.clear()
        self._pending_signal = None
        self.morning_pcr = {}
        self.daily_trades = {idx: 0 for idx in ACTIVE_INDICES}
        self.directions_traded_today = set()
        self.profit_lock_done = False
        self.shadow_signals_today = 0
        self.ml_override_trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.ml_skipped_today = 0
        self.eod_sent = False
        self.vix_alert_sent = False
        self.loss_limit_alert_sent = False
        self.last_heartbeat_hour = -1
        logger.info("New trading day — session state reset.")

    def _get_daily_candle(self, index: str, day) -> Optional[Dict]:
        """Fetch yesterday's OHLC to anchor Fibonacci levels."""
        try:
            from data.feed import DataFeed
            df = self.feed.get_historical_candles(
                str(day), str(day), index=index
            )
            if df is None or df.empty:
                return None
            return {
                "high":  float(df["high"].max()),
                "low":   float(df["low"].min()),
                "open":  float(df["open"].iloc[0]),
                "close": float(df["close"].iloc[-1]),
            }
        except Exception as exc:
            logger.debug("Daily candle fetch failed [%s %s]: %s", index, day, exc)
            return None

    def _notify_engine_start(self, today, now_ist) -> None:
        """Send engine-start Telegram message (once per day after morning setup)."""
        try:
            from telegram_alerts import send_engine_start
            auc: Optional[float] = None
            try:
                _meta = os.path.join("ml", "models", "model_metadata.json")
                if os.path.exists(_meta):
                    with open(_meta) as _f:
                        auc = float(json.load(_f).get("cv_auc", 0)) or None
            except Exception:
                pass

            buckets = 0
            try:
                buckets = len(self.adaptive.memory.all_buckets())
            except Exception:
                pass

            # Average available PCR values (filled by morning_setup just before this call)
            pcr_vals = [v for v in self.morning_pcr.values() if v is not None]
            pcr_avg  = round(sum(pcr_vals) / len(pcr_vals), 2) if pcr_vals else None

            send_engine_start(
                today        = str(today),
                current_time = now_ist.strftime("%H:%M"),
                indices      = ", ".join(ACTIVE_INDICES),
                capital      = self.running_capital,
                auc          = auc,
                pattern_buckets = buckets,
                pcr          = pcr_avg,
                paper        = PAPER_MODE,
            )
        except Exception as exc:
            logger.warning("Engine-start notification failed: %s", exc)

    def _notify_engine_end(
        self,
        trades_today: int,
        pnl_today: float,
        capital: float,
        ml_auc: Optional[float],
    ) -> None:
        """Send engine-end Telegram message (once per day after hard close)."""
        try:
            from telegram_alerts import send_engine_end
            import csv as _csv

            # All-time stats from trades_log.csv
            total_trades = 0
            total_pnl    = 0.0
            try:
                _csv_path = TRADE_LOG_FILE
                if os.path.exists(_csv_path):
                    with open(_csv_path, newline="", encoding="utf-8") as _f:
                        for row in _csv.DictReader(_f):
                            try:
                                total_trades += 1
                                total_pnl    += float(row.get("pnl") or 0)
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

            # ML mode label
            _ML_ACTIVATE_AUC = 0.65
            if ml_auc and ml_auc >= _ML_ACTIVATE_AUC:
                ml_mode = f"Active (AUC {ml_auc:.3f})"
            else:
                ml_mode = "Indicators Only (training)"

            # signals_today = actual trades + shadow/blocked signals
            signals_today = trades_today + self.shadow_signals_today

            send_engine_end(
                today        = datetime.now(IST).strftime("%Y-%m-%d"),
                trades_today = trades_today,
                wins         = self.wins_today,
                losses       = self.losses_today,
                pnl_today    = pnl_today,
                total_trades = total_trades,
                total_pnl    = round(total_pnl, 2),
                capital      = capital,
                auc          = ml_auc,
                ml_mode      = ml_mode,
                signals_today = signals_today,
                ml_skipped   = self.ml_skipped_today,
            )
        except Exception as exc:
            logger.warning("Engine-end notification failed: %s", exc)

    def _get_ml_override_direction(
        self, index: str, result, now_ist
    ) -> Optional[str]:
        """
        Returns a direction string ("CALL" or "PUT") when all ML-override
        conditions are satisfied, otherwise None.

        Requirements:
          - Exactly MIN_CONDITIONS-1 (3/5) conditions met
          - CV AUC >= 0.72 (well-trained model required)
          - Time window 10:00–12:30 IST only
          - Not expiry day (Thu for NIFTY, Wed for BANKNIFTY)
          - ml_override_trades_today == 0  (one override per day max)
          - Unambiguous direction from call_score / put_score
        """
        if result.conditions_met != MIN_CONDITIONS - 1:
            return None

        # AUC gate
        _ML_OVERRIDE_AUC = 0.72
        try:
            _meta_path = os.path.join("ml", "models", "model_metadata.json")
            if not os.path.exists(_meta_path):
                return None
            with open(_meta_path) as _f:
                _auc = float(json.load(_f).get("cv_auc", 0))
        except Exception:
            return None
        if _auc < _ML_OVERRIDE_AUC:
            return None

        # Time window: 10:00–12:30 IST
        hour_min = now_ist.hour * 60 + now_ist.minute
        if not (10 * 60 <= hour_min < 12 * 60 + 30):
            return None

        # Skip expiry day (weekly expiry kills time value fast)
        _EXPIRY_WEEKDAY = {"NIFTY": 3, "BANKNIFTY": 2}  # Thu=3, Wed=2 (Mon=0)
        exp_day = _EXPIRY_WEEKDAY.get(index)
        if exp_day is not None and now_ist.weekday() == exp_day:
            return None

        # Daily override cap
        if self.ml_override_trades_today >= 1:
            return None

        # Determine direction (must be unambiguous)
        d          = result.details
        call_score = int(d.get("call_score") or 0)
        put_score  = int(d.get("put_score")  or 0)

        if call_score == 3 and put_score < 3:
            return "CALL"
        if put_score == 3 and call_score < 3:
            return "PUT"
        return None

    @staticmethod
    def _pcr_bias_label(pcr: Optional[float]) -> str:
        if pcr is None:
            return "neutral"
        from config.settings import PCR_BULL_MIN, PCR_BEAR_MAX
        if pcr > PCR_BULL_MIN:
            return "bullish"
        if pcr < PCR_BEAR_MAX:
            return "bearish"
        return "neutral"

    @staticmethod
    def _print_candle_summary(index: str, result, spot: float) -> None:
        """Clean per-candle console line as specified."""
        now = datetime.now(IST).strftime("%H:%M")
        d   = result.details
        if result.direction in ("CALL", "PUT"):
            adx = d.get("adx", 0) or 0
            logger.info(
                "[%s] %s | SIGNAL %s %d/5 | spot=%.0f | ADX %.1f",
                now, index, result.direction, result.conditions_met, spot, adx,
            )
        else:
            reason = d.get("reason", "")
            rsi    = d.get("rsi") or 0
            if reason:
                logger.info("[%s] %s | No signal (%s)", now, index, reason)
            else:
                logger.info(
                    "[%s] %s | No signal | RSI=%.1f | EMA_bull=%s EMA_bear=%s",
                    now, index, rsi,
                    d.get("ema_bull", False), d.get("ema_bear", False),
                )
