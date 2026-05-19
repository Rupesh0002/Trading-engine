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
import logging
import os
import threading
import time
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import schedule
import pytz

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

        # ── Session state ─────────────────────────────────────────────────
        self.open_positions: List[Dict[str, Any]] = []
        self.morning_pcr:    Dict[str, Optional[float]] = {}
        self.daily_trades:   Dict[str, int] = {idx: 0 for idx in ACTIVE_INDICES}
        self.directions_traded_today: set = set()   # global CALL/PUT cap across all indices
        self.profit_lock_done: bool = False          # 2 PM profit lock fires once per day
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

        if not self.open_positions:
            logger.info("Hard close: no open positions. %s", ist_now_str())
            return

        logger.info(
            "HARD CLOSE — %d position(s) | %s",
            len(self.open_positions), ist_now_str(),
        )
        for pos in list(self.open_positions):
            self._close_position(pos, reason="Hard close (15:00)")
        self.open_positions.clear()
        self.risk_manager.open_positions = 0

    def _day_summary(self) -> None:
        """15:30 IST — print and send Telegram day summary."""
        try:
            from telegram_alerts import send_day_summary
        except ImportError:
            send_day_summary = None

        trades_today = sum(self.daily_trades.values())
        pnl          = self.risk_manager.daily_pnl
        capital_now  = self.running_capital + pnl

        lines = [
            "─" * 50,
            f"  DAY SUMMARY  {ist_now_str()}",
            f"  Trades today : {trades_today}",
            f"  Daily P&L    : ₹{pnl:+,.2f}",
            f"  Capital      : ₹{capital_now:,.0f}",
            f"  Mode         : {'PAPER' if PAPER_MODE else 'LIVE'}",
            "─" * 50,
        ]
        for line in lines:
            logger.info(line)

        if send_day_summary:
            send_day_summary(
                trades=trades_today,
                pnl=pnl,
                capital=capital_now,
                paper=PAPER_MODE,
            )

        # Retrain ML model in background if trades happened today
        self._retrain_eod(trades_today)

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

        # ── Fetch VIX ──────────────────────────────────────────────────────
        vix = self.feed.get_vix()
        now_ist = datetime.now(IST)

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

        # ── Monitor open positions (SL / target) ───────────────────────────
        for pos in list(self.open_positions):
            ltp = self.option_chain.get_ltp(pos["symbol"], pos["index"])
            if ltp is None:
                continue

            # Update trailing SL
            pos["stop_loss"] = self.risk_manager.update_trailing_sl(ltp, pos["stop_loss"])

            exit_reason = None
            if ltp <= pos["stop_loss"]:
                exit_reason = "Stop-loss"
            elif ltp >= pos["target"]:
                rr = round(
                    (ltp - pos["entry_premium"])
                    / (pos["entry_premium"] * self.risk_manager.stop_loss_pct), 2
                )
                exit_reason = f"Target ({rr:.1f}×)"

            if exit_reason:
                pnl = self._close_position(pos, reason=exit_reason, exit_premium=ltp)
                self.open_positions.remove(pos)
                self.risk_manager.open_positions -= 1
                if exit_reason.startswith("Stop") and send_sl_hit:
                    send_sl_hit(pos["index"], pos["symbol"], pnl,
                                self.running_capital + self.risk_manager.daily_pnl)
                elif send_target_hit:
                    send_target_hit(pos["index"], pos["symbol"], pnl,
                                    self.running_capital + self.risk_manager.daily_pnl)

        # ── Risk gate ───────────────────────────────────────────────────────
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
                try:
                    _exp_nm = self.option_chain._nearest_expiry(index)
                    from datetime import date as _date_nm
                    _dte_nm = max((_exp_nm - _date_nm.today()).days, 1)
                    self.adaptive.check_near_miss(index, result.details, _dte_nm, now_ist.hour)
                except Exception as _e:
                    logger.debug("Near-miss check [%s]: %s", index, _e)
                continue

            # Soft cap: 2nd trade only for 5/5 signals
            if (self.daily_trades.get(index, 0) >= MAX_DAILY_TRADES
                    and result.conditions_met < STRONG_SIGNAL_THRESHOLD):
                logger.info(
                    "[%s] Daily trade limit (%d). 2nd trade requires 5/5 signal (got %d/5).",
                    index, MAX_DAILY_TRADES, result.conditions_met,
                )
                continue

            # Global same-direction cap: skip if this direction already traded today
            if result.direction in self.directions_traded_today:
                logger.info(
                    "[%s] %s already traded today — skipping correlated signal.",
                    index, result.direction,
                )
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
                }

        # ── Enter best trade ────────────────────────────────────────────────
        if best:
            self._enter_trade(best, vix=vix, tg_entry=send_trade_entered)

    # ──────────────────────────────────────────────────────────────────────
    # Trade entry
    # ──────────────────────────────────────────────────────────────────────

    def _enter_trade(
        self,
        best: Dict[str, Any],
        vix: Optional[float],
        tg_entry=None,
    ) -> None:
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

        quantity, lots = self.risk_manager.position_size(premium, index)
        if quantity == 0:
            logger.info("[%s] Position size 0 for premium=%.2f — skip.", index, premium)
            return

        # ML + Adaptive confidence gate
        from datetime import date as _date2
        dte = max((expiry - _date2.today()).days, 1)
        signal_details = {**result.details, "conditions_met": result.conditions_met}
        entry_time_str = datetime.now(IST).strftime("%H:%M:%S")
        ml_conf = self.ml_predictor.confidence(
            signal_details=signal_details,
            spot=spot,
            vix=vix,
            pcr=best["pcr"],
            dte=dte,
            entry_time_str=entry_time_str,
        )
        blended_conf, adaptive_reason = self.adaptive.score(
            index, result.direction, signal_details, dte,
            datetime.now(IST).hour, ml_conf,
        )
        if blended_conf < ML_MIN_CONFIDENCE:
            logger.info(
                "[%s] Adaptive+ML blocked: confidence=%.2f < threshold=%.2f | %s",
                index, blended_conf, ML_MIN_CONFIDENCE, adaptive_reason,
            )
            return
        ml_conf = blended_conf

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
            # stored for ML CSV logging on close
            "signal_details": d,
            "actual_risk":   actual_risk,
            "dte":           dte,
            "capital_before": round(self.running_capital, 2),
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

            row = {
                "signal_id":     uuid.uuid4().hex[:8].upper(),
                "date":          pos["entry_time"].strftime("%Y-%m-%d"),
                "time":          pos["entry_time"].strftime("%H:%M:%S"),
                "index":         pos.get("index", ""),
                "direction":     pos.get("direction", ""),
                "direction_int": 1 if pos.get("direction") == "CALL" else 0,
                "conditions_met": pos.get("conditions_met", 0),
                "close":         round(close, 2),
                "vwap":          round(vwap, 2),
                "vwap_distance": round(close - vwap, 2),
                "near_fib":      int(bool(d.get("near_fib"))),
                "fib_label":     d.get("fib_label", ""),
                "fib_level":     round(fib_lvl, 2),
                "fib_distance":  round(abs(close - fib_lvl), 2) if fib_lvl else "",
                "rsi":           round(float(d.get("rsi") or 50), 2),
                "adx":           round(float(d.get("adx") or 0), 2),
                "ema_bull":      int(bool(d.get("ema_bull"))),
                "ema_bear":      int(bool(d.get("ema_bear"))),
                "vol_spike":     int(bool(d.get("vol_spike"))),
                "vol_ratio":     "",
                "pcr":           d.get("pcr", ""),
                "swing_high":    round(float(d.get("swing_high", 0)), 2),
                "swing_low":     round(float(d.get("swing_low", 0)), 2),
                "dte":           pos.get("dte", 1),
                "capital_before": pos.get("capital_before", round(self.running_capital, 2)),
                "capital_used":  round(pos.get("entry_premium", 0) * pos.get("quantity", 0), 2),
                "risk_amount":   round(pos.get("actual_risk", 0), 2),
                "entry_premium": round(pos.get("entry_premium", 0), 2),
                "stop_loss":     round(pos.get("stop_loss", 0), 2),
                "target":        round(pos.get("target", 0), 2),
                "exit_premium":  round(exit_premium, 2),
                "pnl":           round(pnl, 2),
                "pnl_pct":       round(pnl_pct, 2),
                "exit_reason":   reason,
                "trade_taken":   1,
                "outcome":       1 if pnl > 0 else 0,
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

    def _retrain_eod(self, trades_today: int) -> None:
        """Retrain the XGBoost model in a background thread after market close."""
        if trades_today == 0:
            logger.info("No trades today — skipping EOD ML retraining.")
            return

        def _run() -> None:
            try:
                logger.info("EOD ML retraining started (background thread) ...")
                from ml.train import train

                # Collect all backtest signal CSVs + today's live signal CSV
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

        if 9 * 60 <= hour_min < 9 * 60 + 30:
            logger.info("── Morning Setup (candle mode) ── %s", ist_now_str())
            self._morning_setup()

        elif 10 * 60 <= hour_min < 15 * 60:
            self._candle_cycle()

        elif 15 * 60 <= hour_min < 15 * 60 + 30:
            self._hard_close()
            self.open_positions.clear()
            self.risk_manager.open_positions = 0
            self._day_summary()

        else:
            logger.info(
                "Outside trading window (%02d:%02d IST) — no action.",
                now.hour, now.minute,
            )

        self._save_state(today)

    def _load_state(self, today) -> None:
        """Load session state from state.json. Resets automatically on a new day."""
        if not os.path.exists(self._STATE_FILE):
            logger.info("No state file — starting fresh.")
            self._reset_day()
            return

        try:
            with open(self._STATE_FILE) as f:
                s = json.load(f)

            if s.get("date") != str(today):
                logger.info("New trading day — resetting state.")
                self._reset_day()
                return

            self.running_capital          = float(s.get("running_capital", TRADING_CAPITAL))
            self.daily_trades             = s.get("daily_trades", {idx: 0 for idx in ACTIVE_INDICES})
            self.directions_traded_today  = set(s.get("directions_traded_today", []))
            self.profit_lock_done         = bool(s.get("profit_lock_done", False))
            self.morning_pcr              = s.get("morning_pcr", {})

            try:
                self.risk_manager._daily_pnl = float(s.get("daily_pnl", 0.0))
            except AttributeError:
                pass

            positions = []
            for p in s.get("open_positions", []):
                if isinstance(p.get("entry_time"), str):
                    try:
                        p["entry_time"] = datetime.fromisoformat(p["entry_time"])
                        if p["entry_time"].tzinfo is None:
                            p["entry_time"] = IST.localize(p["entry_time"])
                    except Exception:
                        p["entry_time"] = datetime.now(IST)
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

    def _save_state(self, today) -> None:
        """Persist session state to state.json for the next candle run."""
        try:
            positions = []
            for pos in self.open_positions:
                p = dict(pos)
                if isinstance(p.get("entry_time"), datetime):
                    p["entry_time"] = p["entry_time"].isoformat()
                positions.append(p)

            state = {
                "date":                    str(today),
                "running_capital":         round(self.running_capital, 2),
                "daily_pnl":               round(self.risk_manager.daily_pnl, 2),
                "daily_trades":            self.daily_trades,
                "directions_traded_today": list(self.directions_traded_today),
                "profit_lock_done":        self.profit_lock_done,
                "morning_pcr":             self.morning_pcr,
                "open_positions":          positions,
            }
            with open(self._STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("State saved → %s", self._STATE_FILE)

        except Exception as exc:
            logger.warning("State save failed: %s", exc)

    def _reset_day(self) -> None:
        self.risk_manager.reset_daily()
        self.open_positions.clear()
        self.morning_pcr = {}
        self.daily_trades = {idx: 0 for idx in ACTIVE_INDICES}
        self.directions_traded_today = set()
        self.profit_lock_done = False
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
