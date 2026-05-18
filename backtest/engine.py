"""
Backtest engine — NSE F&O options strategy.
Uses Kite Connect historical spot data for signals.
Option premiums are SIMULATED (not real historical option prices):
  - Entry premium  : simplified Black-Scholes ATM estimate
                     premium ≈ 0.4 × spot × (vix/100) × √(dte/365)
  - Intraday P&L   : delta approximation (ATM delta ≈ 0.5)
                     option_move ≈ spot_move × 0.5
This gives a realistic signal-quality test without needing historical
option chain data (which Kite does not provide for free).

All parameters come from .env via config/settings.py.
No hardcoded values.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta, time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config.events_calendar import get_next_expiry
from config.settings import (
    ACTIVE_INDEX,
    BACKTEST_END,
    BACKTEST_START,
    BACKTEST_VIX,
    DAILY_LOSS_LIMIT_PCT,
    HARD_TARGET_PCT,
    INDEX_CONFIG,
    MAX_DAILY_TRADES,
    MAX_DAILY_TRADES_OTM,
    MAX_OPEN_POSITIONS,
    MAX_PREMIUM,
    MIN_CONDITIONS,
    MIN_PREMIUM,
    MIN_PROFIT_RATIO,
    PROFIT_LOCK_TIME,
    REPORTS_DIR,
    RISK_PER_TRADE_PCT,
    SIGNAL_START,
    SOFT_TARGET_PCT,
    STOP_LOSS_PCT,
    STRONG_SIGNAL_THRESHOLD,
    TARGET_PCT,
    TRADE_END,
    TRADING_CAPITAL,
    TRAILING_SL_PCT,
    WEAK_SIGNAL_TARGET_RATIO,
)
from signals.engine import SignalEngine

# CSV written for every signal that fires — used for manual verification + ML training
SIGNAL_CSV_COLUMNS = [
    "signal_id", "date", "time", "index", "direction", "direction_int", "conditions_met",
    "close", "vwap", "vwap_distance",
    "near_fib", "fib_label", "fib_level", "fib_distance",
    "rsi", "adx", "ema_bull", "ema_bear",
    "vol_spike", "vol_ratio", "pcr",
    "swing_high", "swing_low", "dte",
    "capital_before", "capital_used", "risk_amount",
    "entry_premium", "stop_loss", "target",
    "exit_premium", "pnl", "pnl_pct", "exit_reason",
    "trade_taken", "outcome",
]

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_SIGNAL_START_TIME  = time(*[int(x) for x in SIGNAL_START.split(":")])
_TRADE_END_TIME     = time(*[int(x) for x in TRADE_END.split(":")])
_PROFIT_LOCK_TIME   = time(*[int(x) for x in PROFIT_LOCK_TIME.split(":")])
_ATM_DELTA          = 0.5   # ATM option delta approximation


@dataclass
class BacktestTrade:
    date:           str
    index:          str
    direction:      str
    entry_time:     str
    exit_time:      str
    entry_spot:     float
    exit_spot:      float
    entry_premium:  float
    exit_premium:   float
    quantity:       int
    pnl:            float
    pnl_pct:        float
    exit_reason:    str
    conditions_met: int


@dataclass
class BacktestResults:
    trades:       List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float]         = field(default_factory=list)
    index:        str  = ACTIVE_INDEX
    signal_csv:   str  = ""
    start_date:   str  = ""
    end_date:     str  = ""

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / TRADING_CAPITAL * 100

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss   = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        arr  = np.array(self.equity_curve)
        peak = np.maximum.accumulate(arr)
        dd   = (peak - arr) / peak
        return float(dd.max()) * 100

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        rets = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        return float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    def summary(self) -> str:
        sep = "─" * 60
        rr  = TARGET_PCT / STOP_LOSS_PCT
        return "\n".join([
            sep,
            f"  BACKTEST — {self.index} F&O | {self.start_date} → {self.end_date}",
            f"  Strategy : VWAP + Fibonacci + RSI + Volume (min {MIN_CONDITIONS}/5)",
            f"  R:R      : 1:{rr:.1f}  SL={STOP_LOSS_PCT*100:.0f}%  "
            f"Exit={SOFT_TARGET_PCT*100:.1f}%–{HARD_TARGET_PCT*100:.1f}%",
            f"  Premiums : simulated (BS approx, VIX={BACKTEST_VIX:.1f})",
            sep,
            f"  Initial Capital  : ₹{TRADING_CAPITAL:>12,.0f}",
            f"  Final Capital    : ₹{(TRADING_CAPITAL + self.total_pnl):>12,.0f}",
            f"  Total PnL        : ₹{self.total_pnl:>+12,.2f}",
            f"  Total Return     : {self.total_return_pct:>+11.2f}%",
            sep,
            f"  Total Trades     : {self.total_trades:>12}",
            f"  Winning Trades   : {self.winning_trades:>12}",
            f"  Losing Trades    : {self.total_trades - self.winning_trades:>12}",
            f"  Win Rate         : {self.win_rate:>11.1f}%",
            f"  Profit Factor    : {self.profit_factor:>12.2f}",
            sep,
            f"  Max Drawdown     : {self.max_drawdown_pct:>11.2f}%",
            f"  Sharpe Ratio     : {self.sharpe_ratio:>12.2f}",
            sep,
        ])


class BacktestEngine:
    """
    Runs the strategy on historical NSE index data.
    Signal logic is identical to the live engine.
    Option premiums are simulated — see module docstring.
    """

    def __init__(self, kite, index: Optional[str] = None) -> None:
        self.kite          = kite
        self.index         = index or ACTIVE_INDEX
        self.signal_engine = SignalEngine()
        cfg                = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size     = cfg["lot_size"]
        self._adx_threshold = cfg.get("adx_threshold", 20.0)
        self._signal_rows: list[dict] = []   # accumulates one row per signal

    def run(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> BacktestResults:
        from data.feed import DataFeed
        feed  = DataFeed(self.kite)
        start = from_date or BACKTEST_START
        end   = to_date   or BACKTEST_END

        logger.info(
            "Backtest: %s | %s → %s | capital=₹%.0f | VIX=%.1f",
            self.index, start, end, TRADING_CAPITAL, BACKTEST_VIX,
        )

        df_full = feed.get_historical_candles(start, end, index=self.index)
        if df_full is None or df_full.empty:
            logger.error("No historical data for %s. Check token and date range.", self.index)
            return BacktestResults(index=self.index, start_date=start, end_date=end)

        results = BacktestResults(index=self.index, start_date=start, end_date=end)
        capital = TRADING_CAPITAL
        results.equity_curve.append(capital)

        df_full["_date"] = df_full["timestamp"].dt.date

        # Build daily OHLC from intraday data — used for fixed Fibonacci anchors (Fix 1).
        daily_ohlc: dict = {}
        for day, day_df in df_full.groupby("_date"):
            daily_ohlc[day] = {
                "high":  float(day_df["high"].max()),
                "low":   float(day_df["low"].min()),
                "open":  float(day_df["open"].iloc[0]),
                "close": float(day_df["close"].iloc[-1]),
            }
        sorted_days = sorted(daily_ohlc.keys())

        for day, day_df in df_full.groupby("_date"):
            day_df = day_df.reset_index(drop=True)
            # Yesterday's daily candle → fixed Fibonacci anchors for today
            day_idx      = sorted_days.index(day)
            prev_candle  = daily_ohlc[sorted_days[day_idx - 1]] if day_idx > 0 else None
            capital = self._simulate_day(day, day_df, capital, results, prev_candle)
            results.equity_curve.append(capital)

        csv_path = self._write_signal_csv(start, end)
        logger.info(
            "Backtest done: %d trades | PnL=₹%.2f | Signal CSV: %s",
            results.total_trades, results.total_pnl, csv_path,
        )
        results.signal_csv = csv_path
        return results

    def _write_signal_csv(self, start: str, end: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(
            REPORTS_DIR,
            f"signals_{self.index}_{start}_{end}.csv",
        )
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._signal_rows)
        logger.info("Signal CSV written: %s (%d rows)", path, len(self._signal_rows))
        return path

    # ------------------------------------------------------------------
    # Per-day simulation
    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        day,
        day_df: pd.DataFrame,
        capital: float,
        results: BacktestResults,
        prev_candle: Optional[dict] = None,
    ) -> float:
        daily_pnl          = 0.0
        daily_loss_limit   = TRADING_CAPITAL * DAILY_LOSS_LIMIT_PCT
        daily_trade_count  = 0
        open_positions: list[dict] = []
        directions_taken: set      = set()   # same-direction cap: max 1 CALL + 1 PUT per day
        profit_lock_done           = False   # 2 PM profit lock fires once

        # Days to expiry — rolls to next week ONLY on expiry day itself (days_buffer=0).
        # days_buffer=2 was too aggressive: rolled on Tue/Wed/Thu → next-week DTE ~7-9
        # → BS premium inflates to ₹220-250+ → NIFTY lot (50×) never fits ₹1L risk budget.
        # With days_buffer=0: Mon-Wed get DTE 2-4 (affordable), Thu expiry day → next week.
        expiry_date = get_next_expiry(self.index, day, days_buffer=0)
        dte = max((expiry_date - day).days, 1)

        for i in range(2, len(day_df)):
            candle      = day_df.iloc[i]
            candle_time = candle["timestamp"].time()

            # Hard close at TRADE_END
            if candle_time >= _TRADE_END_TIME:
                # Run SL/target checks on this candle first — prevents bypassing
                # the stop-loss when a large adverse move occurs within the closing candle.
                open_positions, pnl_closed = self._check_exits(
                    open_positions, candle, results, day
                )
                daily_pnl += pnl_closed
                for pos in open_positions:
                    exit_spot    = float(candle["close"])
                    exit_premium = self._option_price_at(pos, exit_spot)
                    pnl          = (exit_premium - pos["entry_premium"]) * pos["quantity"]
                    pnl_pct      = (exit_premium - pos["entry_premium"]) / pos["entry_premium"] * 100
                    daily_pnl   += pnl
                    self._update_signal_row(
                        pos, exit_premium, pnl, pnl_pct, "Hard close (15:00)"
                    )
                    results.trades.append(self._make_trade(
                        day, pos, exit_spot, exit_premium,
                        pnl, pnl_pct, candle_time, "Hard close (15:00)",
                    ))
                open_positions.clear()
                break

            # Check SL / target on open positions
            open_positions, pnl_closed = self._check_exits(
                open_positions, candle, results, day
            )
            daily_pnl += pnl_closed

            # 2 PM profit lock — close any profitable positions at first 14:00 candle.
            # Even small gains are worth locking — late-day reversals erase intraday moves.
            if candle_time >= _PROFIT_LOCK_TIME and not profit_lock_done and open_positions:
                profit_lock_done = True
                still_open = []
                for pos in open_positions:
                    cur_px = self._option_price_at(pos, float(candle["close"]))
                    if cur_px > pos["entry_premium"]:
                        pnl     = (cur_px - pos["entry_premium"]) * pos["quantity"]
                        pnl_pct = (cur_px - pos["entry_premium"]) / pos["entry_premium"] * 100
                        daily_pnl += pnl
                        self._update_signal_row(pos, cur_px, pnl, pnl_pct, "Profit lock (14:00)")
                        results.trades.append(self._make_trade(
                            day, pos,
                            self._spot_from_option_price(pos, cur_px),
                            cur_px, pnl, pnl_pct, candle_time, "Profit lock (14:00)",
                        ))
                    else:
                        still_open.append(pos)
                open_positions = still_open

            if daily_pnl <= -daily_loss_limit:
                break

            if candle_time < _SIGNAL_START_TIME:
                continue
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                continue
            if daily_trade_count >= MAX_DAILY_TRADES_OTM:
                continue  # hard cap regardless of signal strength

            # Evaluate signal on all candles up to now (no lookahead).
            # Per-index ADX threshold passed to signal engine.
            # PCR=1.0 (neutral) — historical option chain unavailable in backtest.
            window_df = day_df.iloc[: i + 1].copy()
            signal    = self.signal_engine.evaluate(
                window_df, pcr=1.0, daily_candle=prev_candle,
                adx_threshold=self._adx_threshold,
            )

            if signal.direction not in ("CALL", "PUT"):
                continue
            # Soft cap: 2nd trade only allowed for 5/5 signals
            if daily_trade_count >= MAX_DAILY_TRADES and signal.conditions_met < STRONG_SIGNAL_THRESHOLD:
                continue
            if i + 1 >= len(day_df):
                continue
            # Never enter when the next candle is the hard-close candle.
            if day_df.iloc[i + 1]["timestamp"].time() >= _TRADE_END_TIME:
                continue
            # Same-direction daily cap: one CALL and one PUT per day maximum.
            # Prevents doubling down on correlated macro moves (e.g. all indices PUT same day).
            if signal.direction in directions_taken:
                continue

            d          = signal.details
            spot_close = float(d.get("close", 0))
            vwap_val   = float(d.get("vwap", 0))
            fib_level  = d.get("fib_level") or 0.0
            vol_ma     = float(d.get("vol_ma", 1) or 1)
            vol_cur    = float(d.get("current_vol", 0))

            # Build base signal row — will be updated when trade closes
            signal_id      = uuid.uuid4().hex[:8].upper()
            running_capital = round(capital + daily_pnl, 2)   # capital at this moment
            signal_row = {
                "signal_id":     signal_id,
                "date":          str(day),
                "time":          str(candle["timestamp"].time()),
                "index":         self.index,
                "direction":     signal.direction,
                "direction_int": 1 if signal.direction == "CALL" else 0,
                "conditions_met": signal.conditions_met,
                "close":         round(spot_close, 2),
                "vwap":          round(vwap_val, 2),
                "vwap_distance": round(spot_close - vwap_val, 2),
                "near_fib":      int(bool(d.get("near_fib"))),
                "fib_label":     d.get("fib_label", ""),
                "fib_level":     round(fib_level, 2),
                "fib_distance":  round(abs(spot_close - fib_level), 2) if fib_level else "",
                "rsi":           round(d.get("rsi") or 50, 2),
                "adx":           round(float(d.get("adx") or 0), 2),
                "ema_bull":      int(bool(d.get("ema_bull"))),
                "ema_bear":      int(bool(d.get("ema_bear"))),
                "vol_spike":     int(bool(d.get("vol_spike"))),
                "vol_ratio":     round(vol_cur / vol_ma, 2) if vol_ma else "",
                "pcr":           d.get("pcr", ""),
                "swing_high":    round(float(d.get("swing_high", 0)), 2),
                "swing_low":     round(float(d.get("swing_low", 0)), 2),
                "dte":           dte,
                "capital_before": running_capital,
                "capital_used":  "",   # filled after quantity is known
                "risk_amount":   "",   # filled after quantity is known
                # trade fields — filled in when trade closes
                "entry_premium": "", "stop_loss": "", "target": "",
                "exit_premium":  "", "pnl": "", "pnl_pct": "",
                "exit_reason":   "", "trade_taken": 0, "outcome": -1,
            }

            # Entry at OPEN of next candle
            next_candle   = day_df.iloc[i + 1]
            entry_spot    = float(next_candle["open"])
            entry_premium = self._simulate_entry_premium(entry_spot, dte)
            sl_price      = round(entry_premium * (1.0 - STOP_LOSS_PCT), 2)
            # Tiered target: 5/5 → 2.5× (50%), 4/5 → 1.5× (30%)
            _tgt_ratio = MIN_PROFIT_RATIO if signal.conditions_met >= STRONG_SIGNAL_THRESHOLD else WEAK_SIGNAL_TARGET_RATIO
            tgt_price  = round(entry_premium * (1.0 + STOP_LOSS_PCT * _tgt_ratio), 2)

            signal_row.update({
                "entry_premium": entry_premium,
                "stop_loss":     sl_price,
                "target":        tgt_price,
            })

            if not (MIN_PREMIUM <= entry_premium <= MAX_PREMIUM):
                signal_row["exit_reason"] = "premium_out_of_range"
                self._signal_rows.append(signal_row)
                logger.debug("Premium ₹%.2f outside range on %s — skipped.", entry_premium, day)
                continue

            # Fix 4 — cost-adjusted position sizing.
            # Brokerage is fixed (₹40 round trip) — deduct from risk budget.
            # Slippage and STT are per-unit — added to effective SL per unit instead.
            # This avoids circular dependency (needing quantity to estimate costs).
            _BROKERAGE    = 40.0    # ₹20 × 2 orders (Zerodha flat rate, entry + exit)
            _SLIPPAGE_PTS = 2       # option premium pts per side → 4 pts round-trip per unit
            _STT_PCT      = 0.0003  # 0.03% of exit-side premium per unit

            gross_risk = running_capital * RISK_PER_TRADE_PCT
            net_risk   = max(gross_risk - _BROKERAGE, 0)   # deduct fixed brokerage only

            risk_amount = net_risk
            # Effective SL per unit = option SL + per-unit variable costs
            sl_points   = (entry_premium * STOP_LOSS_PCT) + (_SLIPPAGE_PTS * 2) + (_STT_PCT * entry_premium)
            if sl_points <= 0:
                self._signal_rows.append(signal_row)
                continue
            lots     = math.floor(math.floor(risk_amount / sl_points) / self._lot_size)
            quantity = lots * self._lot_size
            if quantity == 0:
                signal_row["exit_reason"] = "quantity_zero"
                self._signal_rows.append(signal_row)
                continue

            signal_row["risk_amount"]  = round(risk_amount, 2)
            signal_row["capital_used"] = round(entry_premium * quantity, 2)
            signal_row["trade_taken"]  = 1
            daily_trade_count += 1
            directions_taken.add(signal.direction)
            open_positions.append({
                "signal_id":     signal_id,
                "signal_row":    signal_row,      # reference — updated on close
                "direction":     signal.direction,
                "entry_spot":    entry_spot,
                "entry_premium": entry_premium,
                "entry_time":    str(next_candle["timestamp"].time()),
                "stop_loss":     sl_price,
                "target":        tgt_price,
                "quantity":      quantity,
                "conditions":    signal.conditions_met,
            })
            self._signal_rows.append(signal_row)   # append now, mutate reference later
            logger.info(
                "[%s] %s ENTRY | spot=%.0f | premium=₹%.2f | qty=%d | SL=₹%.2f | TGT=₹%.2f",
                day, signal.direction, entry_spot, entry_premium, quantity,
                sl_price, tgt_price,
            )

        capital += daily_pnl
        return capital

    # ------------------------------------------------------------------
    # Exit checking
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        positions: list[dict],
        candle,
        results: BacktestResults,
        day,
    ) -> Tuple[list[dict], float]:
        remaining  = []
        total_pnl  = 0.0
        spot_high  = float(candle["high"])
        spot_low   = float(candle["low"])
        candle_time = str(candle["timestamp"].time())

        for pos in positions:
            # CALL gains when spot rises; PUT gains when spot falls.
            # opt_best  = highest option value this candle (favourable extreme)
            # opt_worst = lowest option value this candle (adverse extreme)
            if pos["direction"] == "PUT":
                opt_best  = self._option_price_at(pos, spot_low)   # PUT best at candle low
                opt_worst = self._option_price_at(pos, spot_high)  # PUT worst at candle high
            else:
                opt_best  = self._option_price_at(pos, spot_high)  # CALL best at candle high
                opt_worst = self._option_price_at(pos, spot_low)   # CALL worst at candle low

            exit_price  = None
            exit_reason = ""

            # Check SL at the CURRENT stop (before any trailing update) — conservative:
            # assumes the adverse intrabar extreme came before the favorable one.
            if opt_worst <= pos["stop_loss"]:
                exit_price  = pos["stop_loss"]
                exit_reason = "Stop-loss"
            elif opt_best >= pos["target"]:
                exit_price  = pos["target"]
                rr = round((exit_price - pos["entry_premium"]) / (pos["entry_premium"] * STOP_LOSS_PCT), 2)
                exit_reason = f"Target ({rr:.1f}×)"
            else:
                # Position still open — trail SL upward based on the BEST price reached.
                # Only update after confirming neither SL nor target was hit this candle.
                new_sl = round(opt_best * (1.0 - TRAILING_SL_PCT), 2)
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            if exit_price is not None:
                pnl     = (exit_price - pos["entry_premium"]) * pos["quantity"]
                pnl_pct = (exit_price - pos["entry_premium"]) / pos["entry_premium"] * 100
                total_pnl += pnl
                exit_spot = self._spot_from_option_price(pos, exit_price)
                self._update_signal_row(pos, exit_price, pnl, pnl_pct, exit_reason)
                results.trades.append(self._make_trade(
                    day, pos, exit_spot, exit_price, pnl, pnl_pct, candle_time, exit_reason,
                ))
            else:
                remaining.append(pos)

        return remaining, total_pnl

    # ------------------------------------------------------------------
    # Signal CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_signal_row(
        pos: dict,
        exit_premium: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        """Mutate the signal_row dict that was already appended to self._signal_rows."""
        row = pos.get("signal_row")
        if row is None:
            return
        row["exit_premium"] = round(exit_premium, 2)
        row["pnl"]          = round(pnl, 2)
        row["pnl_pct"]      = round(pnl_pct, 2)
        row["exit_reason"]  = exit_reason
        row["outcome"]      = 1 if pnl > 0 else 0

    # ------------------------------------------------------------------
    # Option premium simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_entry_premium(spot: float, dte_days: int) -> float:
        """
        Simplified Black-Scholes ATM call/put premium estimate.
        ATM premium ≈ 0.4 × S × σ × √(T)
        where σ = BACKTEST_VIX/100 (annualised), T = dte/365.
        """
        sigma   = BACKTEST_VIX / 100.0
        t       = max(dte_days, 0.5) / 365.0
        premium = 0.4 * spot * sigma * math.sqrt(t)
        return max(round(premium, 2), MIN_PREMIUM)

    @staticmethod
    def _option_price_at(pos: dict, current_spot: float) -> float:
        """
        Estimate current option premium from spot move using delta approximation.
        CALL: premium rises when spot rises.
        PUT:  premium rises when spot falls.
        """
        spot_move = current_spot - pos["entry_spot"]
        if pos["direction"] == "PUT":
            spot_move = -spot_move
        option_price = pos["entry_premium"] + spot_move * _ATM_DELTA
        return max(round(option_price, 2), 0.5)

    @staticmethod
    def _spot_from_option_price(pos: dict, option_price: float) -> float:
        """Reverse-map option price back to approximate spot (for logging)."""
        premium_move = option_price - pos["entry_premium"]
        spot_move    = premium_move / _ATM_DELTA
        if pos["direction"] == "PUT":
            spot_move = -spot_move
        return round(pos["entry_spot"] + spot_move, 2)

    # ------------------------------------------------------------------
    # Trade record builder
    # ------------------------------------------------------------------

    def _make_trade(
        self, day, pos: dict,
        exit_spot: float, exit_premium: float,
        pnl: float, pnl_pct: float,
        exit_time, exit_reason: str,
    ) -> BacktestTrade:
        return BacktestTrade(
            date=str(day),
            index=self.index,
            direction=pos["direction"],
            entry_time=pos["entry_time"],
            exit_time=str(exit_time),
            entry_spot=pos["entry_spot"],
            exit_spot=exit_spot,
            entry_premium=pos["entry_premium"],
            exit_premium=exit_premium,
            quantity=pos["quantity"],
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
            conditions_met=pos["conditions"],
        )
