"""
Backtest engine — Momentum-Confirmed NSE F&O options strategy.

Signal pipeline (in order):
  1. Day Quality Filter  — skip bad days (SKIP if score < 2/4)
  2. ConvictionEngine    — minimum score threshold (from .env)
  3. Momentum Confirm    — 5-min-style: green candle + body + volume + VWAP
  4. Scale-In            — 40% at confirmation; 40% at +3%; 20% at +6%
  5. Scale-Out exits     — 40% at +20%; 40% at +40%; 20% at +60%
  6. Hard SL             — -15% on ALL lots simultaneously
  7. No-Move exit        — flat ±3% for 2 candles → exit ALL

Option premiums are SIMULATED (not real historical option prices):
  - Entry premium  : 0.4 × spot × (vix/100) × √(dte/365)
  - Intraday P&L   : per-candle delta (ATM Δ = 0.5) + theta (-1.2% / candle)
  - Costs          : slippage 2pts/leg, brokerage ₹40 round-trip, STT 0.03% exit

All parameters come from .env via config/settings.py.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, time
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
    CONVICTION_MIN_SCORE,
    DAILY_LOSS_LIMIT_PCT,
    INDEX_CONFIG,
    MAX_DAILY_TRADES,
    MAX_LOTS_CAP,
    MAX_OPEN_POSITIONS,
    MAX_PREMIUM,
    BACKTEST_SLIPPAGE_PTS,
    LOT_SIZING_SL_PCT,
    RISK_PER_TRADE_PCT,
    MIN_PREMIUM,
    PROFIT_LOCK_TIME,
    REPORTS_DIR,
    REQUIRE_PREMIUM_DAY,
    SIGNAL_END,
    SIGNAL_START,
    STOP_LOSS_PCT,
    TRADE_END,
    TRADING_CAPITAL,
)
from data.iv_fetcher import get_backtest_iv_rank
from execution.scale_out import ScaleOutPosition
from signals.conviction_engine import ConvictionEngine
from signals.day_quality import assess_day_quality, DayQuality

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Time sentinels ────────────────────────────────────────────────────────────
_SIGNAL_START_TIME = time(*[int(x) for x in SIGNAL_START.split(":")])
_SIGNAL_END_TIME   = time(*[int(x) for x in SIGNAL_END.split(":")])
_TRADE_END_TIME    = time(*[int(x) for x in TRADE_END.split(":")])
_PROFIT_LOCK_TIME  = time(*[int(x) for x in PROFIT_LOCK_TIME.split(":")])

# ── Simulation constants ──────────────────────────────────────────────────────
_ATM_DELTA            = 0.5
_SLIPPAGE_PTS         = BACKTEST_SLIPPAGE_PTS   # 2 = conservative, 1 = realistic live spread
_BROKERAGE            = 40.0
_STT_PCT              = 0.0003
_THETA_PER_CANDLE_PCT = 0.012

# ── Scale-out thresholds (mirror execution/scale_out.py) ─────────────────────
_HARD_SL_PCT   = -0.15   # -15% on all lots

# ── Momentum-confirm thresholds (15-min backtest version) ────────────────────
# Body threshold removed (0.15% is 5-min specific; 15-min candles are coarser).
# Core checks: green candle + VWAP side + spot momentum (3/5 of the live spec).
_MC_BODY_MIN_PCT  = 0.0      # disabled for 15-min backtest
_MC_VOLUME_MULT   = 1.30     # 1.3× rolling avg (passes automatically when vol=0)
_MC_PREMIUM_TICK_LOOKBACK = 2  # candles back for "spot momentum rising"

# ── Max lots per index ────────────────────────────────────────────────────────
_MAX_LOTS = {"NIFTY": 5, "BANKNIFTY": 8, "FINNIFTY": 6, "SENSEX": 5}

# ── CSV columns ───────────────────────────────────────────────────────────────
SIGNAL_CSV_COLUMNS = [
    "signal_id", "date", "time", "index", "direction", "direction_int",
    "conviction_score", "lot_size", "day_quality",
    "close", "vwap", "vwap_distance", "rsi", "adx", "ema_fast", "ema_slow",
    "pcr", "iv_rank", "pdh", "pdl", "weekly_high", "weekly_low", "dte",
    "capital_before", "capital_used", "risk_amount",
    "entry_premium", "stop_loss", "target",
    "exit_premium", "pnl", "pnl_pct", "exit_reason",
    "trade_taken", "outcome",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    date:             str
    index:            str
    direction:        str
    entry_time:       str
    exit_time:        str
    entry_spot:       float
    exit_spot:        float
    entry_premium:    float
    exit_premium:     float
    quantity:         int
    gross_pnl:        float
    slippage_cost:    float
    brokerage_cost:   float
    stt_cost:         float
    theta_cost:       float
    pnl:              float
    pnl_pct:          float
    exit_reason:      str
    conviction_score: int
    day_quality:      str = "NORMAL"


@dataclass
class BacktestResults:
    trades:       List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float]         = field(default_factory=list)
    index:        str  = ACTIVE_INDEX
    signal_csv:   str  = ""
    start_date:   str  = ""
    end_date:     str  = ""

    @property
    def total_trades(self) -> int:   return len(self.trades)

    @property
    def winning_trades(self) -> int: return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades * 100 if self.trades else 0.0

    @property
    def total_pnl(self) -> float:       return sum(t.pnl for t in self.trades)
    @property
    def total_gross_pnl(self) -> float: return sum(t.gross_pnl for t in self.trades)
    @property
    def total_slippage(self) -> float:  return sum(t.slippage_cost for t in self.trades)
    @property
    def total_brokerage(self) -> float: return sum(t.brokerage_cost for t in self.trades)
    @property
    def total_stt(self) -> float:       return sum(t.stt_cost for t in self.trades)
    @property
    def total_theta(self) -> float:     return sum(t.theta_cost for t in self.trades)
    @property
    def total_costs(self) -> float:
        return self.total_slippage + self.total_brokerage + self.total_stt + self.total_theta

    @property
    def total_return_pct(self) -> float:  return self.total_pnl / TRADING_CAPITAL * 100
    @property
    def gross_return_pct(self) -> float:  return self.total_gross_pnl / TRADING_CAPITAL * 100

    @property
    def profit_factor(self) -> float:
        gp = sum(t.pnl for t in self.trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gp / gl if gl > 0 else float("inf")

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

    @property
    def avg_rr(self) -> float:
        win_pnls  = [t.pnl for t in self.trades if t.pnl > 0]
        loss_pnls = [t.pnl for t in self.trades if t.pnl <= 0]
        avg_win   = sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0.0
        avg_loss  = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
        return abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    def summary(self) -> str:
        dd    = compute_drawdown_stats(self.trades, TRADING_CAPITAL)
        wins  = self.winning_trades
        total = self.total_trades
        sep1  = "  ══════════════════════════════════════════"
        sep2  = "  ──────────────────────────────────────────"
        return "\n".join([
            sep1,
            "  BACKTEST RESULTS — Momentum-Confirmed System",
            sep1,
            f"  Period          : {self.start_date} → {self.end_date}",
            f"  Index           : {self.index}",
            sep2,
            "  TRADE STATS",
            f"  Total trades    : {total}",
            f"  Wins / Losses   : {wins} / {total - wins}",
            f"  Win rate        : {self.win_rate:.1f}%",
            f"  Avg R:R         : {self.avg_rr:.2f}×",
            f"  Profit factor   : {self.profit_factor:.2f}",
            sep2,
            "  RETURNS",
            f"  Starting capital: ₹{TRADING_CAPITAL:,.0f}",
            f"  Final capital   : ₹{TRADING_CAPITAL + self.total_pnl:,.0f}",
            f"  Gross P&L       : ₹{self.total_gross_pnl:+,.0f}",
            f"  Total costs     : ₹{self.total_costs:,.0f}",
            f"    Slippage      : ₹{self.total_slippage:,.0f}",
            f"    Brokerage     : ₹{self.total_brokerage:,.0f}",
            f"    STT           : ₹{self.total_stt:,.0f}",
            f"    Theta decay   : ₹{self.total_theta:,.0f}",
            f"  Net P&L         : ₹{self.total_pnl:+,.0f}",
            f"  Net return      : {self.total_return_pct:.1f}%",
            sep2,
            "  RISK METRICS",
            f"  Max drawdown    : {dd['max_dd_pct']:.1f}% (₹{dd['max_dd_amt']:,.0f})",
            f"  Max loss streak : {dd['max_loss_streak']} trades",
            f"  Sharpe ratio    : {dd['sharpe']:.2f}",
            sep1,
        ])


def compute_drawdown_stats(trades: List[BacktestTrade], starting_capital: float) -> dict:
    equity = starting_capital
    peak   = starting_capital
    max_dd_pct = max_dd_amt = 0.0
    streak = max_loss = max_win = 0
    last_r: Optional[str] = None
    pnls   = []

    for t in trades:
        equity += t.pnl
        pnls.append(t.pnl)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct, max_dd_amt = dd, peak - equity
        r = "W" if t.pnl > 0 else "L"
        streak = streak + 1 if r == last_r else 1
        last_r = r
        if r == "L": max_loss = max(max_loss, streak)
        else:         max_win  = max(max_win,  streak)

    sharpe = (
        np.mean(pnls) / np.std(pnls) * np.sqrt(60)
        if len(pnls) > 1 and np.std(pnls) > 0 else 0.0
    )
    return {
        "max_dd_pct":      round(max_dd_pct, 1),
        "max_dd_amt":      round(max_dd_amt, 0),
        "max_loss_streak": max_loss,
        "max_win_streak":  max_win,
        "recovery_trades": 0,
        "sharpe":          round(sharpe, 2),
        "final_capital":   round(equity, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs the Momentum-Confirmed strategy on historical NSE index data.
    Signal logic mirrors the live engine; premiums are simulated.
    """

    def __init__(self, kite, index: Optional[str] = None) -> None:
        self.kite              = kite
        self.index             = index or ACTIVE_INDEX
        self.conviction_engine = ConvictionEngine()
        cfg                    = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size         = cfg["lot_size"]
        self._adx_threshold    = cfg.get("adx_threshold", 20.0)
        self._backtest_iv_rank = get_backtest_iv_rank(BACKTEST_VIX)
        self._max_lots         = _MAX_LOTS.get(self.index, 5)
        self._signal_rows: list[dict] = []
        self._df_full: Optional[pd.DataFrame] = None  # set in run(); used for multi-day indicator warmup

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str]   = None,
    ) -> BacktestResults:
        from data.feed import DataFeed
        feed  = DataFeed(self.kite)
        start = from_date or BACKTEST_START
        end   = to_date   or BACKTEST_END

        logger.info("Backtest: %s | %s → %s | VIX=%.1f", self.index, start, end, BACKTEST_VIX)

        df_full = feed.get_historical_candles(start, end, index=self.index)
        self._df_full = df_full   # store for multi-day indicator warmup inside _simulate_day
        if df_full is None or df_full.empty:
            logger.error("No historical data for %s.", self.index)
            return BacktestResults(index=self.index, start_date=start, end_date=end)

        results = BacktestResults(index=self.index, start_date=start, end_date=end)
        capital = TRADING_CAPITAL
        results.equity_curve.append(capital)

        df_full["_date"] = df_full["timestamp"].dt.date

        # ── Pre-compute daily summaries ───────────────────────────────────────
        daily_ohlc: dict = {}
        first_candle_vol: dict = {}
        for day, day_df in df_full.groupby("_date"):
            daily_ohlc[day] = {
                "high":  float(day_df["high"].max()),
                "low":   float(day_df["low"].min()),
                "open":  float(day_df["open"].iloc[0]),
                "close": float(day_df["close"].iloc[-1]),
            }
            first_candle_vol[day] = float(day_df["volume"].iloc[0]) if len(day_df) > 0 else 0.0

        sorted_days = sorted(daily_ohlc.keys())

        # Rolling 5-day weekly high/low
        weekly_levels: dict = {}
        for i, day in enumerate(sorted_days):
            past_5 = sorted_days[max(0, i - 4): i + 1]
            weekly_levels[day] = {
                "high": max(daily_ohlc[d]["high"] for d in past_5),
                "low":  min(daily_ohlc[d]["low"]  for d in past_5),
            }

        # Rolling 20-day average first-candle volume
        avg_vol_20d: dict = {}
        for i, day in enumerate(sorted_days):
            past_20 = sorted_days[max(0, i - 19): i + 1]
            vols = [first_candle_vol.get(d, 0) for d in past_20]
            avg_vol_20d[day] = sum(vols) / len(vols) if vols else 0.0

        # ── Day loop ──────────────────────────────────────────────────────────
        for day, day_df in df_full.groupby("_date"):
            day_df    = day_df.reset_index(drop=True)
            day_idx   = sorted_days.index(day)
            prev_ohlc = daily_ohlc[sorted_days[day_idx - 1]] if day_idx > 0 else None
            weekly    = weekly_levels.get(day, {"high": 0.0, "low": 0.0})
            avg_vol   = avg_vol_20d.get(day, 0.0)

            capital = self._simulate_day(
                day, day_df, capital, results,
                prev_ohlc, weekly, avg_vol,
            )
            results.equity_curve.append(capital)

        csv_path = self._write_signal_csv(start, end)
        logger.info(
            "Backtest done: %d trades | Net PnL=₹%.2f | Signal CSV: %s",
            results.total_trades, results.total_pnl, csv_path,
        )
        results.signal_csv = csv_path
        return results

    def _write_signal_csv(self, start: str, end: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"signals_{self.index}_{start}_{end}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._signal_rows)
        return path

    # ── Per-day simulation ────────────────────────────────────────────────────

    def _simulate_day(
        self,
        day,
        day_df: pd.DataFrame,
        capital: float,
        results: BacktestResults,
        prev_candle: Optional[dict],
        weekly_candle: Optional[dict],
        avg_first_vol: float,
    ) -> float:
        # ── 1. Day Quality Filter ─────────────────────────────────────────────
        day_qual = self._day_quality(day_df, prev_candle, avg_first_vol)
        if not day_qual.tradeable:
            logger.debug("[%s] SKIP day (quality %d/4)", day, day_qual.score)
            return capital
        if REQUIRE_PREMIUM_DAY and day_qual.label != "PREMIUM":
            logger.debug("[%s] SKIP day (REQUIRE_PREMIUM_DAY, label=%s)", day, day_qual.label)
            return capital

        daily_pnl         = 0.0
        daily_loss_limit  = TRADING_CAPITAL * DAILY_LOSS_LIMIT_PCT
        daily_trade_count = 0
        open_positions: list[dict] = []
        directions_taken: set      = set()
        profit_lock_done           = False

        expiry_date = get_next_expiry(self.index, day, days_buffer=0)
        dte         = max((expiry_date - day).days, 1)

        # Opening range (first 3 candles = 9:15-10:00) for ORB detection
        orb_slice = day_df.iloc[:min(3, len(day_df))]
        orb_high  = float(orb_slice["high"].max())
        orb_low   = float(orb_slice["low"].min())

        # Rolling avg of intra-day candle volumes (for momentum confirmation)
        all_volumes: list[float] = []

        # Retest entry: signal waits for a pullback candle before committing
        pending_signal: Optional[dict] = None

        for i in range(2, len(day_df)):
            candle      = day_df.iloc[i]
            candle_time = candle["timestamp"].time()
            candle_close = float(candle["close"])

            all_volumes.append(float(candle["volume"]))

            # ── Hard close at TRADE_END ────────────────────────────────────
            if candle_time >= _TRADE_END_TIME:
                open_positions, pnl_closed = self._check_exits(
                    open_positions, candle, results, day, force_close=True,
                )
                daily_pnl += pnl_closed
                break

            # ── Check exits on open positions ──────────────────────────────
            open_positions, pnl_closed = self._check_exits(
                open_positions, candle, results, day,
            )
            daily_pnl += pnl_closed

            # ── 2 PM profit lock ───────────────────────────────────────────
            if candle_time >= _PROFIT_LOCK_TIME and not profit_lock_done and open_positions:
                profit_lock_done = True
                still_open       = []
                for pos in open_positions:
                    self._step_premium(pos, candle_close)
                    cur_px = pos["current_premium"]
                    entry  = pos["entry_premium"]
                    if cur_px > entry:   # only close if in profit
                        qty_rem = pos["scale_out"].lots_remaining * self._lot_size
                        costs   = self._compute_net_pnl(entry, cur_px, qty_rem, pos["total_theta"])
                        pnl     = pos["realized_pnl"] + costs["net_pnl"]
                        pnl_pct = pnl / (entry * (pos["initial_lots"] * self._lot_size)) * 100 if entry > 0 else 0.0
                        daily_pnl += pnl
                        self._update_signal_row(pos, cur_px, pnl, pnl_pct, "Profit lock (14:00)")
                        results.trades.append(self._make_trade(
                            day, pos, cur_px, pnl, pnl_pct,
                            str(candle_time), "Profit lock (14:00)",
                            costs, pos["realized_costs"],
                        ))
                    else:
                        still_open.append(pos)
                open_positions = still_open

            if daily_pnl <= -daily_loss_limit:
                break

            # ── Retest entry: resolve pending signal ───────────────────────
            if pending_signal is not None and (
                daily_trade_count >= MAX_DAILY_TRADES
                or pending_signal["direction"] in directions_taken
            ):
                pending_signal = None   # no longer eligible — discard

            if (pending_signal is not None
                    and len(open_positions) < MAX_OPEN_POSITIONS):
                decision = self._check_retest_entry(pending_signal, candle)
                if decision == "ENTER":
                    ps = pending_signal
                    pending_signal = None
                    entry_premium_r = self._simulate_entry_premium(
                        float(candle["open"]), ps["dte"]
                    )
                    if MIN_PREMIUM <= entry_premium_r <= MAX_PREMIUM:
                        adx_val_r   = ps["adx_val"]
                        risk_budget_r = capital * RISK_PER_TRADE_PCT
                        sl_per_lot_r  = entry_premium_r * LOT_SIZING_SL_PCT * self._lot_size
                        base_lots_r   = max(1, int(risk_budget_r / sl_per_lot_r)) if sl_per_lot_r > 0 else 1
                        if adx_val_r > 30:   adx_mult_r = 1.0
                        elif adx_val_r > 25: adx_mult_r = 0.75
                        else:                adx_mult_r = 0.5
                        lots_r    = max(1, min(round(base_lots_r * adx_mult_r), MAX_LOTS_CAP))
                        quantity_r = lots_r * self._lot_size
                        signal_row_r = ps["signal_row"].copy()
                        signal_row_r["entry_premium"] = entry_premium_r
                        signal_row_r["stop_loss"]     = round(entry_premium_r * (1 + _HARD_SL_PCT), 2)
                        signal_row_r["target"]        = round(entry_premium_r * 1.60, 2)
                        signal_row_r["lot_size"]      = lots_r
                        signal_row_r["capital_used"]  = round(entry_premium_r * lots_r * self._lot_size, 2)
                        signal_row_r["risk_amount"]   = round(entry_premium_r * abs(_HARD_SL_PCT) * lots_r * self._lot_size, 2)
                        self._signal_rows.append(signal_row_r)
                        scale_out_r = ScaleOutPosition(
                            total_lots=lots_r,
                            entry_premium=entry_premium_r,
                            direction=ps["direction"],
                        )
                        open_positions.append({
                            "signal_id":      signal_row_r["signal_id"],
                            "signal_row":     signal_row_r,
                            "direction":      ps["direction"],
                            "entry_spot":     float(candle["open"]),
                            "entry_premium":  entry_premium_r,
                            "entry_time":     str(candle["timestamp"].time()),
                            "quantity":       quantity_r,
                            "initial_lots":   lots_r,
                            "conditions":     ps["conditions"],
                            "day_quality":    day_qual.label,
                            "current_premium": entry_premium_r,
                            "peak_premium":   entry_premium_r,
                            "prev_spot":      float(candle["open"]),
                            "total_theta":    0.0,
                            "candles_held":   0,
                            "scale_out":      scale_out_r,
                            "realized_pnl":   0.0,
                            "realized_costs": {
                                "gross_pnl": 0.0, "slippage": 0.0,
                                "brokerage": 0.0, "stt": 0.0, "theta_cost": 0.0,
                            },
                        })
                        daily_trade_count += 1
                        directions_taken.add(ps["direction"])
                        logger.info(
                            "[%s] %s RETEST ENTRY | score=%d lots=%d | prem=₹%.2f",
                            day, ps["direction"], ps["conditions"], lots_r, entry_premium_r,
                        )
                elif decision == "CANCEL":
                    logger.debug("[%s] %s RETEST CANCELLED", day, pending_signal["direction"])
                    pending_signal = None
                # else "WAIT" — carry pending_signal forward

            if candle_time < _SIGNAL_START_TIME:
                continue
            if candle_time > _SIGNAL_END_TIME:
                continue   # no new entries after signal window closes
            if i + 1 >= len(day_df):
                continue
            if day_df.iloc[i + 1]["timestamp"].time() >= _TRADE_END_TIME:
                continue

            # ── 2. Conviction evaluation ───────────────────────────────────
            # Use a multi-day rolling window (last 100 candles) so ADX/EMA/RSI
            # are properly warmed up even for early-morning entries.
            current_ts = candle["timestamp"]
            if self._df_full is not None:
                window_df = self._df_full[
                    self._df_full["timestamp"] <= current_ts
                ].iloc[-100:].copy()
            else:
                window_df = day_df.iloc[:i + 1].copy()
            pdh = float(prev_candle["high"])  if prev_candle else None
            pdl = float(prev_candle["low"])   if prev_candle else None
            wh  = float(weekly_candle["high"]) if weekly_candle else None
            wl  = float(weekly_candle["low"])  if weekly_candle else None

            conviction = self.conviction_engine.evaluate(
                window_df, index=self.index,
                pcr=1.0, iv_rank=self._backtest_iv_rank,
                pdh=pdh, pdl=pdl, weekly_high=wh, weekly_low=wl,
            )
            if conviction.direction not in ("CALL", "PUT"):
                continue

            # ── Capacity / duplicate checks ────────────────────────────────
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                continue
            if daily_trade_count >= MAX_DAILY_TRADES:
                continue
            if conviction.direction in directions_taken:
                continue

            # ── 3. Momentum Confirmation ───────────────────────────────────
            avg_vol = sum(all_volumes) / len(all_volumes) if all_volumes else 0.0
            vwap_val = conviction.momentum.vwap_val   # pre-computed by ConvictionEngine
            if not self._momentum_confirmed(candle, window_df, conviction.direction, avg_vol, vwap_val):
                continue

            # ── Park signal as pending — retest entry confirms on next candle ──
            if pending_signal is not None:
                continue   # already waiting for a retest

            mom        = conviction.momentum
            running_cap = round(capital + daily_pnl, 2)
            signal_id  = uuid.uuid4().hex[:8].upper()

            # Build signal row with placeholder entry values (updated on actual entry)
            pending_entry_premium = self._simulate_entry_premium(
                float(candle["close"]), dte
            )
            pending_signal = {
                "signal_id":  signal_id,
                "direction":  conviction.direction,
                "conditions": conviction.total_score,
                "adx_val":    mom.adx_val,
                "dte":        dte,
                "candles_waited": 0,
                "signal_row": {
                    "signal_id":       signal_id,
                    "date":            str(day),
                    "time":            str(candle["timestamp"].time()),
                    "index":           self.index,
                    "direction":       conviction.direction,
                    "direction_int":   1 if conviction.direction == "CALL" else 0,
                    "conviction_score": conviction.total_score,
                    "lot_size":        1,          # updated on entry
                    "day_quality":     day_qual.label,
                    "close":           round(mom.close, 2),
                    "vwap":            round(mom.vwap_val, 2),
                    "vwap_distance":   round(mom.close - mom.vwap_val, 2),
                    "rsi":             round(mom.rsi_val, 2),
                    "adx":             round(mom.adx_val, 2),
                    "ema_fast":        round(mom.ema_fast, 2),
                    "ema_slow":        round(mom.ema_slow, 2),
                    "pcr":             conviction.options.pcr,
                    "iv_rank":         conviction.options.iv_rank,
                    "pdh":             round(pdh, 2) if pdh else "",
                    "pdl":             round(pdl, 2) if pdl else "",
                    "weekly_high":     round(wh, 2) if wh else "",
                    "weekly_low":      round(wl, 2) if wl else "",
                    "dte":             dte,
                    "capital_before":  running_cap,
                    "capital_used":    0.0,        # updated on entry
                    "risk_amount":     0.0,        # updated on entry
                    "entry_premium":   pending_entry_premium,
                    "stop_loss":       round(pending_entry_premium * (1 + _HARD_SL_PCT), 2),
                    "target":          round(pending_entry_premium * 1.60, 2),
                    "exit_premium":    "", "pnl": "", "pnl_pct": "",
                    "exit_reason":     "", "trade_taken": 1, "outcome": -1,
                },
            }
            logger.debug(
                "[%s] %s PENDING (retest) | score=%d adx=%.1f",
                day, conviction.direction, conviction.total_score, mom.adx_val,
            )

        capital += daily_pnl
        return capital

    # ── Retest entry decision ─────────────────────────────────────────────────

    def _check_retest_entry(self, pending: dict, candle) -> str:
        """
        Returns "ENTER", "WAIT", or "CANCEL".
        CALL: wait for a red pullback candle, then enter on the next green.
              If candle 1 is green → enter immediately (no pullback needed).
        PUT:  mirror — wait for green bounce, then enter on next red.
        Cancel after 3 candles without entry.
        """
        is_green = float(candle["close"]) >= float(candle["open"])
        is_red   = not is_green
        pending["candles_waited"] = pending.get("candles_waited", 0) + 1
        direction = pending["direction"]

        if pending["candles_waited"] > 3:
            return "CANCEL"

        if direction == "CALL":
            if not pending.get("saw_pullback"):
                if is_red:
                    pending["saw_pullback"] = True
                    return "WAIT"       # pullback seen — wait for green
                else:
                    return "ENTER"      # straight green — enter now
            else:
                if is_green:
                    return "ENTER"      # green after pullback — enter
                else:
                    return "CANCEL"     # second red — momentum failed
        else:  # PUT
            if not pending.get("saw_pullback"):
                if is_green:
                    pending["saw_pullback"] = True
                    return "WAIT"       # bounce seen — wait for red
                else:
                    return "ENTER"      # straight red — enter now
            else:
                if is_red:
                    return "ENTER"      # red after bounce — enter
                else:
                    return "CANCEL"     # second green — momentum failed

    # ── Exit management (scale-out aware) ────────────────────────────────────

    def _check_exits(
        self,
        positions: list[dict],
        candle,
        results: BacktestResults,
        day,
        force_close: bool = False,
    ) -> Tuple[list[dict], float]:
        remaining   = []
        total_pnl   = 0.0
        candle_time = str(candle["timestamp"].time())
        candle_close = float(candle["close"])
        is_red = float(candle["close"]) < float(candle["open"])

        for pos in positions:
            self._step_premium(pos, candle_close)
            pos["candles_held"] += 1
            current_px = pos["current_premium"]

            if force_close:
                # Hard close: exit ALL remaining lots at market
                qty_rem = pos["scale_out"].lots_remaining * self._lot_size
                costs   = self._compute_net_pnl(
                    pos["entry_premium"], current_px, qty_rem, pos["total_theta"],
                )
                self._merge_costs(pos["realized_costs"], costs)
                pnl     = pos["realized_pnl"] + costs["net_pnl"]
                pnl_pct = pnl / (pos["entry_premium"] * pos["initial_lots"] * self._lot_size) * 100
                total_pnl += pnl
                self._update_signal_row(pos, current_px, pnl, pnl_pct, "Hard close (15:00)")
                results.trades.append(self._make_trade(
                    day, pos, current_px, pnl, pnl_pct,
                    candle_time, "Hard close (15:00)",
                    costs, pos["realized_costs"],
                ))
                continue   # don't keep position

            decision = pos["scale_out"].check_exit(current_px, is_red)

            if not decision.should_exit:
                remaining.append(pos)
                continue

            # Use SL price override when premium gapped through stop
            exit_px    = decision.exit_price_override if decision.exit_price_override > 0 else current_px
            qty_closed = decision.lots_to_close * self._lot_size
            costs      = self._compute_net_pnl(
                pos["entry_premium"], exit_px, qty_closed, 0.0,
            )
            # Allocate theta proportionally to closed lots
            if pos["initial_lots"] > 0:
                theta_share = pos["total_theta"] * decision.lots_to_close / pos["initial_lots"]
            else:
                theta_share = 0.0
            costs["theta_cost"] = round(theta_share * self._lot_size, 2)
            costs["net_pnl"]   -= costs["theta_cost"]
            self._merge_costs(pos["realized_costs"], costs)
            pos["realized_pnl"] += costs["net_pnl"]

            if decision.exit_all:
                # Final exit — record the BacktestTrade
                pnl     = pos["realized_pnl"]
                pnl_pct = pnl / (pos["entry_premium"] * pos["initial_lots"] * self._lot_size) * 100
                total_pnl += pnl
                self._update_signal_row(pos, exit_px, pnl, pnl_pct, decision.exit_reason)
                results.trades.append(self._make_trade(
                    day, pos, exit_px, pnl, pnl_pct,
                    candle_time, decision.exit_reason,
                    costs, pos["realized_costs"],
                ))
            else:
                # Partial exit — position continues with fewer lots
                pos["quantity"]    = pos["scale_out"].lots_remaining * self._lot_size
                pos["current_premium"] = exit_px   # update current to actual exit price
                remaining.append(pos)

        return remaining, total_pnl

    # ── Day quality (backtest simulation) ────────────────────────────────────

    def _day_quality(
        self,
        day_df: pd.DataFrame,
        prev_candle: Optional[dict],
        avg_first_vol: float,
    ) -> DayQuality:
        if len(day_df) < 1:
            from signals.day_quality import DayQuality as DQ
            return DQ(label="SKIP", score=0, gap_ok=False, range_ok=False,
                      vix_ok=False, volume_ok=False, lot_multiplier=0.0)

        first    = day_df.iloc[0]
        open_px  = float(first["open"])
        high_px  = float(first["high"])
        low_px   = float(first["low"])
        vol      = float(first["volume"])
        prev_cls = float(prev_candle["close"]) if prev_candle else open_px

        return assess_day_quality(
            prev_close=prev_cls,
            open_price=open_px,
            first_high=high_px,
            first_low=low_px,
            first_volume=vol,
            avg_volume_20d=avg_first_vol,
            vix=BACKTEST_VIX,
        )

    # ── Momentum Confirmation (backtest simulation) ───────────────────────────

    def _momentum_confirmed(
        self,
        candle,
        window_df: pd.DataFrame,
        direction: str,
        avg_vol: float,
        vwap_val: float,
    ) -> bool:
        """
        Check momentum confirmation on the current 15-min candle:
          1. Green/red candle in signal direction
          2. Body > 0.15% of spot
          3. Volume > 1.3× rolling avg  (skipped if volume data = 0)
          4. Close on correct side of VWAP (computed by ConvictionEngine)
          5. Spot moved in signal direction vs 2 candles ago
        """
        open_  = float(candle["open"])
        close_ = float(candle["close"])
        vol    = float(candle["volume"])
        spot   = close_

        if direction == "CALL":
            green     = close_ > open_
            body_pct  = (close_ - open_) / spot if spot > 0 else 0.0
        else:
            green     = close_ < open_
            body_pct  = (open_ - close_) / spot if spot > 0 else 0.0

        body_ok = body_pct >= _MC_BODY_MIN_PCT
        vol_ok  = avg_vol <= 0 or vol <= 0 or vol >= avg_vol * _MC_VOLUME_MULT

        # VWAP: use value already computed by ConvictionEngine (handles 0-volume fallback)
        if direction == "CALL":
            vwap_ok = close_ > vwap_val
        else:
            vwap_ok = close_ < vwap_val

        # Spot momentum: moved in signal direction vs 2 candles ago
        if len(window_df) >= _MC_PREMIUM_TICK_LOOKBACK + 1:
            past_close  = float(window_df["close"].iloc[-_MC_PREMIUM_TICK_LOOKBACK - 1])
            prem_rising = (direction == "CALL" and close_ > past_close) or \
                          (direction == "PUT"  and close_ < past_close)
        else:
            prem_rising = True

        confirmed = green and body_ok and vol_ok and vwap_ok and prem_rising

        if not confirmed:
            logger.debug(
                "MomConfirm FAIL[%s] green=%s body=%.3f%% vol_ok=%s vwap_ok=%s prem=%s",
                direction, green, body_pct * 100, vol_ok, vwap_ok, prem_rising,
            )
        return confirmed

    # ── Premium simulation ────────────────────────────────────────────────────

    @staticmethod
    def _step_premium(pos: dict, candle_close: float) -> None:
        spot_move = candle_close - pos["prev_spot"]
        if pos["direction"] == "PUT":
            spot_move = -spot_move
        delta_impact            = spot_move * _ATM_DELTA
        theta_impact            = -(pos["current_premium"] * _THETA_PER_CANDLE_PCT)
        pos["total_theta"]     += abs(theta_impact)
        pos["current_premium"]  = max(
            pos["current_premium"] + delta_impact + theta_impact, 1.0
        )
        if pos["current_premium"] > pos.get("peak_premium", pos["current_premium"]):
            pos["peak_premium"] = pos["current_premium"]
        pos["prev_spot"] = candle_close

    @staticmethod
    def _simulate_entry_premium(spot: float, dte_days: int) -> float:
        sigma   = BACKTEST_VIX / 100.0
        t       = max(dte_days, 0.5) / 365.0
        premium = 0.4 * spot * sigma * math.sqrt(t)
        return max(round(premium, 2), MIN_PREMIUM)

    @staticmethod
    def _compute_net_pnl(
        entry_premium: float,
        exit_premium_raw: float,
        quantity: int,
        theta_cost_pts: float = 0.0,
    ) -> dict:
        entry_cost = entry_premium + _SLIPPAGE_PTS
        exit_value = max(exit_premium_raw - _SLIPPAGE_PTS, 0.1)
        gross_pnl  = (exit_value - entry_cost) * quantity
        stt        = exit_value * quantity * _STT_PCT
        net_pnl    = gross_pnl - _BROKERAGE - stt
        slippage   = _SLIPPAGE_PTS * 2 * quantity
        return {
            "gross_pnl":  round(gross_pnl, 2),
            "net_pnl":    round(net_pnl, 2),
            "slippage":   round(slippage, 2),
            "brokerage":  _BROKERAGE,
            "stt":        round(stt, 2),
            "theta_cost": round(theta_cost_pts * quantity, 2),
        }

    @staticmethod
    def _merge_costs(acc: dict, new: dict) -> None:
        for k in ("gross_pnl", "slippage", "brokerage", "stt", "theta_cost"):
            acc[k] = round(acc.get(k, 0.0) + new.get(k, 0.0), 2)

    # ── Signal CSV helper ─────────────────────────────────────────────────────

    @staticmethod
    def _update_signal_row(
        pos: dict,
        exit_premium: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        row = pos.get("signal_row")
        if row is None:
            return
        row["exit_premium"] = round(exit_premium, 2)
        row["pnl"]          = round(pnl, 2)
        row["pnl_pct"]      = round(pnl_pct, 2)
        row["exit_reason"]  = exit_reason
        row["outcome"]      = 1 if pnl > 0 else 0

    # ── Trade record builder ──────────────────────────────────────────────────

    def _make_trade(
        self,
        day,
        pos: dict,
        exit_premium: float,
        pnl: float,
        pnl_pct: float,
        exit_time: str,
        exit_reason: str,
        last_costs: dict,
        accum_costs: dict,
    ) -> BacktestTrade:
        return BacktestTrade(
            date=str(day),
            index=self.index,
            direction=pos["direction"],
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            entry_spot=pos["entry_spot"],
            exit_spot=round(pos["entry_spot"] + (exit_premium - pos["entry_premium"]) / _ATM_DELTA *
                            (1 if pos["direction"] == "CALL" else -1), 2),
            entry_premium=pos["entry_premium"],
            exit_premium=round(exit_premium, 2),
            quantity=pos["initial_lots"] * self._lot_size,
            gross_pnl=round(accum_costs.get("gross_pnl", pnl), 2),
            slippage_cost=round(accum_costs.get("slippage", 0.0), 2),
            brokerage_cost=round(accum_costs.get("brokerage", _BROKERAGE), 2),
            stt_cost=round(accum_costs.get("stt", 0.0), 2),
            theta_cost=round(pos["total_theta"] * pos["initial_lots"] * self._lot_size, 2),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
            conviction_score=pos["conditions"],
            day_quality=pos.get("day_quality", "NORMAL"),
        )
