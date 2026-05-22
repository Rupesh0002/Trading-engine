"""
Backtest Engine V2 — Gap & Go setup + dynamic ExitManager on 5-min candles.

Key differences from simple_engine.py:
  - Exit managed by ExitManager (trailing stop, no-move, partial spike)
  - Exit simulation runs on 5-min candles (3× finer granularity)
  - No fixed TP/SL targets — trade stays open as long as momentum continues
  - Partial exits tracked separately for full P&L accounting

Usage:
  python run_backtest.py --v2 --index BANKNIFTY --from 2024-01-01 --to 2026-04-30
"""
from __future__ import annotations

import csv
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config.events_calendar import get_next_expiry
from config.settings import (
    ACTIVE_INDEX,
    BACKTEST_END,
    BACKTEST_START,
    BACKTEST_VIX,
    INDEX_CONFIG,
    MAX_PREMIUM,
    MIN_PREMIUM,
    REPORTS_DIR,
    TRADING_CAPITAL,
)
from engine_v2.exit_manager import ExitManager, ExitSignal
from signals.indicators import compute_adx, compute_ema, compute_rsi_wilder, compute_vwap
from signals.setup_detector import (
    GAP_EMA_PERIOD,
    check_gap_setup,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Simulation constants ──────────────────────────────────────────────────────
_ATM_DELTA             = 0.5
_SLIPPAGE_PTS          = 2
_BROKERAGE             = 40.0
_STT_PCT               = 0.0003
_THETA_PER_15MIN_PCT   = 0.012          # 1.2% per 15-min candle
_THETA_PER_5MIN_PCT    = _THETA_PER_15MIN_PCT / 3  # 0.4% per 5-min candle

# Lot sizing (same as simple_engine)
_LOTS_GAP_BEST   = 3
_LOTS_GAP_NORMAL = 2

_WARMUP_DAYS = 5    # prior trading days for indicator pre-warming

# 5-min entry detection
# EMA period on 5-min: 9-period = 45-min lookback (same as 3-period on 15-min)
_5MIN_EMA_PERIOD    = 9
_ENTRY_WINDOW_START = time(9, 30)    # earliest 5-min entry candle
_ENTRY_WINDOW_END   = time(10, 30)   # latest 5-min entry candle (10:30 close)

V2_CSV_COLUMNS = [
    "date", "index", "direction", "gap_pct", "vix", "adx_val", "rsi_val",
    "entry_time", "final_exit_time",
    "entry_spot", "entry_premium",
    "lots", "quantity",
    "candles_held_5min",
    # partial exit fields
    "partial_lots", "partial_premium", "partial_pnl",
    # final exit fields
    "final_lots", "final_premium", "final_exit_reason",
    # totals
    "gross_pnl", "theta_cost", "slippage_cost", "brokerage_cost", "stt_cost",
    "net_pnl", "net_pnl_pct",
    "highest_premium", "trailing_sl_at_exit",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V2Trade:
    date:             str
    index:            str
    direction:        str
    gap_pct:          float
    vix:              float
    adx_val:          float
    rsi_val:          float
    entry_time:       str
    final_exit_time:  str
    entry_spot:       float
    entry_premium:    float
    lots:             int
    quantity:         int
    candles_held_5min: int
    # partial exit (may be 0 if no partial fired)
    partial_lots:     int
    partial_premium:  float
    partial_pnl:      float
    # final exit
    final_lots:       int
    final_premium:    float
    final_exit_reason: str
    # totals
    gross_pnl:        float
    theta_cost:       float
    slippage_cost:    float
    brokerage_cost:   float
    stt_cost:         float
    net_pnl:          float
    net_pnl_pct:      float
    # trail metadata
    highest_premium:  float
    trailing_sl_at_exit: float


@dataclass
class V2Results:
    trades:       List[V2Trade] = field(default_factory=list)
    equity_curve: List[float]   = field(default_factory=list)
    index:        str = ACTIVE_INDEX
    csv_path:     str = ""
    start_date:   str = ""
    end_date:     str = ""

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / TRADING_CAPITAL * 100

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades) * 100

    @property
    def avg_rr(self) -> float:
        wins   = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl <= 0]
        aw = sum(wins)   / len(wins)   if wins   else 0.0
        al = sum(losses) / len(losses) if losses else 0.0
        return abs(aw / al) if al != 0 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        arr  = np.array(self.equity_curve)
        peak = np.maximum.accumulate(arr)
        dd   = (peak - arr) / peak
        return float(dd.max()) * 100

    def exit_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for t in self.trades:
            counts[t.final_exit_reason] += 1
            if t.partial_lots > 0:
                counts["PARTIAL_SPIKE"] += 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        monthly = self.total_trades / max(_months_between(self.start_date, self.end_date), 1)
        bd = self.exit_breakdown()
        partial_count = sum(1 for t in self.trades if t.partial_lots > 0)

        sep1 = "  ══════════════════════════════════════════════"
        sep2 = "  ──────────────────────────────────────────────"
        lines = [
            sep1,
            "  BACKTEST V2 — Gap&Go + Dynamic Trailing Exit",
            sep1,
            f"  Period       : {self.start_date} → {self.end_date}",
            f"  Index        : {self.index}",
            sep2,
            f"  Total trades : {self.total_trades} ({monthly:.1f}/month)",
            f"  Win rate     : {self.win_rate:.1f}%",
            f"  Avg R:R      : {self.avg_rr:.2f}×",
            f"  Net return   : {self.total_return_pct:+.1f}%  (₹{self.total_pnl:+,.0f})",
            f"  Max drawdown : {self.max_drawdown_pct:.1f}%",
            sep2,
            "  EXIT TYPE BREAKDOWN",
        ]
        total = self.total_trades or 1
        for reason, count in bd.items():
            if reason == "PARTIAL_SPIKE":
                lines.append(f"  {'PARTIAL_SPIKE':<18}: {partial_count:>3} trades (bonus partial exits)")
            else:
                pct = count / total * 100
                wins = sum(1 for t in self.trades
                           if t.final_exit_reason == reason and t.net_pnl > 0)
                lines.append(f"  {reason:<18}: {count:>3} trades  {pct:4.0f}%  win={wins}/{count}")
        lines.append(sep1)
        return "\n".join(lines)


def _months_between(start: str, end: str) -> float:
    try:
        d1 = datetime.strptime(start[:10], "%Y-%m-%d")
        d2 = datetime.strptime(end[:10],   "%Y-%m-%d")
        return max((d2 - d1).days / 30.44, 1.0)
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngineV2:
    """
    Gap & Go setup detection on 15-min candles.
    Exit management via ExitManager on 5-min candles.
    """

    def __init__(self, kite, index: Optional[str] = None) -> None:
        self.kite    = kite
        self.index   = index or ACTIVE_INDEX
        cfg          = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_sz = cfg["lot_size"]
        self._csv_rows: List[dict] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str]   = None,
    ) -> V2Results:
        from data.feed import DataFeed
        feed  = DataFeed(self.kite)
        start = from_date or BACKTEST_START
        end   = to_date   or BACKTEST_END

        logger.info("V2 backtest: %s | %s → %s", self.index, start, end)

        # Fetch both resolutions
        df_15 = feed.get_historical_candles(start, end, index=self.index, interval="15minute")
        df_5  = feed.get_historical_candles(start, end, index=self.index, interval="5minute")

        if df_15 is None or df_15.empty:
            logger.error("No 15-min data for %s", self.index)
            return V2Results(index=self.index, start_date=start, end_date=end)
        if df_5 is None or df_5.empty:
            logger.warning("No 5-min data — falling back to 15-min for exit simulation")
            df_5 = None

        results = V2Results(index=self.index, start_date=start, end_date=end)
        capital = TRADING_CAPITAL
        results.equity_curve.append(capital)

        df_15["_date"] = df_15["timestamp"].dt.date
        sorted_days    = sorted(df_15["_date"].unique())
        daily_close    = {day: float(ddf["close"].iloc[-1])
                          for day, ddf in df_15.groupby("_date")}

        if df_5 is not None:
            df_5["_date"] = df_5["timestamp"].dt.date

        for i, day in enumerate(sorted_days):
            day_df_15 = df_15[df_15["_date"] == day].reset_index(drop=True)
            day_df_5  = (df_5[df_5["_date"] == day].reset_index(drop=True)
                         if df_5 is not None else None)

            prev_days  = sorted_days[max(0, i - _WARMUP_DAYS): i]
            prev_close = (daily_close[prev_days[-1]] if prev_days
                          else float(day_df_15["open"].iloc[0]))

            warmup_15  = df_15[df_15["_date"].isin(prev_days)].reset_index(drop=True)
            combined   = pd.concat([warmup_15, day_df_15], ignore_index=True)

            # Build pre-warmed 5-min combined series for EMA computation
            if df_5 is not None and day_df_5 is not None:
                warmup_5   = df_5[df_5["_date"].isin(prev_days)].reset_index(drop=True)
                combined_5 = pd.concat([warmup_5, day_df_5], ignore_index=True)
            else:
                combined_5 = None

            trade = self._simulate_day(day, day_df_15, day_df_5, combined, combined_5, prev_close)

            if trade:
                results.trades.append(trade)
                capital += trade.net_pnl
                self._csv_rows.append(_trade_to_row(trade))

            results.equity_curve.append(capital)

        csv_path = self._write_csv(start, end)
        results.csv_path = csv_path
        logger.info("V2 backtest done: %d trades | net=₹%.2f",
                    results.total_trades, results.total_pnl)
        return results

    # ── Per-day simulation ────────────────────────────────────────────────────

    def _simulate_day(
        self,
        day,
        day_df_15: pd.DataFrame,
        day_df_5: Optional[pd.DataFrame],
        combined_15: pd.DataFrame,
        combined_5: Optional[pd.DataFrame],
        prev_close: float,
    ) -> Optional[V2Trade]:
        if len(day_df_15) < 2:
            return None

        n_today_15 = len(day_df_15)

        # ── Step 1: Gap qualification on 15-min ──────────────────────────────
        # Pre-warm 15-min indicators (ADX, RSI for reporting; EMA3 for gap check)
        ema3_full = compute_ema(combined_15["close"], GAP_EMA_PERIOD)
        adx_full  = compute_adx(combined_15)
        rsi_full  = compute_rsi_wilder(combined_15["close"])

        ema3_today = ema3_full.iloc[-n_today_15:].reset_index(drop=True)
        adx_today  = adx_full.iloc[-n_today_15:].reset_index(drop=True)
        rsi_today  = rsi_full.iloc[-n_today_15:].reset_index(drop=True)

        expiry = get_next_expiry(self.index, day, days_buffer=0)
        dte    = max((expiry - day).days, 1)

        # check_gap_setup validates: gap%, VIX, first-candle direction, EMA touch.
        # We use its direction/gap_pct/lot_band; entry fields are replaced below.
        gap_setup = check_gap_setup(day_df_15, prev_close, BACKTEST_VIX, ema3_today)
        if not gap_setup:
            return None

        # ADX/RSI at time of gap (candle 0 of today = 9:15)
        adx_at = float(adx_today.iloc[0]) if not pd.isna(adx_today.iloc[0]) else 0.0
        rsi_at = float(rsi_today.iloc[0]) if not pd.isna(rsi_today.iloc[0]) else 50.0

        lots     = _LOTS_GAP_BEST if gap_setup.lot_band == "BEST" else _LOTS_GAP_NORMAL
        quantity = lots * self._lot_sz

        # ── Step 2: 5-min confirmed entry ─────────────────────────────────────
        # Require a 5-min candle whose LOW touches the EMA (pullback happened)
        # AND whose CLOSE is ABOVE the EMA (recovery confirmed).
        # This filters fake pullbacks where price just wicked through and reversed.
        if day_df_5 is None or combined_5 is None or len(day_df_5) < 3:
            return None   # no 5-min data → skip (don't fall back to dirty 15-min entry)

        n_today_5    = len(day_df_5)
        ema9_5_full  = compute_ema(combined_5["close"], _5MIN_EMA_PERIOD)
        ema9_today_5 = ema9_5_full.iloc[-n_today_5:].reset_index(drop=True)

        entry_result = _find_5min_entry(
            day_df_5, ema9_today_5, gap_setup.direction,
            _ENTRY_WINDOW_START, _ENTRY_WINDOW_END,
        )
        if entry_result is None:
            logger.debug("[%s] No 5-min confirmed entry for %s gap",
                         day, gap_setup.direction)
            return None

        entry_5min_idx, entry_spot, sl_spot_level = entry_result
        entry_premium = _sim_premium(entry_spot, dte)

        if not (MIN_PREMIUM <= entry_premium <= MAX_PREMIUM):
            return None

        entry_ts   = day_df_5.iloc[entry_5min_idx]["timestamp"]
        # Exit management starts on the NEXT 5-min candle after entry close
        exit_candles = day_df_5.iloc[entry_5min_idx + 1:].reset_index(drop=True)

        return self._run_trade(
            day=day, entry_ts=entry_ts, entry_spot=entry_spot,
            entry_premium=entry_premium, exit_candles=exit_candles,
            minutes_per_candle=5,
            lots=lots, quantity=quantity,
            direction=gap_setup.direction,
            gap_pct=gap_setup.gap_pct, adx_val=adx_at, rsi_val=rsi_at,
        )

    # ── Candle-by-candle exit simulation ─────────────────────────────────────

    def _run_trade(
        self,
        day,
        entry_ts,
        entry_spot: float,
        entry_premium: float,
        exit_candles: pd.DataFrame,
        minutes_per_candle: int,
        lots: int,
        quantity: int,
        direction: str,
        gap_pct: float,
        adx_val: float,
        rsi_val: float,
    ) -> V2Trade:
        theta_pct = (_THETA_PER_5MIN_PCT if minutes_per_candle == 5
                     else _THETA_PER_15MIN_PCT)

        entry_time = entry_ts.time()
        manager    = ExitManager(entry_premium, lots, entry_time)

        current_px   = entry_premium
        prev_spot    = entry_spot
        total_theta  = 0.0
        candles_held = 0

        # Partial exit bookkeeping
        partial_lots    = 0
        partial_premium = 0.0
        partial_pnl     = 0.0

        # Final exit defaults (end of data)
        final_ts     = entry_ts
        final_px     = entry_premium
        final_reason = "EOD"
        final_lots   = lots

        for _, row in exit_candles.iterrows():
            candle_close = float(row["close"])
            candle_time  = row["timestamp"].time()
            candles_held += 1

            # Simulate premium movement: delta + theta
            spot_move = candle_close - prev_spot
            if direction == "PUT":
                spot_move = -spot_move
            delta_impact = spot_move * _ATM_DELTA
            theta_impact = -(current_px * theta_pct)
            total_theta += abs(theta_impact)
            current_px   = max(current_px + delta_impact + theta_impact, 0.5)
            prev_spot    = candle_close

            signal: Optional[ExitSignal] = manager.check(current_px, candle_time)

            if signal is None:
                continue

            if not signal.exit_all:
                # Partial exit — book P&L, continue with remaining lots
                p_qty     = signal.lots * self._lot_sz
                p_entry   = entry_premium + _SLIPPAGE_PTS
                p_exit    = max(signal.premium - _SLIPPAGE_PTS, 0.1)
                p_gross   = (p_exit - p_entry) * p_qty
                p_stt     = p_exit * p_qty * _STT_PCT
                p_net     = p_gross - p_stt  # brokerage shared at trade end
                partial_lots    += signal.lots
                partial_premium  = signal.premium
                partial_pnl     += p_net
                final_lots       = manager.remaining_lots
                continue

            # Full exit
            final_ts     = row["timestamp"]
            final_px     = signal.premium
            final_reason = signal.reason
            final_lots   = signal.lots
            break
        else:
            # Ran out of candles — EOD exit at last price
            final_px   = current_px
            final_lots = manager.remaining_lots
            if exit_candles is not None and len(exit_candles) > 0:
                final_ts = exit_candles.iloc[-1]["timestamp"]

        # ── P&L calculation ───────────────────────────────────────────────────
        # Remaining lots (after partial) at final_px
        rem_qty     = final_lots * self._lot_sz
        rem_entry   = entry_premium + _SLIPPAGE_PTS
        rem_exit    = max(final_px - _SLIPPAGE_PTS, 0.1)
        rem_gross   = (rem_exit - rem_entry) * rem_qty

        total_qty   = quantity
        total_gross = partial_pnl + rem_gross  # partial already net of partial slippage
        theta_cost  = total_theta * total_qty
        stt         = rem_exit * rem_qty * _STT_PCT
        slippage    = _SLIPPAGE_PTS * 2 * total_qty
        net_pnl     = total_gross - _BROKERAGE - stt - theta_cost
        pnl_pct     = net_pnl / (entry_premium * total_qty) * 100 if entry_premium * total_qty > 0 else 0.0

        return V2Trade(
            date=str(day),
            index=self.index,
            direction=direction,
            gap_pct=round(gap_pct, 2),
            vix=BACKTEST_VIX,
            adx_val=round(adx_val, 1),
            rsi_val=round(rsi_val, 1),
            entry_time=str(entry_time),
            final_exit_time=str(final_ts.time() if hasattr(final_ts, "time") else entry_time),
            entry_spot=round(entry_spot, 2),
            entry_premium=round(entry_premium, 2),
            lots=lots,
            quantity=total_qty,
            candles_held_5min=candles_held,
            partial_lots=partial_lots,
            partial_premium=round(partial_premium, 2),
            partial_pnl=round(partial_pnl, 2),
            final_lots=final_lots,
            final_premium=round(final_px, 2),
            final_exit_reason=final_reason,
            gross_pnl=round(total_gross, 2),
            theta_cost=round(theta_cost, 2),
            slippage_cost=round(slippage, 2),
            brokerage_cost=_BROKERAGE,
            stt_cost=round(stt, 2),
            net_pnl=round(net_pnl, 2),
            net_pnl_pct=round(pnl_pct, 2),
            highest_premium=round(manager.highest, 2),
            trailing_sl_at_exit=round(manager.trailing_sl or 0.0, 2),
        )

    # ── CSV writer ────────────────────────────────────────────────────────────

    def _write_csv(self, start: str, end: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"v2_{self.index}_{start}_{end}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=V2_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._csv_rows)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_5min_entry(
    day_df_5: pd.DataFrame,
    ema9: pd.Series,
    direction: str,
    window_start: time,
    window_end: time,
) -> Optional[Tuple[int, float, float]]:
    """
    Scan 5-min candles for a confirmed EMA pullback entry.

    Confirmation rule:
      CALL: candle LOW  <= EMA  (pullback touched EMA)
            AND candle CLOSE > EMA  (price recovered above EMA)
      PUT:  candle HIGH >= EMA  (pullback touched EMA)
            AND candle CLOSE < EMA  (price recovered below EMA)

    Returns (candle_index, entry_spot, sl_level) or None.
    Entry spot = close of the confirmation candle.
    SL level   = low (CALL) or high (PUT) of the confirmation candle.
    """
    for i in range(len(day_df_5)):
        row = day_df_5.iloc[i]
        candle_time = row["timestamp"].time()

        if candle_time < window_start:
            continue
        if candle_time > window_end:
            break

        if i >= len(ema9):
            break
        ema_val = float(ema9.iloc[i])
        if pd.isna(ema_val) or ema_val <= 0:
            continue

        low_  = float(row["low"])
        high_ = float(row["high"])
        close = float(row["close"])

        if direction == "CALL":
            if low_ <= ema_val and close > ema_val:
                return (i, close, low_)
        else:  # PUT
            if high_ >= ema_val and close < ema_val:
                return (i, close, high_)

    return None


def _sim_premium(spot: float, dte_days: int) -> float:
    sigma   = BACKTEST_VIX / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), MIN_PREMIUM)


def _trade_to_row(t: V2Trade) -> dict:
    return {c: getattr(t, c, "") for c in V2_CSV_COLUMNS}
