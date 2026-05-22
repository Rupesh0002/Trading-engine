"""
Simplified backtest engine — 2 setups only.

Setup A — Gap & Go       : gap-day momentum, pullback entry, fast exit
Setup B — 11AM Trend     : confirmed trend, timed entry, 60-min max hold

Design principles:
  - ADX, RSI, EMA pre-warmed from 5 prior trading days
  - SL based on candle high/low (not % of premium)
  - TP = N × SL distance converted to premium via ATM delta
  - Max 1 trade per day; Gap&Go has priority
  - Full cost model: slippage, brokerage, STT, theta
  - Per-setup win rate reported separately

Usage:
  python run_backtest.py --simple --from 2024-01-01 --to 2026-04-30 --index NIFTY
"""
from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date
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
    INDEX_CONFIG,
    MAX_PREMIUM,
    MIN_PREMIUM,
    REPORTS_DIR,
    TRADING_CAPITAL,
)
from signals.indicators import compute_adx, compute_ema, compute_rsi_wilder, compute_vwap
from signals.setup_detector import (
    GapSetup, TrendSetup,
    GAP_EMA_PERIOD, MAX_HOLD_GAP, MAX_HOLD_TREND,
    TP_MULT_GAP, TP_MULT_TREND,
    check_gap_setup, check_11am_trend,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_ATM_DELTA            = 0.5
_SLIPPAGE_PTS         = 2
_BROKERAGE            = 40.0
_STT_PCT              = 0.0003
_THETA_PER_CANDLE_PCT = 0.012

# Lot sizing
_LOTS_GAP_BEST   = 3
_LOTS_GAP_NORMAL = 2
_LOTS_TREND      = 2

_WARMUP_DAYS = 5   # prior trading days used to pre-warm indicators

SIMPLE_CSV_COLUMNS = [
    "date", "index", "setup_type", "direction",
    "gap_pct", "vix", "adx_val", "rsi_val",
    "entry_time", "exit_time",
    "entry_spot", "exit_spot",
    "entry_premium", "exit_premium",
    "sl_pts", "tp_pts", "lots", "quantity",
    "candles_held", "gross_pnl", "theta_cost",
    "slippage_cost", "brokerage_cost", "stt_cost",
    "pnl", "pnl_pct", "exit_reason",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimpleBacktestTrade:
    date:           str
    index:          str
    setup_type:     str       # "GAP_GO" | "TREND_11AM"
    direction:      str
    gap_pct:        float
    vix:            float
    adx_val:        float
    rsi_val:        float
    entry_time:     str
    exit_time:      str
    entry_spot:     float
    exit_spot:      float
    entry_premium:  float
    exit_premium:   float
    sl_pts:         float     # premium-point distance to SL
    tp_pts:         float     # premium-point distance to TP
    lots:           int
    quantity:       int
    candles_held:   int
    gross_pnl:      float
    theta_cost:     float
    slippage_cost:  float
    brokerage_cost: float
    stt_cost:       float
    pnl:            float
    pnl_pct:        float
    exit_reason:    str


@dataclass
class SimpleBacktestResults:
    trades:       List[SimpleBacktestTrade] = field(default_factory=list)
    equity_curve: List[float]               = field(default_factory=list)
    index:        str  = ACTIVE_INDEX
    csv_path:     str  = ""
    start_date:   str  = ""
    end_date:     str  = ""

    # ── Per-setup stats ──────────────────────────────────────────────────────

    def _setup(self, setup_type: str) -> List[SimpleBacktestTrade]:
        return [t for t in self.trades if t.setup_type == setup_type]

    def _win_rate(self, ts: List[SimpleBacktestTrade]) -> float:
        if not ts: return 0.0
        return sum(1 for t in ts if t.pnl > 0) / len(ts) * 100

    def _avg_rr(self, ts: List[SimpleBacktestTrade]) -> float:
        wins  = [t.pnl for t in ts if t.pnl > 0]
        losses= [t.pnl for t in ts if t.pnl <= 0]
        aw = sum(wins)  / len(wins)   if wins   else 0.0
        al = sum(losses)/ len(losses) if losses else 0.0
        return abs(aw / al) if al != 0 else 0.0

    def _net_pnl(self, ts: List[SimpleBacktestTrade]) -> float:
        return sum(t.pnl for t in ts)

    @property
    def total_trades(self) -> int:  return len(self.trades)
    @property
    def total_pnl(self) -> float:   return sum(t.pnl for t in self.trades)
    @property
    def total_return_pct(self) -> float: return self.total_pnl / TRADING_CAPITAL * 100
    @property
    def win_rate(self) -> float:    return self._win_rate(self.trades)
    @property
    def avg_rr(self) -> float:      return self._avg_rr(self.trades)
    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve: return 0.0
        arr  = np.array(self.equity_curve)
        peak = np.maximum.accumulate(arr)
        dd   = (peak - arr) / peak
        return float(dd.max()) * 100

    def summary(self) -> str:
        gap_t   = self._setup("GAP_GO")
        trnd_t  = self._setup("TREND_11AM")
        monthly = self.total_trades / max((
            _months_between(self.start_date, self.end_date)), 1)

        sep1 = "  ══════════════════════════════════════════"
        sep2 = "  ──────────────────────────────────────────"
        lines = [
            sep1,
            "  SIMPLIFIED BACKTEST — Gap&Go + 11AM Trend",
            sep1,
            f"  Period         : {self.start_date} → {self.end_date}",
            f"  Index          : {self.index}",
            sep2,
            "  OVERALL",
            f"  Total trades   : {self.total_trades} ({monthly:.1f}/month)",
            f"  Win rate       : {self.win_rate:.1f}%",
            f"  Avg R:R        : {self.avg_rr:.2f}×",
            f"  Net return     : {self.total_return_pct:+.1f}%  (₹{self.total_pnl:+,.0f})",
            f"  Max drawdown   : {self.max_drawdown_pct:.1f}%",
            sep2,
            "  GAP & GO",
            f"  Trades         : {len(gap_t)}",
            f"  Win rate       : {self._win_rate(gap_t):.1f}%",
            f"  Avg R:R        : {self._avg_rr(gap_t):.2f}×",
            f"  Net P&L        : ₹{self._net_pnl(gap_t):+,.0f}",
            sep2,
            "  11 AM TREND",
            f"  Trades         : {len(trnd_t)}",
            f"  Win rate       : {self._win_rate(trnd_t):.1f}%",
            f"  Avg R:R        : {self._avg_rr(trnd_t):.2f}×",
            f"  Net P&L        : ₹{self._net_pnl(trnd_t):+,.0f}",
            sep1,
        ]
        return "\n".join(lines)


def _months_between(start: str, end: str) -> float:
    try:
        from datetime import datetime
        d1 = datetime.strptime(start[:10], "%Y-%m-%d")
        d2 = datetime.strptime(end[:10],   "%Y-%m-%d")
        return max((d2 - d1).days / 30.44, 1.0)
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class SimplifiedEngine:
    """
    Simplified backtest engine — 2 setups, pre-warmed indicators, 1 trade/day.
    """

    def __init__(self, kite, index: Optional[str] = None) -> None:
        self.kite     = kite
        self.index    = index or ACTIVE_INDEX
        cfg           = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_sz  = cfg["lot_size"]
        self._csv_rows: list[dict] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str]   = None,
    ) -> SimpleBacktestResults:
        from data.feed import DataFeed
        feed  = DataFeed(self.kite)
        start = from_date or BACKTEST_START
        end   = to_date   or BACKTEST_END

        logger.info("Simple backtest: %s | %s → %s", self.index, start, end)

        df_full = feed.get_historical_candles(start, end, index=self.index)
        if df_full is None or df_full.empty:
            logger.error("No data for %s", self.index)
            return SimpleBacktestResults(index=self.index, start_date=start, end_date=end)

        results = SimpleBacktestResults(index=self.index, start_date=start, end_date=end)
        capital = TRADING_CAPITAL
        results.equity_curve.append(capital)

        df_full["_date"] = df_full["timestamp"].dt.date
        sorted_days      = sorted(df_full["_date"].unique())
        daily_close      = {}
        for day, ddf in df_full.groupby("_date"):
            daily_close[day] = float(ddf["close"].iloc[-1])

        # Main day loop
        for i, day in enumerate(sorted_days):
            day_df   = df_full[df_full["_date"] == day].reset_index(drop=True)
            prev_days = sorted_days[max(0, i - _WARMUP_DAYS): i]
            prev_close = daily_close[prev_days[-1]] if prev_days else float(day_df["open"].iloc[0])

            # Build combined warmup + today df for indicator pre-warming
            warmup_df  = df_full[df_full["_date"].isin(prev_days)].reset_index(drop=True)
            combined   = pd.concat([warmup_df, day_df], ignore_index=True)

            trade = self._simulate_day(day, day_df, combined, prev_close)

            if trade:
                results.trades.append(trade)
                capital += trade.pnl
                self._csv_rows.append(_trade_to_row(trade))

            results.equity_curve.append(capital)

        csv_path = self._write_csv(start, end)
        results.csv_path = csv_path
        logger.info(
            "Simple backtest done: %d trades | net=₹%.2f",
            results.total_trades, results.total_pnl,
        )
        return results

    # ── Per-day simulation ────────────────────────────────────────────────────

    def _simulate_day(
        self,
        day,
        day_df: pd.DataFrame,
        combined: pd.DataFrame,
        prev_close: float,
    ) -> Optional[SimpleBacktestTrade]:
        """Returns at most one trade for the day."""
        if len(day_df) < 2:
            return None

        n_today = len(day_df)

        # ── Pre-warm indicators on combined data ──────────────────────────────
        ema3_full  = compute_ema(combined["close"], GAP_EMA_PERIOD)
        adx_full   = compute_adx(combined)
        rsi_full   = compute_rsi_wilder(combined["close"])
        vwap_full  = compute_vwap(combined)

        ema3_today  = ema3_full.iloc[-n_today:].reset_index(drop=True)
        adx_today   = adx_full.iloc[-n_today:].reset_index(drop=True)
        rsi_today   = rsi_full.iloc[-n_today:].reset_index(drop=True)
        vwap_today  = vwap_full.iloc[-n_today:].reset_index(drop=True)

        expiry = get_next_expiry(self.index, day, days_buffer=0)
        dte    = max((expiry - day).days, 1)

        # ── Setup A: Gap & Go ─────────────────────────────────────────────────
        gap_setup = check_gap_setup(day_df, prev_close, BACKTEST_VIX, ema3_today)

        if gap_setup:
            idx = gap_setup.entry_candle_idx
            adx_at = float(adx_today.iloc[idx]) if idx < len(adx_today) and not pd.isna(adx_today.iloc[idx]) else 0.0
            rsi_at = float(rsi_today.iloc[idx]) if idx < len(rsi_today) and not pd.isna(rsi_today.iloc[idx]) else 50.0

            lots     = _LOTS_GAP_BEST if gap_setup.lot_band == "BEST" else _LOTS_GAP_NORMAL
            quantity = lots * self._lot_sz

            entry_spot    = gap_setup.entry_spot
            entry_premium = _sim_premium(entry_spot, dte)

            if MIN_PREMIUM <= entry_premium <= MAX_PREMIUM:
                sl_pts, tp_pts = _compute_sl_tp(
                    entry_spot, entry_premium, gap_setup.sl_spot_level,
                    gap_setup.direction, TP_MULT_GAP,
                )
                sl_price = entry_premium - sl_pts
                tp_price = entry_premium + tp_pts

                return self._run_trade(
                    day=day, day_df=day_df,
                    entry_idx=idx, entry_spot=entry_spot,
                    entry_premium=entry_premium, sl_price=sl_price, tp_price=tp_price,
                    max_hold=MAX_HOLD_GAP + 1,
                    lots=lots, quantity=quantity,
                    setup_type="GAP_GO", direction=gap_setup.direction,
                    gap_pct=gap_setup.gap_pct, adx_val=adx_at, rsi_val=rsi_at,
                    sl_pts=sl_pts, tp_pts=tp_pts,
                )

        # ── Setup B: 11 AM Trend — DISABLED (21.7% win rate in backtest) ──────
        # Kept in code for future re-evaluation with live data.
        # To re-enable: remove the `if False:` guard below.
        if False:
            check_idx = 6
            if len(adx_today) > check_idx and len(rsi_today) > check_idx:
                adx_at = float(adx_today.iloc[check_idx]) if not pd.isna(adx_today.iloc[check_idx]) else 0.0
                rsi_at = float(rsi_today.iloc[check_idx]) if not pd.isna(rsi_today.iloc[check_idx]) else 50.0
            else:
                return None

            trend_setup = check_11am_trend(day_df, adx_at, rsi_at, vwap_today, BACKTEST_VIX)

            if trend_setup:
                lots     = _LOTS_TREND
                quantity = lots * self._lot_sz

                entry_spot    = trend_setup.entry_spot
                entry_premium = _sim_premium(entry_spot, dte)

            if MIN_PREMIUM <= entry_premium <= MAX_PREMIUM:
                sl_pts, tp_pts = _compute_sl_tp(
                    entry_spot, entry_premium, trend_setup.sl_spot_level,
                    trend_setup.direction, TP_MULT_TREND,
                )
                sl_price = entry_premium - sl_pts
                tp_price = entry_premium + tp_pts

                return self._run_trade(
                    day=day, day_df=day_df,
                    entry_idx=trend_setup.entry_candle_idx, entry_spot=entry_spot,
                    entry_premium=entry_premium, sl_price=sl_price, tp_price=tp_price,
                    max_hold=MAX_HOLD_TREND, lots=lots, quantity=quantity,
                    setup_type="TREND_11AM", direction=trend_setup.direction,
                    gap_pct=0.0, adx_val=trend_setup.adx_val, rsi_val=trend_setup.rsi_val,
                    sl_pts=sl_pts, tp_pts=tp_pts,
                )

        return None

    # ── Candle-by-candle trade simulation ────────────────────────────────────

    def _run_trade(
        self,
        day,
        day_df: pd.DataFrame,
        entry_idx: int,
        entry_spot: float,
        entry_premium: float,
        sl_price: float,
        tp_price: float,
        max_hold: int,
        lots: int,
        quantity: int,
        setup_type: str,
        direction: str,
        gap_pct: float,
        adx_val: float,
        rsi_val: float,
        sl_pts: float,
        tp_pts: float,
    ) -> SimpleBacktestTrade:
        """Simulate candle-by-candle until SL / TP / max-hold."""
        entry_time = str(day_df.iloc[entry_idx]["timestamp"].time())
        current_px = entry_premium
        prev_spot  = entry_spot
        total_theta = 0.0
        candles_held = 0
        exit_reason  = ""
        exit_px      = current_px
        exit_spot    = entry_spot

        for k in range(entry_idx + 1, len(day_df)):
            candle_close = float(day_df.iloc[k]["close"])
            candles_held += 1

            # Step premium (delta + theta)
            spot_move = candle_close - prev_spot
            if direction == "PUT":
                spot_move = -spot_move
            delta_impact  = spot_move * _ATM_DELTA
            theta_impact  = -(current_px * _THETA_PER_CANDLE_PCT)
            total_theta  += abs(theta_impact)
            current_px    = max(current_px + delta_impact + theta_impact, 0.5)
            prev_spot     = candle_close

            # Check SL
            if current_px <= sl_price:
                exit_px     = sl_price   # exit at SL level, not gap-through
                exit_spot   = candle_close
                exit_reason = f"SL (-{sl_pts:.1f}pts)"
                break

            # Check TP
            if current_px >= tp_price:
                exit_px     = tp_price
                exit_spot   = candle_close
                exit_reason = f"TP (+{tp_pts:.1f}pts)"
                break

            # Max hold
            if candles_held >= max_hold:
                exit_px     = current_px
                exit_spot   = candle_close
                exit_reason = f"MaxHold ({candles_held}c)"
                break
        else:
            # End of day
            exit_px     = current_px
            exit_spot   = float(day_df.iloc[-1]["close"])
            exit_reason = "EOD"

        exit_time = str(day_df.iloc[min(entry_idx + candles_held, len(day_df) - 1)]["timestamp"].time())

        costs    = _compute_costs(entry_premium, exit_px, quantity, total_theta)
        pnl      = costs["net_pnl"]
        pnl_pct  = pnl / (entry_premium * quantity) * 100 if entry_premium * quantity > 0 else 0.0

        return SimpleBacktestTrade(
            date=str(day),
            index=self.index,
            setup_type=setup_type,
            direction=direction,
            gap_pct=round(gap_pct, 2),
            vix=BACKTEST_VIX,
            adx_val=round(adx_val, 1),
            rsi_val=round(rsi_val, 1),
            entry_time=entry_time,
            exit_time=exit_time,
            entry_spot=round(entry_spot, 2),
            exit_spot=round(exit_spot, 2),
            entry_premium=round(entry_premium, 2),
            exit_premium=round(exit_px, 2),
            sl_pts=round(sl_pts, 2),
            tp_pts=round(tp_pts, 2),
            lots=lots,
            quantity=quantity,
            candles_held=candles_held,
            gross_pnl=round(costs["gross_pnl"], 2),
            theta_cost=round(total_theta * quantity, 2),
            slippage_cost=round(costs["slippage"], 2),
            brokerage_cost=round(costs["brokerage"], 2),
            stt_cost=round(costs["stt"], 2),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
        )

    # ── CSV writer ────────────────────────────────────────────────────────────

    def _write_csv(self, start: str, end: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"simple_{self.index}_{start}_{end}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SIMPLE_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._csv_rows)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim_premium(spot: float, dte_days: int) -> float:
    sigma   = BACKTEST_VIX / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), MIN_PREMIUM)


def _compute_sl_tp(
    entry_spot: float,
    entry_premium: float,
    sl_spot_level: float,
    direction: str,
    tp_multiplier: float,
) -> Tuple[float, float]:
    """
    Returns (sl_pts, tp_pts) in premium terms.
    sl_pts: premium-point distance to SL
    tp_pts: tp_multiplier × sl_pts
    Minimum sl_pts capped at 2 pts to avoid degenerate trades.
    """
    if direction == "CALL":
        spot_sl_dist = max(entry_spot - sl_spot_level, 0.0)
    else:
        spot_sl_dist = max(sl_spot_level - entry_spot, 0.0)

    sl_pts = max(spot_sl_dist * _ATM_DELTA, 2.0)
    tp_pts = sl_pts * tp_multiplier
    return round(sl_pts, 2), round(tp_pts, 2)


def _compute_costs(
    entry_premium: float,
    exit_premium: float,
    quantity: int,
    theta_pts: float = 0.0,
) -> dict:
    entry_cost = entry_premium + _SLIPPAGE_PTS
    exit_value = max(exit_premium - _SLIPPAGE_PTS, 0.1)
    gross_pnl  = (exit_value - entry_cost) * quantity
    stt        = exit_value * quantity * _STT_PCT
    net_pnl    = gross_pnl - _BROKERAGE - stt
    slippage   = _SLIPPAGE_PTS * 2 * quantity
    return {
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl":   round(net_pnl, 2),
        "slippage":  round(slippage, 2),
        "brokerage": _BROKERAGE,
        "stt":       round(stt, 2),
    }


def _trade_to_row(t: SimpleBacktestTrade) -> dict:
    return {c: getattr(t, c, "") for c in SIMPLE_CSV_COLUMNS}
