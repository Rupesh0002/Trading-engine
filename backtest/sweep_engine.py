"""
Liquidity Sweep Backtest Engine.

Signal: a 5-min candle whose wick pierces through a key S/R level (PDH/PDL,
weekly H/L, swing pivot, OR_H/OR_L, round number) and whose body closes back
inside that level — indicating stop-hunt / liquidity grab by smart money.

Entry    = sweep candle close (immediate, no extra confirmation needed)
SL       = 0.03% beyond the sweep extreme  (very tight structural SL)
TP       = nearest structural level in the reversal direction
Option   = ATM strike simulated via BS approximation; delta = 0.50

Levels refreshed each day:
  Prior S/R  — from detect_levels() on all 15-min candles before today
  OR_H/OR_L  — high/low of today's opening 9:15 candle
"""
from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, time
from typing import List, Optional

import pandas as pd
import pytz

from config.events_calendar import get_next_expiry
from config.settings import (
    BACKTEST_VIX,
    INDEX_CONFIG,
    MAX_LOTS_CAP,
    MAX_PREMIUM,
    MIN_PREMIUM,
    REPORTS_DIR,
    RISK_PER_TRADE_PCT,
    TRADING_CAPITAL,
)
from signals.levels import SRLevel, detect_levels, _merge
from signals.liquidity_sweep import LiquiditySweepEngine, SweepSignal
from signals.mtf import MTFEngine

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_ATM_DELTA       = 0.50
_THETA_5M        = 0.004    # 0.4% per 5-min candle
_OPTION_BACKSTOP = 0.35     # force-close if premium drops 35%
_NO_RUN_CANDLES  = 6        # exit if stalled after 6 candles
_NO_RUN_BAND     = 0.08     # "stalled" = within ±8% of entry premium
_SLIPPAGE_PTS    = 2.0
_BROKERAGE       = 40.0
_STT_PCT         = 0.0003
_HARD_CLOSE_TIME = time(15, 0)


@dataclass
class SweepTrade:
    date:          str
    index:         str
    direction:     str
    sweep_time:    str
    level_price:   float
    level_kind:    str
    wick_pts:      float
    rr:            float
    adx:           float
    trend:         str
    entry_spot:    float
    sl_spot:       float
    tp_spot:       float
    entry_premium: float
    exit_premium:  float
    lots:          int
    lot_size:      int
    quantity:      int
    gross_pnl:     float
    net_pnl:       float
    pnl_pct:       float
    exit_reason:   str
    dte:           int


@dataclass
class SweepBacktestResult:
    trades:   List[SweepTrade] = field(default_factory=list)
    csv_path: str              = ""

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades * 100 if self.total_trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def avg_winner(self) -> float:
        w = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loser(self) -> float:
        l = [t.net_pnl for t in self.trades if t.net_pnl <= 0]
        return sum(l) / len(l) if l else 0.0

    @property
    def avg_rr(self) -> float:
        return abs(self.avg_winner / self.avg_loser) if self.avg_loser != 0 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        cap, peak, dd = TRADING_CAPITAL, TRADING_CAPITAL, 0.0
        for t in self.trades:
            cap  += t.net_pnl
            peak  = max(peak, cap)
            dd    = max(dd, (peak - cap) / peak * 100)
        return dd

    def summary(self) -> str:
        ret      = self.total_pnl / TRADING_CAPITAL * 100
        call_t   = [t for t in self.trades if t.direction == "CALL"]
        put_t    = [t for t in self.trades if t.direction == "PUT"]
        lines = [
            "─" * 68,
            "  LIQUIDITY SWEEP BACKTEST",
            "  15-min context (S/R) + 5-min stop-hunt detection",
            "─" * 68,
            f"  Trades       : {self.total_trades}  "
            f"(CALL={len(call_t)}  PUT={len(put_t)})",
            f"  Win rate     : {self.win_rate:.1f}%  "
            f"({self.winners}W / {self.total_trades - self.winners}L)",
            f"  Avg winner   : ₹{self.avg_winner:+,.0f}",
            f"  Avg loser    : ₹{self.avg_loser:+,.0f}",
            f"  Avg R:R      : {self.avg_rr:.2f}×",
            f"  Total P&L    : ₹{self.total_pnl:+,.0f}  ({ret:+.1f}%)",
            f"  Max drawdown : {self.max_drawdown_pct:.1f}%",
            f"  CSV          : {self.csv_path}",
            "─" * 68,
        ]
        if call_t:
            cwr = sum(1 for t in call_t if t.net_pnl > 0) / len(call_t) * 100
            lines.append(f"  CALL sweeps  : {len(call_t)} trades | win {cwr:.1f}% | "
                         f"net ₹{sum(t.net_pnl for t in call_t):+,.0f}")
        if put_t:
            pwr = sum(1 for t in put_t if t.net_pnl > 0) / len(put_t) * 100
            lines.append(f"  PUT  sweeps  : {len(put_t)} trades | win {pwr:.1f}% | "
                         f"net ₹{sum(t.net_pnl for t in put_t):+,.0f}")
        lines.append("─" * 68)
        return "\n".join(lines)


class SweepBacktestEngine:

    def __init__(self, kite, index: str = "NIFTY") -> None:
        self.kite      = kite
        self.index     = index.upper()
        cfg            = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size = cfg["lot_size"]
        self._sweep    = LiquiditySweepEngine()
        self._mtf      = MTFEngine()   # reuse context (ADX/trend)

    def run(self, from_date: str, to_date: str) -> SweepBacktestResult:
        from datetime import datetime, timedelta
        from data.feed import DataFeed

        feed = DataFeed(self.kite)

        from_dt     = IST.localize(datetime.strptime(from_date, "%Y-%m-%d"))
        warmup_from = (from_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        from_date_d = from_dt.date()

        logger.info("[SWEEP-BT] 15-min %s→%s (warmup %s) [%s]",
                    from_date, to_date, warmup_from, self.index)
        df15 = feed.get_historical_candles(
            warmup_from, to_date, index=self.index, interval="15minute"
        )

        logger.info("[SWEEP-BT] 5-min  %s→%s [%s]", from_date, to_date, self.index)
        df5 = feed.get_historical_candles(
            from_date, to_date, index=self.index, interval="5minute"
        )

        if df15 is None or df15.empty:
            logger.error("[SWEEP-BT] No 15-min data.")
            return SweepBacktestResult()
        if df5 is None or df5.empty:
            logger.error("[SWEEP-BT] No 5-min data.")
            return SweepBacktestResult()

        for df in (df15, df5):
            if "date" in df.columns and "timestamp" not in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_days     = sorted(df15["timestamp"].dt.date.unique())
        trading_days = [d for d in all_days if d >= from_date_d]
        logger.info("[SWEEP-BT] %d trading days | 15m=%d rows | 5m=%d rows",
                    len(trading_days), len(df15), len(df5))

        result  = SweepBacktestResult()
        capital = TRADING_CAPITAL

        for day in trading_days:
            prior_15 = df15[df15["timestamp"].dt.date < day].copy()
            today_15 = (
                df15[df15["timestamp"].dt.date == day]
                .sort_values("timestamp").reset_index(drop=True)
            )
            today_5 = (
                df5[df5["timestamp"].dt.date == day]
                .sort_values("timestamp").reset_index(drop=True)
            )

            expiry = get_next_expiry(self.index, day, days_buffer=0)
            dte    = max((expiry - day).days, 1)

            trade = self._simulate_day(prior_15, today_15, today_5, day, dte, capital)
            if trade is not None:
                result.trades.append(trade)
                capital += trade.net_pnl

        result.csv_path = self._save_csv(result.trades, from_date, to_date)
        return result

    # ── Per-day simulation ────────────────────────────────────────────────────

    def _simulate_day(
        self,
        prior_15: pd.DataFrame,
        today_15: pd.DataFrame,
        today_5:  pd.DataFrame,
        day:      date,
        dte:      int,
        capital:  float,
    ) -> Optional[SweepTrade]:

        if today_5.empty or today_15.empty:
            return None

        # Reference spot for level detection (first 5-min candle close)
        spot = float(today_5.iloc[0]["close"])

        # Build prior S/R levels (no look-ahead — only data before today)
        levels: List[SRLevel] = detect_levels(prior_15, spot, index=self.index)

        # Add today's opening-range levels (first 15-min candle = 9:15 AM)
        or_candle = today_15.iloc[0]
        or_h      = float(or_candle["high"])
        or_l      = float(or_candle["low"])
        or_levels = [SRLevel(or_h, "OR_H", 2), SRLevel(or_l, "OR_L", 2)]

        # Merge OR levels into the existing level list
        all_levels = _merge(levels + or_levels, spot)

        if not all_levels:
            return None

        # ADX and trend from MTF context (reuses prior 15-min candles)
        ctx = self._mtf.analyze_context(prior_15, today_15, index=self.index)
        adx   = ctx.adx   if ctx is not None else 0.0
        trend = ctx.trend  if ctx is not None else "NEUTRAL"

        # Scan 5-min candles for first qualifying liquidity sweep
        signal: Optional[SweepSignal] = self._sweep.scan_5min(
            today_5, all_levels,
            context_adx=adx,
            context_trend=trend,
        )

        if signal is None:
            return None

        entry_premium = _sim_premium(signal.entry_price, dte)
        if not (MIN_PREMIUM <= entry_premium <= MAX_PREMIUM):
            return None

        sl_spot = signal.sl_spot
        tp_spot = signal.tp_spot

        # Lot sizing from risk budget using spot-derived option SL delta
        sl_option_delta = abs(signal.entry_price - sl_spot) * _ATM_DELTA
        risk_budget = capital * RISK_PER_TRADE_PCT
        sl_per_lot  = sl_option_delta * self._lot_size
        lots = max(1, min(
            int(risk_budget / sl_per_lot) if sl_per_lot > 0 else 1,
            MAX_LOTS_CAP,
        ))
        qty = lots * self._lot_size

        logger.info(
            "[SWEEP-BT] %s %s %s | lv=%s(%.2f) entry=%.2f prem=₹%.2f "
            "SL=%.2f TP=%.2f R:R=%.2f lots=%d",
            day, self.index, signal.direction,
            signal.level_kind, signal.level_price,
            signal.entry_price, entry_premium,
            sl_spot, tp_spot, signal.rr, lots,
        )

        # ── Monitor 5-min candles after the sweep candle ──────────────────
        current_prem = entry_premium
        prev_spot    = signal.entry_price
        candles_held = 0
        direction    = signal.direction

        for _, c5 in today_5.iterrows():
            t5 = pd.Timestamp(c5["timestamp"]).time()
            if t5 <= signal.sweep_time:
                continue

            close5 = float(c5["close"])

            if t5 >= _HARD_CLOSE_TIME:
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "Hard close", day, dte, adx, trend,
                )

            # Theta decay + delta-based spot move
            current_prem = max(current_prem * (1 - _THETA_5M), 0.1)
            px_d = (close5 - prev_spot) * _ATM_DELTA if direction == "CALL" \
                   else (prev_spot - close5) * _ATM_DELTA
            current_prem = max(current_prem + px_d, 0.1)
            prev_spot    = close5
            candles_held += 1

            # Option backstop — theta drag with no spot progress
            if current_prem <= entry_premium * (1 - _OPTION_BACKSTOP):
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "SL hit (theta)", day, dte, adx, trend,
                )

            # No-run exit — position stalled, avoid further theta bleed
            if (candles_held >= _NO_RUN_CANDLES
                    and abs(current_prem - entry_premium) / entry_premium <= _NO_RUN_BAND):
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "No-run exit", day, dte, adx, trend,
                )

            # Spot-based SL
            if direction == "CALL" and close5 <= sl_spot:
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "SL hit", day, dte, adx, trend,
                )
            if direction == "PUT" and close5 >= sl_spot:
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "SL hit", day, dte, adx, trend,
                )

            # Spot-based TP
            if direction == "CALL" and close5 >= tp_spot:
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "Target hit", day, dte, adx, trend,
                )
            if direction == "PUT" and close5 <= tp_spot:
                return self._close_trade(
                    signal, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "Target hit", day, dte, adx, trend,
                )

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _close_trade(
        self,
        signal:        SweepSignal,
        entry_px:      float,
        sl_spot:       float,
        tp_spot:       float,
        exit_px:       float,
        lots:          int,
        qty:           int,
        reason:        str,
        day:           date,
        dte:           int,
        adx:           float,
        trend:         str,
    ) -> SweepTrade:
        exit_adj  = max(exit_px - _SLIPPAGE_PTS, 0.1)
        entry_adj = entry_px + _SLIPPAGE_PTS
        gross     = (exit_adj - entry_adj) * qty
        stt       = exit_adj * qty * _STT_PCT
        net       = gross - _BROKERAGE - stt
        pnl_pct   = net / (entry_px * qty) * 100 if entry_px * qty > 0 else 0.0

        logger.info(
            "[SWEEP-BT] %s %s/%s EXIT %s | prem ₹%.2f→₹%.2f | net ₹%.2f",
            day, signal.level_kind, signal.direction, reason, entry_px, exit_px, net,
        )
        return SweepTrade(
            date          = str(day),
            index         = self.index,
            direction     = signal.direction,
            sweep_time    = str(signal.sweep_time),
            level_price   = round(signal.level_price, 2),
            level_kind    = signal.level_kind,
            wick_pts      = round(signal.wick_pts, 2),
            rr            = round(signal.rr, 2),
            adx           = round(adx, 1),
            trend         = trend,
            entry_spot    = round(signal.entry_price, 2),
            sl_spot       = round(sl_spot, 2),
            tp_spot       = round(tp_spot, 2),
            entry_premium = round(entry_px, 2),
            exit_premium  = round(exit_px, 2),
            lots          = lots,
            lot_size      = self._lot_size,
            quantity      = qty,
            gross_pnl     = round(gross, 2),
            net_pnl       = round(net, 2),
            pnl_pct       = round(pnl_pct, 2),
            exit_reason   = reason,
            dte           = dte,
        )

    def _save_csv(self, trades: List[SweepTrade], from_date: str, to_date: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(
            REPORTS_DIR,
            f"sweep_{self.index}_{from_date}_{to_date}.csv",
        )
        if not trades:
            return fname
        fields = [f.name for f in SweepTrade.__dataclass_fields__.values()]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({fn: getattr(t, fn) for fn in fields})
        logger.info("[SWEEP-BT] CSV → %s", fname)
        return fname


def _sim_premium(spot: float, dte_days: int) -> float:
    sigma   = BACKTEST_VIX / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), MIN_PREMIUM)
