"""
ORB Backtest Engine — Opening Range Breakout on NSE F&O (15-min candles).

Opening range  = high/low of the 9:15 candle (first 15-min bar of the day).
Breakout rule  = candle CLOSES beyond OR boundary + 0.1% buffer.
Entry window   = 10:00–13:00 IST (first breakout only per index per day).
Exit           = SL -20% | Target +50% | Hard close 15:00.
Premium sim    = same Black-Scholes approx as conviction backtest.
Costs          = slippage 2 pts/leg, brokerage ₹40, STT 0.03% on exit.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import List, Optional

import pandas as pd
import pytz

from config.events_calendar import get_next_expiry
from config.settings import (
    ACTIVE_INDICES,
    BACKTEST_VIX,
    INDEX_CONFIG,
    MAX_LOTS_CAP,
    MIN_PREMIUM,
    MAX_PREMIUM,
    ORB_BUFFER_PCT,
    ORB_MIN_RANGE_PCT,
    ORB_MAX_RANGE_PCT,
    REPORTS_DIR,
    RISK_PER_TRADE_PCT,
    TRADING_CAPITAL,
    LOT_SIZING_SL_PCT,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_SL_PCT          = 0.20    # -20% on entry premium → stop loss
_TARGET_PCT      = 0.50    # +50% on entry premium → target
_THETA_PCT       = 0.012   # theta decay per 15-min candle
_ATM_DELTA       = 0.5
_SLIPPAGE_PTS    = 2.0
_BROKERAGE       = 40.0
_STT_PCT         = 0.0003

_SIGNAL_START    = time(10, 0)
_ORB_TRADE_END   = time(13, 0)
_HARD_CLOSE_TIME = time(15, 0)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class ORBTrade:
    date:            str
    index:           str
    direction:       str
    or_high:         float
    or_low:          float
    or_range_pct:    float
    breakout_time:   str
    breakout_price:  float
    entry_premium:   float
    exit_premium:    float
    lots:            int
    lot_size:        int
    quantity:        int
    gross_pnl:       float
    net_pnl:         float
    pnl_pct:         float
    exit_reason:     str
    dte:             int


@dataclass
class ORBBacktestResult:
    trades:    List[ORBTrade]  = field(default_factory=list)
    csv_path:  str             = ""

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
    def max_drawdown_pct(self) -> float:
        cap = TRADING_CAPITAL
        peak = cap
        dd   = 0.0
        for t in self.trades:
            cap += t.net_pnl
            peak = max(peak, cap)
            dd   = max(dd, (peak - cap) / peak * 100)
        return dd

    def summary(self) -> str:
        total_return = self.total_pnl / TRADING_CAPITAL * 100
        rr = abs(self.avg_winner / self.avg_loser) if self.avg_loser != 0 else 0.0
        lines = [
            "─" * 60,
            "  ORB BACKTEST RESULTS",
            "─" * 60,
            f"  Trades       : {self.total_trades}",
            f"  Win rate     : {self.win_rate:.1f}%  ({self.winners}W / {self.total_trades - self.winners}L)",
            f"  Avg winner   : ₹{self.avg_winner:+,.0f}",
            f"  Avg loser    : ₹{self.avg_loser:+,.0f}",
            f"  Avg R:R      : {rr:.2f}×",
            f"  Total P&L    : ₹{self.total_pnl:+,.0f}  ({total_return:+.1f}%)",
            f"  Max drawdown : {self.max_drawdown_pct:.1f}%",
            f"  CSV          : {self.csv_path}",
            "─" * 60,
        ]
        return "\n".join(lines)


# ── Backtest engine ───────────────────────────────────────────────────────────

class ORBBacktestEngine:

    def __init__(self, kite, index: str = "NIFTY") -> None:
        self.kite      = kite
        self.index     = index.upper()
        cfg            = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size = cfg["lot_size"]

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, from_date: str, to_date: str) -> ORBBacktestResult:
        from data.feed import DataFeed
        feed = DataFeed(self.kite)

        logger.info("[ORB-BT] Fetching candles %s → %s [%s]", from_date, to_date, self.index)
        df_full = feed.get_historical_candles(from_date, to_date, index=self.index)
        if df_full is None or df_full.empty:
            logger.error("[ORB-BT] No candle data returned.")
            return ORBBacktestResult()

        # Normalise timestamp column
        if "date" in df_full.columns and "timestamp" not in df_full.columns:
            df_full = df_full.rename(columns={"date": "timestamp"})
        df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])

        trading_days = sorted(df_full["timestamp"].dt.date.unique())
        logger.info("[ORB-BT] %d trading days found.", len(trading_days))

        result  = ORBBacktestResult()
        capital = TRADING_CAPITAL

        for day in trading_days:
            day_df = df_full[df_full["timestamp"].dt.date == day].copy()
            day_df = day_df.sort_values("timestamp").reset_index(drop=True)

            expiry_date = get_next_expiry(self.index, day, days_buffer=0)
            dte         = max((expiry_date - day).days, 1)

            trade = self._simulate_day(day_df, day, dte, capital)
            if trade is not None:
                result.trades.append(trade)
                capital += trade.net_pnl

        result.csv_path = self._save_csv(result.trades, from_date, to_date)
        return result

    # ── Per-day simulation ────────────────────────────────────────────────

    def _simulate_day(
        self,
        day_df: pd.DataFrame,
        day: date,
        dte:  int,
        capital: float,
    ) -> Optional[ORBTrade]:

        # ── Step 1: extract opening range ─────────────────────────────────
        or_row = None
        for _, row in day_df.iterrows():
            t = row["timestamp"].time()
            if t.hour == 9 and t.minute == 15:
                or_row = row
                break

        if or_row is None:
            return None  # no 9:15 candle (holiday / data gap)

        or_high    = float(or_row["high"])
        or_low     = float(or_row["low"])
        or_range   = or_high - or_low
        or_rng_pct = or_range / or_low if or_low > 0 else 0.0

        if or_rng_pct < ORB_MIN_RANGE_PCT:
            logger.debug("[ORB-BT] %s range %.2f%% too tight — skip.", day, or_rng_pct * 100)
            return None
        if or_rng_pct > ORB_MAX_RANGE_PCT:
            logger.debug("[ORB-BT] %s range %.2f%% gap day — skip.", day, or_rng_pct * 100)
            return None

        buf_up   = or_high * (1 + ORB_BUFFER_PCT)
        buf_down = or_low  * (1 - ORB_BUFFER_PCT)

        # ── Step 2: scan for breakout ──────────────────────────────────────
        open_pos   = None
        traded     = False

        for _, candle in day_df.iterrows():
            t     = candle["timestamp"].time()
            close = float(candle["close"])

            # Hard close: exit any open position at current simulated premium
            if t >= _HARD_CLOSE_TIME:
                if open_pos is not None:
                    return self._close_trade(open_pos, open_pos["current_premium"], "Hard close", day, dte, capital)
                break

            # Monitor open position
            if open_pos is not None:
                open_pos["candles_held"] += 1
                # theta decay
                open_pos["current_premium"] = max(
                    open_pos["current_premium"] * (1 - _THETA_PCT), 0.1
                )
                # Simulate option move: ATM delta × spot change
                spot_prev   = open_pos["prev_spot"]
                if open_pos["direction"] == "CALL":
                    px_delta = (close - spot_prev) * _ATM_DELTA
                else:
                    px_delta = (spot_prev - close) * _ATM_DELTA
                open_pos["current_premium"] = max(open_pos["current_premium"] + px_delta, 0.1)

                ep = open_pos["entry_premium"]
                cp = open_pos["current_premium"]

                if cp <= ep * (1 - _SL_PCT):
                    return self._close_trade(open_pos, cp, "SL hit", day, dte, capital)
                if cp >= ep * (1 + _TARGET_PCT):
                    return self._close_trade(open_pos, cp, "Target hit", day, dte, capital)

                open_pos["prev_spot"] = close
                continue

            # Check breakout entry window
            if traded or t < _SIGNAL_START or t >= _ORB_TRADE_END:
                continue

            direction = None
            if close > buf_up:
                direction = "CALL"
            elif close < buf_down:
                direction = "PUT"

            if direction is None:
                continue

            # ── Entry ──────────────────────────────────────────────────────
            entry_premium = self._sim_premium(close, dte)
            if not (MIN_PREMIUM <= entry_premium <= MAX_PREMIUM):
                continue

            risk_budget = capital * RISK_PER_TRADE_PCT
            sl_per_lot  = entry_premium * _SL_PCT * self._lot_size
            lots        = max(1, min(int(risk_budget / sl_per_lot) if sl_per_lot > 0 else 1, MAX_LOTS_CAP))
            qty         = lots * self._lot_size

            open_pos = {
                "signal_id":        str(uuid.uuid4())[:8],
                "direction":        direction,
                "or_high":          or_high,
                "or_low":           or_low,
                "or_range_pct":     or_rng_pct,
                "breakout_time":    str(t),
                "breakout_price":   close,
                "entry_premium":    entry_premium,
                "current_premium":  entry_premium,
                "prev_spot":        close,
                "lots":             lots,
                "quantity":         qty,
                "candles_held":     0,
            }
            traded = True
            logger.info(
                "[ORB-BT] %s %s %s breakout | OR H=%.2f L=%.2f | prem=₹%.2f lots=%d",
                day, self.index, direction, or_high, or_low, entry_premium, lots,
            )

        return None  # day ended without completing a trade

    # ── Helpers ──────────────────────────────────────────────────────────

    def _close_trade(
        self,
        pos:    dict,
        exit_px: float,
        reason: str,
        day:    date,
        dte:    int,
        capital: float,
    ) -> ORBTrade:
        qty         = pos["quantity"]
        entry       = pos["entry_premium"]
        exit_adj    = max(exit_px - _SLIPPAGE_PTS, 0.1)
        entry_adj   = entry + _SLIPPAGE_PTS
        gross       = (exit_adj - entry_adj) * qty
        stt         = exit_adj * qty * _STT_PCT
        net         = gross - _BROKERAGE - stt
        pnl_pct     = net / (entry * qty) * 100 if entry * qty > 0 else 0.0

        logger.info(
            "[ORB-BT] %s %s EXIT %s | entry=₹%.2f exit=₹%.2f net=₹%.2f",
            day, pos["direction"], reason, entry, exit_px, net,
        )
        return ORBTrade(
            date           = str(day),
            index          = self.index,
            direction      = pos["direction"],
            or_high        = pos["or_high"],
            or_low         = pos["or_low"],
            or_range_pct   = pos["or_range_pct"],
            breakout_time  = pos["breakout_time"],
            breakout_price = pos["breakout_price"],
            entry_premium  = entry,
            exit_premium   = round(exit_px, 2),
            lots           = pos["lots"],
            lot_size       = self._lot_size,
            quantity       = qty,
            gross_pnl      = round(gross, 2),
            net_pnl        = round(net, 2),
            pnl_pct        = round(pnl_pct, 2),
            exit_reason    = reason,
            dte            = dte,
        )

    @staticmethod
    def _sim_premium(spot: float, dte_days: int) -> float:
        sigma   = BACKTEST_VIX / 100.0
        t       = max(dte_days, 0.5) / 365.0
        premium = 0.4 * spot * sigma * math.sqrt(t)
        return max(round(premium, 2), MIN_PREMIUM)

    def _save_csv(self, trades: List[ORBTrade], from_date: str, to_date: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(
            REPORTS_DIR,
            f"orb_{self.index}_{from_date}_{to_date}.csv",
        )
        if not trades:
            return fname
        fields = [f.name for f in ORBTrade.__dataclass_fields__.values()]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({fn: getattr(t, fn) for fn in fields})
        logger.info("[ORB-BT] CSV saved → %s", fname)
        return fname
