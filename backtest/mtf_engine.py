"""
MTF (Multi-Timeframe) Backtest Engine — v2.

Signal types:
  ORB        — 15-min close beyond OR boundary + body-ratio + trend filter.
  EMA_BOUNCE — 15-min pullback to EMA20 then bounces in trend direction.

Exit logic (spot-based, not fixed premium %):
  ORB  SL   = spot crosses back through OR boundary (invalidation level).
  ORB  TP   = entry_spot + OR_range × 1.5  (structural target, CALL)
               entry_spot − OR_range × 1.5  (PUT)
  EMA  SL   = spot crosses 15-min candle low/high that fired the signal.
  EMA  TP   = entry_spot + (entry − SL) × 2  (2:1 R:R in spot)

Option backstop: if option premium falls 35% below entry (theta drag with no
spot progress) the trade is closed regardless of spot SL/TP levels.
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
from signals.mtf import MTFEngine, MTFEntry, WatchSignal

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_ATM_DELTA        = 0.50
_THETA_5M         = 0.004   # 0.4% theta per 5-min candle (= 1.2%/3 from 15-min model)
_OPTION_BACKSTOP  = 0.35    # force-close if premium drops 35% (theta with no spot move)
_NO_RUN_CANDLES   = 6       # after this many 5-min candles with no progress → exit flat
_NO_RUN_BAND      = 0.08    # "no progress" = premium within ±8% of entry
_SLIPPAGE_PTS     = 2.0
_BROKERAGE        = 40.0
_STT_PCT          = 0.0003
_HARD_CLOSE_TIME  = time(15, 0)
_SL_BUFFER        = 0.001   # 0.1% inside OR boundary for ORB SL


@dataclass
class MTFTrade:
    date:          str
    index:         str
    direction:     str
    source:        str      # ORB | EMA_BOUNCE
    or_high:       float
    or_low:        float
    or_range_pct:  float
    watch_time:    str
    entry_time:    str
    entry_mode:    str      # MOMENTUM | RETEST | EMA_BOUNCE
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
class MTFBacktestResult:
    trades:   List[MTFTrade] = field(default_factory=list)
    csv_path: str            = ""

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
        cap, peak, dd = TRADING_CAPITAL, TRADING_CAPITAL, 0.0
        for t in self.trades:
            cap  += t.net_pnl
            peak  = max(peak, cap)
            dd    = max(dd, (peak - cap) / peak * 100)
        return dd

    @property
    def avg_rr(self) -> float:
        return abs(self.avg_winner / self.avg_loser) if self.avg_loser != 0 else 0.0

    def summary(self) -> str:
        ret = self.total_pnl / TRADING_CAPITAL * 100
        orb_t = [t for t in self.trades if t.source == "ORB"]
        ema_t = [t for t in self.trades if t.source == "EMA_BOUNCE"]
        lines = [
            "─" * 68,
            "  MTF BACKTEST v2  (15-min context  +  5-min execution)",
            "  Spot-based SL/TP  |  ORB + EMA-bounce entries",
            "─" * 68,
            f"  Trades       : {self.total_trades}  "
            f"(ORB={len(orb_t)}  EMA-bounce={len(ema_t)})",
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
        if orb_t:
            orb_wr = sum(1 for t in orb_t if t.net_pnl > 0) / len(orb_t) * 100
            lines.append(f"  ORB entries  : {len(orb_t)} trades | win {orb_wr:.1f}% | "
                         f"net ₹{sum(t.net_pnl for t in orb_t):+,.0f}")
        if ema_t:
            ema_wr = sum(1 for t in ema_t if t.net_pnl > 0) / len(ema_t) * 100
            lines.append(f"  EMA entries  : {len(ema_t)} trades | win {ema_wr:.1f}% | "
                         f"net ₹{sum(t.net_pnl for t in ema_t):+,.0f}")
        lines.append("─" * 68)
        return "\n".join(lines)


class MTFBacktestEngine:

    def __init__(self, kite, index: str = "NIFTY") -> None:
        self.kite      = kite
        self.index     = index.upper()
        cfg            = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size = cfg["lot_size"]
        self._mtf      = MTFEngine()

    def run(self, from_date: str, to_date: str) -> MTFBacktestResult:
        from datetime import datetime, timedelta

        from data.feed import DataFeed
        feed = DataFeed(self.kite)

        from_dt     = IST.localize(datetime.strptime(from_date, "%Y-%m-%d"))
        warmup_from = (from_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        from_date_d = from_dt.date()

        logger.info("[MTF-BT] 15-min %s→%s (warmup %s) [%s]",
                    from_date, to_date, warmup_from, self.index)
        df15 = feed.get_historical_candles(warmup_from, to_date, index=self.index, interval="15minute")

        logger.info("[MTF-BT] 5-min  %s→%s [%s]", from_date, to_date, self.index)
        df5  = feed.get_historical_candles(from_date, to_date, index=self.index, interval="5minute")

        if df15 is None or df15.empty:
            logger.error("[MTF-BT] No 15-min data.")
            return MTFBacktestResult()
        if df5 is None or df5.empty:
            logger.error("[MTF-BT] No 5-min data.")
            return MTFBacktestResult()

        for df in (df15, df5):
            if "date" in df.columns and "timestamp" not in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_days     = sorted(df15["timestamp"].dt.date.unique())
        trading_days = [d for d in all_days if d >= from_date_d]
        logger.info("[MTF-BT] %d days | 15m=%d | 5m=%d",
                    len(trading_days), len(df15), len(df5))

        result  = MTFBacktestResult()
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
    ) -> Optional[MTFTrade]:

        ctx = self._mtf.analyze_context(prior_15, today_15, index=self.index)
        if ctx is None:
            return None

        # Pre-compute EMA20 for today's 15-min candles (rolling, no look-ahead)
        # Uses prior data for warmup then extends candle-by-candle for today.
        ema20_by_idx = self._compute_ema20_today(prior_15, today_15)

        watch: Optional[WatchSignal] = None
        prev_candle = None

        for idx, candle in today_15.iterrows():
            # ── ORB signal ────────────────────────────────────────────────
            if ctx.or_valid:
                w = self._mtf.check_watch_signal(candle, ctx)
                if w:
                    watch = w
                    break

            # ── EMA bounce signal ─────────────────────────────────────────
            if prev_candle is not None:
                ema20_curr = ema20_by_idx.get(idx, 0.0)
                ema20_prev = ema20_by_idx.get(idx - 1, 0.0)
                if ema20_curr > 0 and ema20_prev > 0:
                    w = self._mtf.check_ema_bounce_signal(
                        candle, prev_candle, ema20_curr, ema20_prev, ctx
                    )
                    if w:
                        watch = w
                        break

            prev_candle = candle

        if watch is None:
            return None

        entry = self._mtf.find_entry_5min(today_5, watch)
        if entry is None:
            return None

        entry_premium = _sim_premium(entry.entry_price, dte)
        if not (MIN_PREMIUM <= entry_premium <= MAX_PREMIUM):
            return None

        # ── Spot-based SL and TP ──────────────────────────────────────────
        direction = watch.direction
        or_range  = watch.or_high - watch.or_low

        if watch.source == "ORB":
            # SL = back through OR boundary (breakout invalidation)
            if direction == "CALL":
                sl_spot = watch.or_high * (1 - _SL_BUFFER)
            else:
                sl_spot = watch.or_low * (1 + _SL_BUFFER)
            # TP = OR_range × 1.5 from entry
            if direction == "CALL":
                tp_spot = entry.entry_price + or_range * 1.5
            else:
                tp_spot = entry.entry_price - or_range * 1.5

        else:  # EMA_BOUNCE
            # SL = suggested sl from 15-min candle (low for CALL, high for PUT)
            sl_spot = watch.suggested_sl_spot if watch.suggested_sl_spot > 0 else (
                entry.entry_price * 0.998 if direction == "CALL" else entry.entry_price * 1.002
            )
            # TP = 2:1 R:R in spot terms from entry
            sl_dist = abs(entry.entry_price - sl_spot)
            if direction == "CALL":
                tp_spot = entry.entry_price + sl_dist * 2
            else:
                tp_spot = entry.entry_price - sl_dist * 2

        # Ensure SL / TP are on the right side of entry
        if direction == "CALL":
            sl_spot = min(sl_spot, entry.entry_price * 0.998)
            tp_spot = max(tp_spot, entry.entry_price * 1.002)
        else:
            sl_spot = max(sl_spot, entry.entry_price * 1.002)
            tp_spot = min(tp_spot, entry.entry_price * 0.998)

        # Lot sizing from risk budget (uses spot-derived option SL delta)
        sl_option_delta = abs(entry.entry_price - sl_spot) * _ATM_DELTA
        risk_budget = capital * RISK_PER_TRADE_PCT
        sl_per_lot  = sl_option_delta * self._lot_size
        lots = max(1, min(
            int(risk_budget / sl_per_lot) if sl_per_lot > 0 else 1,
            MAX_LOTS_CAP,
        ))
        qty = lots * self._lot_size

        logger.info(
            "[MTF-BT] %s %s %s/%s | watch=%s entry=%s prem=₹%.2f "
            "SL_spot=%.2f TP_spot=%.2f lots=%d",
            day, self.index, watch.source, direction,
            watch.trigger_time, entry.entry_time,
            entry_premium, sl_spot, tp_spot, lots,
        )

        # ── Monitor 5-min candles ─────────────────────────────────────────
        current_prem  = entry_premium
        prev_spot     = entry.entry_price
        candles_held  = 0

        for _, c5 in today_5.iterrows():
            t5 = pd.Timestamp(c5["timestamp"]).time()
            if t5 <= entry.entry_time:
                continue

            close5 = float(c5["close"])

            if t5 >= _HARD_CLOSE_TIME:
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "Hard close", day, dte,
                )

            # Theta decay + delta move
            current_prem = max(current_prem * (1 - _THETA_5M), 0.1)
            px_d = (close5 - prev_spot) * _ATM_DELTA if direction == "CALL" \
                   else (prev_spot - close5) * _ATM_DELTA
            current_prem = max(current_prem + px_d, 0.1)
            prev_spot    = close5
            candles_held += 1

            # Option backstop (theta drag with no spot progress)
            if current_prem <= entry_premium * (1 - _OPTION_BACKSTOP):
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "SL hit (theta)", day, dte,
                )

            # No-run exit: held N candles with premium barely moving → stop wasting theta
            if (candles_held >= _NO_RUN_CANDLES
                    and abs(current_prem - entry_premium) / entry_premium <= _NO_RUN_BAND):
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "No-run exit", day, dte,
                )

            # Spot-based SL
            if direction == "CALL" and close5 <= sl_spot:
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "SL hit", day, dte,
                )
            if direction == "PUT" and close5 >= sl_spot:
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "SL hit", day, dte,
                )

            # Spot-based TP
            if direction == "CALL" and close5 >= tp_spot:
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "Target hit", day, dte,
                )
            if direction == "PUT" and close5 <= tp_spot:
                return self._close_trade(
                    watch, entry, entry_premium, sl_spot, tp_spot,
                    current_prem, lots, qty, "Target hit", day, dte,
                )

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _close_trade(
        self,
        watch:         WatchSignal,
        entry:         MTFEntry,
        entry_px:      float,
        sl_spot:       float,
        tp_spot:       float,
        exit_px:       float,
        lots:          int,
        qty:           int,
        reason:        str,
        day:           date,
        dte:           int,
    ) -> MTFTrade:
        exit_adj  = max(exit_px - _SLIPPAGE_PTS, 0.1)
        entry_adj = entry_px + _SLIPPAGE_PTS
        gross     = (exit_adj - entry_adj) * qty
        stt       = exit_adj * qty * _STT_PCT
        net       = gross - _BROKERAGE - stt
        pnl_pct   = net / (entry_px * qty) * 100 if entry_px * qty > 0 else 0.0

        ctx = watch.context
        logger.info(
            "[MTF-BT] %s %s/%s EXIT %s | prem ₹%.2f→₹%.2f | net ₹%.2f",
            day, watch.source, watch.direction, reason, entry_px, exit_px, net,
        )
        return MTFTrade(
            date          = str(day),
            index         = self.index,
            direction     = watch.direction,
            source        = watch.source,
            or_high       = watch.or_high,
            or_low        = watch.or_low,
            or_range_pct  = watch.or_range_pct,
            watch_time    = str(watch.trigger_time),
            entry_time    = str(entry.entry_time),
            entry_mode    = entry.mode,
            adx           = ctx.adx,
            trend         = ctx.trend,
            entry_spot    = entry.entry_price,
            sl_spot       = round(sl_spot, 2),
            tp_spot       = round(tp_spot, 2),
            entry_premium = entry_px,
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

    def _compute_ema20_today(
        self,
        prior_15: pd.DataFrame,
        today_15: pd.DataFrame,
    ) -> dict[int, float]:
        """
        Compute EMA20 for today's candles using prior data as warmup.
        Returns {today_15_index: ema20_value}.
        """
        if today_15.empty:
            return {}

        prior_closes = (
            prior_15["close"].astype(float).tolist()
            if prior_15 is not None and not prior_15.empty else []
        )
        span  = 20
        alpha = 2.0 / (span + 1)

        # Seed EMA from prior data
        if prior_closes:
            ema = prior_closes[0]
            for v in prior_closes[1:]:
                ema = v * alpha + ema * (1 - alpha)
        else:
            ema = float(today_15["close"].iloc[0])

        result: dict[int, float] = {}
        for idx, row in today_15.iterrows():
            ema = float(row["close"]) * alpha + ema * (1 - alpha)
            result[idx] = round(ema, 2)

        return result

    def _save_csv(self, trades: List[MTFTrade], from_date: str, to_date: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(
            REPORTS_DIR,
            f"mtf_{self.index}_{from_date}_{to_date}.csv",
        )
        if not trades:
            return fname
        fields = [f.name for f in MTFTrade.__dataclass_fields__.values()]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({fn: getattr(t, fn) for fn in fields})
        logger.info("[MTF-BT] CSV → %s", fname)
        return fname


def _sim_premium(spot: float, dte_days: int) -> float:
    sigma   = BACKTEST_VIX / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), MIN_PREMIUM)
