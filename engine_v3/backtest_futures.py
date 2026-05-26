"""
engine_v3 Futures Backtest.

Same 4 setups as backtest.py (ORB, PDH/PDL, VWAP, SMC) but P&L is computed
on the underlying futures contract — no theta, no premium, pure spot P&L.

P&L = (exit_spot - entry_spot) × lot_size × lots      # CALL
P&L = (entry_spot - exit_spot) × lot_size × lots      # PUT

SL in spot points:
  ORB    : entry ± ORB_HIGH/LOW × 0.003  (0.3% of breakout level)
  PDH/PDL: entry ± level × 0.002         (0.2% of level)
  VWAP   : entry ± entry × 0.002         (0.2% of entry)
  SMC    : wick extreme ± index-aware noise buffer

TP remains the same spot levels as in the options backtest:
  ORB    : ORB_HIGH + 1.5 × ORB_RANGE  (CALL)
  PDH/PDL: VWAP at signal time
  VWAP   : entry ± entry × 0.004 as initial (2× SL), trail after
  SMC    : entry ± 2 × SL_distance

Position sizing:
  risk_amount  = capital × 1.5%
  sl_pts       = abs(entry - sl_spot)
  sl_per_lot   = sl_pts × lot_size
  raw_lots     = floor(risk_amount / sl_per_lot)
  margin_lots  = floor(capital × 0.85 / margin_per_lot)
  lots         = max(1, min(raw_lots, margin_lots, MAX_LOTS_CAP))

Costs:
  Brokerage : ₹40 per trade
  Slippage  : 0.5 pts per leg (1 pt round trip)
  STT       : 0.01% on exit side only
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import pytz

from config.events_calendar import get_next_expiry
from config.settings import (
    INDEX_CONFIG, MAX_LOTS_CAP, REPORTS_DIR, TRADING_CAPITAL,
)
from engine_v3.levels import (
    build_prior_levels, compute_ema, compute_orb, update_vwap, compute_rsi,
)
from engine_v3.setup_orb  import check_orb,      ORBSignal
from engine_v3.setup_pdh  import check_pdh_pdl,  PDHSignal
from engine_v3.setup_vwap import check_vwap,     VWAPSignal
from engine_v3.setup_smc  import check_smc,      SMCSignal
from engine_v3.bias import compute_bias, lots_multiplier

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Futures-specific constants ────────────────────────────────────────────────
_FUTURES_MARGIN = {"NIFTY": 65_000, "BANKNIFTY": 50_000, "SENSEX": 130_000}
_MAX_LOTS       = 2          # hard cap regardless of capital
_CAPITAL_USE    = 0.85       # use max 85% of capital for margin
_RISK_PCT       = 0.015      # 1.5% risk per trade
_SLIPPAGE_PTS   = 0.5        # per leg
_BROKERAGE      = 40.0
_STT_PCT        = 0.0001     # 0.01% on exit (sell side)

# SL distances (as fraction of reference price)
_SL_PCT = {"ORB": 0.003, "PDH_PDL": 0.002, "VWAP": 0.002}  # SMC uses its own

# Setup-specific no-move handling
_NO_MOVE_CANDLES = {
    "ORB": 12,
    "PDH_PDL": 16,
    "VWAP": 10,
    "SMC": 8,
}
_NO_MOVE_PCT = {
    "ORB": 0.0020,
    "PDH_PDL": 0.0015,
    "VWAP": 0.0012,
    "SMC": 0.0012,
}
_NO_MOVE_EXTENSION_CANDLES = 6

_ORB_END_TIME  = time(9, 45)
_HARD_CLOSE    = time(15, 15)
_MAX_TRADES    = 2


def _is_no_move_reason(reason: str) -> bool:
    normalized = reason.upper().replace("-", "_").replace(" ", "_")
    return normalized.startswith("NO_MOVE")


@dataclass
class FuturesTrade:
    date:         str
    index:        str
    setup:        str    # ORB | PDH_PDL | VWAP | SMC
    direction:    str    # CALL | PUT
    bias:         str
    gap_pct:      float
    signal_time:  str
    entry_time:   str
    entry_spot:   float
    sl_spot:      float
    tp_spot:      float
    sl_pts:       float
    lots:         int
    lot_size:     int
    quantity:     int
    exit_spot:    float
    gross_pnl:    float
    net_pnl:      float
    pnl_pct:      float
    exit_reason:  str


@dataclass
class FuturesBacktestResult:
    trades:    List[FuturesTrade] = field(default_factory=list)
    csv_path:  str                = ""
    index:     str                = ""
    from_date: str                = ""
    to_date:   str                = ""

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

    def _setup_stats(self, setup: str) -> Optional[dict]:
        t = [x for x in self.trades if x.setup == setup]
        if not t:
            return None
        w   = sum(1 for x in t if x.net_pnl > 0)
        pnl = sum(x.net_pnl for x in t)
        nm  = sum(1 for x in t if _is_no_move_reason(x.exit_reason))
        aw  = sum(x.net_pnl for x in t if x.net_pnl > 0) / max(w, 1)
        al  = sum(x.net_pnl for x in t if x.net_pnl <= 0) / max(len(t) - w, 1)
        return {"n": len(t), "wr": w / len(t) * 100, "pnl": pnl,
                "aw": aw, "al": al, "rr": abs(aw / al) if al else 0.0,
                "nm_pct": nm / len(t) * 100}

    def summary(self,
                opt_win_rate: float = 0.0,
                opt_annual:   float = 0.0,
                opt_dd:       float = 0.0) -> str:
        if not self.trades:
            return "  No futures trades generated."

        from datetime import datetime
        d0     = datetime.strptime(self.trades[0].date,  "%Y-%m-%d")
        d1     = datetime.strptime(self.trades[-1].date, "%Y-%m-%d")
        months = max(1.0, (d1 - d0).days / 30.44)
        ann    = self.total_pnl / TRADING_CAPITAL * 100 / months * 12

        orb  = self._setup_stats("ORB")
        pdh  = self._setup_stats("PDH_PDL")
        vwap = self._setup_stats("VWAP")
        smc  = self._setup_stats("SMC")

        def _v(st, key, fmt):
            return fmt.format(st[key]) if st else "     —"

        def _row(label, o, p, v, s, all_v, fmt="{:.1f}"):
            cols = [o, p, v, s, all_v]
            vals = [fmt.format(c) if c is not None else "     —" for c in cols]
            return f"  {label:<20}: {vals[0]:>7}  {vals[1]:>7}  {vals[2]:>7}  {vals[3]:>7}  {vals[4]:>7}"

        lines = [
            "━" * 72,
            f"  FUTURES BACKTEST — {self.index}",
            "  Same setups · No theta · Pure spot P&L",
            "━" * 72,
            "",
            f"  {'':20}  {'ORB':>7}  {'PDH/PDL':>7}  {'VWAP':>7}  {'SMC':>7}  {'ALL':>7}",
            "  " + "─" * 68,
        ]

        orb_pm  = orb["n"]  / months if orb  else None
        pdh_pm  = pdh["n"]  / months if pdh  else None
        vwap_pm = vwap["n"] / months if vwap else None
        smc_pm  = smc["n"]  / months if smc  else None

        lines.append(_row("Trades/month",
            orb_pm, pdh_pm, vwap_pm, smc_pm, self.total_trades / months, "{:.1f}"))
        lines.append(_row("Win rate %",
            orb["wr"]  if orb  else None,
            pdh["wr"]  if pdh  else None,
            vwap["wr"] if vwap else None,
            smc["wr"]  if smc  else None,
            self.win_rate))
        lines.append(_row("Avg winner ₹",
            orb["aw"]  if orb  else None,
            pdh["aw"]  if pdh  else None,
            vwap["aw"] if vwap else None,
            smc["aw"]  if smc  else None,
            self.avg_winner, "{:+,.0f}"))
        lines.append(_row("Avg loser ₹",
            orb["al"]  if orb  else None,
            pdh["al"]  if pdh  else None,
            vwap["al"] if vwap else None,
            smc["al"]  if smc  else None,
            self.avg_loser, "{:+,.0f}"))
        lines.append(_row("Avg R multiple",
            orb["rr"]  if orb  else None,
            pdh["rr"]  if pdh  else None,
            vwap["rr"] if vwap else None,
            smc["rr"]  if smc  else None,
            self.avg_rr, "{:.2f}×"))

        orb_ann  = (orb["pnl"]  / TRADING_CAPITAL * 100 / months * 12) if orb  else None
        pdh_ann  = (pdh["pnl"]  / TRADING_CAPITAL * 100 / months * 12) if pdh  else None
        vwap_ann = (vwap["pnl"] / TRADING_CAPITAL * 100 / months * 12) if vwap else None
        smc_ann  = (smc["pnl"]  / TRADING_CAPITAL * 100 / months * 12) if smc  else None
        lines.append(_row("Annual return %",
            orb_ann, pdh_ann, vwap_ann, smc_ann, ann, "{:+.1f}%"))
        lines.append(_row("No-move exits %",
            orb["nm_pct"] if orb else None,
            pdh["nm_pct"] if pdh else None,
            vwap["nm_pct"] if vwap else None,
            smc["nm_pct"] if smc else None,
            sum(1 for t in self.trades if _is_no_move_reason(t.exit_reason)) / self.total_trades * 100,
            "{:.1f}%"))

        lines += [
            "  " + "─" * 68,
            f"  Max drawdown   : {self.max_drawdown_pct:.1f}%",
            f"  Total P&L      : ₹{self.total_pnl:+,.0f}",
            "━" * 72,
        ]

        # ── Options vs Futures comparison ────────────────────────────
        if opt_win_rate:
            fut_wr  = self.win_rate
            fut_ann = ann
            fut_dd  = self.max_drawdown_pct
            lines += [
                "",
                "  OPTIONS vs FUTURES (same signals)",
                "  " + "─" * 50,
                f"  {'':18}  {'Options':>10}  {'Futures':>10}  {'Δ':>8}",
                "  " + "─" * 50,
                f"  {'Win rate':18}  {opt_win_rate:>9.1f}%  {fut_wr:>9.1f}%  {fut_wr-opt_win_rate:>+7.1f}pp",
                f"  {'Annual return':18}  {opt_annual:>9.1f}%  {fut_ann:>9.1f}%  {fut_ann-opt_annual:>+7.1f}pp",
                f"  {'Max drawdown':18}  {opt_dd:>9.1f}%  {fut_dd:>9.1f}%  {fut_dd-opt_dd:>+7.1f}pp",
                "  " + "─" * 50,
            ]

        return "\n".join(lines)


class FuturesBacktestEngine:

    def __init__(self, kite, index: str = "NIFTY") -> None:
        self.kite      = kite
        self.index     = index.upper()
        cfg            = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size = cfg["lot_size"]
        self._margin   = _FUTURES_MARGIN.get(self.index, 65_000)

    def run(self, from_date: str, to_date: str) -> FuturesBacktestResult:
        from datetime import datetime
        from data.feed import DataFeed
        feed = DataFeed(self.kite)

        from_dt     = IST.localize(datetime.strptime(from_date, "%Y-%m-%d"))
        warmup_from = (from_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        from_date_d = from_dt.date()

        logger.info("[FUT-BT] 15-min %s→%s (warmup %s) [%s]",
                    from_date, to_date, warmup_from, self.index)
        df15 = feed.get_historical_candles(
            warmup_from, to_date, index=self.index, interval="15minute"
        )
        logger.info("[FUT-BT] 5-min %s→%s [%s]", from_date, to_date, self.index)
        df5  = feed.get_historical_candles(
            from_date, to_date, index=self.index, interval="5minute"
        )

        for df in (df15, df5):
            if "date" in df.columns and "timestamp" not in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_days     = sorted(df5["timestamp"].dt.date.unique())
        trading_days = [d for d in all_days if d >= from_date_d]
        logger.info("[FUT-BT] %d days | 15m=%d | 5m=%d",
                    len(trading_days), len(df15), len(df5))

        result  = FuturesBacktestResult(index=self.index,
                                        from_date=from_date, to_date=to_date)
        capital = TRADING_CAPITAL

        for day in trading_days:
            prior_15 = df15[df15["timestamp"].dt.date < day].copy()
            today_5  = (
                df5[df5["timestamp"].dt.date == day]
                .sort_values("timestamp").reset_index(drop=True)
            )
            day_trades = self._simulate_day(prior_15, today_5, day, capital)
            for t in day_trades:
                result.trades.append(t)
                capital += t.net_pnl

        result.csv_path = self._save_csv(result.trades, from_date, to_date)
        return result

    # ── Per-day simulation ────────────────────────────────────────────

    def _simulate_day(
        self,
        prior_15: pd.DataFrame,
        today_5:  pd.DataFrame,
        day:      date,
        capital:  float,
    ) -> List[FuturesTrade]:

        if today_5.empty:
            return []

        today_open = float(today_5.iloc[0]["open"])
        levels     = build_prior_levels(prior_15, today_open, index=self.index)
        bias_obj   = compute_bias(today_open, levels.pdc)

        vwap_num, vwap_den = 0.0, 0.0
        prev_vwap          = today_open
        vwap_val           = today_open
        orb_ready          = False
        seen_rows: List[pd.Series] = []

        pending_entry = None       # (signal, setup_name, direction)  — next candle open
        immediate_entry = None     # (signal, setup_name, direction)  — THIS candle close (SMC)

        open_trade: Optional[Tuple[FuturesTrade, float, int, bool]] = None  # (trade, best_spot, candles_held, no_move_extended)
        completed:  List[FuturesTrade] = []
        trades_opened = 0
        daily_pnl     = 0.0

        rows = list(today_5.iterrows())

        for idx, (_, row) in enumerate(rows):
            t5   = pd.Timestamp(row["timestamp"]).time()
            c5   = float(row["close"])
            o5   = float(row["open"])
            prev_vwap_step = vwap_val

            # Update VWAP
            vwap_val, vwap_num, vwap_den = update_vwap(vwap_num, vwap_den, row)
            levels.vwap = vwap_val

            # Build ORB once
            if not orb_ready and t5 >= _ORB_END_TIME:
                oh, ol = compute_orb(today_5)
                if oh > 0:
                    levels.orb_high  = oh
                    levels.orb_low   = ol
                    levels.orb_range = oh - ol
                    levels.orb_ready = True
                    orb_ready        = True

            prev_df = pd.DataFrame(seen_rows) if seen_rows else pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close"]
            )

            # ── SMC immediate entry (at THIS candle's close) ──────────
            if immediate_entry is not None and open_trade is None:
                sig, setup, direction = immediate_entry
                entry_spot = sig.signal_spot   # already the close
                t_entry    = t5

                sl_spot, tp_spot = self._compute_sl_tp(
                    setup, direction, entry_spot, sig, levels
                )
                lots = self._size_lots(entry_spot, sl_spot, capital)

                if lots > 0:
                    trade = self._make_trade(
                        day, setup, direction, bias_obj, sig.signal_time, t_entry,
                        entry_spot, sl_spot, tp_spot, lots
                    )
                    open_trade    = (trade, entry_spot, 0, False)
                    trades_opened += 1
                    logger.info("[FUT-BT] %s %s %s/%s | entry=%.2f SL=%.2f TP=%.2f lots=%d",
                                day, self.index, setup, direction,
                                entry_spot, sl_spot, tp_spot, lots)
                immediate_entry = None

            # ── Next-candle entry (ORB/PDH/VWAP) ────────────────────
            if pending_entry is not None and open_trade is None:
                sig, setup, direction = pending_entry
                entry_spot = o5          # open of THIS candle
                t_entry    = t5

                sl_spot, tp_spot = self._compute_sl_tp(
                    setup, direction, entry_spot, sig, levels
                )
                lots = self._size_lots(entry_spot, sl_spot, capital)

                if lots > 0:
                    trade = self._make_trade(
                        day, setup, direction, bias_obj, sig.signal_time, t_entry,
                        entry_spot, sl_spot, tp_spot, lots
                    )
                    open_trade    = (trade, entry_spot, 0, False)
                    trades_opened += 1
                    logger.info("[FUT-BT] %s %s %s/%s | entry=%.2f SL=%.2f TP=%.2f lots=%d",
                                day, self.index, setup, direction,
                                entry_spot, sl_spot, tp_spot, lots)
                pending_entry = None

            # ── Monitor open trade ────────────────────────────────────
            if open_trade is not None:
                trade, best_spot, candles, no_move_extended = open_trade
                direction = trade.direction

                # Track best spot
                if direction == "CALL":
                    new_best = max(best_spot, c5)
                else:
                    new_best = min(best_spot, c5)
                candles += 1
                open_trade = (trade, new_best, candles, no_move_extended)

                # Hard close
                if t5 >= _HARD_CLOSE:
                    t = self._close(trade, c5, "Hard close (15:15)")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

                # SL hit
                if direction == "CALL" and c5 <= trade.sl_spot:
                    t = self._close(trade, c5, "SL hit")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue
                if direction == "PUT" and c5 >= trade.sl_spot:
                    t = self._close(trade, c5, "SL hit")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

                # TP hit (ORB fixed target, PDH VWAP target)
                if trade.tp_spot > 0:
                    if direction == "CALL" and c5 >= trade.tp_spot:
                        t = self._close(trade, c5, "TP hit")
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue
                    if direction == "PUT" and c5 <= trade.tp_spot:
                        t = self._close(trade, c5, "TP hit")
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue

                # Trail SL for VWAP (no fixed TP)
                if trade.setup == "VWAP" and trade.tp_spot == 0:
                    fav_move = abs(new_best - trade.entry_spot)
                    if fav_move >= trade.entry_spot * 0.005:   # 0.5% gained
                        if direction == "CALL":
                            trail = new_best * (1 - 0.003)    # trail 0.3% behind high
                            if c5 <= max(trail, trade.sl_spot):
                                t = self._close(trade, c5, "Trail SL")
                                completed.append(t)
                                daily_pnl += t.net_pnl
                                open_trade = None
                                seen_rows.append(row)
                                continue
                        else:
                            trail = new_best * (1 + 0.003)
                            if c5 >= min(trail, trade.sl_spot):
                                t = self._close(trade, c5, "Trail SL")
                                completed.append(t)
                                daily_pnl += t.net_pnl
                                open_trade = None
                                seen_rows.append(row)
                                continue

                # Setup-specific no-move timeout with one EMA-based extension
                base_deadline = _NO_MOVE_CANDLES.get(trade.setup, _NO_MOVE_CANDLES["VWAP"])
                threshold = _NO_MOVE_PCT.get(trade.setup, _NO_MOVE_PCT["VWAP"])
                move_pct = abs(c5 - trade.entry_spot) / trade.entry_spot if trade.entry_spot > 0 else 0.0
                if candles == base_deadline and move_pct < threshold:
                    closes_with_current = [float(r["close"]) for r in seen_rows] + [c5]
                    ema21_now = compute_ema(closes_with_current[-40:], 21) if closes_with_current else c5
                    trend_ok = c5 > ema21_now if direction == "CALL" else c5 < ema21_now
                    if trend_ok and not no_move_extended:
                        open_trade = (trade, new_best, candles, True)
                    else:
                        t = self._close(trade, c5, "NO_MOVE_TREND_LOST")
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue
                elif no_move_extended and candles == base_deadline + _NO_MOVE_EXTENSION_CANDLES and move_pct < threshold:
                    t = self._close(trade, c5, "NO_MOVE_EXTENDED")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

            # ── Daily loss limit ──────────────────────────────────────
            if daily_pnl <= -capital * 0.03:
                if open_trade is not None:
                    trade, _, _, _ = open_trade
                    t = self._close(trade, c5, "Daily loss limit")
                    completed.append(t)
                    open_trade = None
                seen_rows.append(row)
                break

            # ── Check for new signals ─────────────────────────────────
            if (open_trade is None
                    and pending_entry is None
                    and immediate_entry is None
                    and trades_opened < _MAX_TRADES
                    and levels.orb_ready):

                closes = [float(r["close"]) for r in seen_rows]
                rsi    = compute_rsi(closes[-20:]) if len(closes) >= 5 else 50.0

                # Priority: ORB → PDH/PDL → VWAP → SMC
                sig_orb = check_orb(row, prev_df, levels, vwap_val, rsi, index=self.index)
                if sig_orb:
                    pending_entry = (sig_orb, "ORB", sig_orb.direction)
                else:
                    sig_pdh = check_pdh_pdl(row, prev_df, levels, vwap_val)
                    if sig_pdh:
                        pending_entry = (sig_pdh, "PDH_PDL", sig_pdh.direction)
                    else:
                        sig_vwap = check_vwap(row, prev_df, vwap_val, prev_vwap_step)
                        if sig_vwap:
                            pending_entry = (sig_vwap, "VWAP", sig_vwap.direction)
                        else:
                            sig_smc = check_smc(row, prev_df, index=self.index)
                            if sig_smc:
                                immediate_entry = (sig_smc, "SMC", sig_smc.direction)

            seen_rows.append(row)
            prev_vwap = vwap_val

        # End of day: close any open trade
        if open_trade is not None:
            trade, _, _, _ = open_trade
            last_close = float(today_5.iloc[-1]["close"]) if not today_5.empty else trade.entry_spot
            t = self._close(trade, last_close, "Hard close (15:15)")
            completed.append(t)

        return completed

    # ── Helpers ───────────────────────────────────────────────────────

    def _compute_sl_tp(
        self,
        setup:     str,
        direction: str,
        entry:     float,
        sig,
        levels,
    ) -> Tuple[float, float]:
        """
        Override SL and TP for futures positioning.
        SL is wider than options (spot-percentage based, not range-fraction).
        """
        if setup == "SMC":
            # SMC signal already has correct wick-based SL and TP
            return sig.sl_spot, sig.tp_spot

        pct = _SL_PCT.get(setup, 0.002)

        if setup == "ORB":
            ref = levels.orb_high if direction == "CALL" else levels.orb_low
            sl  = (entry - ref * pct) if direction == "CALL" else (entry + ref * pct)
            # TP from setup signal
            tp  = sig.tp_spot

        elif setup == "PDH_PDL":
            ref = sig.level
            sl  = (entry - ref * pct) if direction == "CALL" else (entry + ref * pct)
            tp  = sig.tp_spot   # VWAP

        else:  # VWAP
            sl = (entry - entry * pct) if direction == "CALL" else (entry + entry * pct)
            # VWAP has no fixed TP; trail-based. Set tp_spot=0 to signal trail mode.
            tp = 0.0

        # Ensure SL is on correct side
        if direction == "CALL":
            sl = min(sl, entry * 0.998)
            tp = max(tp, entry * 1.002) if tp > 0 else 0.0
        else:
            sl = max(sl, entry * 1.002)
            tp = min(tp, entry * 0.998) if tp > 0 else 0.0

        return round(sl, 2), round(tp, 2)

    def _size_lots(self, entry: float, sl: float, capital: float) -> int:
        sl_pts    = abs(entry - sl)
        if sl_pts <= 0:
            return 1
        sl_per_lot = sl_pts * self._lot_size
        risk_amt   = capital * _RISK_PCT
        raw_lots   = int(risk_amt / sl_per_lot) if sl_per_lot > 0 else 1

        # Margin constraint
        margin_lots = int(capital * _CAPITAL_USE / self._margin) if self._margin > 0 else _MAX_LOTS

        return max(1, min(raw_lots, margin_lots, _MAX_LOTS))

    def _make_trade(
        self,
        day, setup, direction, bias_obj, signal_time, entry_time,
        entry_spot, sl_spot, tp_spot, lots,
    ) -> FuturesTrade:
        sl_pts = round(abs(entry_spot - sl_spot), 2)
        qty    = lots * self._lot_size
        return FuturesTrade(
            date        = str(day),
            index       = self.index,
            setup       = setup,
            direction   = direction,
            bias        = bias_obj.bias,
            gap_pct     = bias_obj.gap_pct,
            signal_time = str(signal_time),
            entry_time  = str(entry_time),
            entry_spot  = round(entry_spot, 2),
            sl_spot     = sl_spot,
            tp_spot     = tp_spot,
            sl_pts      = sl_pts,
            lots        = lots,
            lot_size    = self._lot_size,
            quantity    = qty,
            exit_spot   = 0.0,
            gross_pnl   = 0.0,
            net_pnl     = 0.0,
            pnl_pct     = 0.0,
            exit_reason = "",
        )

    def _close(
        self,
        trade:    FuturesTrade,
        exit_spot: float,
        reason:   str,
    ) -> FuturesTrade:
        direction = trade.direction
        qty       = trade.quantity

        # Slippage: entry paid 0.5 pt more, exit received 0.5 pt less
        entry_adj = trade.entry_spot + _SLIPPAGE_PTS
        exit_adj  = exit_spot - _SLIPPAGE_PTS

        if direction == "CALL":
            gross = (exit_adj - entry_adj) * qty
        else:
            gross = (entry_adj - exit_adj) * qty

        stt = abs(exit_adj) * qty * _STT_PCT
        net = gross - _BROKERAGE - stt

        base  = trade.entry_spot * qty
        pct   = net / base * 100 if base > 0 else 0.0

        trade.exit_spot   = round(exit_spot, 2)
        trade.gross_pnl   = round(gross, 2)
        trade.net_pnl     = round(net, 2)
        trade.pnl_pct     = round(pct, 2)
        trade.exit_reason = reason

        logger.info("[FUT-BT] %s %s/%s EXIT %s | %.2f→%.2f | net ₹%.0f",
                    trade.date, trade.setup, direction, reason,
                    trade.entry_spot, exit_spot, net)
        return trade

    def _save_csv(self, trades: List[FuturesTrade], from_date: str, to_date: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(REPORTS_DIR,
                             f"futures_{self.index}_{from_date}_{to_date}.csv")
        if not trades:
            return fname
        fields = [f.name for f in FuturesTrade.__dataclass_fields__.values()]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({fn: getattr(t, fn) for fn in fields})
        logger.info("[FUT-BT] CSV → %s", fname)
        return fname
