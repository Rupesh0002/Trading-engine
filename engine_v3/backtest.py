"""
engine_v3 Backtest — 3 setups on 5-min candles.

  Setup 1: ORB  Breakout  (10:00–12:00)
  Setup 2: PDH/PDL Retest (10:00–14:00)
  Setup 3: VWAP Reclaim   (10:30–13:30)

Entry executed at OPEN of the next 5-min candle after signal fires.
Max 2 trades per index per day. Daily loss limit: 3% of capital.
"""
from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import pytz

from config.events_calendar import get_next_expiry
from config.settings import (
    BACKTEST_VIX, INDEX_CONFIG, MAX_LOTS_CAP, MAX_PREMIUM, MIN_PREMIUM,
    REPORTS_DIR, TRADING_CAPITAL,
)
from engine_v3.levels import (
    build_prior_levels, compute_orb, update_vwap, compute_rsi, compute_ema,
)
from engine_v3.setup_orb  import check_orb,      ORBSignal
from engine_v3.setup_pdh  import check_pdh_pdl,  PDHSignal
from engine_v3.setup_vwap import check_vwap,     VWAPSignal
from engine_v3.exit_manager import TradeState, update_and_check
from engine_v3.bias import compute_bias, lots_multiplier

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_RISK_PCT       = 0.015    # 1.5% risk per trade
_DAILY_LOSS_LIM = 0.03     # 3% daily loss limit
_MAX_TRADES_DAY = 2
_ATM_DELTA      = 0.50
_THETA_5M       = 0.004    # 0.4% per 5-min candle
_SLIPPAGE_PTS   = 1.0
_BROKERAGE      = 40.0
_STT_PCT        = 0.0003
_ORB_END_TIME   = time(9, 45)

# ── Vertical spread mode (buy ATM, sell OTM) ─────────────────────────────────
# Sell OTM for 55% of ATM premium → net debit = ATM × 0.45
# Net theta ≈ 0.0018/candle (vs 0.004 naked) — OTM sale offsets ~55% of decay
# Net delta ≈ 0.22           (ATM 0.50 − OTM 0.28)
# Profit capped at spread_width pts (100 NIFTY, 200 BANKNIFTY)
_SPREAD_OTM_CREDIT = 0.55
_SPREAD_NET_DELTA  = 0.22
_SPREAD_NET_THETA  = 0.0018
_SPREAD_WIDTH      = {"NIFTY": 100.0, "BANKNIFTY": 200.0, "SENSEX": 200.0}


@dataclass
class V3Trade:
    date:           str
    index:          str
    setup:          str    # ORB | PDH_PDL | VWAP
    direction:      str
    bias:           str
    gap_pct:        float
    signal_time:    str
    entry_time:     str
    entry_spot:     float
    entry_premium:  float
    sl_spot:        float
    tp_spot:        float
    lots:           int
    lot_size:       int
    quantity:       int
    partial1_lots:  int
    partial1_px:    float
    partial2_lots:  int
    partial2_px:    float
    exit_premium:   float
    gross_pnl:      float
    net_pnl:        float
    pnl_pct:        float
    exit_reason:    str
    dte:            int


@dataclass
class V3BacktestResult:
    trades:          List[V3Trade] = field(default_factory=list)
    csv_path:        str           = ""
    starting_capital: float        = TRADING_CAPITAL
    spread_mode:     bool          = False

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
        start = self.starting_capital
        cap, peak, dd = start, start, 0.0
        for t in self.trades:
            cap  += t.net_pnl
            peak  = max(peak, cap)
            dd    = max(dd, (peak - cap) / peak * 100)
        return dd

    def _setup_stats(self, setup: str):
        t = [x for x in self.trades if x.setup == setup]
        if not t:
            return None
        w   = sum(1 for x in t if x.net_pnl > 0)
        pnl = sum(x.net_pnl for x in t)
        aw  = sum(x.net_pnl for x in t if x.net_pnl > 0) / max(w, 1)
        al  = sum(x.net_pnl for x in t if x.net_pnl <= 0) / max(len(t) - w, 1)
        return {"n": len(t), "wr": w/len(t)*100, "pnl": pnl,
                "aw": aw, "al": al, "rr": abs(aw/al) if al else 0}

    def summary(self, from_date: str = "", to_date: str = "") -> str:
        CAP     = self.starting_capital
        months  = max(1, self.total_trades / 4) if self.total_trades else 1  # rough
        ret_pct = self.total_pnl / CAP * 100

        orb  = self._setup_stats("ORB")
        pdh  = self._setup_stats("PDH_PDL")
        vwap = self._setup_stats("VWAP")

        def _row(label, orb_v, pdh_v, vwap_v, all_v, fmt="{:.1f}"):
            cols = [orb_v, pdh_v, vwap_v, all_v]
            vals = []
            for c in cols:
                vals.append(fmt.format(c) if c is not None else "  —  ")
            return f"  {label:<22}: {vals[0]:>8}  {vals[1]:>8}  {vals[2]:>8}  {vals[3]:>8}"

        # Trades per month
        # Calculate actual month span from dates
        if self.trades:
            from datetime import datetime
            d0 = datetime.strptime(self.trades[0].date,  "%Y-%m-%d")
            d1 = datetime.strptime(self.trades[-1].date, "%Y-%m-%d")
            months = max(1, (d1 - d0).days / 30.44)
        else:
            months = 1

        mode_tag = "SPREAD (buy ATM + sell OTM)" if self.spread_mode else "Naked ATM options"
        lines = [
            "━" * 70,
            "  ENGINE V3 BACKTEST — ORB + PDH/PDL + VWAP",
            f"  Mode: {mode_tag}  ·  Capital: ₹{CAP:,.0f}",
            "━" * 70,
            "",
            f"  {'':22}  {'ORB':>8}  {'PDH/PDL':>8}  {'VWAP':>8}  {'ALL':>8}",
            "  " + "─" * 65,
        ]

        def _s(st):
            return st if st else {}

        orb_n   = orb["n"]   if orb  else 0
        pdh_n   = pdh["n"]   if pdh  else 0
        vwap_n  = vwap["n"]  if vwap else 0
        all_n   = self.total_trades

        orb_pm  = orb_n  / months if orb  else 0
        pdh_pm  = pdh_n  / months if pdh  else 0
        vwap_pm = vwap_n / months if vwap else 0
        all_pm  = all_n  / months

        lines.append(_row("Trades/month",
            orb_pm, pdh_pm, vwap_pm, all_pm, "{:.1f}"))

        lines.append(_row("Win rate %",
            orb["wr"]  if orb  else None,
            pdh["wr"]  if pdh  else None,
            vwap["wr"] if vwap else None,
            self.win_rate))

        lines.append(_row("Avg winner ₹",
            orb["aw"]  if orb  else None,
            pdh["aw"]  if pdh  else None,
            vwap["aw"] if vwap else None,
            self.avg_winner, "{:+,.0f}"))

        lines.append(_row("Avg loser ₹",
            orb["al"]  if orb  else None,
            pdh["al"]  if pdh  else None,
            vwap["al"] if vwap else None,
            self.avg_loser, "{:+,.0f}"))

        lines.append(_row("Avg R multiple",
            orb["rr"]  if orb  else None,
            pdh["rr"]  if pdh  else None,
            vwap["rr"] if vwap else None,
            self.avg_rr, "{:.2f}×"))

        # Annualised return
        ann      = ret_pct / months * 12
        orb_ann  = (orb["pnl"]  / CAP * 100 / months * 12) if orb  else None
        pdh_ann  = (pdh["pnl"]  / CAP * 100 / months * 12) if pdh  else None
        vwap_ann = (vwap["pnl"] / CAP * 100 / months * 12) if vwap else None
        lines.append(_row("Net annual ret %",
            orb_ann, pdh_ann, vwap_ann, ann, "{:+.1f}%"))

        lines += [
            "  " + "─" * 65,
            f"  Max drawdown    : {self.max_drawdown_pct:.1f}%",
            f"  Total P&L       : ₹{self.total_pnl:+,.0f}  ({ret_pct:+.1f}%  over period)",
            f"  CSV             : {self.csv_path}",
            "━" * 70,
        ]
        return "\n".join(lines)


class V3BacktestEngine:

    def __init__(self, kite, index: str = "NIFTY",
                 spread: bool = False, capital: float = 0.0) -> None:
        self.kite      = kite
        self.index     = index.upper()
        cfg            = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size = cfg["lot_size"]
        self._spread   = spread
        self._capital  = capital if capital > 0 else TRADING_CAPITAL

    def run(self, from_date: str, to_date: str) -> V3BacktestResult:
        from data.feed import DataFeed
        feed = DataFeed(self.kite)

        from datetime import datetime
        from_dt     = IST.localize(datetime.strptime(from_date, "%Y-%m-%d"))
        warmup_from = (from_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        from_date_d = from_dt.date()

        logger.info("[V3-BT] 15-min %s→%s (warmup from %s) [%s]",
                    from_date, to_date, warmup_from, self.index)
        df15 = feed.get_historical_candles(
            warmup_from, to_date, index=self.index, interval="15minute"
        )
        logger.info("[V3-BT] 5-min  %s→%s [%s]", from_date, to_date, self.index)
        df5  = feed.get_historical_candles(
            from_date, to_date, index=self.index, interval="5minute"
        )

        if df15 is None or df15.empty:
            logger.error("[V3-BT] No 15-min data.")
            return V3BacktestResult()
        if df5 is None or df5.empty:
            logger.error("[V3-BT] No 5-min data.")
            return V3BacktestResult()

        for df in (df15, df5):
            if "date" in df.columns and "timestamp" not in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_days     = sorted(df5["timestamp"].dt.date.unique())
        trading_days = [d for d in all_days if d >= from_date_d]
        logger.info("[V3-BT] %d trading days | 15m=%d | 5m=%d",
                    len(trading_days), len(df15), len(df5))

        result  = V3BacktestResult(starting_capital=self._capital,
                                   spread_mode=self._spread)
        capital = self._capital

        for day in trading_days:
            prior_15 = df15[df15["timestamp"].dt.date < day].copy()
            today_5  = (
                df5[df5["timestamp"].dt.date == day]
                .sort_values("timestamp").reset_index(drop=True)
            )
            expiry = get_next_expiry(self.index, day, days_buffer=0)
            dte    = max((expiry - day).days, 1)

            day_trades = self._simulate_day(prior_15, today_5, day, dte, capital)
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
        dte:      int,
        capital:  float,
    ) -> List[V3Trade]:

        if today_5.empty:
            return []

        today_open = float(today_5.iloc[0]["open"])
        levels     = build_prior_levels(prior_15, today_open, index=self.index)
        bias_obj   = compute_bias(today_open, levels.pdc)

        # VWAP state
        vwap_num, vwap_den = 0.0, 0.0
        prev_vwap          = today_open
        vwap_val           = today_open

        # Rolling 5-min candle history for volume proxy + RSI + EMA
        seen_rows: List[pd.Series] = []

        # Pending entry: (signal_obj, setup_name, direction)
        pending_entry = None

        # Open trade (one at a time per spec: max 2 per day)
        open_trade:   Optional[Tuple[V3Trade, TradeState]] = None
        completed:    List[V3Trade] = []
        trades_opened = 0
        daily_pnl     = 0.0
        orb_ready      = False

        rows = list(today_5.iterrows())

        for idx, (_, row) in enumerate(rows):
            t5   = pd.Timestamp(row["timestamp"]).time()
            prev_vwap_this_step = vwap_val

            # Update VWAP
            vwap_val, vwap_num, vwap_den = update_vwap(vwap_num, vwap_den, row)
            levels.vwap = vwap_val

            # Build ORB once the 9:40 candle has been processed
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

            # ── Execute pending entry (next candle open) ──────────────
            if pending_entry is not None and open_trade is None:
                sig, setup, direction = pending_entry
                entry_spot    = float(row["open"])
                atm_premium   = _sim_premium(entry_spot, dte)
                entry_premium = (atm_premium * (1 - _SPREAD_OTM_CREDIT)
                                 if self._spread else atm_premium)

                if MIN_PREMIUM <= entry_premium <= MAX_PREMIUM:
                    # Lot sizing: 1.5% risk per trade, bias-adjusted
                    lot_mult   = lots_multiplier(bias_obj.bias, direction)
                    sl_spot    = sig.sl_spot
                    eff_delta  = _SPREAD_NET_DELTA if self._spread else _ATM_DELTA
                    sl_delta   = abs(entry_spot - sl_spot) * eff_delta
                    sl_per_lot = sl_delta * self._lot_size
                    risk_amt   = capital * _RISK_PCT * lot_mult
                    raw_lots   = int(risk_amt / sl_per_lot) if sl_per_lot > 0 else 1
                    lots       = max(1, min(raw_lots, MAX_LOTS_CAP))
                    qty       = lots * self._lot_size

                    tp_spot = sig.tp_spot

                    trade = V3Trade(
                        date          = str(day),
                        index         = self.index,
                        setup         = setup,
                        direction     = direction,
                        bias          = bias_obj.bias,
                        gap_pct       = bias_obj.gap_pct,
                        signal_time   = str(sig.signal_time),
                        entry_time    = str(t5),
                        entry_spot    = round(entry_spot, 2),
                        entry_premium = round(entry_premium, 2),
                        sl_spot       = round(sl_spot, 2),
                        tp_spot       = round(tp_spot, 2),
                        lots          = lots,
                        lot_size      = self._lot_size,
                        quantity      = qty,
                        partial1_lots = 0, partial1_px = 0.0,
                        partial2_lots = 0, partial2_px = 0.0,
                        exit_premium  = 0.0,
                        gross_pnl     = 0.0, net_pnl = 0.0, pnl_pct = 0.0,
                        exit_reason   = "",
                        dte           = dte,
                    )
                    state = TradeState(
                        entry_premium = entry_premium,
                        entry_spot    = entry_spot,
                        setup         = setup,
                        total_lots    = lots,
                    )
                    open_trade    = (trade, state)
                    trades_opened += 1
                    logger.info(
                        "[V3-BT] %s %s %s/%s | entry=%.2f prem=₹%.2f "
                        "SL=%.2f TP=%.2f lots=%d bias=%s",
                        day, self.index, setup, direction,
                        entry_spot, entry_premium, sl_spot, tp_spot,
                        lots, bias_obj.bias,
                    )
                pending_entry = None

            # ── Monitor open trade ────────────────────────────────────
            if open_trade is not None:
                trade, state = open_trade
                direction    = trade.direction

                # Theta + delta update
                c5       = float(row["close"])
                prev_close = float(seen_rows[-1]["close"]) if seen_rows else trade.entry_spot
                current_prem = state.entry_premium if state.candles_held == 0 else \
                               getattr(state, "_current_prem", state.entry_premium)

                # Rebuild current premium from theta+delta model
                if state.candles_held == 0:
                    current_prem = trade.entry_premium
                else:
                    current_prem = getattr(state, "_current_prem", trade.entry_premium)

                theta  = _SPREAD_NET_THETA if self._spread else _THETA_5M
                delta  = _SPREAD_NET_DELTA if self._spread else _ATM_DELTA
                current_prem = max(current_prem * (1 - theta), 0.1)
                px_d = (c5 - prev_close) * delta if direction == "CALL" \
                       else (prev_close - c5) * delta
                current_prem = max(current_prem + px_d, 0.1)
                # Spread: premium capped at spread width (max value of spread)
                if self._spread:
                    sw = _SPREAD_WIDTH.get(self.index, 100.0)
                    current_prem = min(current_prem, sw)
                state._current_prem = current_prem  # type: ignore[attr-defined]

                # Fixed TP check for ORB (spot-based)
                if trade.setup == "ORB" and trade.tp_spot > 0:
                    if direction == "CALL" and c5 >= trade.tp_spot:
                        t = self._close(trade, state, current_prem, "TP hit", day)
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue
                    if direction == "PUT" and c5 <= trade.tp_spot:
                        t = self._close(trade, state, current_prem, "TP hit", day)
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue

                # PDH/PDL VWAP TP (spot-based)
                if trade.setup == "PDH_PDL" and trade.tp_spot > 0:
                    if direction == "CALL" and c5 >= trade.tp_spot:
                        t = self._close(trade, state, current_prem, "VWAP TP", day)
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue
                    if direction == "PUT" and c5 <= trade.tp_spot:
                        t = self._close(trade, state, current_prem, "VWAP TP", day)
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None
                        seen_rows.append(row)
                        continue

                # Spot-based hard SL
                if direction == "CALL" and c5 <= trade.sl_spot:
                    t = self._close(trade, state, current_prem, "SL hit", day)
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue
                if direction == "PUT" and c5 >= trade.sl_spot:
                    t = self._close(trade, state, current_prem, "SL hit", day)
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

                # Exit manager (premium-based exits)
                closes_with_current = [float(r["close"]) for r in seen_rows] + [c5]
                result = update_and_check(
                    state,
                    current_prem,
                    t5,
                    direction,
                    c5,
                    closes_with_current,
                )
                if result is not None:
                    if result.reason.startswith("Partial"):
                        # Record partial booking in trade, don't close yet
                        if state.partial1_done and trade.partial1_lots == 0:
                            trade.partial1_lots = state.partial1_lots
                            trade.partial1_px   = state.partial1_px
                        elif state.partial2_done and trade.partial2_lots == 0:
                            trade.partial2_lots = state.partial2_lots
                            trade.partial2_px   = state.partial2_px
                        if result.is_full:
                            t = self._close(trade, state, current_prem, result.reason, day)
                            completed.append(t)
                            daily_pnl += t.net_pnl
                            open_trade = None
                    else:
                        t = self._close(trade, state, current_prem, result.reason, day)
                        completed.append(t)
                        daily_pnl += t.net_pnl
                        open_trade = None

            # ── Check daily loss limit ────────────────────────────────
            if daily_pnl <= -capital * _DAILY_LOSS_LIM:
                if open_trade is not None:
                    trade, state = open_trade
                    cprem = getattr(state, "_current_prem", trade.entry_premium)
                    t = self._close(trade, state, cprem, "Daily loss limit", day)
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                seen_rows.append(row)
                break  # stop processing this day

            # ── Check for new signals ─────────────────────────────────
            if (pending_entry is None
                    and open_trade is None
                    and trades_opened < _MAX_TRADES_DAY
                    and levels.orb_ready):

                closes_so_far = [float(r["close"]) for r in seen_rows]

                rsi = compute_rsi(closes_so_far[-20:]) if len(closes_so_far) >= 5 else 50.0

                # Priority: ORB > PDH/PDL > VWAP
                sig_orb  = check_orb(row, prev_df, levels, vwap_val, rsi, index=self.index)
                if sig_orb:
                    pending_entry = (sig_orb, "ORB", sig_orb.direction)
                else:
                    sig_pdh = check_pdh_pdl(row, prev_df, levels, vwap_val)
                    if sig_pdh:
                        pending_entry = (sig_pdh, "PDH_PDL", sig_pdh.direction)
                    else:
                        sig_vwap = check_vwap(
                            row, prev_df, vwap_val, prev_vwap_this_step
                        )
                        if sig_vwap:
                            pending_entry = (sig_vwap, "VWAP", sig_vwap.direction)

            seen_rows.append(row)
            prev_vwap = vwap_val

        # End of day: close any open trade
        if open_trade is not None:
            trade, state = open_trade
            cprem = getattr(state, "_current_prem", trade.entry_premium)
            t = self._close(trade, state, cprem, "Hard close (15:15)", day)
            completed.append(t)

        return completed

    # ── Helpers ───────────────────────────────────────────────────────

    def _close(
        self,
        trade:   V3Trade,
        state:   TradeState,
        exit_px: float,
        reason:  str,
        day:     date,
    ) -> V3Trade:
        lot_size = self._lot_size
        ep       = trade.entry_premium

        # Compute total P&L including any partial exits
        total_entry  = (ep + _SLIPPAGE_PTS) * trade.quantity
        total_exit   = 0.0
        total_exit  += (state.partial1_px - _SLIPPAGE_PTS) * state.partial1_lots * lot_size \
                        if state.partial1_done else 0.0
        total_exit  += (state.partial2_px - _SLIPPAGE_PTS) * state.partial2_lots * lot_size \
                        if state.partial2_done else 0.0
        exit_adj     = max(exit_px - _SLIPPAGE_PTS, 0.1)
        total_exit  += exit_adj * state.remaining_lots * lot_size

        gross    = total_exit - total_entry
        stt      = total_exit * _STT_PCT
        net      = gross - _BROKERAGE - stt
        pnl_pct  = net / total_entry * 100 if total_entry > 0 else 0.0

        trade.partial1_lots = state.partial1_lots
        trade.partial1_px   = state.partial1_px
        trade.partial2_lots = state.partial2_lots
        trade.partial2_px   = state.partial2_px
        trade.exit_premium  = round(exit_px, 2)
        trade.gross_pnl     = round(gross, 2)
        trade.net_pnl       = round(net, 2)
        trade.pnl_pct       = round(pnl_pct, 2)
        trade.exit_reason   = reason

        logger.info(
            "[V3-BT] %s %s/%s EXIT %s | prem ₹%.2f→₹%.2f | net ₹%.0f",
            day, trade.setup, trade.direction, reason,
            ep, exit_px, net,
        )
        return trade

    def _save_csv(self, trades: List[V3Trade], from_date: str, to_date: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(
            REPORTS_DIR, f"v3_{self.index}_{from_date}_{to_date}.csv"
        )
        if not trades:
            return fname
        fields = [f.name for f in V3Trade.__dataclass_fields__.values()]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({fn: getattr(t, fn) for fn in fields})
        logger.info("[V3-BT] CSV → %s", fname)
        return fname


def _sim_premium(spot: float, dte_days: int) -> float:
    sigma   = BACKTEST_VIX / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), MIN_PREMIUM)
