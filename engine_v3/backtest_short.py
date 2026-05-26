"""
engine_v3 Short Options Backtest.

Same directional signals (ORB, PDH/PDL, VWAP, SMC) but instead of BUYING
the option in the signal direction, we SELL the opposite-side ATM option.

  CALL signal → Sell ATM PUT  (profit when spot rises or stays flat)
  PUT  signal → Sell ATM CALL (profit when spot falls or stays flat)

Theta now works FOR us every candle.
A flat or mildly adverse spot move still wins via time decay.

Exit:
  Short SL  : sold premium rises to 150% of entry (50% of credit = max loss)
  Short TP  : sold premium falls to 30% of entry  (captured 70% of credit)
  Hard close: 14:30 IST  (before end-of-day gamma expansion)
  Daily loss: 3% of capital

P&L = (entry_premium - exit_premium) × quantity
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
    build_prior_levels, compute_orb, update_vwap, compute_rsi,
)
from engine_v3.setup_orb  import check_orb,      ORBSignal
from engine_v3.setup_pdh  import check_pdh_pdl,  PDHSignal
from engine_v3.setup_vwap import check_vwap,     VWAPSignal
from engine_v3.setup_smc  import check_smc,      SMCSignal
from engine_v3.bias import compute_bias, lots_multiplier

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

_ATM_DELTA      = 0.50
_THETA_5M       = 0.004     # 0.4%/candle — now benefits the seller
_SHORT_SL_PCT   = 1.50      # exit if premium reaches 150% of entry (50% loss on credit)
_SHORT_TP_PCT   = 0.30      # exit if premium falls to 30% of entry  (70% credit captured)
_HARD_CLOSE     = time(14, 30)
_DAILY_LOSS_LIM = 0.03
_MAX_TRADES_DAY = 2
_SLIPPAGE_PTS   = 1.0
_BROKERAGE      = 40.0
_STT_PCT        = 0.0003
_ORB_END_TIME   = time(9, 45)
_RISK_PCT       = 0.015

# Margin required per lot for selling options (approximate)
_MARGIN = {"NIFTY": 65_000, "BANKNIFTY": 50_000, "SENSEX": 130_000}
_CAPITAL_USE = 0.85


@dataclass
class ShortTrade:
    date:           str
    index:          str
    setup:          str
    direction:      str    # CALL (sold PUT) | PUT (sold CALL)
    bias:           str
    gap_pct:        float
    signal_time:    str
    entry_time:     str
    entry_spot:     float
    entry_premium:  float  # premium received when selling
    sl_spot:        float  # reference only (not used for exit — exit is premium-based)
    lots:           int
    lot_size:       int
    quantity:       int
    exit_premium:   float  # premium paid to close
    gross_pnl:      float
    net_pnl:        float
    pnl_pct:        float
    exit_reason:    str
    dte:            int


@dataclass
class ShortBacktestResult:
    trades:          List[ShortTrade] = field(default_factory=list)
    csv_path:        str              = ""
    starting_capital: float          = TRADING_CAPITAL
    index:           str              = ""

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
        cap, peak, dd = self.starting_capital, self.starting_capital, 0.0
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
        return {"n": len(t), "wr": w / len(t) * 100, "pnl": pnl,
                "aw": aw, "al": al, "rr": abs(aw / al) if al else 0.0}

    def summary(self) -> str:
        CAP = self.starting_capital
        if not self.trades:
            return "  No short-option trades generated."

        from datetime import datetime
        d0     = datetime.strptime(self.trades[0].date,  "%Y-%m-%d")
        d1     = datetime.strptime(self.trades[-1].date, "%Y-%m-%d")
        months = max(1.0, (d1 - d0).days / 30.44)
        ret    = self.total_pnl / CAP * 100
        ann    = ret / months * 12

        orb  = self._setup_stats("ORB")
        pdh  = self._setup_stats("PDH_PDL")
        vwap = self._setup_stats("VWAP")
        smc  = self._setup_stats("SMC")

        def _row(label, o, p, v, s, all_v, fmt="{:.1f}"):
            cols = [o, p, v, s, all_v]
            vals = [fmt.format(c) if c is not None else "     —" for c in cols]
            return (f"  {label:<20}: {vals[0]:>7}  {vals[1]:>7}  "
                    f"{vals[2]:>7}  {vals[3]:>7}  {vals[4]:>7}")

        lines = [
            "━" * 72,
            f"  SHORT OPTIONS BACKTEST — {self.index}",
            "  CALL signal → Sell ATM PUT  ·  PUT signal → Sell ATM CALL",
            f"  Capital: ₹{CAP:,.0f}  ·  Theta works FOR us",
            "━" * 72,
            "",
            f"  {'':20}  {'ORB':>7}  {'PDH/PDL':>7}  {'VWAP':>7}  {'SMC':>7}  {'ALL':>7}",
            "  " + "─" * 68,
        ]

        def _n(st): return st["n"] / months if st else None
        def _v(st, k): return st[k] if st else None

        lines.append(_row("Trades/month",
            _n(orb), _n(pdh), _n(vwap), _n(smc), self.total_trades / months, "{:.1f}"))
        lines.append(_row("Win rate %",
            _v(orb,"wr"), _v(pdh,"wr"), _v(vwap,"wr"), _v(smc,"wr"), self.win_rate))
        lines.append(_row("Avg winner ₹",
            _v(orb,"aw"), _v(pdh,"aw"), _v(vwap,"aw"), _v(smc,"aw"),
            self.avg_winner, "{:+,.0f}"))
        lines.append(_row("Avg loser ₹",
            _v(orb,"al"), _v(pdh,"al"), _v(vwap,"al"), _v(smc,"al"),
            self.avg_loser, "{:+,.0f}"))
        lines.append(_row("Avg R multiple",
            _v(orb,"rr"), _v(pdh,"rr"), _v(vwap,"rr"), _v(smc,"rr"),
            self.avg_rr, "{:.2f}×"))

        orb_ann  = (_v(orb,"pnl")  / CAP * 100 / months * 12) if orb  else None
        pdh_ann  = (_v(pdh,"pnl")  / CAP * 100 / months * 12) if pdh  else None
        vwap_ann = (_v(vwap,"pnl") / CAP * 100 / months * 12) if vwap else None
        smc_ann  = (_v(smc,"pnl")  / CAP * 100 / months * 12) if smc  else None
        lines.append(_row("Annual return %",
            orb_ann, pdh_ann, vwap_ann, smc_ann, ann, "{:+.1f}%"))

        lines += [
            "  " + "─" * 68,
            f"  Max drawdown   : {self.max_drawdown_pct:.1f}%",
            f"  Total P&L      : ₹{self.total_pnl:+,.0f}  ({ret:+.1f}% over period)",
            "━" * 72,
        ]
        return "\n".join(lines)


class ShortOptionsEngine:

    def __init__(self, kite, index: str = "NIFTY", capital: float = 0.0) -> None:
        self.kite      = kite
        self.index     = index.upper()
        cfg            = INDEX_CONFIG.get(self.index, INDEX_CONFIG["NIFTY"])
        self._lot_size = cfg["lot_size"]
        self._margin   = _MARGIN.get(self.index, 65_000)
        self._capital  = capital if capital > 0 else TRADING_CAPITAL

    def run(self, from_date: str, to_date: str) -> ShortBacktestResult:
        from datetime import datetime
        from data.feed import DataFeed
        feed = DataFeed(self.kite)

        from_dt     = IST.localize(datetime.strptime(from_date, "%Y-%m-%d"))
        warmup_from = (from_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        from_date_d = from_dt.date()

        logger.info("[SHORT-BT] 15-min %s→%s (warmup %s) [%s]",
                    from_date, to_date, warmup_from, self.index)
        df15 = feed.get_historical_candles(
            warmup_from, to_date, index=self.index, interval="15minute"
        )
        logger.info("[SHORT-BT] 5-min %s→%s [%s]", from_date, to_date, self.index)
        df5  = feed.get_historical_candles(
            from_date, to_date, index=self.index, interval="5minute"
        )

        for df in (df15, df5):
            if "date" in df.columns and "timestamp" not in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        all_days     = sorted(df5["timestamp"].dt.date.unique())
        trading_days = [d for d in all_days if d >= from_date_d]
        logger.info("[SHORT-BT] %d days | 15m=%d | 5m=%d",
                    len(trading_days), len(df15), len(df5))

        result  = ShortBacktestResult(starting_capital=self._capital, index=self.index)
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
    ) -> List[ShortTrade]:

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

        pending_entry = None
        open_trade: Optional[Tuple[ShortTrade, float]] = None  # (trade, current_prem)
        completed:  List[ShortTrade] = []
        trades_opened = 0
        daily_pnl     = 0.0

        rows = list(today_5.iterrows())

        for idx, (_, row) in enumerate(rows):
            t5   = pd.Timestamp(row["timestamp"]).time()
            c5   = float(row["close"])
            prev_vwap_step = vwap_val

            vwap_val, vwap_num, vwap_den = update_vwap(vwap_num, vwap_den, row)
            levels.vwap = vwap_val

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
                entry_premium = _sim_premium(entry_spot, dte)

                if MIN_PREMIUM <= entry_premium <= MAX_PREMIUM:
                    lot_mult  = lots_multiplier(bias_obj.bias, direction)
                    # SL loss per lot = 50% of entry_premium × lot_size
                    sl_loss_per_lot = entry_premium * (_SHORT_SL_PCT - 1.0) * self._lot_size
                    risk_amt  = capital * _RISK_PCT * lot_mult
                    risk_lots = int(risk_amt / sl_loss_per_lot) if sl_loss_per_lot > 0 else 1
                    margin_lots = int(capital * _CAPITAL_USE / self._margin)
                    lots      = max(1, min(risk_lots, margin_lots, MAX_LOTS_CAP))
                    qty       = lots * self._lot_size

                    trade = ShortTrade(
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
                        sl_spot       = round(sig.sl_spot, 2),
                        lots          = lots,
                        lot_size      = self._lot_size,
                        quantity      = qty,
                        exit_premium  = 0.0,
                        gross_pnl     = 0.0, net_pnl = 0.0, pnl_pct = 0.0,
                        exit_reason   = "",
                        dte           = dte,
                    )
                    open_trade    = (trade, entry_premium)
                    trades_opened += 1
                    logger.info(
                        "[SHORT-BT] %s %s %s/%s SELL | spot=%.2f prem=₹%.2f lots=%d",
                        day, self.index, setup, direction,
                        entry_spot, entry_premium, lots,
                    )
                pending_entry = None

            # ── Monitor open trade ────────────────────────────────────
            if open_trade is not None:
                trade, current_prem = open_trade
                direction = trade.direction
                prev_close = float(seen_rows[-1]["close"]) if seen_rows else trade.entry_spot

                # Premium model for SOLD option:
                # Theta decays the sold premium → good for seller
                # Spot move in signal direction → sold option falls → good
                # Spot move against signal → sold option rises → bad
                current_prem = max(current_prem * (1 - _THETA_5M), 0.1)
                if direction == "CALL":     # sold PUT: up move → PUT falls
                    px_d = (prev_close - c5) * _ATM_DELTA
                else:                       # sold CALL: down move → CALL falls
                    px_d = (c5 - prev_close) * _ATM_DELTA
                current_prem = max(current_prem + px_d, 0.1)
                open_trade = (trade, current_prem)

                # Hard close 14:30
                if t5 >= _HARD_CLOSE:
                    t = self._close(trade, current_prem, "Hard close (14:30)")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

                # SL: sold premium rose to 150% of entry
                if current_prem >= trade.entry_premium * _SHORT_SL_PCT:
                    t = self._close(trade, current_prem, "Short SL")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

                # TP: sold premium fell to 30% of entry (captured 70%)
                if current_prem <= trade.entry_premium * _SHORT_TP_PCT:
                    t = self._close(trade, current_prem, "Short TP")
                    completed.append(t)
                    daily_pnl += t.net_pnl
                    open_trade = None
                    seen_rows.append(row)
                    continue

            # Daily loss limit
            if daily_pnl <= -capital * _DAILY_LOSS_LIM:
                if open_trade is not None:
                    trade, current_prem = open_trade
                    t = self._close(trade, current_prem, "Daily loss limit")
                    completed.append(t)
                    open_trade = None
                seen_rows.append(row)
                break

            # ── Check for new signals ─────────────────────────────────
            if (pending_entry is None
                    and open_trade is None
                    and trades_opened < _MAX_TRADES_DAY
                    and levels.orb_ready):

                closes = [float(r["close"]) for r in seen_rows]
                rsi    = compute_rsi(closes[-20:]) if len(closes) >= 5 else 50.0

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
                            sig_smc = check_smc(row, prev_df)
                            if sig_smc:
                                pending_entry = (sig_smc, "SMC", sig_smc.direction)

            seen_rows.append(row)
            prev_vwap = vwap_val

        # End of day: close any open trade
        if open_trade is not None:
            trade, current_prem = open_trade
            t = self._close(trade, current_prem, "Hard close (14:30)")
            completed.append(t)

        return completed

    # ── Helpers ───────────────────────────────────────────────────────

    def _close(self, trade: ShortTrade, exit_prem: float, reason: str) -> ShortTrade:
        qty   = trade.quantity
        ep    = trade.entry_premium

        # Received premium at entry, paid to close at exit
        entry_recv = (ep - _SLIPPAGE_PTS) * qty        # slippage: received less
        exit_pay   = (exit_prem + _SLIPPAGE_PTS) * qty  # slippage: paid more to close

        gross = entry_recv - exit_pay
        stt   = exit_pay * _STT_PCT
        net   = gross - _BROKERAGE - stt
        pct   = net / entry_recv * 100 if entry_recv > 0 else 0.0

        trade.exit_premium = round(exit_prem, 2)
        trade.gross_pnl    = round(gross, 2)
        trade.net_pnl      = round(net, 2)
        trade.pnl_pct      = round(pct, 2)
        trade.exit_reason  = reason

        logger.info("[SHORT-BT] %s %s/%s EXIT %s | ₹%.2f→₹%.2f | net ₹%.0f",
                    trade.date, trade.setup, trade.direction,
                    reason, ep, exit_prem, net)
        return trade

    def _save_csv(self, trades: List[ShortTrade], from_date: str, to_date: str) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(REPORTS_DIR,
                             f"short_{self.index}_{from_date}_{to_date}.csv")
        if not trades:
            return fname
        fields = [f.name for f in ShortTrade.__dataclass_fields__.values()]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({fn: getattr(t, fn) for fn in fields})
        logger.info("[SHORT-BT] CSV → %s", fname)
        return fname


def _sim_premium(spot: float, dte_days: int) -> float:
    sigma   = BACKTEST_VIX / 100.0
    t       = max(dte_days, 0.5) / 365.0
    premium = 0.4 * spot * sigma * math.sqrt(t)
    return max(round(premium, 2), MIN_PREMIUM)
