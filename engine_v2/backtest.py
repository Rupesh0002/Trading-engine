"""
engine_v2 backtest — 4-indicator signal on 5-min candles + dynamic exit.

Pipeline per day:
  1. Gap filter        → abs(today_open - prev_close) / prev_close > 0.3%
  2. Signal scan       → first all-4-pass candle between 9:30–13:00
                         direction must agree with gap direction
  3. Entry             → next 5-min candle open after signal
  4. Premium sim       → Black-Scholes ATM entry, then delta+theta per candle
  5. ExitManager       → 6-state dynamic exit on every 5-min candle

Cost model:
  - Theta:     0.4% per 5-min candle
  - Slippage:  1.5 pts / leg (entry + exit = 3 pts total)
  - Brokerage: ₹40 flat
  - STT:       0.03% of exit value (buy side only — options buyer)

Phase 1 lot sizing: always 1 lot (ML filter not yet active).
"""
from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config.settings import BACKTEST_VIX, TRADING_CAPITAL, INDEX_CONFIG, REPORTS_DIR
from data.feed import DataFeed
from engine_v2.exit_manager import ExitManager, ExitSignal
from engine_v2.signal import SignalResult, compute_signal

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Cost constants ────────────────────────────────────────────────────────────
_THETA_5MIN   = 0.004    # 0.4% per 5-min candle
_SLIPPAGE_PTS = 1.5      # per leg
_BROKERAGE    = 40.0
_STT_PCT      = 0.0003

# ── Premium simulation ────────────────────────────────────────────────────────
_ATM_FACTOR   = 0.4      # ATM premium ≈ 0.4 × spot × (VIX/100) × √(DTE/365)
_DELTA        = 0.5      # ATM delta

# ── Filters ───────────────────────────────────────────────────────────────────
_GAP_MIN_PCT  = 0.30     # minimum gap % (open vs prev_close)
_MIN_PREMIUM  = 40.0     # reject if simulated ATM premium too thin
_MAX_PREMIUM  = 400.0    # reject if premium out of retail range
_PREWARM_DAYS = 3        # days of 5-min data prepended for indicator warm-up

# ── Phase 1 lot sizing ────────────────────────────────────────────────────────
_PHASE1_LOTS  = 1


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class V3Trade:
    date:             str
    direction:        str
    gap_pct:          float
    entry_premium:    float
    entry_spot:       float
    lots:             int
    lot_size:         int
    candles_held_5min: int

    # Partial exit details
    partial1_lots:    int   = 0
    partial1_premium: float = 0.0
    partial2_lots:    int   = 0
    partial2_premium: float = 0.0

    # Final exit
    final_premium:    float = 0.0
    final_reason:     str   = ""
    highest_premium:  float = 0.0
    trailing_sl_at_exit: float = 0.0

    # Signal detail
    adx_val:          float = 0.0
    rsi_val:          float = 0.0
    range_ratio:      float = 0.0
    ema9:             float = 0.0

    # ML features (extracted at signal time)
    features:         dict  = field(default_factory=dict)

    # P&L components
    gross_pnl:        float = 0.0
    theta_cost:       float = 0.0
    slippage_cost:    float = 0.0
    brokerage:        float = 0.0
    stt_cost:         float = 0.0
    net_pnl:          float = 0.0
    net_pnl_pct:      float = 0.0


@dataclass
class V3Results:
    trades:            List[V3Trade]
    total_trades:      int
    winning_trades:    int
    losing_trades:     int
    win_rate:          float
    avg_winner:        float
    avg_loser:         float
    avg_rr:            float
    total_pnl:         float
    total_return_pct:  float
    max_drawdown_pct:  float
    exit_breakdown:    Dict[str, int]
    avg_hold_candles:  float
    csv_path:          str

    def summary(self) -> str:
        lines = [
            "  ── ENGINE V3 BACKTEST RESULTS ──────────────────────────────",
            f"  Trades       : {self.total_trades}  "
            f"(W {self.winning_trades} / L {self.losing_trades})",
            f"  Win Rate     : {self.win_rate:.1f}%",
            f"  Avg Winner   : ₹{self.avg_winner:+,.0f}",
            f"  Avg Loser    : ₹{self.avg_loser:+,.0f}",
            f"  Avg R:R      : {self.avg_rr:.2f}×",
            f"  Total P&L    : ₹{self.total_pnl:+,.0f}  ({self.total_return_pct:+.1f}%)",
            f"  Max Drawdown : {self.max_drawdown_pct:.1f}%",
            f"  Avg Hold     : {self.avg_hold_candles:.1f} candles ({self.avg_hold_candles * 5:.0f} min)",
            f"  Exit Reasons : {dict(sorted(self.exit_breakdown.items(), key=lambda x: -x[1]))}",
        ]
        return "\n".join(lines)


# ── Main engine ───────────────────────────────────────────────────────────────

class BacktestEngineV3:
    def __init__(self, kite, index: str = "NIFTY") -> None:
        self.kite  = kite
        self.index = index
        self.feed  = DataFeed(kite)
        cfg        = INDEX_CONFIG[index]
        self.lot_size    = cfg["lot_size"]
        self.strike_step = cfg["strike_step"]

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, from_date: str, to_date: str) -> V3Results:
        logger.info("V3 backtest: %s %s → %s", self.index, from_date, to_date)

        # Fetch 5-min candles with extra prewarm (fetch from 5 days before start)
        pre_start = (
            datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=10)
        ).strftime("%Y-%m-%d")

        df5 = self.feed.get_historical_candles(
            pre_start, to_date, index=self.index, interval="5minute"
        )
        if df5 is None or df5.empty:
            logger.error("No 5-min data returned for %s", self.index)
            return self._empty_results()

        df5 = df5.copy()
        df5["date"] = df5["timestamp"].dt.date

        # Group into trading days
        all_dates = sorted(df5["date"].unique())
        # Only backtest within the requested range
        from_d = datetime.strptime(from_date, "%Y-%m-%d").date()
        to_d   = datetime.strptime(to_date,   "%Y-%m-%d").date()
        trade_dates = [d for d in all_dates if from_d <= d <= to_d]

        trades: List[V3Trade] = []
        prev_close: Optional[float] = None

        for day in trade_dates:
            today_rows = df5[df5["date"] == day]
            if today_rows.empty:
                continue

            # prev_close: last candle of previous trading day
            prev_rows = df5[df5["date"] < day]
            if prev_rows.empty:
                prev_close = None
            else:
                prev_close = float(prev_rows["close"].iloc[-1])

            # prewarm: last _PREWARM_DAYS calendar days before today
            prewarm_start = day - timedelta(days=_PREWARM_DAYS + 4)
            prewarm_rows  = df5[(df5["date"] >= prewarm_start) & (df5["date"] < day)]

            trade = self._simulate_day(today_rows, prewarm_rows, prev_close, day)
            if trade is not None:
                trades.append(trade)

        return self._build_results(trades)

    # ── Day simulation ──────────────────────────────────────────────────────

    def _simulate_day(
        self,
        today: pd.DataFrame,
        prewarm: pd.DataFrame,
        prev_close: Optional[float],
        day: date,
    ) -> Optional[V3Trade]:

        # Need market open candle (9:15) to determine today's open
        open_row = today[today["timestamp"].dt.time == time(9, 15)]
        if open_row.empty:
            open_row = today.iloc[[0]]
        today_open = float(open_row["open"].iloc[0])

        # ── Gap filter ──────────────────────────────────────────────────────
        if prev_close is None or prev_close == 0:
            return None
        gap_pct = (today_open - prev_close) / prev_close * 100
        if abs(gap_pct) < _GAP_MIN_PCT:
            return None

        gap_direction = "CALL" if gap_pct > 0 else "PUT"

        # ── Build pre-warmed frame ───────────────────────────────────────────
        combined = pd.concat([prewarm, today], ignore_index=True)

        # ── Scan for signal in 9:30–13:00 window ────────────────────────────
        today_idx_start = len(prewarm)
        signal_row: Optional[int] = None  # absolute row index in combined
        sig_result: Optional[SignalResult] = None

        for i in range(today_idx_start, len(combined)):
            row_time = combined["timestamp"].iloc[i].time()
            if row_time < time(9, 30):
                continue
            if row_time > time(13, 0):
                break

            sr = compute_signal(combined, candle_idx=i)
            if sr.direction == gap_direction:
                signal_row = i
                sig_result  = sr
                break

        if signal_row is None:
            return None  # no signal today

        # ── Extract ML features at signal candle ─────────────────────────────
        try:
            from engine_v2.ml_filter import extract_features as _ef
            # FCR: first 5-min candle of today (9:15)
            first_row  = today[today["timestamp"].dt.time == time(9, 15)]
            if first_row.empty:
                first_row = today.iloc[[0]]
            fcr_high = float(first_row["high"].iloc[0])
            fcr_low  = float(first_row["low"].iloc[0])
            fcr_open = float(first_row["open"].iloc[0])
            fcr_pct  = (fcr_high - fcr_low) / fcr_open if fcr_open else 0.0

            ml_features = _ef(
                combined, candle_idx=signal_row,
                adx_val=sig_result.adx, rsi_val=sig_result.rsi,
                ema9=sig_result.ema9, ema21=sig_result.ema21, ema50=sig_result.ema50,
                vix=BACKTEST_VIX, gap_pct=gap_pct, fcr_pct=fcr_pct,
                index=self.index,
            )
        except Exception:
            ml_features = {}

        # Entry is on the NEXT candle after signal
        entry_row_idx = signal_row + 1
        if entry_row_idx >= len(combined):
            return None

        entry_row  = combined.iloc[entry_row_idx]
        entry_time = entry_row["timestamp"].time()
        entry_spot = float(entry_row["open"])

        # ── Simulate entry premium ───────────────────────────────────────────
        entry_premium = self._sim_entry_premium(entry_spot, day)
        if not (_MIN_PREMIUM <= entry_premium <= _MAX_PREMIUM):
            return None

        # Apply entry slippage
        entry_premium += _SLIPPAGE_PTS  # paid on buy

        lots     = _PHASE1_LOTS
        em       = ExitManager(entry_premium, lots, entry_time)
        candles_5min = 0

        partial1_lots = 0;  partial1_prem = 0.0
        partial2_lots = 0;  partial2_prem = 0.0
        partial1_theta = 0.0; partial2_theta = 0.0
        theta_total   = 0.0
        exit_premium  = entry_premium
        exit_reason   = "HARD_CLOSE"

        # ── Walk forward through remaining candles ───────────────────────────
        sim_premium = entry_premium
        for j in range(entry_row_idx + 1, len(combined)):
            row       = combined.iloc[j]
            row_time  = row["timestamp"].time()

            # Only process today's candles
            if row["timestamp"].date() != day:
                break

            candles_5min += 1
            spot_close = float(row["close"])

            # Simulate premium: delta move + theta decay
            sim_premium = self._sim_premium(
                entry_premium_no_slip=entry_premium - _SLIPPAGE_PTS,
                entry_spot=entry_spot,
                current_spot=spot_close,
                candles_held=candles_5min,
                direction=gap_direction,
            )

            # df_recent: last 20 rows up to and including current
            df_recent = combined.iloc[max(0, j - 20) : j + 1]

            signal = em.check(sim_premium, row_time, df_recent=df_recent)
            if signal is None:
                continue

            if not signal.exit_all:
                # Partial exit
                prem_received = max(signal.premium - _SLIPPAGE_PTS, 0)
                theta_part    = candles_5min * _THETA_5MIN * entry_premium * signal.lots / lots
                if partial1_lots == 0:
                    partial1_lots = signal.lots
                    partial1_prem = prem_received
                    partial1_theta = theta_part
                else:
                    partial2_lots = signal.lots
                    partial2_prem = prem_received
                    partial2_theta = theta_part
                continue

            # Final exit
            exit_premium = max(signal.premium - _SLIPPAGE_PTS, 0)
            exit_reason  = signal.reason
            break
        else:
            # End of day without exit
            last_row = combined[combined["timestamp"].dt.date == day]
            if not last_row.empty:
                sim_premium = self._sim_premium(
                    entry_premium - _SLIPPAGE_PTS, entry_spot,
                    float(last_row["close"].iloc[-1]), candles_5min, gap_direction,
                )
            exit_premium = max(sim_premium - _SLIPPAGE_PTS, 0)
            exit_reason  = "EOD"

        # ── P&L calculation ──────────────────────────────────────────────────
        final_lots = em.remaining_lots
        theta_cost = candles_5min * _THETA_5MIN * entry_premium * final_lots

        gross_pnl = 0.0
        if partial1_lots > 0:
            gross_pnl += (partial1_prem - entry_premium) * partial1_lots * self.lot_size
        if partial2_lots > 0:
            gross_pnl += (partial2_prem - entry_premium) * partial2_lots * self.lot_size
        gross_pnl += (exit_premium - entry_premium) * final_lots * self.lot_size

        total_lots_closed = final_lots + partial1_lots + partial2_lots
        slippage = _SLIPPAGE_PTS * total_lots_closed * self.lot_size  # exit slips only (entry already added)
        stt      = exit_premium * final_lots * self.lot_size * _STT_PCT
        brok     = _BROKERAGE

        net_pnl     = gross_pnl - slippage - brok - stt
        net_pnl_pct = net_pnl / TRADING_CAPITAL * 100

        return V3Trade(
            date=str(day),
            direction=gap_direction,
            gap_pct=round(gap_pct, 2),
            entry_premium=round(entry_premium, 2),
            entry_spot=round(entry_spot, 2),
            lots=lots,
            lot_size=self.lot_size,
            candles_held_5min=candles_5min,
            partial1_lots=partial1_lots,
            partial1_premium=round(partial1_prem, 2),
            partial2_lots=partial2_lots,
            partial2_premium=round(partial2_prem, 2),
            final_premium=round(exit_premium, 2),
            final_reason=exit_reason,
            highest_premium=round(em.highest, 2),
            trailing_sl_at_exit=round(em.trailing_sl or 0.0, 2),
            adx_val=round(sig_result.adx if sig_result else 0.0, 1),
            rsi_val=round(sig_result.rsi if sig_result else 0.0, 1),
            range_ratio=round(sig_result.range_ratio if sig_result else 0.0, 2),
            ema9=round(sig_result.ema9 if sig_result else 0.0, 1),
            gross_pnl=round(gross_pnl, 2),
            theta_cost=round(theta_cost, 2),
            slippage_cost=round(slippage, 2),
            brokerage=round(brok, 2),
            stt_cost=round(stt, 2),
            net_pnl=round(net_pnl, 2),
            net_pnl_pct=round(net_pnl_pct, 2),
            features=ml_features,
        )

    # ── Premium simulation ──────────────────────────────────────────────────

    def _sim_entry_premium(self, spot: float, day: date) -> float:
        dte = self._dte(day)
        return _ATM_FACTOR * spot * (BACKTEST_VIX / 100) * math.sqrt(max(dte, 1) / 365)

    def _sim_premium(
        self,
        entry_premium_no_slip: float,
        entry_spot: float,
        current_spot: float,
        candles_held: int,
        direction: str,
    ) -> float:
        spot_move = current_spot - entry_spot
        if direction == "PUT":
            spot_move = -spot_move
        prem_move = _DELTA * spot_move
        theta_drag = candles_held * _THETA_5MIN
        raw = entry_premium_no_slip + prem_move - theta_drag * entry_premium_no_slip
        return max(raw, entry_premium_no_slip * 0.05)

    def _dte(self, day: date) -> int:
        cfg     = INDEX_CONFIG[self.index]
        exp_day = cfg.get("expiry_day", "thursday")
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4,
        }
        target = day_map.get(exp_day.lower(), 3)
        d = day
        for _ in range(8):
            if d.weekday() == target:
                return max((d - day).days, 1)
            d += timedelta(days=1)
        return 3

    # ── Build results ────────────────────────────────────────────────────────

    def _build_results(self, trades: List[V3Trade]) -> V3Results:
        if not trades:
            return self._empty_results()

        winners = [t for t in trades if t.net_pnl > 0]
        losers  = [t for t in trades if t.net_pnl <= 0]
        total   = len(trades)
        win_r   = len(winners) / total * 100

        avg_w = sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0
        avg_l = sum(t.net_pnl for t in losers)  / len(losers)  if losers  else 0.0
        avg_rr = abs(avg_w / avg_l) if avg_l != 0 else 0.0

        total_pnl = sum(t.net_pnl for t in trades)
        total_ret = total_pnl / TRADING_CAPITAL * 100

        # Max drawdown (running peak)
        equity = 0.0
        peak   = 0.0
        max_dd = 0.0
        for t in trades:
            equity += t.net_pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / TRADING_CAPITAL * 100
            if dd > max_dd:
                max_dd = dd

        exit_bd: Dict[str, int] = {}
        for t in trades:
            exit_bd[t.final_reason] = exit_bd.get(t.final_reason, 0) + 1

        avg_hold = sum(t.candles_held_5min for t in trades) / total

        csv_path = self._write_csv(trades)

        return V3Results(
            trades=trades,
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_r, 1),
            avg_winner=round(avg_w, 2),
            avg_loser=round(avg_l, 2),
            avg_rr=round(avg_rr, 2),
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_ret, 2),
            max_drawdown_pct=round(max_dd, 1),
            exit_breakdown=exit_bd,
            avg_hold_candles=round(avg_hold, 1),
            csv_path=csv_path,
        )

    def _empty_results(self) -> V3Results:
        return V3Results(
            trades=[], total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_winner=0.0, avg_loser=0.0, avg_rr=0.0,
            total_pnl=0.0, total_return_pct=0.0, max_drawdown_pct=0.0,
            exit_breakdown={}, avg_hold_candles=0.0,
            csv_path="",
        )

    def _write_csv(self, trades: List[V3Trade]) -> str:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"v3_{self.index}_signals.csv")
        fields = [
            "date", "direction", "gap_pct", "entry_premium", "entry_spot",
            "lots", "candles_held_5min",
            "partial1_lots", "partial1_premium",
            "partial2_lots", "partial2_premium",
            "final_premium", "final_reason",
            "highest_premium", "trailing_sl_at_exit",
            "adx_val", "rsi_val", "range_ratio", "ema9",
            "gross_pnl", "theta_cost", "slippage_cost", "brokerage", "stt_cost",
            "net_pnl", "net_pnl_pct",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({k: getattr(t, k) for k in fields})
        return path

    # ── ML Phase 2 simulation ────────────────────────────────────────────────

    def simulate_ml_phase2(self, phase1_trades: List[V3Trade]) -> Dict:
        """
        Simulate ML Phase 2 on the Phase 1 backtest results.

        Uses time-series forward-walk: train on first 40 trades, score 41+.
        Retrain every 15 new trades (same as production schedule).
        Returns dict with filtered trade list + performance delta.
        """
        try:
            import pandas as _pd
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from engine_v2.ml_filter import FEATURE_COLS
        except ImportError as e:
            return {"error": str(e), "filtered_trades": phase1_trades}

        _PHASE2_START = 40
        _RETRAIN_EVERY = 15
        _THRESHOLD = 0.55

        all_feats  = [t.features for t in phase1_trades]
        all_labels = [1 if t.final_reason == "NO_MOVE" else 0 for t in phase1_trades]

        current_model = None
        last_train_at = -1

        filtered_trades: List[V3Trade] = []
        skipped: List[V3Trade] = []
        ml_probs: List[float]  = []

        for i, trade in enumerate(phase1_trades):
            if i < _PHASE2_START:
                filtered_trades.append(trade)
                ml_probs.append(1.0)
                continue

            # Retrain if at 40 or every 15 thereafter
            due_retrain = (last_train_at < 0) or ((i - _PHASE2_START) % _RETRAIN_EVERY == 0)
            if due_retrain:
                X_raw = _pd.DataFrame(
                    [[all_feats[j].get(c, 0.0) for c in FEATURE_COLS] for j in range(i)],
                    columns=FEATURE_COLS,
                ).fillna(0)
                y_tr = _pd.Series(all_labels[:i])
                n_pos = int(y_tr.sum())
                n_neg = len(y_tr) - n_pos
                class_w = {0: 1.0, 1: max(1.0, n_neg / n_pos) if n_pos else 1.0}

                scaler = StandardScaler()
                X_tr   = _pd.DataFrame(scaler.fit_transform(X_raw), columns=FEATURE_COLS)

                current_model = LogisticRegression(
                    C=0.5,                # moderate regularization
                    class_weight=class_w,
                    max_iter=500,
                    random_state=42,
                )
                current_model.fit(X_tr, y_tr)
                current_model._scaler = scaler   # attach scaler for predict step
                last_train_at = i
                logger.info("ML sim: retrained at trade %d (n=%d, NO_MOVE=%d)", i, len(y_tr), n_pos)

            # Score this trade (apply same scaler used in training)
            raw_row = _pd.DataFrame(
                [[all_feats[i].get(c, 0.0) for c in FEATURE_COLS]], columns=FEATURE_COLS
            ).fillna(0)
            scaler   = getattr(current_model, "_scaler", None)
            row      = _pd.DataFrame(
                scaler.transform(raw_row), columns=FEATURE_COLS
            ) if scaler is not None else raw_row
            prob_nm  = float(current_model.predict_proba(row)[0][1])
            ml_probs.append(prob_nm)

            if prob_nm >= _THRESHOLD:
                skipped.append(trade)
            else:
                filtered_trades.append(trade)

        # Build comparison stats
        def _stats(trades: List[V3Trade]) -> Dict:
            if not trades:
                return {"n": 0, "win_rate": 0, "total_pnl": 0, "avg_rr": 0, "max_dd": 0}
            w = [t for t in trades if t.net_pnl > 0]
            l = [t for t in trades if t.net_pnl <= 0]
            aw = sum(t.net_pnl for t in w) / len(w) if w else 0.0
            al = sum(t.net_pnl for t in l) / len(l) if l else 0.0
            eq = 0.0; pk = 0.0; dd = 0.0
            for t in trades:
                eq += t.net_pnl
                pk  = max(pk, eq)
                dd  = max(dd, (pk - eq) / TRADING_CAPITAL * 100)
            return {
                "n":         len(trades),
                "win_rate":  round(len(w) / len(trades) * 100, 1),
                "total_pnl": round(sum(t.net_pnl for t in trades), 0),
                "avg_rr":    round(abs(aw / al) if al else 0.0, 2),
                "max_dd":    round(dd, 1),
            }

        phase1_stats  = _stats(phase1_trades)
        phase2_stats  = _stats(filtered_trades)
        skipped_stats = _stats(skipped)

        # Feature importance (coefficients for LogReg, importance for tree models)
        feat_imp = {}
        if current_model is not None:
            try:
                imp_vals = current_model.feature_importances_
            except AttributeError:
                # LogisticRegression: use abs(coef) as importance
                imp_vals = abs(current_model.coef_[0])
            feat_imp = dict(zip(FEATURE_COLS, imp_vals))
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))

        return {
            "filtered_trades":  filtered_trades,
            "skipped_trades":   skipped,
            "phase1":           phase1_stats,
            "phase2":           phase2_stats,
            "skipped_summary":  skipped_stats,
            "feature_importance": feat_imp,
            "ml_probs":         ml_probs,
        }
