#!/usr/bin/env python3
"""
Backtest runner — NSE F&O options strategy.

Usage:
  python3 run_backtest.py                            # conviction system, .env dates
  python3 run_backtest.py --from 2024-01-01 --to 2026-04-30
  python3 run_backtest.py --from 2024-01-01 --to 2026-04-30 --index BANKNIFTY

  python3 run_backtest.py --simple                   # 2-setup simplified system
  python3 run_backtest.py --simple --from 2024-01-01 --to 2026-04-30
  python3 run_backtest.py --simple --index BANKNIFTY

All other parameters (capital, VIX) always come from .env.
"""
from __future__ import annotations

import argparse

from config.settings import (
    ACTIVE_INDEX,
    BACKTEST_END,
    BACKTEST_START,
    BACKTEST_VIX,
    TRADING_CAPITAL,
    validate_settings,
)
from utils.logger import get_logger

logger = get_logger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the NSE F&O trading engine.")
    parser.add_argument("--from",   dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (overrides .env BACKTEST_START)")
    parser.add_argument("--to",     dest="to_date",   default=None,
                        help="End date YYYY-MM-DD   (overrides .env BACKTEST_END)")
    parser.add_argument("--index",  dest="index",     default=ACTIVE_INDEX,
                        help="Index: NIFTY / BANKNIFTY / SENSEX")
    parser.add_argument("--simple", action="store_true",
                        help="Run simplified 2-setup backtest (Gap&Go + 11AM Trend)")
    parser.add_argument("--v2",    action="store_true",
                        help="Run V2 backtest (Gap&Go + dynamic trailing exit on 5-min candles)")
    parser.add_argument("--v3",    action="store_true",
                        help="Run V3 backtest (4-indicator signal + dynamic exit, engine_v2/)")
    parser.add_argument("--v3-ml", action="store_true",
                        help="Run V3 backtest then simulate ML Phase 2 filtering")
    parser.add_argument("--v4",      action="store_true",
                        help="Run engine_v3 backtest (ORB + PDH/PDL + VWAP + SMC · options)")
    parser.add_argument("--spread",  action="store_true",
                        help="Use vertical spread mode (buy ATM + sell OTM) to reduce theta")
    parser.add_argument("--short",   action="store_true",
                        help="Sell options (CALL signal→sell PUT, PUT signal→sell CALL)")
    parser.add_argument("--capital", dest="capital", type=float, default=0,
                        help="Override trading capital (e.g. 300000)")
    args = parser.parse_args()

    from_date = args.from_date or BACKTEST_START
    to_date   = args.to_date   or BACKTEST_END
    index     = args.index.upper()

    validate_settings()

    print()
    print("─" * 60)
    if getattr(args, "v3_ml", False):
        print("  BACKTEST V3 + ML PHASE 2 SIMULATION")
    elif args.short:
        print("  ENGINE V3 SHORT OPTIONS  (sell opposite-side ATM on signal)")
    elif args.v4:
        mode = "SPREAD (buy ATM + sell OTM)" if args.spread else "Naked ATM"
        cap  = f"₹{args.capital:,.0f}" if args.capital else f"₹{TRADING_CAPITAL:,.0f}"
        print(f"  ENGINE V3 BACKTEST  (ORB + PDH/PDL + VWAP · {mode} · {cap} capital)")
    elif args.v3:
        print("  BACKTEST V3  (4-Indicator Signal + Dynamic Exit)")
    elif args.v2:
        print("  BACKTEST V2  (Gap & Go  +  Dynamic Trailing Exit)")
    elif args.simple:
        print("  SIMPLIFIED BACKTEST  (Gap & Go  +  11AM Trend)")
    else:
        print("  BACKTEST  (Conviction + Momentum-Confirmed system)")
    print(f"  Index   : {index}")
    print(f"  Period  : {from_date} → {to_date}")
    print(f"  Capital : ₹{TRADING_CAPITAL:,.0f}")
    print(f"  VIX     : {BACKTEST_VIX:.1f} (assumed for premium simulation)")
    print("─" * 60)
    print()

    from config.auth import get_kite_client
    kite = get_kite_client()

    if getattr(args, "v3_ml", False):
        _run_v3_ml(kite, index, from_date, to_date)
    elif args.short:
        _run_short(kite, index, from_date, to_date, capital=args.capital)
    elif args.v4:
        _run_v4(kite, index, from_date, to_date,
                spread=args.spread, capital=args.capital)
    elif args.v3:
        _run_v3(kite, index, from_date, to_date)
    elif args.v2:
        _run_v2(kite, index, from_date, to_date)
    elif args.simple:
        _run_simple(kite, index, from_date, to_date)
    else:
        _run_conviction(kite, index, from_date, to_date)


# ── engine_v3: Short options (sell opposite-side ATM on signal) ──────────────

def _run_short(kite, index: str, from_date: str, to_date: str,
               capital: float = 0) -> None:
    from engine_v3.backtest_short import ShortOptionsEngine
    engine  = ShortOptionsEngine(kite, index=index, capital=capital)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary())
    print()

    if not results.trades:
        print("  No trades generated.")
        print()
        return

    print("  TRADE LOG")
    print("  " + "─" * 155)
    print(
        f"  {'#':>3}  {'Date':<12} {'Setup':<8} {'Dir':<5} {'Bias':<8} "
        f"{'SigT':<8} {'EntT':<8} "
        f"{'EntSpot':>9} {'EntPrem':>8} {'ExPrem':>8} "
        f"{'Lots':>4} {'Qty':>5} "
        f"{'GrossP&L':>10} {'NetP&L':>10} {'%':>7}  Reason"
    )
    print("  " + "─" * 155)
    for i, t in enumerate(results.trades, 1):
        print(
            f"  {i:>3}  {t.date:<12} {t.setup:<8} {t.direction:<5} {t.bias:<8} "
            f"{t.signal_time:<8} {t.entry_time:<8} "
            f"{t.entry_spot:>9.2f} {t.entry_premium:>8.2f} {t.exit_premium:>8.2f} "
            f"{t.lots:>4} {t.quantity:>5} "
            f"{t.gross_pnl:>+10.2f} {t.net_pnl:>+10.2f} {t.pnl_pct:>+6.1f}%  {t.exit_reason}"
        )
    print("  " + "─" * 155)

    sl_t = [t for t in results.trades if "SL"    in t.exit_reason]
    tp_t = [t for t in results.trades if "TP"    in t.exit_reason]
    hc_t = [t for t in results.trades if "close" in t.exit_reason.lower()
                                      or "limit" in t.exit_reason.lower()]

    call_t = [t for t in results.trades if t.direction == "CALL"]
    put_t  = [t for t in results.trades if t.direction == "PUT"]

    print(f"\n  DIRECTION:")
    for lst, lbl in [(call_t, "CALL (sold PUT)"), (put_t, "PUT (sold CALL)")]:
        if lst:
            wr = sum(1 for t in lst if t.net_pnl > 0) / len(lst) * 100
            print(f"  {lbl:<18}: {len(lst):>3} | Win {wr:.1f}% | "
                  f"Net ₹{sum(t.net_pnl for t in lst):+,.0f}")

    print(f"\n  EXIT REASONS:  Short-SL={len(sl_t)}  Short-TP={len(tp_t)}  "
          f"Hard/DailyLim={len(hc_t)}")

    print(f"\n  PER-SETUP WIN RATES:")
    for sname in ("ORB", "PDH_PDL", "VWAP", "SMC"):
        st = [t for t in results.trades if t.setup == sname]
        if st:
            wr  = sum(1 for t in st if t.net_pnl > 0) / len(st) * 100
            sl  = sum(1 for t in st if "SL" in t.exit_reason)
            tp  = sum(1 for t in st if "TP" in t.exit_reason)
            hc  = sum(1 for t in st if "close" in t.exit_reason.lower())
            print(f"  {sname:<8}: {len(st):>3} trades | Win {wr:.1f}% | "
                  f"SL={sl}  TP={tp}  Hard={hc} | "
                  f"Net ₹{sum(t.net_pnl for t in st):+,.0f}")

    print(f"\n  CSV: {results.csv_path}")
    print()


# ── engine_v3: ORB + PDH/PDL + VWAP + SMC (options) ─────────────────────────

def _run_v4(kite, index: str, from_date: str, to_date: str,
            spread: bool = False, capital: float = 0) -> None:
    from engine_v3.backtest import V3BacktestEngine
    engine  = V3BacktestEngine(kite, index=index, spread=spread, capital=capital)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary(from_date, to_date))
    print()

    if not results.trades:
        print("  No trades generated.")
        print()
        return

    print("  TRADE LOG")
    print("  " + "─" * 178)
    print(
        f"  {'#':>3}  {'Date':<12} {'Setup':<8} {'Dir':<5} {'Bias':<8} "
        f"{'SigT':<8} {'EntT':<8} "
        f"{'EntSpot':>9} {'SL':>9} {'TP':>9} "
        f"{'EntPrem':>8} {'ExPrem':>8} "
        f"{'P1Px':>7} {'P2Px':>7} "
        f"{'Lots':>4} {'NetP&L':>10} {'%':>7}  Reason"
    )
    print("  " + "─" * 178)
    for i, t in enumerate(results.trades, 1):
        p1 = f"{t.partial1_px:>7.1f}" if t.partial1_lots > 0 else "      —"
        p2 = f"{t.partial2_px:>7.1f}" if t.partial2_lots > 0 else "      —"
        print(
            f"  {i:>3}  {t.date:<12} {t.setup:<8} {t.direction:<5} {t.bias:<8} "
            f"{t.signal_time:<8} {t.entry_time:<8} "
            f"{t.entry_spot:>9.2f} {t.sl_spot:>9.2f} {t.tp_spot:>9.2f} "
            f"{t.entry_premium:>8.2f} {t.exit_premium:>8.2f} "
            f"{p1} {p2} "
            f"{t.lots:>4} {t.net_pnl:>+10.2f} {t.pnl_pct:>+6.1f}%  {t.exit_reason}"
        )
    print("  " + "─" * 178)

    call_t = [t for t in results.trades if t.direction == "CALL"]
    put_t  = [t for t in results.trades if t.direction == "PUT"]
    sl_t   = [t for t in results.trades if "SL"      in t.exit_reason]
    tp_t   = [t for t in results.trades if "TP"      in t.exit_reason or "VWAP" in t.exit_reason]
    nm_t   = [t for t in results.trades if "No-move" in t.exit_reason]
    tr_t   = [t for t in results.trades if "Trail"   in t.exit_reason]
    hc_t   = [t for t in results.trades if "close"   in t.exit_reason.lower()
                                        or "limit"   in t.exit_reason.lower()]

    print(f"\n  DIRECTION:")
    for lst, lbl in [(call_t, "CALL"), (put_t, "PUT")]:
        if lst:
            wr = sum(1 for t in lst if t.net_pnl > 0) / len(lst) * 100
            print(f"  {lbl:<5}: {len(lst):>3} | Win {wr:.1f}% | "
                  f"Net ₹{sum(t.net_pnl for t in lst):+,.0f}")

    print(f"\n  EXIT REASONS:  SL={len(sl_t)}  TP/VWAP={len(tp_t)}  "
          f"Trail={len(tr_t)}  No-move={len(nm_t)}  Hard={len(hc_t)}")

    print(f"\n  PER-SETUP WIN RATES:")
    for sname in ("ORB", "PDH_PDL", "VWAP", "SMC"):
        st = [t for t in results.trades if t.setup == sname]
        if st:
            wr = sum(1 for t in st if t.net_pnl > 0) / len(st) * 100
            nm = sum(1 for t in st if "No-move" in t.exit_reason)
            sl = sum(1 for t in st if "SL" in t.exit_reason)
            print(f"  {sname:<8}: {len(st):>3} trades | Win {wr:.1f}% | "
                  f"SL={sl}  No-move={nm} | "
                  f"Net ₹{sum(t.net_pnl for t in st):+,.0f}")

    print(f"\n  CSV: {results.csv_path}")
    print()


# ── Simplified 2-setup backtest ───────────────────────────────────────────────

def _run_simple(kite, index: str, from_date: str, to_date: str) -> None:
    from backtest.simple_engine import SimplifiedEngine
    engine  = SimplifiedEngine(kite, index=index)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary())
    print()

    if results.trades:
        print("  TRADE LOG")
        print("  " + "─" * 130)
        print(
            f"  {'#':>3}  {'Date':<12} {'Setup':<10} {'Dir':<5} "
            f"{'ADX':>6} {'Gap%':>6} "
            f"{'EntPrem':>8} {'ExPrem':>8} "
            f"{'SL':>7} {'TP':>7} "
            f"{'Lots':>5} {'Hold':>5} "
            f"{'NetP&L':>10} {'%':>7}  Reason"
        )
        print("  " + "─" * 136)
        for i, t in enumerate(results.trades, 1):
            print(
                f"  {i:>3}  {t.date:<12} {t.setup_type:<10} {t.direction:<5} "
                f"{t.adx_val:>6.1f} {t.gap_pct:>6.2f} "
                f"{t.entry_premium:>8.2f} {t.exit_premium:>8.2f} "
                f"{t.sl_pts:>7.1f} {t.tp_pts:>7.1f} "
                f"{t.lots:>5} {t.candles_held:>5} "
                f"{t.pnl:>+10.2f} {t.pnl_pct:>+6.1f}%  {t.exit_reason}"
            )
        print("  " + "─" * 136)

        # Per-setup breakdown
        gap_t  = [t for t in results.trades if t.setup_type == "GAP_GO"]
        trnd_t = [t for t in results.trades if t.setup_type == "TREND_11AM"]
        print(f"\n  TOTAL  : {results.total_trades} trades | "
              f"Win {results.win_rate:.1f}% | "
              f"R:R {results.avg_rr:.2f}× | "
              f"Net ₹{results.total_pnl:+,.0f} ({results.total_return_pct:+.1f}%)")
        if gap_t:
            gw = sum(1 for t in gap_t if t.pnl > 0)
            print(f"  GAP&GO : {len(gap_t)} trades | Win {gw/len(gap_t)*100:.1f}% | "
                  f"Net ₹{sum(t.pnl for t in gap_t):+,.0f}")
        if trnd_t:
            tw = sum(1 for t in trnd_t if t.pnl > 0)
            print(f"  11AM   : {len(trnd_t)} trades | Win {tw/len(trnd_t)*100:.1f}% | "
                  f"Net ₹{sum(t.pnl for t in trnd_t):+,.0f}")
        print(f"\n  Signal CSV: {results.csv_path}")
    else:
        print("  No trades generated.")
        print("  Check: BACKTEST_VIX in .env, date range, and index data availability.")
    print()


# ── V3: 4-Indicator Signal + Dynamic Exit ────────────────────────────────────

def _run_v3(kite, index: str, from_date: str, to_date: str) -> None:
    from engine_v2.backtest import BacktestEngineV3
    engine  = BacktestEngineV3(kite, index=index)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary())
    print()

    if results.trades:
        hdr = (
            f"  {'#':>3}  {'Date':<12} {'Dir':<5} {'Gap%':>6} "
            f"{'ADX':>5} {'RSI':>5} {'Rng×':>5} "
            f"{'EntPx':>7} {'High':>7} {'Trail':>7} "
            f"{'P1L':>4} {'P1Px':>6} {'P2L':>4} {'P2Px':>6} "
            f"{'FinPx':>7} {'Reason':<18} {'NetP&L':>10} {'%':>7}"
        )
        print("  TRADE LOG")
        print("  " + "─" * 155)
        print(hdr)
        print("  " + "─" * 155)
        for i, t in enumerate(results.trades, 1):
            print(
                f"  {i:>3}  {t.date:<12} {t.direction:<5} {t.gap_pct:>6.2f} "
                f"{t.adx_val:>5.1f} {t.rsi_val:>5.1f} {t.range_ratio:>5.2f} "
                f"{t.entry_premium:>7.2f} {t.highest_premium:>7.2f} "
                f"{t.trailing_sl_at_exit:>7.2f} "
                f"{t.partial1_lots:>4} {t.partial1_premium:>6.2f} "
                f"{t.partial2_lots:>4} {t.partial2_premium:>6.2f} "
                f"{t.final_premium:>7.2f} {t.final_reason:<18} "
                f"{t.net_pnl:>+10.2f} {t.net_pnl_pct:>+6.1f}%"
            )
        print("  " + "─" * 155)
        print(f"\n  Signal CSV: {results.csv_path}")
    else:
        print("  No trades generated.")
        print("  Possible causes: no gap >0.3% days, or 4-indicator filter too strict.")
    print()


# ── V3 + ML Phase 2 simulation ────────────────────────────────────────────────

def _run_v3_ml(kite, index: str, from_date: str, to_date: str) -> None:
    from engine_v2.backtest import BacktestEngineV3
    engine  = BacktestEngineV3(kite, index=index)

    print("  Running Phase 1 backtest (collecting features)...")
    results = engine.run(from_date=from_date, to_date=to_date)

    if not results.trades:
        print("  No trades generated.")
        return

    print(f"  Phase 1 complete: {results.total_trades} trades collected.")
    print()

    # ── Phase 1 summary ──────────────────────────────────────────────────────
    print("  ─────────────────────────────────────────────────────")
    print("  PHASE 1 BASELINE (no ML)")
    print("  ─────────────────────────────────────────────────────")
    print(results.summary())
    print()

    # ── ML Phase 2 simulation ────────────────────────────────────────────────
    print("  Running ML Phase 2 simulation (XGBoost, online learning)...")
    sim = engine.simulate_ml_phase2(results.trades)

    if "error" in sim:
        print(f"  ML simulation failed: {sim['error']}")
        return

    p1 = sim["phase1"]
    p2 = sim["phase2"]
    sk = sim["skipped_summary"]

    print()
    print("  ─────────────────────────────────────────────────────")
    print("  PHASE 2 RESULTS (ML filter active from trade 41)")
    print("  ─────────────────────────────────────────────────────")
    print(f"  Trades taken   : {p2['n']}  (filtered out {p1['n'] - p2['n']})")
    print(f"  Win Rate       : {p2['win_rate']:.1f}%  "
          f"(was {p1['win_rate']:.1f}%  Δ {p2['win_rate']-p1['win_rate']:+.1f}pp)")
    print(f"  Total P&L      : ₹{p2['total_pnl']:+,.0f}  "
          f"(was ₹{p1['total_pnl']:+,.0f}  Δ ₹{p2['total_pnl']-p1['total_pnl']:+,.0f})")
    print(f"  Avg R:R        : {p2['avg_rr']:.2f}×  (was {p1['avg_rr']:.2f}×)")
    print(f"  Max Drawdown   : {p2['max_dd']:.1f}%  (was {p1['max_dd']:.1f}%)")
    print(f"  Return %       : {p2['total_pnl']/100000*100:.1f}%  "
          f"(was {p1['total_pnl']/100000*100:.1f}%)")

    if sk["n"] > 0:
        print()
        print(f"  ── Skipped trades analysis ({sk['n']} filtered out by ML) ──")
        print(f"  Win rate if taken : {sk['win_rate']:.1f}%  "
              f"(ML correctly skipped low-quality trades)")
        print(f"  Avg P&L if taken  : ₹{sk['total_pnl']/sk['n']:+,.0f} per trade  "
              f"({'bad' if sk['total_pnl']/sk['n'] < 0 else 'acceptable'})")

    print()
    print("  ── Feature Importance (top 6) ───────────────────────")
    imp = sim.get("feature_importance", {})
    for feat, val in list(imp.items())[:6]:
        bar = "█" * int(val * 40)
        print(f"  {feat:<25} {val:.4f}  {bar}")

    print()
    print(f"  Signal CSV: {results.csv_path}")
    print()


# ── V2: Gap & Go + Dynamic Trailing Exit ────────────────────────────────────

def _run_v2(kite, index: str, from_date: str, to_date: str) -> None:
    from backtest.engine_v2 import BacktestEngineV2
    engine  = BacktestEngineV2(kite, index=index)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary())
    print()

    if results.trades:
        print("  TRADE LOG")
        print("  " + "─" * 148)
        print(
            f"  {'#':>3}  {'Date':<12} {'Dir':<5} {'Gap%':>6} "
            f"{'Prem':>7} {'High':>7} {'Trail':>7} "
            f"{'Lots':>5} {'5mC':>4} "
            f"{'PartLots':>8} {'PartPx':>7} "
            f"{'FinalPx':>8} {'FinalReason':<16} "
            f"{'NetP&L':>10} {'%':>7}"
        )
        print("  " + "─" * 148)
        for i, t in enumerate(results.trades, 1):
            print(
                f"  {i:>3}  {t.date:<12} {t.direction:<5} {t.gap_pct:>6.2f} "
                f"{t.entry_premium:>7.2f} {t.highest_premium:>7.2f} {t.trailing_sl_at_exit:>7.2f} "
                f"{t.lots:>5} {t.candles_held_5min:>4} "
                f"{t.partial_lots:>8} {t.partial_premium:>7.2f} "
                f"{t.final_premium:>8.2f} {t.final_exit_reason:<16} "
                f"{t.net_pnl:>+10.2f} {t.net_pnl_pct:>+6.1f}%"
            )
        print("  " + "─" * 148)
        print(f"\n  TOTAL  : {results.total_trades} trades | "
              f"Win {results.win_rate:.1f}% | "
              f"R:R {results.avg_rr:.2f}× | "
              f"Net ₹{results.total_pnl:+,.0f} ({results.total_return_pct:+.1f}%)")
        print(f"\n  Signal CSV: {results.csv_path}")
    else:
        print("  No trades generated.")
    print()


# ── Conviction + Momentum-Confirmed backtest ──────────────────────────────────

def _run_conviction(kite, index: str, from_date: str, to_date: str) -> None:
    from config.settings import (
        MIN_CONDITIONS, RISK_REWARD_RATIO,
        SOFT_TARGET_PCT, HARD_TARGET_PCT, STOP_LOSS_PCT,
    )

    from backtest.engine import BacktestEngine
    engine  = BacktestEngine(kite, index=index)
    results = engine.run(from_date=from_date, to_date=to_date)

    print()
    print(results.summary())
    print()

    if results.trades:
        print("  TRADE LOG  (net P&L after all costs)")
        print("  " + "─" * 120)
        print(
            f"  {'#':>3}  {'Date':<12} {'Dir':<5} {'Score':>6} "
            f"{'EntryPrem':>10} {'ExitPrem':>10} "
            f"{'Qty':>5} {'GrossP&L':>10} {'Theta':>8} {'Slip':>7} {'NetP&L':>10} {'%':>7}  Reason"
        )
        print("  " + "─" * 126)
        for i, t in enumerate(results.trades, 1):
            print(
                f"  {i:>3}  {t.date:<12} {t.direction:<5} {t.conviction_score:>5} "
                f"{t.entry_premium:>10.2f} {t.exit_premium:>10.2f} "
                f"{t.quantity:>5} {t.gross_pnl:>+10.2f} "
                f"{-t.theta_cost:>+8.0f} {-t.slippage_cost:>+7.0f} "
                f"{t.pnl:>+10.2f} {t.pnl_pct:>+6.1f}%  {t.exit_reason}"
            )
        print("  " + "─" * 126)
        print(f"\n  Total: {results.total_trades} trades | "
              f"Win rate: {results.win_rate:.1f}% | "
              f"Net P&L: ₹{results.total_pnl:+,.2f} ({results.total_return_pct:+.2f}%)")
    else:
        print("  No trades generated.")
    print()


if __name__ == "__main__":
    main()
