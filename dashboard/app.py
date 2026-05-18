"""
Streamlit live dashboard.
Launch with: streamlit run dashboard/app.py --server.port $DASHBOARD_PORT
All display parameters come from config/settings.py → .env.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from config.settings import (
    ACTIVE_INDEX,
    CANDLE_INTERVAL,
    DASHBOARD_REFRESH,
    PAPER_MODE,
    RISK_REWARD_RATIO,
    STOP_LOSS_PCT,
    TARGET_PCT,
    TRADING_CAPITAL,
    TRADE_LOG_FILE,
    SIGNAL_LOG_FILE,
)

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return pd.DataFrame()


def run_dashboard() -> None:
    if not _STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title=f"Trading Engine — {ACTIVE_INDEX}",
        page_icon="📈",
        layout="wide",
    )

    # ── Header ──────────────────────────────────────────────────────────
    mode_badge = "🟡 PAPER MODE" if PAPER_MODE else "🔴 LIVE MODE"
    st.title(f"Trading Engine — {ACTIVE_INDEX}  {mode_badge}")
    st.caption(
        f"Capital: ₹{TRADING_CAPITAL:,.0f}  |  "
        f"R:R = 1:{RISK_REWARD_RATIO:.1f}  |  "
        f"SL {STOP_LOSS_PCT*100:.0f}%  →  Target {TARGET_PCT*100:.0f}%  |  "
        f"Candle: {CANDLE_INTERVAL}"
    )

    # ── Key metrics row ─────────────────────────────────────────────────
    trades_df = _load_csv(TRADE_LOG_FILE)

    if not trades_df.empty and "pnl" in trades_df.columns:
        total_pnl    = trades_df["pnl"].sum()
        total_trades = len(trades_df)
        wins         = (trades_df["pnl"] > 0).sum()
        win_rate     = wins / total_trades * 100 if total_trades > 0 else 0.0
    else:
        total_pnl = total_trades = win_rate = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily PnL",     f"₹{total_pnl:+,.2f}")
    col2.metric("Trades Today",  int(total_trades))
    col3.metric("Win Rate",      f"{win_rate:.1f}%")
    col4.metric("Capital",       f"₹{TRADING_CAPITAL:,.0f}")

    st.divider()

    # ── Trade log ───────────────────────────────────────────────────────
    st.subheader("Trade Log")
    if not trades_df.empty:
        st.dataframe(trades_df.tail(50), use_container_width=True)
    else:
        st.info("No trades recorded yet. Trade log will appear here once positions are taken.")

    # ── Signal log ──────────────────────────────────────────────────────
    st.subheader("Signal Log")
    signals_df = _load_csv(SIGNAL_LOG_FILE)
    if not signals_df.empty:
        st.dataframe(signals_df.tail(100), use_container_width=True)
    else:
        st.info("No signals logged yet.")

    # ── Auto-refresh ─────────────────────────────────────────────────────
    st.caption(f"Auto-refreshes every {DASHBOARD_REFRESH}s")
    time.sleep(DASHBOARD_REFRESH)
    st.rerun()


if __name__ == "__main__":
    run_dashboard()
