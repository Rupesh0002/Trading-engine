"""
Trading Engine — Live Dashboard
Reads directly from GitHub raw URLs so it can run anywhere
without a local copy of the engine.

Run:
    pip install streamlit pandas plotly requests
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import io
import json

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Data sources ──────────────────────────────────────────────────────────────
TRADES_CSV = "https://raw.githubusercontent.com/Rupesh0002/Trading-engine/main/trades_log.csv"
STATE_JSON = "https://raw.githubusercontent.com/Rupesh0002/Trading-engine/main/state.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Engine",
    page_icon="📈",
    layout="wide",
)


# ── Data loading (cached 5 min) ───────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_trades() -> pd.DataFrame:
    try:
        r = requests.get(TRADES_CSV, timeout=10)
        r.raise_for_status()
        if not r.text.strip() or r.text.strip().count("\n") == 0:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        # Build datetime for sorting / equity curve
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
            )
            df = df.sort_values("datetime").reset_index(drop=True)
        if "pnl" in df.columns:
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_state() -> dict:
    try:
        r = requests.get(STATE_JSON, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pnl_color(val):
    if isinstance(val, (int, float)):
        color = "#1a9641" if val >= 0 else "#d7191c"
        return f"color: {color}; font-weight: bold"
    return ""


def _result_color(val):
    if str(val).upper() == "WIN":
        return "background-color: #d4edda; color: #155724"
    if str(val).upper() == "LOSS":
        return "background-color: #f8d7da; color: #721c24"
    return ""


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("📈 Trading Engine — Live Dashboard")
st.caption("Auto-refreshes every 5 minutes  ·  Data from GitHub")

df     = load_trades()
state  = load_state()

# ── Top metrics ───────────────────────────────────────────────────────────────
st.subheader("Overview")

if df.empty or "pnl" not in df.columns:
    total_trades = 0
    win_rate     = 0.0
    total_pnl    = 0.0
    total_wins   = 0
else:
    closed = df[df["result"].isin(["WIN", "LOSS"])] if "result" in df.columns else df
    total_trades = len(closed)
    wins         = int((closed["result"] == "WIN").sum()) if "result" in closed.columns else 0
    total_wins   = wins
    win_rate     = wins / total_trades * 100 if total_trades else 0.0
    total_pnl    = float(df["pnl"].sum())

capital_now = float(state.get("running_capital", 100_000))
daily_pnl   = float(state.get("daily_pnl", 0.0))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades",  total_trades)
c2.metric("Win Rate",      f"{win_rate:.1f}%")
c3.metric("Total P&L",     f"₹{total_pnl:+,.0f}",  delta=None)
c4.metric("Today's P&L",   f"₹{daily_pnl:+,.0f}")
c5.metric("Capital",       f"₹{capital_now:,.0f}")

st.divider()

# ── Open positions ─────────────────────────────────────────────────────────────
st.subheader("Open Positions")

positions = state.get("open_positions", [])
if not positions:
    st.info("No open positions right now.")
else:
    for pos in positions:
        direction = pos.get("direction", "")
        index     = pos.get("index", "")
        arrow     = "🟢" if direction == "CALL" else "🔴"
        entry_p   = pos.get("entry_premium", 0)
        sl        = pos.get("stop_loss", 0)
        target    = pos.get("target", 0)
        lots      = pos.get("lots", 0)
        strike    = pos.get("strike", "")
        entry_t   = pos.get("entry_time", "")

        with st.container(border=True):
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric(f"{arrow} {index} {direction}", f"Strike {strike}")
            h2.metric("Entry Premium", f"₹{entry_p:.2f}")
            h3.metric("Stop Loss",     f"₹{sl:.2f}")
            h4.metric("Target",        f"₹{target:.2f}")
            h5.metric("Lots",          lots)
            st.caption(f"Entered: {entry_t}")

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")

if df.empty or "pnl" not in df.columns or total_trades == 0:
    st.info("No trades yet — equity curve will appear here.")
else:
    plot_df = df[df["pnl"] != 0].copy()
    plot_df["cumulative_pnl"] = plot_df["pnl"].cumsum()
    x_axis = plot_df.get("datetime", plot_df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=plot_df["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color="#2196F3", width=2),
        marker=dict(
            size=6,
            color=plot_df["pnl"].apply(lambda v: "#1a9641" if v >= 0 else "#d7191c"),
        ),
        hovertemplate="<b>%{x}</b><br>Cumulative P&L: ₹%{y:+,.0f}<extra></extra>",
        name="Cumulative P&L",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_title="Cumulative P&L (₹)",
        xaxis_title="",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        yaxis=dict(gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Trade table ───────────────────────────────────────────────────────────────
st.subheader("Last 50 Trades")

if df.empty or total_trades == 0:
    st.info("No trades yet.")
else:
    display_cols = [c for c in [
        "date", "time", "index", "direction",
        "strike", "lots",
        "entry_premium", "exit_premium", "pnl",
        "result", "exit_reason",
        "adx", "rsi",
    ] if c in df.columns]

    last50 = df.tail(50)[display_cols].copy()

    styled = (
        last50.style
        .applymap(_result_color, subset=["result"] if "result" in last50.columns else [])
        .applymap(_pnl_color,    subset=["pnl"]    if "pnl"    in last50.columns else [])
        .format({"pnl": "₹{:+,.2f}", "entry_premium": "₹{:.2f}", "exit_premium": "₹{:.2f}"},
                na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=500)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
mode = "📄 PAPER" if state.get("paper_mode", True) else "💰 LIVE"
st.caption(
    f"{mode}  ·  "
    f"Trades: {total_trades}  ·  "
    f"Wins: {total_wins}  ·  "
    f"Data: [trades_log.csv]({TRADES_CSV})  ·  "
    f"[state.json]({STATE_JSON})"
)
