"""
Momentum scorer — EMA stack, RSI, ADX trend strength, VWAP distance.
Maximum 40 points:  EMA (0-10) + RSI (0-10) + ADX (0-10) + VWAP (0-10).

Hard block: ADX < ADX_THRESHOLD → hard_blocked=True.
Callers must check hard_blocked before using the score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    ADX_THRESHOLD,
    EMA_FAST_PERIOD,
    EMA_SLOW_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)
from signals.indicators import compute_adx, compute_ema, compute_rsi_wilder, compute_vwap

_EMA_LONG_PERIOD = 200  # third EMA for perfect-stack confirmation


@dataclass
class MomentumScore:
    ema_score:    int    # 0-10
    rsi_score:    int    # 0-10
    adx_score:    int    # 0-10
    vwap_score:   int    # 0-10
    total:        int    # 0-40
    hard_blocked: bool   # True when ADX < threshold — caller must skip trade
    adx_val:      float
    rsi_val:      float
    ema_fast:     float
    ema_slow:     float
    vwap_val:     float
    close:        float
    direction:    str


def score_momentum(
    df: pd.DataFrame,
    direction: str,
    adx_threshold: Optional[float] = None,
    candle_idx: int = -2,
) -> MomentumScore:
    """
    Score momentum conditions for the last closed candle (candle_idx=-2).
    direction      : "CALL" for bullish scoring, "PUT" for bearish.
    adx_threshold  : per-index override; defaults to ADX_THRESHOLD from .env.
    """
    thr    = adx_threshold if adx_threshold is not None else ADX_THRESHOLD
    candle = df.iloc[candle_idx]
    close  = float(candle["close"])

    # ── EMA stack ─────────────────────────────────────────────────────────────
    # 10 pts: close fully stacked above/below all 3 EMAs (fast > mid > slow)
    # 7 pts:  close above/below fast and mid EMAs (2-stack)
    # 4 pts:  close above/below fast EMA only
    # 0 pts:  price on wrong side of fast EMA
    ema_f  = compute_ema(df["close"], EMA_FAST_PERIOD)
    ema_s  = compute_ema(df["close"], EMA_SLOW_PERIOD)
    ema_l  = compute_ema(df["close"], _EMA_LONG_PERIOD)
    ef     = float(ema_f.iloc[candle_idx])
    es     = float(ema_s.iloc[candle_idx])
    el     = float(ema_l.iloc[candle_idx])

    if direction == "CALL":
        if close > ef > es > el:
            ema_score = 10
        elif close > ef > es:
            ema_score = 7
        elif close > ef:
            ema_score = 4
        else:
            ema_score = 0
    else:  # PUT
        if close < ef < es < el:
            ema_score = 10
        elif close < ef < es:
            ema_score = 7
        elif close < ef:
            ema_score = 4
        else:
            ema_score = 0

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_s   = compute_rsi_wilder(df["close"])
    rsi_val = 50.0
    if len(rsi_s) >= abs(candle_idx):
        v = float(rsi_s.iloc[candle_idx])
        if not np.isnan(v):
            rsi_val = v

    if direction == "CALL":
        if rsi_val > 65:
            rsi_score = 10
        elif rsi_val > 58:
            rsi_score = 7
        elif rsi_val > RSI_OVERSOLD:    # default 55
            rsi_score = 4
        else:
            rsi_score = 0
    else:  # PUT
        if rsi_val < 35:
            rsi_score = 10
        elif rsi_val < 42:
            rsi_score = 7
        elif rsi_val < RSI_OVERBOUGHT:  # default 45
            rsi_score = 4
        else:
            rsi_score = 0

    # ── ADX ───────────────────────────────────────────────────────────────────
    # Hard block below threshold; 10/7/4/0 for strength bands above threshold.
    adx_s   = compute_adx(df)
    adx_val = 0.0
    if len(adx_s) >= abs(candle_idx):
        v = float(adx_s.iloc[candle_idx])
        if not np.isnan(v):
            adx_val = v

    hard_blocked = adx_val < thr

    if adx_val >= 35:
        adx_score = 10
    elif adx_val >= 28:
        adx_score = 7
    elif adx_val >= thr:
        adx_score = 4
    else:
        adx_score = 0

    # ── VWAP distance ─────────────────────────────────────────────────────────
    # Price must be clearly on the right side of VWAP to score.
    # 10 pts: >0.3% away; 7 pts: >0.2%; 4 pts: >0.1%; 0: below threshold.
    vwap_s   = compute_vwap(df)
    vwap_val = close
    if len(vwap_s) >= abs(candle_idx):
        v = float(vwap_s.iloc[candle_idx])
        if not np.isnan(v) and v > 0:
            vwap_val = v

    dist_pct = (close - vwap_val) / vwap_val * 100.0  # positive = above VWAP

    if direction == "CALL":
        if dist_pct > 0.3:
            vwap_score = 10
        elif dist_pct > 0.2:
            vwap_score = 7
        elif dist_pct > 0.1:
            vwap_score = 4
        else:
            vwap_score = 0
    else:  # PUT
        neg = -dist_pct
        if neg > 0.3:
            vwap_score = 10
        elif neg > 0.2:
            vwap_score = 7
        elif neg > 0.1:
            vwap_score = 4
        else:
            vwap_score = 0

    return MomentumScore(
        ema_score=ema_score,
        rsi_score=rsi_score,
        adx_score=adx_score,
        vwap_score=vwap_score,
        total=ema_score + rsi_score + adx_score + vwap_score,
        hard_blocked=hard_blocked,
        adx_val=round(adx_val, 2),
        rsi_val=round(rsi_val, 2),
        ema_fast=round(ef, 2),
        ema_slow=round(es, 2),
        vwap_val=round(vwap_val, 2),
        close=round(close, 2),
        direction=direction,
    )
