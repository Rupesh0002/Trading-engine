"""
Multi-Timeframe (MTF) signal engine.

  Context   (15-min prior candles) — EMA20/50 trend, ADX, key S/R levels, gap.

  Watch signals — two types:
    ORB        15-min close beyond OR boundary + body-ratio + trend + space check.
    EMA_BOUNCE 15-min pullback to EMA20 then bounce in trend direction.

  Entry     (5-min today candles)  — momentum body or OR-retest confirmation.

Volume is intentionally NOT used — index instruments always report 0 on Kite.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import List, Optional

import pandas as pd

from signals.levels import SRLevel, detect_levels, nearest_resistance

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_BODY_RATIO_ORB   = 0.45   # ORB breakout candle body ≥ 45% of H-L range
_BODY_RATIO_EMA   = 0.52   # EMA bounce candle body ≥ 52% (tighter — needs conviction)
_SPACE_PCT        = 0.002  # next S/R must be ≥ 0.2% away (ORB only)
_ADX_MIN_ORB      = 15.0   # ADX floor for ORB signals
_ADX_MIN_EMA      = 22.0   # ADX floor for EMA bounce (higher — must be real trend)
_EMA_MIN_SL_PCT   = 0.002  # EMA bounce SL must be ≥ 0.2% of price (filters tiny candles)
_EMA_FAST         = 20     # EMA20 — used for trend + bounce detection
_EMA_SLOW         = 50     # EMA50 — used for trend direction
_ORB_BUFFER       = 0.001  # 0.1% close-beyond-OR to confirm ORB breakout
_OR_MIN_PCT       = 0.003  # skip OR < 0.3% (choppy open)
_OR_MAX_PCT       = 0.020  # skip OR > 2.0% (gap day)
_ENTRY_MAX_WAIT   = 8      # max 5-min candles to look for entry
_RETEST_ZONE_PCT  = 0.002  # within 0.2% of OR boundary = retest (ORB only)
_EMA_TOUCH_PCT    = 0.001  # candle low/high within 0.1% of EMA20 = "touching"


@dataclass
class MTFContext:
    or_high:   float
    or_low:    float
    or_valid:  bool
    trend:     str          # "BULL" | "BEAR" | "NEUTRAL"
    adx:       float
    ema20:     float = 0.0  # last known EMA20 from prior data
    levels:    List[SRLevel] = field(default_factory=list)
    gap_pct:   float = 0.0


@dataclass
class WatchSignal:
    direction:         str    # "CALL" | "PUT"
    trigger_time:      time   # 15-min bar START time (bar closes 15 min later)
    trigger_price:     float  # 15-min close
    or_high:           float
    or_low:            float
    or_range_pct:      float
    space_to_move:     float  # % gap to next S/R (ORB), or to candle SL (EMA)
    context:           MTFContext
    source:            str    = "ORB"   # "ORB" | "EMA_BOUNCE"
    suggested_sl_spot: float  = 0.0    # signal's SL hint (EMA_BOUNCE uses candle low/high)


@dataclass
class MTFEntry:
    entry_price: float    # 5-min close used as option entry reference spot
    entry_time:  time
    sl_spot:     float    # spot SL from 5-min swing low (CALL) or high (PUT)
    mode:        str      # "MOMENTUM" | "RETEST" | "EMA_BOUNCE"


# ── Engine ────────────────────────────────────────────────────────────────────

class MTFEngine:
    """
    Stateless computation engine.
    Call analyze_context() once per day, then check signals per 15-min candle,
    then find_entry_5min() when a watch signal fires.
    """

    def analyze_context(
        self,
        df_prior_15: pd.DataFrame,
        df_today_15: pd.DataFrame,
        index: str = "NIFTY",
    ) -> Optional[MTFContext]:
        """Build context from prior data (no look-ahead into today)."""
        if df_today_15 is None or df_today_15.empty:
            return None

        or_high, or_low, or_valid = self._extract_or(df_today_15)

        trend, adx, ema20 = self._compute_trend_adx(df_prior_15)

        spot = float(df_today_15.iloc[0]["open"]) if len(df_today_15) > 0 else (or_high + or_low) / 2
        levels = detect_levels(df_prior_15, spot, index=index) if (
            df_prior_15 is not None and not df_prior_15.empty
        ) else []

        gap_pct = self._compute_gap(df_prior_15, df_today_15)

        return MTFContext(
            or_high=or_high, or_low=or_low, or_valid=or_valid,
            trend=trend, adx=adx, ema20=ema20,
            levels=levels, gap_pct=gap_pct,
        )

    # ── ORB watch signal ──────────────────────────────────────────────────────

    def check_watch_signal(
        self,
        candle: pd.Series,
        context: MTFContext,
        signal_start: time = time(10, 0),
        signal_end:   time = time(13, 0),
    ) -> Optional[WatchSignal]:
        """
        ORB breakout signal: 15-min candle closes beyond OR with body + trend filter.
        """
        if not context.or_valid:
            return None

        ts = pd.Timestamp(candle["timestamp"])
        t  = ts.time()
        if t < signal_start or t >= signal_end:
            return None

        close = float(candle["close"])
        open_ = float(candle["open"])
        high  = float(candle["high"])
        low   = float(candle["low"])

        rng = high - low
        if rng <= 0 or abs(close - open_) / rng < _BODY_RATIO_ORB:
            return None
        if context.adx < _ADX_MIN_ORB:
            return None

        direction = None
        if close > context.or_high * (1 + _ORB_BUFFER):
            direction = "CALL"
        elif close < context.or_low * (1 - _ORB_BUFFER):
            direction = "PUT"
        if direction is None:
            return None

        if direction == "CALL" and close < open_:
            return None
        if direction == "PUT" and close > open_:
            return None

        if direction == "CALL" and context.trend == "BEAR":
            return None
        if direction == "PUT" and context.trend == "BULL":
            return None

        nearest = nearest_resistance(context.levels, close, direction)
        space = (
            (nearest - close) / close if direction == "CALL"
            else (close - nearest) / close
        )
        if space < _SPACE_PCT:
            logger.debug("[MTF/ORB] blocked: space=%.2f%% dir=%s close=%.2f nearest_SR=%.2f",
                         space * 100, direction, close, nearest)
            return None

        or_range_pct = (context.or_high - context.or_low) / context.or_low
        logger.info(
            "[MTF/ORB] %s | close=%.2f OR H=%.2f L=%.2f ADX=%.1f trend=%s space=%.2f%%",
            direction, close, context.or_high, context.or_low, context.adx, context.trend, space * 100,
        )
        return WatchSignal(
            direction=direction, trigger_time=t, trigger_price=close,
            or_high=context.or_high, or_low=context.or_low,
            or_range_pct=or_range_pct, space_to_move=space,
            context=context, source="ORB",
        )

    # ── EMA bounce watch signal ───────────────────────────────────────────────

    def check_ema_bounce_signal(
        self,
        candle:       pd.Series,
        prev_candle:  pd.Series,
        ema20_curr:   float,
        ema20_prev:   float,
        context:      MTFContext,
        signal_start: time = time(10, 0),
        signal_end:   time = time(14, 0),
    ) -> Optional[WatchSignal]:
        """
        EMA20 bounce: previous candle touched EMA20 (low ≤ EMA for BULL, high ≥ EMA for BEAR),
        current candle closes back above/below EMA with body ratio.
        Fires in BULL and BEAR trends — no S/R space filter (normal momentum trade).
        """
        if context.trend == "NEUTRAL" or context.adx < _ADX_MIN_EMA:
            return None

        ts = pd.Timestamp(candle["timestamp"])
        t  = ts.time()
        if t < signal_start or t >= signal_end:
            return None

        close = float(candle["close"])
        open_ = float(candle["open"])
        high  = float(candle["high"])
        low   = float(candle["low"])
        prev_low  = float(prev_candle["low"])
        prev_high = float(prev_candle["high"])

        rng = high - low
        if rng <= 0 or abs(close - open_) / rng < _BODY_RATIO_EMA:
            return None

        direction = None
        sl_hint   = 0.0

        if context.trend == "BULL":
            # Previous candle dipped to/below EMA20, current bounces above it
            touched = prev_low <= ema20_prev * (1 + _EMA_TOUCH_PCT)
            bounced = close > ema20_curr and close > open_
            if touched and bounced:
                direction = "CALL"
                sl_hint   = low  # this candle's low as SL

        elif context.trend == "BEAR":
            # Previous candle rose to/above EMA20, current closes back below it
            touched = prev_high >= ema20_prev * (1 - _EMA_TOUCH_PCT)
            bounced = close < ema20_curr and close < open_
            if touched and bounced:
                direction = "PUT"
                sl_hint   = high

        if direction is None:
            return None

        # Minimum SL distance: signal candle must be large enough to give real room
        min_sl_pts = close * _EMA_MIN_SL_PCT
        if abs(close - sl_hint) < min_sl_pts:
            logger.debug(
                "[MTF/EMA] blocked: SL distance %.1f pts < min %.1f pts  dir=%s",
                abs(close - sl_hint), min_sl_pts, direction,
            )
            return None

        or_range_pct = (context.or_high - context.or_low) / context.or_low if context.or_low > 0 else 0.0
        space = abs(close - sl_hint) / close if close > 0 else 0.0

        logger.info(
            "[MTF/EMA] %s | close=%.2f EMA20=%.2f ADX=%.1f trend=%s",
            direction, close, ema20_curr, context.adx, context.trend,
        )
        return WatchSignal(
            direction=direction, trigger_time=t, trigger_price=close,
            or_high=context.or_high, or_low=context.or_low,
            or_range_pct=or_range_pct, space_to_move=space,
            context=context, source="EMA_BOUNCE",
            suggested_sl_spot=sl_hint,
        )

    # ── 5-min entry ───────────────────────────────────────────────────────────

    def find_entry_5min(
        self,
        df_5: pd.DataFrame,
        watch: WatchSignal,
    ) -> Optional[MTFEntry]:
        """
        Scan 5-min candles after the watch signal fires.
        For EMA_BOUNCE: just need a momentum candle in direction.
        For ORB: momentum or OR-retest.
        The 15-min bar with timestamp T closes at T+15min → start from T+15.
        """
        if df_5 is None or df_5.empty:
            return None

        trigger_dt    = datetime(2000, 1, 1, watch.trigger_time.hour, watch.trigger_time.minute)
        entry_start_t = (trigger_dt + timedelta(minutes=15)).time()

        relevant = df_5[
            df_5["timestamp"].apply(lambda x: pd.Timestamp(x).time()) >= entry_start_t
        ].head(_ENTRY_MAX_WAIT)

        if relevant.empty:
            return None

        direction   = watch.direction
        or_boundary = watch.or_high if direction == "CALL" else watch.or_low
        recent_lows:  list[float] = []
        recent_highs: list[float] = []

        for _, c in relevant.iterrows():
            t5  = pd.Timestamp(c["timestamp"]).time()
            o5  = float(c["open"])
            h5  = float(c["high"])
            l5  = float(c["low"])
            c5  = float(c["close"])
            rng = h5 - l5

            recent_lows.append(l5)
            recent_highs.append(h5)

            if rng <= 0:
                continue
            body_ratio = abs(c5 - o5) / rng

            # Momentum candle in direction
            if direction == "CALL" and c5 > o5 and body_ratio >= _BODY_RATIO_ORB:
                mode = "EMA_BOUNCE" if watch.source == "EMA_BOUNCE" else "MOMENTUM"
                return MTFEntry(
                    entry_price=c5, entry_time=t5,
                    sl_spot=min(recent_lows), mode=mode,
                )
            if direction == "PUT" and c5 < o5 and body_ratio >= _BODY_RATIO_ORB:
                mode = "EMA_BOUNCE" if watch.source == "EMA_BOUNCE" else "MOMENTUM"
                return MTFEntry(
                    entry_price=c5, entry_time=t5,
                    sl_spot=max(recent_highs), mode=mode,
                )

            # ORB only: retest of OR boundary then bounce
            if watch.source == "ORB":
                in_retest = abs(c5 - or_boundary) / or_boundary < _RETEST_ZONE_PCT
                if in_retest:
                    if direction == "CALL" and c5 > o5:
                        return MTFEntry(
                            entry_price=c5, entry_time=t5,
                            sl_spot=min(recent_lows), mode="RETEST",
                        )
                    if direction == "PUT" and c5 < o5:
                        return MTFEntry(
                            entry_price=c5, entry_time=t5,
                            sl_spot=max(recent_highs), mode="RETEST",
                        )

        return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_or(self, df: pd.DataFrame) -> tuple[float, float, bool]:
        for _, row in df.iterrows():
            t = pd.Timestamp(row["timestamp"])
            if t.hour == 9 and t.minute == 15:
                h   = float(row["high"])
                l   = float(row["low"])
                rng = (h - l) / l if l > 0 else 0.0
                return h, l, _OR_MIN_PCT <= rng <= _OR_MAX_PCT
        return 0.0, 0.0, False

    def _compute_trend_adx(self, df: pd.DataFrame) -> tuple[str, float, float]:
        if df is None or len(df) < _EMA_SLOW + 5:
            return "NEUTRAL", 0.0, 0.0

        closes = df["close"].astype(float)
        ema_f  = closes.ewm(span=_EMA_FAST, adjust=False).mean()
        ema_s  = closes.ewm(span=_EMA_SLOW, adjust=False).mean()
        last   = closes.iloc[-1]
        ema_f_last = ema_f.iloc[-1]
        ema_s_last = ema_s.iloc[-1]

        if ema_f_last > ema_s_last and last > ema_f_last:
            trend = "BULL"
        elif ema_f_last < ema_s_last and last < ema_f_last:
            trend = "BEAR"
        else:
            trend = "NEUTRAL"

        adx = _compute_adx(df)
        return trend, adx, round(ema_f_last, 2)

    def _compute_gap(self, df_prior: pd.DataFrame, df_today: pd.DataFrame) -> float:
        if df_prior is None or df_prior.empty or df_today is None or df_today.empty:
            return 0.0
        prev_close = float(df_prior["close"].iloc[-1])
        today_open = float(df_today["open"].iloc[0])
        if prev_close == 0:
            return 0.0
        return (today_open - prev_close) / prev_close * 100


# ── Standalone ADX ────────────────────────────────────────────────────────────

def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period * 2:
        return 0.0

    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    n = len(h)

    tr_arr, dp_arr, dm_arr = [], [], []
    for i in range(1, n):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        tr_arr.append(tr)
        dp_arr.append(up if up > dn and up > 0 else 0.0)
        dm_arr.append(dn if dn > up and dn > 0 else 0.0)

    def smma(arr: list[float], p: int) -> list[float]:
        res = [0.0] * len(arr)
        if len(arr) < p:
            return res
        res[p - 1] = sum(arr[:p])
        for i in range(p, len(arr)):
            res[i] = res[i - 1] - res[i - 1] / p + arr[i]
        return res

    atr = smma(tr_arr, period)
    sdp = smma(dp_arr, period)
    sdm = smma(dm_arr, period)

    dx_list: list[float] = []
    for i in range(period - 1, len(atr)):
        if atr[i] == 0:
            continue
        dip = sdp[i] / atr[i] * 100
        dim = sdm[i] / atr[i] * 100
        tot = dip + dim
        if tot == 0:
            continue
        dx_list.append(abs(dip - dim) / tot * 100)

    if not dx_list:
        return 0.0
    tail = dx_list[-period:] if len(dx_list) >= period else dx_list
    return round(sum(tail) / len(tail), 2)
