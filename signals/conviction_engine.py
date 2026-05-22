"""
Conviction Engine — combines Momentum, Options Flow, and Price Structure scorers.
Returns a ConvictionResult with total score (0-100), direction, lot size, and all sub-scores.

Score bands → lot sizes:
  < 65  → NO TRADE  (score too low — signal not high-conviction)
  65-71 → 1 lot     (risk 1% of capital)
  72-79 → 2 lots    (risk 2%)
  80-86 → 3 lots    (risk 3%)
  87-92 → 4 lots    (risk 4%)
  93+   → 5 lots    (risk 5%, max position)

Hard blocks (any → direction = WAIT):
  - ADX < per-index threshold   (momentum scorer)
  - IV Rank > IV_RANK_MAX=80   (options scorer)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import CONVICTION_MIN_SCORE, INDEX_CONFIG, IV_RANK_MAX
from signals.momentum_scorer import MomentumScore, score_momentum
from signals.options_scorer import OptionsScore, score_options
from signals.structure_scorer import StructureScore, score_structure

_LOT_BANDS = [
    (93, 5),
    (87, 4),
    (80, 3),
    (72, 2),
    (65, 1),
    (0,  0),
]


def score_to_lots(score: int) -> int:
    """Map conviction score (0-100) to lot count (0-5)."""
    for threshold, lots in _LOT_BANDS:
        if score >= threshold:
            return lots
    return 0


@dataclass
class ConvictionResult:
    direction:    str             # "CALL", "PUT", or "WAIT"
    total_score:  int             # 0-100 (may exceed 100 on perfect setup)
    lot_size:     int             # 0-5
    hard_blocked: bool
    block_reason: str
    momentum:     MomentumScore
    options:      OptionsScore
    structure:    StructureScore

    def __bool__(self) -> bool:
        return self.direction in ("CALL", "PUT")

    def __repr__(self) -> str:
        return (
            f"Conviction({self.direction} | score={self.total_score} | "
            f"lots={self.lot_size} | "
            f"M={self.momentum.total}/40 O={self.options.total}/35 S={self.structure.total}/25)"
        )


class ConvictionEngine:
    """
    Evaluates both CALL and PUT conviction scores and returns the stronger direction.
    If the winning score < CONVICTION_MIN_SCORE or any hard block fires → WAIT.
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        index: str = "NIFTY",
        pcr: Optional[float] = None,
        iv_rank: Optional[float] = None,
        ce_oi_change: Optional[float] = None,
        pe_oi_change: Optional[float] = None,
        pdh: Optional[float] = None,
        pdl: Optional[float] = None,
        weekly_high: Optional[float] = None,
        weekly_low: Optional[float] = None,
        candle_idx: int = -2,
    ) -> ConvictionResult:
        """
        df          : OHLCV DataFrame (50+ bars for reliable indicator warm-up)
        index       : "NIFTY", "BANKNIFTY", or "SENSEX"
        pcr         : Put-Call Ratio (optional — 0 pts if None)
        iv_rank     : India VIX IV Rank 0-100 (optional — assumes 30 if None)
        ce_oi_change: CE OI % change vs last candle (optional)
        pe_oi_change: PE OI % change vs last candle (optional)
        pdh / pdl   : Previous day high/low (inferred from df if None)
        weekly_high : Week-to-date high (inferred from last 5 days if None)
        weekly_low  : Week-to-date low
        """
        cfg           = INDEX_CONFIG.get(index, INDEX_CONFIG["NIFTY"])
        adx_threshold = cfg.get("adx_threshold", 20.0)

        # Infer PDH/PDL if not provided
        _pdh = pdh if pdh is not None else float(df["high"].max())
        _pdl = pdl if pdl is not None else float(df["low"].min())

        # Infer weekly H/L from rolling 5-day window if not provided
        # ~8 candles/day × 5 days = 40 candles — good rolling-week approximation
        if weekly_high is None or weekly_low is None:
            _weekly_high = float(df.tail(40)["high"].max())
            _weekly_low  = float(df.tail(40)["low"].min())
        else:
            _weekly_high = weekly_high
            _weekly_low  = weekly_low

        close = float(df.iloc[candle_idx]["close"])

        # ── Score both directions, pick the stronger ───────────────────────────
        scored: dict[str, dict] = {}
        for direction in ("CALL", "PUT"):
            mom    = score_momentum(df, direction, adx_threshold=adx_threshold, candle_idx=candle_idx)
            opt    = score_options(direction, pcr, iv_rank, ce_oi_change, pe_oi_change)
            struct = score_structure(close, direction, _pdh, _pdl, _weekly_high, _weekly_low, index)

            total        = mom.total + opt.total + struct.total
            hard_blocked = mom.hard_blocked or opt.hard_blocked
            block_reason = ""
            if mom.hard_blocked:
                block_reason = f"ADX {mom.adx_val:.1f} < {adx_threshold:.0f} (hard block)"
            elif opt.hard_blocked:
                block_reason = f"IV Rank {opt.iv_rank:.0f} > {IV_RANK_MAX:.0f} (hard block)"

            scored[direction] = {
                "total":        total,
                "hard_blocked": hard_blocked,
                "block_reason": block_reason,
                "momentum":     mom,
                "options":      opt,
                "structure":    struct,
            }

        # Both directions share the same ADX hard block — check once
        call_res = scored["CALL"]
        put_res  = scored["PUT"]

        # Hard block fires if momentum scorer blocks (same ADX value for both)
        if call_res["hard_blocked"]:
            best = call_res  # same block reason for both
            return ConvictionResult(
                direction="WAIT",
                total_score=best["total"],
                lot_size=0,
                hard_blocked=True,
                block_reason=best["block_reason"],
                momentum=best["momentum"],
                options=best["options"],
                structure=best["structure"],
            )

        # IV Rank hard block
        if call_res["options"].hard_blocked:
            return ConvictionResult(
                direction="WAIT",
                total_score=call_res["total"],
                lot_size=0,
                hard_blocked=True,
                block_reason=call_res["block_reason"],
                momentum=call_res["momentum"],
                options=call_res["options"],
                structure=call_res["structure"],
            )

        # Pick strongest direction
        if call_res["total"] >= put_res["total"]:
            best_dir, best = "CALL", call_res
        else:
            best_dir, best = "PUT", put_res

        total_score = best["total"]

        if total_score < CONVICTION_MIN_SCORE:
            return ConvictionResult(
                direction="WAIT",
                total_score=total_score,
                lot_size=0,
                hard_blocked=False,
                block_reason=f"Score {total_score} < min {CONVICTION_MIN_SCORE}",
                momentum=best["momentum"],
                options=best["options"],
                structure=best["structure"],
            )

        return ConvictionResult(
            direction=best_dir,
            total_score=total_score,
            lot_size=score_to_lots(total_score),
            hard_blocked=False,
            block_reason="",
            momentum=best["momentum"],
            options=best["options"],
            structure=best["structure"],
        )
