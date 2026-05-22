"""
Options flow scorer — PCR, IV Rank, OI buildup.
Maximum 35 points:  PCR (0-10) + IV Rank (0-15) + OI buildup (0-10).

Hard block: IV Rank > IV_RANK_MAX (default 80) → options too expensive.
When PCR or OI data is None (no real data in backtest), those components score 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import IV_RANK_MAX, IV_RANK_SWEET_MAX, IV_RANK_SWEET_MIN


@dataclass
class OptionsScore:
    pcr_score:    int            # 0-10
    ivrank_score: int            # 0-15
    oi_score:     int            # 0-10
    total:        int            # 0-35
    hard_blocked: bool           # True when IV Rank > IV_RANK_MAX
    iv_rank:      float
    pcr:          Optional[float]
    direction:    str


def score_options(
    direction: str,
    pcr: Optional[float],
    iv_rank: Optional[float],
    ce_oi_change: Optional[float] = None,
    pe_oi_change: Optional[float] = None,
) -> OptionsScore:
    """
    Score options flow conditions.

    direction     : "CALL" or "PUT"
    pcr           : Put-Call Ratio (None → 0 pts, does not hard-block)
    iv_rank       : IV Rank 0-100 (None → assume 30, sweet spot)
    ce_oi_change  : CE OI % change since last candle (None → 0 pts)
    pe_oi_change  : PE OI % change since last candle (None → 0 pts)
    """
    # ── PCR ───────────────────────────────────────────────────────────────────
    # High PCR (fear) confirms CALL; low PCR (greed) confirms PUT.
    if pcr is not None:
        if direction == "CALL":
            if pcr > 1.3:
                pcr_score = 10
            elif pcr > 1.1:
                pcr_score = 7
            elif pcr >= 0.9:
                pcr_score = 4
            else:
                pcr_score = 0
        else:  # PUT
            if pcr < 0.7:
                pcr_score = 10
            elif pcr < 0.9:
                pcr_score = 7
            elif pcr <= 1.1:
                pcr_score = 4
            else:
                pcr_score = 0
    else:
        pcr_score = 0

    # ── IV Rank ───────────────────────────────────────────────────────────────
    # Sweet spot 20-50: options fairly priced — best edge for buyers.
    # < 20: IV crush risk (buying when premium is at 52w low).
    # > 80: hard block — premium too expensive to buy.
    iv = iv_rank if iv_rank is not None else 30.0   # neutral assumption when unavailable
    hard_blocked = iv > IV_RANK_MAX

    if hard_blocked:
        ivrank_score = 0
    elif IV_RANK_SWEET_MIN <= iv <= IV_RANK_SWEET_MAX:  # 20-50
        ivrank_score = 15
    elif iv < IV_RANK_SWEET_MIN:                        # < 20: cheap but risky
        ivrank_score = 0
    elif iv <= 65:                                      # 50-65: still good
        ivrank_score = 10
    else:                                               # 65-80: expensive but tradeable
        ivrank_score = 5

    # ── OI Buildup ────────────────────────────────────────────────────────────
    # CALL: CE OI decreasing + PE OI increasing → bears covering, bulls building → 10 pts
    # PUT:  CE OI increasing + PE OI decreasing → bears adding, bulls covering → 10 pts
    oi_score = 0
    if ce_oi_change is not None and pe_oi_change is not None:
        if direction == "CALL":
            ce_dec = ce_oi_change < 0
            pe_inc = pe_oi_change > 0
            if ce_dec and pe_inc:
                oi_score = 10
            elif ce_dec or pe_inc:
                oi_score = 5
        else:  # PUT
            ce_inc = ce_oi_change > 0
            pe_dec = pe_oi_change < 0
            if ce_inc and pe_dec:
                oi_score = 10
            elif ce_inc or pe_dec:
                oi_score = 5

    return OptionsScore(
        pcr_score=pcr_score,
        ivrank_score=ivrank_score,
        oi_score=oi_score,
        total=pcr_score + ivrank_score + oi_score,
        hard_blocked=hard_blocked,
        iv_rank=round(iv, 1),
        pcr=pcr,
        direction=direction,
    )
