"""
Adaptive pattern memory — learns WHY trades succeed or fail.

How it works:
  Every closed trade is bucketed by its market conditions and the outcome
  (WIN / LOSS) is recorded. On the next similar trade, the engine blends
  the XGBoost ML confidence with the historical win-rate for those conditions.

Decision logic:
  count < MIN_SAMPLES (5)  → trust ML only (not enough data)
  count >= 5, win_rate < 0.30  → HARD BLOCK regardless of ML score
  count >= 5, win_rate >= 0.30 → BLEND: (1-w)*ML + w*hist_win_rate
     blend-weight grows with sample count: 5 samples → 20%, 20+ samples → 50%

Near-miss detection (when signal returns WAIT):
  If call_score or put_score == 3 (one condition short), check the bucket
  for the direction that almost fired. If win_rate > 0.65 on 5+ past trades,
  log a WARNING so the user can manually review.

Bucket dimensions (6):
  index       : NIFTY | BANKNIFTY | SENSEX
  direction   : CALL | PUT
  adx_bucket  : flat(<12) | moderate(12-20) | strong(20+)
  rsi_bucket  : bear(<40) | neutral(40-60) | bull(60+)
  dte_bucket  : expiry(0-2d) | near(3-7d) | far(7+d)
  time_bucket : morning(10-12h) | afternoon(12-15h)

Storage: ml/models/pattern_memory.json  (plain JSON, human-readable)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_MEMORY_PATH    = os.path.join("ml", "models", "pattern_memory.json")
_MIN_SAMPLES    = 5      # minimum trades in a bucket before memory is trusted
_HARD_BLOCK_WR  = 0.30   # win-rate below this → block trade regardless of ML
_NEAR_MISS_WR   = 0.65   # win-rate above this → flag near-miss opportunity
_MAX_BLEND_W    = 0.50   # max weight given to pattern memory (vs ML)


# ─────────────────────────────────────────────────────────────────────────────
# Bucket helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adx_bucket(adx: float) -> str:
    if adx < 12:   return "flat"
    if adx < 20:   return "moderate"
    return "strong"


def _rsi_bucket(rsi: float) -> str:
    if rsi < 40:   return "bear"
    if rsi < 60:   return "neutral"
    return "bull"


def _dte_bucket(dte: int) -> str:
    if dte <= 2:   return "expiry"
    if dte <= 7:   return "near"
    return "far"


def _time_bucket(hour: int) -> str:
    return "morning" if hour < 12 else "afternoon"


def _make_key(index: str, direction: str, adx: float,
              rsi: float, dte: int, hour: int) -> str:
    return "|".join([
        index,
        direction,
        _adx_bucket(adx),
        _rsi_bucket(rsi),
        _dte_bucket(dte),
        _time_bucket(hour),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Memory store
# ─────────────────────────────────────────────────────────────────────────────

class PatternMemory:
    """
    Persistent dict keyed by condition bucket.
    Each bucket stores: wins, losses, total_pnl.
    """

    def __init__(self, path: str = _MEMORY_PATH) -> None:
        self._path = path
        self._data: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
                logger.info("Pattern memory loaded: %d buckets", len(self._data))
            except Exception as exc:
                logger.warning("Could not load pattern memory: %s — starting fresh", exc)
                self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save pattern memory: %s", exc)

    def record(self, key: str, pnl: float) -> None:
        """Record one trade outcome into the bucket."""
        b = self._data.setdefault(key, {"wins": 0, "losses": 0, "total_pnl": 0.0})
        if pnl > 0:
            b["wins"]  += 1
        else:
            b["losses"] += 1
        b["total_pnl"] = round(b["total_pnl"] + pnl, 2)
        self._save()

    def lookup(self, key: str) -> Dict[str, Any]:
        """
        Returns stats for a bucket.
        Fields: wins, losses, count, win_rate, avg_pnl.
        Returns count=0 if bucket is empty.
        """
        b = self._data.get(key, {})
        wins   = int(b.get("wins",   0))
        losses = int(b.get("losses", 0))
        count  = wins + losses
        return {
            "key":      key,
            "wins":     wins,
            "losses":   losses,
            "count":    count,
            "win_rate": wins / count if count else 0.0,
            "avg_pnl":  round(b.get("total_pnl", 0.0) / count, 2) if count else 0.0,
        }

    def all_buckets(self) -> Dict[str, Dict]:
        """Return full memory for inspection / reporting."""
        return {k: self.lookup(k) for k in self._data}


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive scorer
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMemory:
    """
    Main adaptive layer.  Used by MLPredictor and the scheduler.

    Typical flow:
      1. scheduler._enter_trade()  calls  score()  to blend ML + history
      2. scheduler._close_position() calls  record_outcome()
      3. scheduler._candle_cycle() calls  check_near_miss() on WAIT signals
    """

    def __init__(self) -> None:
        self.memory = PatternMemory()

    # ── Record outcome after a trade closes ───────────────────────────────

    def record_outcome(
        self,
        index: str,
        direction: str,
        signal_details: Dict[str, Any],
        dte: int,
        entry_hour: int,
        pnl: float,
    ) -> None:
        """
        Record the outcome of a closed trade and log a WHY analysis.
        Called by scheduler._close_position().
        """
        adx  = float(signal_details.get("adx") or 0)
        rsi  = float(signal_details.get("rsi") or 50)
        key  = _make_key(index, direction, adx, rsi, dte, entry_hour)

        # Record into persistent memory
        self.memory.record(key, pnl)
        stats = self.memory.lookup(key)

        outcome_str = "WIN" if pnl > 0 else "LOSS"
        self._log_why(outcome_str, index, direction, signal_details, dte, entry_hour, pnl, stats)

    def _log_why(
        self,
        outcome: str,
        index: str,
        direction: str,
        d: Dict[str, Any],
        dte: int,
        hour: int,
        pnl: float,
        stats: Dict[str, Any],
    ) -> None:
        """Log a clear explanation of why this trade won or lost."""
        adx  = float(d.get("adx") or 0)
        rsi  = float(d.get("rsi") or 50)
        vwap_dist = float(d.get("vwap_distance") or 0)
        fib_dist  = float(d.get("fib_distance") or 0)
        conds     = int(d.get("conditions_met") or 4)
        label     = d.get("fib_label") or d.get("pdh_pdl_label", "")

        lines = [
            f"[ADAPTIVE] {outcome} — {index} {direction} | PnL=₹{pnl:+,.0f}",
            f"  Conditions : {conds}/5  |  Level: {label}",
            f"  ADX={adx:.1f} ({_adx_bucket(adx)})  |  RSI={rsi:.1f} ({_rsi_bucket(rsi)})",
            f"  VWAP dist={vwap_dist:+.1f}  |  PDH/PDL dist={fib_dist:.1f}",
            f"  DTE={dte}d ({_dte_bucket(dte)})  |  Time={hour:02d}:xx ({_time_bucket(hour)})",
        ]

        if stats["count"] >= _MIN_SAMPLES:
            lines.append(
                f"  Memory: {stats['wins']}/{stats['count']} wins "
                f"({stats['win_rate']:.0%}) in this bucket | "
                f"avg PnL=₹{stats['avg_pnl']:+,.0f}"
            )
            if outcome == "LOSS":
                if stats["win_rate"] < 0.40:
                    lines.append(
                        f"  ⚠ This condition bucket historically loses "
                        f"({stats['win_rate']:.0%} win rate) — "
                        f"future trades here will be blocked or reduced."
                    )
            else:  # WIN
                if stats["win_rate"] > 0.65:
                    lines.append(
                        f"  ✓ Strong historical pattern — this bucket "
                        f"wins {stats['win_rate']:.0%} of the time."
                    )
        else:
            lines.append(
                f"  Memory: {stats['count']}/{_MIN_SAMPLES} samples collected "
                f"(need {_MIN_SAMPLES} to activate adaptive logic)"
            )

        for line in lines:
            logger.info(line)

    # ── Score a new trade signal ──────────────────────────────────────────

    def score(
        self,
        index: str,
        direction: str,
        signal_details: Dict[str, Any],
        dte: int,
        entry_hour: int,
        ml_confidence: float,
    ) -> Tuple[float, str]:
        """
        Blend XGBoost ML confidence with historical pattern win-rate.

        Returns:
          (final_confidence, reason_string)

        Logic:
          - count <  MIN_SAMPLES : return ml_confidence unchanged
          - win_rate < HARD_BLOCK: return 0.0 (hard block)
          - else                 : blend = (1-w)*ml + w*win_rate
            where w grows from 0.20 (5 samples) to MAX_BLEND_W (20+ samples)
        """
        adx  = float(signal_details.get("adx") or 0)
        rsi  = float(signal_details.get("rsi") or 50)
        key  = _make_key(index, direction, adx, rsi, dte, entry_hour)
        stats = self.memory.lookup(key)
        count = stats["count"]

        if count < _MIN_SAMPLES:
            reason = (
                f"pattern memory: {count}/{_MIN_SAMPLES} samples — ML only ({ml_confidence:.2f})"
            )
            return ml_confidence, reason

        win_rate = stats["win_rate"]

        # Hard block: historically very bad conditions
        if win_rate < _HARD_BLOCK_WR:
            reason = (
                f"HARD BLOCK — pattern memory: {stats['wins']}/{count} wins "
                f"({win_rate:.0%}) in bucket [{key}]. "
                f"Threshold: {_HARD_BLOCK_WR:.0%}"
            )
            logger.warning("[ADAPTIVE] %s", reason)
            return 0.0, reason

        # Blend weight grows with sample count: 5→20%, 20+→50%
        blend_w = min(_MAX_BLEND_W, 0.10 + 0.02 * count)
        combined = (1.0 - blend_w) * ml_confidence + blend_w * win_rate
        combined = round(combined, 4)

        reason = (
            f"pattern memory: {stats['wins']}/{count} wins ({win_rate:.0%}) | "
            f"ML={ml_confidence:.2f} + hist={win_rate:.2f} "
            f"(w={blend_w:.0%}) → combined={combined:.2f}"
        )
        logger.debug("[ADAPTIVE] %s", reason)
        return combined, reason

    # ── Near-miss detection ───────────────────────────────────────────────

    def check_near_miss(
        self,
        index: str,
        signal_details: Dict[str, Any],
        dte: int,
        entry_hour: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Called when signal engine returns WAIT.
        Checks if either CALL or PUT in similar conditions historically wins.

        Returns a near-miss dict if found, else None.
        Near-miss dict: {direction, win_rate, count, avg_pnl, key, reason}
        """
        adx = float(signal_details.get("adx") or 0)
        rsi = float(signal_details.get("rsi") or 50)

        call_score = int(signal_details.get("call_score") or 0)
        put_score  = int(signal_details.get("put_score") or 0)

        # Only check directions that were close to firing (score == 3, one short)
        candidates = []
        if call_score == 3:
            candidates.append("CALL")
        if put_score == 3:
            candidates.append("PUT")

        if not candidates:
            return None

        best = None
        for direction in candidates:
            key   = _make_key(index, direction, adx, rsi, dte, entry_hour)
            stats = self.memory.lookup(key)

            if stats["count"] < _MIN_SAMPLES:
                continue
            if stats["win_rate"] < _NEAR_MISS_WR:
                continue

            # Found a near-miss with strong historical win rate
            info = {
                "direction": direction,
                "win_rate":  stats["win_rate"],
                "count":     stats["count"],
                "avg_pnl":   stats["avg_pnl"],
                "key":       key,
                "reason": (
                    f"Near-miss {direction}: {stats['wins']}/{stats['count']} wins "
                    f"({stats['win_rate']:.0%}) in bucket [{key}] | "
                    f"avg PnL=₹{stats['avg_pnl']:+,.0f}. "
                    f"Indicators were 3/5 — one condition short."
                ),
            }
            # Keep the one with higher win rate if both qualify
            if best is None or stats["win_rate"] > best["win_rate"]:
                best = info

        if best:
            logger.warning(
                "[ADAPTIVE] NEAR-MISS %s on %s: %s",
                best["direction"], index, best["reason"],
            )

        return best

    # ── Inspection / reporting ────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of all pattern memory buckets."""
        buckets = self.memory.all_buckets()
        if not buckets:
            return "Pattern memory is empty — no trades recorded yet."

        lines = [
            f"Pattern Memory — {len(buckets)} buckets",
            f"{'Bucket':<55} {'W':>4} {'L':>4} {'WR':>6} {'AvgPnL':>9}",
            "-" * 82,
        ]
        for key, s in sorted(buckets.items(), key=lambda x: -x[1]["count"]):
            marker = ""
            if s["count"] >= _MIN_SAMPLES:
                if s["win_rate"] < _HARD_BLOCK_WR:
                    marker = " ⛔"
                elif s["win_rate"] >= _NEAR_MISS_WR:
                    marker = " ✓"
            lines.append(
                f"{key:<55} {s['wins']:>4} {s['losses']:>4} "
                f"{s['win_rate']:>5.0%}  ₹{s['avg_pnl']:>+8,.0f}{marker}"
            )
        return "\n".join(lines)
