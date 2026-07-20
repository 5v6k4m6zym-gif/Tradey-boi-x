"""
Pattern Similarity Agent (v5)

Compares the current setup's indicator fingerprint against every resolved
ELITE/STRONG BUY signal in signal_log.json.  Signals similar to past winners
get a score boost; signals similar to past losers get a penalty.

Feature vector (7 dimensions, all normalised to [0, 1]):
  0  rsi_norm       — RSI / 100
  1  vol_ratio_norm — min(vol_ratio / 5, 1.0)
  2  prob_norm      — (prob - 0.40) / 0.40  (0.40→0, 0.80→1)
  3  score_norm     — score / 12
  4  breakout       — 1 if 52-week breakout, else 0
  5  quality_norm   — quality_score / 100
  6  base_score_norm— base_score / 11  (max base_score is ~11)

Cosine similarity is used; a perfect match = 1.0.

Score adjustments:
  avg top-3 similarity >= 0.95  AND majority winners  → +2
  avg top-3 similarity >= 0.85  AND majority winners  → +1
  avg top-3 similarity >= 0.80  AND majority losers   → -1
  avg top-3 similarity >= 0.90  AND majority losers   → -2
  otherwise                                           →  0
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
LOG_FILE = BASE_DIR / "signal_log.json"

WIN_OUTCOMES = {"WIN", "HIT_TARGET", "EXPIRED_GAIN"}

TOP_K = 5   # compare against top-K most similar historical setups


def _feature_vector(entry: dict) -> list[float] | None:
    """
    Build a normalised 7-d feature vector from a signal_log entry or a
    decide() result dict.  Returns None if essential fields are missing.
    """
    features = entry.get("features") or {}

    rsi         = entry.get("rsi") or features.get("rsi")
    vol_ratio   = features.get("vol_ratio")
    prob        = entry.get("prob")
    score       = entry.get("score")
    breakout    = int(bool(features.get("breakout") or features.get("breakout_52w")))
    quality     = entry.get("quality_score") or features.get("quality_score")
    base_score  = entry.get("base_score")

    if any(v is None for v in [rsi, prob, score]):
        return None

    return [
        float(rsi)  / 100.0,
        min(float(vol_ratio) / 5.0, 1.0) if vol_ratio else 0.5,
        max(0.0, min((float(prob) - 0.40) / 0.40, 1.0)),
        min(float(score) / 12.0, 1.0),
        float(breakout),
        float(quality) / 100.0 if quality else 0.5,
        min(float(base_score) / 11.0, 1.0) if base_score else 0.5,
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _load_history() -> list[dict]:
    try:
        entries = json.loads(LOG_FILE.read_text())
    except Exception:
        return []
    return [
        e for e in entries
        if e.get("outcome") is not None
        and e.get("tier") in ("ELITE", "STRONG BUY")
    ]


def pattern_similarity_signal(current: dict,
                               history: list[dict] | None = None) -> tuple[int, str]:
    """
    Compare `current` (a decide() result dict or a signal_log entry) against
    past resolved ELITE/STRONG BUY setups.

    Returns (score_adjustment: int, reason: str).

    Adjustment table:
      +2  — very high similarity (≥0.95) to mostly-winning setups
      +1  — high similarity (≥0.85) to mostly-winning setups
       0  — insufficient history or ambiguous
      -1  — high similarity (≥0.80) to mostly-losing setups
      -2  — very high similarity (≥0.90) to mostly-losing setups
    """
    if history is None:
        history = _load_history()

    if len(history) < 10:
        return 0, ""

    cur_vec = _feature_vector(current)
    if cur_vec is None:
        return 0, ""

    # Compute similarity against all historical entries that have vectors
    similarities: list[tuple[float, bool]] = []
    for h in history:
        h_vec = _feature_vector(h)
        if h_vec is None:
            continue
        sim     = _cosine(cur_vec, h_vec)
        is_win  = h.get("outcome") in WIN_OUTCOMES
        similarities.append((sim, is_win))

    if not similarities:
        return 0, ""

    similarities.sort(key=lambda x: -x[0])
    top_k = similarities[:TOP_K]

    avg_sim   = sum(s for s, _ in top_k) / len(top_k)
    win_count = sum(1 for _, w in top_k if w)
    win_rate  = win_count / len(top_k)
    majority_win  = win_rate >= 0.6
    majority_loss = win_rate <= 0.4

    if avg_sim >= 0.95 and majority_win:
        return +2, f"Pattern match: top-{TOP_K} similarity {avg_sim:.2f} — closely resembles past winners (WR={win_rate:.0%})"
    if avg_sim >= 0.85 and majority_win:
        return +1, f"Pattern match: top-{TOP_K} similarity {avg_sim:.2f} — similar to past winning setups (WR={win_rate:.0%})"
    if avg_sim >= 0.90 and majority_loss:
        return -2, f"Pattern caution: top-{TOP_K} similarity {avg_sim:.2f} — closely resembles past losers (WR={win_rate:.0%})"
    if avg_sim >= 0.80 and majority_loss:
        return -1, f"Pattern caution: top-{TOP_K} similarity {avg_sim:.2f} — similar to past losing setups (WR={win_rate:.0%})"

    return 0, ""


def similar_past_trades(current: dict,
                        n: int = 3,
                        history: list[dict] | None = None) -> list[dict]:
    """
    Returns the N most-similar resolved past trades.
    Useful for dashboard display and the Discord alert narrative.
    """
    if history is None:
        history = _load_history()

    cur_vec = _feature_vector(current)
    if cur_vec is None or not history:
        return []

    scored = []
    for h in history:
        h_vec = _feature_vector(h)
        if h_vec is None:
            continue
        scored.append({
            "similarity": round(_cosine(cur_vec, h_vec), 3),
            "ticker":     h.get("ticker"),
            "date":       h.get("signal_date"),
            "outcome":    h.get("outcome"),
            "actual_pct": h.get("actual_pct"),
            "tier":       h.get("tier"),
        })

    scored.sort(key=lambda x: -x["similarity"])
    return scored[:n]
