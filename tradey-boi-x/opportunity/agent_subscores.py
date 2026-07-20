"""
Multi-Agent Performance Evaluation System (v5)

Groups the 18 existing signal adjustments inside decide() into 7 named agent
"councils". Each agent produces a subscore that is logged alongside every
ELITE/STRONG BUY signal in signal_log.json.

Weekly, agent_weight_update() reads resolved outcomes and computes how well
each agent's subscore predicted wins vs losses. Weights are saved to
config/agent_weights.json and picked up by the next decide() call.

The weighted shadow score runs in parallel to the existing scoring pipeline —
it does NOT replace the current alerts until it proves a higher profit factor
in the walk-forward sim (via challenge_mode_comparison()).

Agent groupings:
  technical   — ML probability + core technicals (EMA, RSI, volume, breakout)
  regime      — market regime, sector rotation, fear/greed, relative strength
  fundamental — fundamental health (P/E, FCF, debt/equity)
  sentiment   — news sentiment + velocity (VADER + LM lexicon)
  risk        — support/resistance quality + multi-timeframe alignment
  momentum    — squeeze setups, gap patterns, multibagger potential
  smart_money — options flow, insider signals, short interest, VWAP, commodity
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR     = Path(__file__).parent.parent
WEIGHTS_FILE = BASE_DIR / "config" / "agent_weights.json"
LOG_FILE     = BASE_DIR / "signal_log.json"

AGENT_NAMES = [
    "technical", "regime", "fundamental",
    "sentiment", "risk", "momentum", "smart_money",
]

DEFAULT_WEIGHTS = {name: 1.0 for name in AGENT_NAMES}

WIN_OUTCOMES = {"WIN", "HIT_TARGET", "EXPIRED_GAIN"}


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_agent_weights() -> dict[str, float]:
    try:
        cfg = json.loads(WEIGHTS_FILE.read_text())
        weights = cfg.get("weights", {})
        return {k: float(weights.get(k, 1.0)) for k in AGENT_NAMES}
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def _save_agent_weights(weights: dict[str, float], meta: dict | None = None):
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    try:
        existing = json.loads(WEIGHTS_FILE.read_text())
    except Exception:
        pass
    history = existing.get("weight_history", [])
    history.append({
        "date":    datetime.now().strftime("%Y-%m-%d"),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        **(meta or {}),
    })
    existing.update({
        "weights":        {k: round(v, 4) for k, v in weights.items()},
        "last_updated":   datetime.now().strftime("%Y-%m-%d"),
        "weight_history": history[-24:],   # keep ~6 months of weekly history
        **(meta or {}),
    })
    WEIGHTS_FILE.write_text(json.dumps(existing, indent=2))


# ─── Subscore extraction from decide() result ─────────────────────────────────

def compute_subscores(result: dict) -> dict[str, int]:
    """
    Extract per-agent subscores from a decide() result dict.
    Returns a dict with one key per agent (integer score, can be negative).
    """
    return result.get("subscores", {k: 0 for k in AGENT_NAMES})


# ─── Weighted shadow score ────────────────────────────────────────────────────

def weighted_score(subscores: dict[str, int],
                   weights: dict[str, float] | None = None) -> float:
    """
    Compute a weighted composite score from per-agent subscores.
    Uses learned weights; falls back to equal weighting.
    Result is on the same 0-12 integer-ish scale as the existing score.
    """
    w = weights if weights is not None else load_agent_weights()
    raw = sum(subscores.get(agent, 0) * w.get(agent, 1.0) for agent in AGENT_NAMES)
    total_weight = sum(w.get(a, 1.0) for a in AGENT_NAMES)
    mean_w = total_weight / len(AGENT_NAMES) if AGENT_NAMES else 1.0
    return raw / mean_w if mean_w != 0 else raw


# ─── Agent weight learner ─────────────────────────────────────────────────────

def agent_weight_update(min_trades: int = 15) -> dict:
    """
    Weekly agent weight updater.

    For each agent, computes:
      directional_accuracy = fraction of trades where
        (agent_subscore > 0 AND outcome=WIN) OR (agent_subscore <= 0 AND outcome=LOSS)

    Weights are set to the accuracy, then normalised so their mean = 1.0
    (preserving the overall score scale).

    Safety: requires >= min_trades resolved ELITE/STRONG BUY entries.
    Agents without enough signal are kept at their current weight.

    Returns a summary dict suitable for Discord reporting.
    """
    try:
        entries = json.loads(LOG_FILE.read_text())
    except Exception:
        return {"skipped": True, "reason": "signal_log.json unreadable"}

    resolved = [
        e for e in entries
        if e.get("outcome") is not None
        and e.get("tier") in ("ELITE", "STRONG BUY")
        and e.get("subscores")
    ]

    if len(resolved) < min_trades:
        result = {
            "skipped": True,
            "reason":  f"Only {len(resolved)} resolved ELITE/SB entries with subscores (need ≥{min_trades})",
            "resolved_n": len(resolved),
        }
        _save_agent_weights(load_agent_weights(), meta=result)
        return result

    recent = resolved[-60:]   # cap at last 60 trades for rolling window

    current_weights = load_agent_weights()
    new_weights: dict[str, float] = {}
    agent_stats: dict[str, dict]  = {}

    for agent in AGENT_NAMES:
        scores   = [e["subscores"].get(agent, 0) for e in recent]
        outcomes = [e["outcome"] in WIN_OUTCOMES for e in recent]

        # Only compute accuracy for trades where the agent had a non-zero opinion
        opinionated = [(s, o) for s, o in zip(scores, outcomes) if s != 0]
        if len(opinionated) < 5:
            new_weights[agent] = current_weights.get(agent, 1.0)
            agent_stats[agent] = {"accuracy": None, "n": 0, "weight": new_weights[agent]}
            continue

        correct = sum(1 for s, o in opinionated if (s > 0 and o) or (s < 0 and not o))
        accuracy = correct / len(opinionated)
        # Floor at 0.35 (random baseline for a directional bet is 0.5, but regime can structurally tilt)
        # Cap at 1.5× current to prevent one noisy good week spiking a weight
        new_weights[agent] = max(0.35, min(accuracy, current_weights.get(agent, 1.0) * 1.5))
        agent_stats[agent] = {
            "accuracy": round(accuracy, 3),
            "n":        len(opinionated),
            "weight":   round(new_weights[agent], 4),
        }

    # Normalise so mean weight = 1.0 (preserves score scale)
    mean_w = sum(new_weights.values()) / len(new_weights) if new_weights else 1.0
    if mean_w > 0:
        new_weights = {k: round(v / mean_w, 4) for k, v in new_weights.items()}
        for agent in agent_stats:
            if agent_stats[agent].get("weight") is not None:
                agent_stats[agent]["weight"] = new_weights[agent]

    meta = {
        "recent_n":    len(recent),
        "agent_stats": agent_stats,
    }
    _save_agent_weights(new_weights, meta=meta)
    return {"skipped": False, "weights": new_weights, "agent_stats": agent_stats, "n": len(recent)}


# ─── Challenge mode: compare unweighted vs weighted PF ───────────────────────

def challenge_mode_comparison(weights: dict[str, float] | None = None) -> dict:
    """
    Walk-forward comparison: run both the current flat score and the weighted
    score against all resolved entries, compute PF for each.

    Returns a dict with:
      current_pf   — profit factor using existing flat score
      weighted_pf  — profit factor using agent-weighted score
      promote      — True if weighted_pf >= current_pf AND weighted_pf >= 1.2
      summary_lines — list of strings for Discord
    """
    try:
        entries = json.loads(LOG_FILE.read_text())
    except Exception:
        return {"error": "signal_log.json unreadable"}

    resolved = [
        e for e in entries
        if e.get("outcome") is not None
        and e.get("tier") in ("ELITE", "STRONG BUY")
        and e.get("subscores")
        and e.get("actual_pct") is not None
    ]

    if len(resolved) < 15:
        return {"skipped": True, "reason": f"Only {len(resolved)} resolved entries with subscores+outcomes"}

    w = weights if weights is not None else load_agent_weights()

    def pf_from_trades(trades):
        gains  = sum(t["actual_pct"] for t in trades if t["actual_pct"] >= 0)
        losses = abs(sum(t["actual_pct"] for t in trades if t["actual_pct"] < 0))
        return gains / losses if losses > 0 else float("inf")

    # Current system: no weighting (uses existing score as-is)
    current_pf = pf_from_trades(resolved)

    # Weighted system: apply weights to subscores, filter using same thresholds
    from opportunity.agent_subscores import weighted_score
    weighted_filtered = [
        e for e in resolved
        if weighted_score(e["subscores"], w) >= 7   # same sb_base threshold
    ]
    weighted_pf = pf_from_trades(weighted_filtered) if weighted_filtered else 0.0

    promote = (
        len(weighted_filtered) >= 10
        and weighted_pf >= current_pf
        and weighted_pf >= 1.2
    )

    summary = [
        "**🥊 Challenge Mode — Flat vs Weighted Score**",
        f"  Flat score:    n={len(resolved)},     PF={current_pf:.3f}",
        f"  Weighted score: n={len(weighted_filtered)}, PF={weighted_pf:.3f}",
        f"  → {'✅ PROMOTE weighted score' if promote else '❌ Keep flat score (no improvement yet)'}",
    ]
    return {
        "skipped":          False,
        "current_pf":       round(current_pf,  3),
        "weighted_pf":      round(weighted_pf, 3),
        "current_n":        len(resolved),
        "weighted_n":       len(weighted_filtered),
        "promote":          promote,
        "summary_lines":    summary,
    }
