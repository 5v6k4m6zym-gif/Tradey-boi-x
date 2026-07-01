"""
Phase 6 — Strategy Challenger Sandbox
Runs a read-only shadow copy of the scoring weights in parallel with production,
backtests it against historical signals, and produces a comparison report.

Guardrails:
  - ZERO write access to engine.py, scanner.py, or signal_log.json
  - Cannot send Discord alerts
  - All output is JSON files in reports/challenger/
  - Human approval required before any recommendation is acted on

Feature flag: ENABLE_STRATEGY_CHALLENGER  (default: false → complete no-op)
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from opportunity.config import (
    ENABLE_STRATEGY_CHALLENGER,
    WEIGHTS as PRODUCTION_WEIGHTS,
)
from opportunity.backtester import (
    compute_metrics,
    WIN_OUTCOMES,
    _resolved,
    _load_log,
)

BASE_DIR      = Path(__file__).parent.parent
CHALLENGER_DIR = BASE_DIR / "reports" / "challenger"

# ── Default candidate weight adjustment: slightly amplify high-return factor ──
DEFAULT_CANDIDATE_WEIGHTS: dict[str, float] = {
    "expected_return":    0.40,
    "technical_strength": 0.18,
    "volume_expansion":   0.13,
    "momentum":           0.10,
    "news_catalyst":      0.09,
    "institutional":      0.05,
    "risk_reward":        0.05,
}


# ─── Shadow scoring ───────────────────────────────────────────────────────────

def _challenger_opportunity_score(
    entry: dict,
    candidate_weights: dict[str, float],
) -> float | None:
    """
    Re-score a resolved signal log entry using candidate weights.
    Uses the original `prob` as a proxy for technical strength and momentum
    (since we don't have raw component scores in the log).

    Returns a 0–100 score or None if data is insufficient.
    """
    prob = float(entry.get("prob", 0) or 0)
    if prob == 0:
        return None

    actual_pct = float(entry.get("actual_pct", 0) or 0)
    is_win     = entry.get("outcome") in WIN_OUTCOMES

    # Proxy component scores from available log fields
    expected_return_score    = min(100.0, max(0.0, abs(actual_pct) * 250))
    technical_strength_score = prob * 100
    volume_expansion_score   = prob * 80       # approximation
    momentum_score           = 60.0 if is_win else 30.0
    news_catalyst_score      = prob * 70
    institutional_score      = prob * 65
    risk_reward_score        = min(100.0, abs(actual_pct) * 500) if is_win else 10.0

    components = {
        "expected_return":    expected_return_score,
        "technical_strength": technical_strength_score,
        "volume_expansion":   volume_expansion_score,
        "momentum":           momentum_score,
        "news_catalyst":      news_catalyst_score,
        "institutional":      institutional_score,
        "risk_reward":        risk_reward_score,
    }

    total_weight  = sum(candidate_weights.values())
    weighted_sum  = sum(
        components.get(k, 50.0) * v
        for k, v in candidate_weights.items()
    )
    return round(weighted_sum / total_weight, 1) if total_weight > 0 else None


def _prod_opportunity_score(entry: dict) -> float | None:
    """Production score using the same proxy method but PRODUCTION_WEIGHTS."""
    return _challenger_opportunity_score(entry, PRODUCTION_WEIGHTS)


# ─── Comparison report ────────────────────────────────────────────────────────

def compare_strategies(
    entries:           list[dict],
    candidate_weights: dict[str, float],
    min_score:         float = 60.0,
) -> dict[str, Any]:
    """
    Compare production weights vs candidate weights against resolved signal history.
    Simulates how many trades each strategy would have *selected* (score ≥ min_score)
    and the resulting metrics.

    Returns a comparison dict — never modifies production weights or log.
    """
    prod_selected  = []
    chal_selected  = []

    for e in entries:
        p_score = _prod_opportunity_score(e)
        c_score = _challenger_opportunity_score(e, candidate_weights)

        if p_score is not None and p_score >= min_score:
            prod_selected.append(e)
        if c_score is not None and c_score >= min_score:
            chal_selected.append(e)

    prod_metrics = compute_metrics(prod_selected)
    chal_metrics = compute_metrics(chal_selected)

    # Delta (challenger - production)
    def _delta(key: str) -> float:
        return round(
            float(chal_metrics.get(key, 0)) - float(prod_metrics.get(key, 0)), 4
        )

    def _pct_change(key: str) -> float:
        base = float(prod_metrics.get(key, 0))
        return round((_delta(key) / base) if base != 0 else 0.0, 4)

    comparison_keys = [
        "win_rate", "profit_factor", "sharpe_ratio", "expectancy_r",
        "avg_gain_pct", "avg_loss_pct", "max_drawdown_pct",
        "annualised_return_pct",
    ]

    recommendation = _recommendation(prod_metrics, chal_metrics)

    return {
        "generated_at":         datetime.utcnow().isoformat(),
        "total_history":        len(entries),
        "min_score_threshold":  min_score,
        "production": {
            "weights":       PRODUCTION_WEIGHTS,
            "selected":      len(prod_selected),
            "metrics":       prod_metrics,
        },
        "challenger": {
            "weights":       candidate_weights,
            "selected":      len(chal_selected),
            "metrics":       chal_metrics,
        },
        "delta": {k: _delta(k) for k in comparison_keys},
        "pct_change": {k: _pct_change(k) for k in comparison_keys},
        "recommendation": recommendation,
        "requires_human_approval": True,
    }


def _recommendation(prod: dict, chal: dict) -> str:
    """
    Rule-based recommendation — purely advisory, never auto-deployed.
    """
    p_exp  = float(prod.get("expectancy_r", 0))
    c_exp  = float(chal.get("expectancy_r", 0))
    p_sha  = float(prod.get("sharpe_ratio", 0))
    c_sha  = float(chal.get("sharpe_ratio", 0))
    p_dd   = float(prod.get("max_drawdown_pct", 0))
    c_dd   = float(chal.get("max_drawdown_pct", 0))

    wins_on_expectancy = c_exp > p_exp + 0.05
    wins_on_sharpe     = c_sha > p_sha + 0.10
    lower_drawdown     = c_dd  < p_dd  - 0.5

    score = sum([wins_on_expectancy, wins_on_sharpe, lower_drawdown])

    if score >= 2:
        return ("CONSIDER_ADOPTING — challenger weights show improved expectancy "
                "and/or Sharpe. Recommend 30-day paper trade before any live change.")
    elif score == 1:
        return ("BORDERLINE — challenger shows marginal improvement on one metric. "
                "Continue paper trading for more data.")
    else:
        return ("KEEP_PRODUCTION — challenger weights show no clear improvement. "
                "Production weights remain preferred.")


# ─── Report persistence ───────────────────────────────────────────────────────

def save_challenger_report(comparison: dict) -> Path:
    CHALLENGER_DIR.mkdir(parents=True, exist_ok=True)
    fname = CHALLENGER_DIR / f"challenger_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    fname.write_text(json.dumps(comparison, indent=2))
    return fname


# ─── Public API ───────────────────────────────────────────────────────────────

def run_challenger(
    candidate_weights: dict[str, float] | None = None,
    min_score: float = 60.0,
) -> dict | None:
    """
    Run the challenger sandbox. Returns comparison dict or None if disabled.

    Parameters
    ----------
    candidate_weights : dict mapping weight keys to floats (must sum ≈ 1.0)
                        Defaults to DEFAULT_CANDIDATE_WEIGHTS when None.
    min_score         : minimum opportunity score threshold for trade selection.

    This function is ENTIRELY READ-ONLY with respect to the production system.
    Output is written only to reports/challenger/ as JSON.
    """
    if not ENABLE_STRATEGY_CHALLENGER:
        return None

    weights = candidate_weights if candidate_weights is not None \
              else DEFAULT_CANDIDATE_WEIGHTS

    entries = _resolved(_load_log())
    if not entries:
        print("  🔬 Challenger: no resolved entries to compare against.")
        return None

    comparison  = compare_strategies(entries, weights, min_score)
    report_path = save_challenger_report(comparison)

    prod_m = comparison["production"]["metrics"]
    chal_m = comparison["challenger"]["metrics"]
    print(
        f"  🔬 Challenger: prod win_rate {prod_m.get('win_rate',0)*100:.0f}%  "
        f"vs chal {chal_m.get('win_rate',0)*100:.0f}% | "
        f"Recommendation: {comparison['recommendation'][:40]}…"
    )
    print(f"     └─ Report: {report_path}")

    return comparison
