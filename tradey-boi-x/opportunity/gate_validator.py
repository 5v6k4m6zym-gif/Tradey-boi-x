"""
Gate Validator — v4.0 Walk-Forward + Monte Carlo + Ensemble Validation.

Every proposed gate change (prob_floor / sb_base_score) must clear:

  1. Minimum sample size (≥15 resolved trades)
  2. Ensemble walk-forward (3 time splits, ≥2 must agree)
  3. Monte Carlo significance (beat ≥40% of 300 random-shuffle PFs)
  4. Multi-metric acceptance (PF and expectancy must not materially degrade)

A change that fails any check is automatically rejected.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Sequence

from .metrics import compute_metrics

MONTE_CARLO_RUNS  = 300
MIN_SAMPLE_SIZE   = 15
ENSEMBLE_SPLITS   = [(0.60, 0.40), (0.65, 0.35), (0.70, 0.30)]
ENSEMBLE_REQUIRED = 2
MC_MIN_PERCENTILE = 0.40
PF_MAX_DROP_RATIO = 0.95
EXP_MAX_DROP      = 0.10


@dataclass
class ValidationResult:
    passed:         bool
    confidence:     float = 0.0
    oos_pf_delta:   float = 0.0
    oos_exp_delta:  float = 0.0
    splits_agreed:  int   = 0
    mc_percentile:  float = 0.0
    reason:         str   = ""
    metrics_before: dict  = field(default_factory=dict)
    metrics_after:  dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed":        self.passed,
            "confidence":    round(self.confidence,    3),
            "oos_pf_delta":  round(self.oos_pf_delta,  3),
            "oos_exp_delta": round(self.oos_exp_delta, 3),
            "splits_agreed": self.splits_agreed,
            "mc_percentile": round(self.mc_percentile, 3),
            "reason":        self.reason,
        }


def _filter(trades: list[dict], prob_floor: float, sb_floor: int) -> list[dict]:
    return [t for t in trades
            if float(t.get("prob", 1.0)) >= prob_floor
            and int(t.get("score", 99))  >= sb_floor]


def validate_gate_change(
    current_prob:  float,
    current_sb:    int,
    proposed_prob: float,
    proposed_sb:   int,
    trades:        Sequence[dict],
) -> ValidationResult:
    """
    Validate a proposed change to prob_floor / sb_base_score.

    trades — resolved entries from signal_log with actual_pct, prob, score, outcome.

    Returns ValidationResult; .passed is True only when all checks clear.
    """
    resolved = [t for t in trades
                if t.get("actual_pct") is not None and t.get("outcome")]

    if len(resolved) < MIN_SAMPLE_SIZE:
        return ValidationResult(
            passed  = False,
            reason  = f"Insufficient sample — {len(resolved)} resolved trades (need ≥{MIN_SAMPLE_SIZE})",
        )

    base_set  = _filter(resolved, current_prob,  current_sb)
    prop_set  = _filter(resolved, proposed_prob, proposed_sb)

    if not prop_set:
        return ValidationResult(
            passed = False,
            reason = "Proposed gates reject ALL resolved trades — too restrictive",
        )

    m_before = compute_metrics(base_set)
    m_after  = compute_metrics(prop_set)

    splits_agreed = 0
    for train_frac, _ in ENSEMBLE_SPLITS:
        split = int(len(resolved) * train_frac)
        test  = resolved[split:]
        if len(test) < 5:
            continue
        tb = _filter(test, current_prob,  current_sb)
        tp = _filter(test, proposed_prob, proposed_sb)
        if not tp:
            continue
        mb = compute_metrics(tb)
        mp = compute_metrics(tp)
        if (mp["profit_factor"] >= mb["profit_factor"]
                and mp["expectancy"] >= mb["expectancy"]):
            splits_agreed += 1

    pcts_after = [t.get("actual_pct", 0.0) for t in prop_set]
    real_pf    = m_after["profit_factor"]
    rng        = random.Random(42)
    mc_better  = 0
    for _ in range(MONTE_CARLO_RUNS):
        sh    = pcts_after[:]
        rng.shuffle(sh)
        gw    = sum(p for p in sh if p >= 0)
        gl    = abs(sum(p for p in sh if p <  0))
        mc_pf = gw / gl if gl > 0 else 99.0
        if real_pf > mc_pf:
            mc_better += 1
    mc_pct = mc_better / MONTE_CARLO_RUNS

    pf_ok  = m_after["profit_factor"] >= max(m_before["profit_factor"] * PF_MAX_DROP_RATIO, 0.90)
    exp_ok = m_after["expectancy"]    >= m_before["expectancy"] - EXP_MAX_DROP

    passed = (
        splits_agreed >= ENSEMBLE_REQUIRED
        and mc_pct    >= MC_MIN_PERCENTILE
        and pf_ok
        and exp_ok
    )

    reasons = []
    if splits_agreed < ENSEMBLE_REQUIRED:
        reasons.append(
            f"only {splits_agreed}/{len(ENSEMBLE_SPLITS)} walk-forward splits agreed "
            f"(need ≥{ENSEMBLE_REQUIRED})"
        )
    if mc_pct < MC_MIN_PERCENTILE:
        reasons.append(f"MC significance {mc_pct:.0%} < {MC_MIN_PERCENTILE:.0%} — may be noise")
    if not pf_ok:
        reasons.append(
            f"PF {m_before['profit_factor']:.2f}→{m_after['profit_factor']:.2f} "
            f"(drop exceeds 5% tolerance)"
        )
    if not exp_ok:
        reasons.append(
            f"Expectancy {m_before['expectancy']:.3f}→{m_after['expectancy']:.3f} "
            f"(drop >{EXP_MAX_DROP:.2f})"
        )

    return ValidationResult(
        passed         = passed,
        confidence     = mc_pct,
        oos_pf_delta   = m_after["profit_factor"]  - m_before["profit_factor"],
        oos_exp_delta  = m_after["expectancy"]      - m_before["expectancy"],
        splits_agreed  = splits_agreed,
        mc_percentile  = mc_pct,
        reason         = "All checks passed" if passed else "Rejected: " + "; ".join(reasons),
        metrics_before = m_before,
        metrics_after  = m_after,
    )
