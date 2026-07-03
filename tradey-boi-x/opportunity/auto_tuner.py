"""
Auto Threshold Tuner — SAFE, constrained (Trade Evaluation Layer upgrade)
==========================================================================
Every AUTO_TUNER_INTERVAL_TRADES resolved trade-evaluator decisions, nudges
`TRADE_EVAL_THRESHOLDS` (imported and mutated IN PLACE, so `TradeEvaluator`
picks up the change immediately — no restart needed) by at most
AUTO_TUNER_MAX_STEP_PCT (default 5%) per cycle, clamped to
`AUTO_TUNER_BOUNDS`, changing only ONE threshold family per cycle.

Guardrails (all absolute — cannot be bypassed by the adjustment logic):
  - No-op unless ENABLE_AUTO_TUNER is True (default False).
  - No-op while SHADOW_MODE is True — tuning only runs once the evaluator is
    live-gating trades, per the spec's validation requirement.
  - Every adjustment is clamped to AUTO_TUNER_BOUNDS regardless of what the
    rule computed.
  - Any error anywhere in this module is caught and logged; it NEVER raises
    into the caller (process_trade_signal / scanner loop).

Adjustment rules (checked in priority order, first match wins — enforces
"never change all thresholds at once"):
  1. Too few resolved trades in the window -> loosen (all four thresholds
     move a small step towards "easier to pass") so the filter doesn't
     starve itself of data.
  2. Win rate dropped vs the prior window -> raise min_edge_score (tighten
     signal quality bar).
  3. Win rate rose but avg R per trade dropped -> raise min_risk_reward
     (winners are getting smaller relative to risk).
  4. Otherwise -> no change this cycle.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opportunity.config import (
    ENABLE_AUTO_TUNER,
    SHADOW_MODE,
    TRADE_EVAL_THRESHOLDS,
    AUTO_TUNER_INTERVAL_TRADES,
    AUTO_TUNER_MAX_STEP_PCT,
    AUTO_TUNER_MIN_TRADES_FLOOR,
    AUTO_TUNER_BOUNDS,
    AUTO_TUNER_STATE_PATH,
    AUTO_TUNER_LOG_PATH,
)
from opportunity.performance_tracker import PerformanceTracker

BASE_DIR = Path(__file__).parent.parent


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else BASE_DIR / path


def _load_state() -> dict[str, Any]:
    path = _resolve_path(AUTO_TUNER_STATE_PATH)
    if not path.exists():
        return {"last_tuned_count": 0}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"last_tuned_count": 0}


def _save_state(state: dict[str, Any]) -> None:
    path = _resolve_path(AUTO_TUNER_STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _log_decision(record: dict[str, Any]) -> None:
    try:
        path = _resolve_path(AUTO_TUNER_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"  ⚠️  auto_tuner: failed to log decision ({e})")


def _clamp(value: float, key: str) -> float:
    lo, hi = AUTO_TUNER_BOUNDS[key]
    return round(min(max(value, lo), hi), 4)


def _step(current: float, direction: int, key: str) -> float:
    """Move `current` by at most AUTO_TUNER_MAX_STEP_PCT in `direction`
    (+1 tighten / -1 loosen), then clamp to the safe bounds."""
    delta = current * AUTO_TUNER_MAX_STEP_PCT * direction
    return _clamp(current + delta, key)


class AutoThresholdTuner:
    """Stateless decision logic + a tiny persisted counter so we only act
    once every AUTO_TUNER_INTERVAL_TRADES resolved decisions."""

    def __init__(self, thresholds: dict[str, float] | None = None):
        # Intentionally the SAME dict object as config.TRADE_EVAL_THRESHOLDS
        # (unless overridden for tests) so mutating it in place propagates
        # to the live TradeEvaluator instance immediately.
        self.thresholds = thresholds if thresholds is not None else TRADE_EVAL_THRESHOLDS

    def decide_adjustment(
        self, current_stats: dict, previous_stats: dict,
    ) -> dict[str, Any] | None:
        """
        Returns a dict describing the single adjustment to make (or None
        for "no change"), following the priority-ordered rules. Does NOT
        mutate anything — callers apply the change.
        """
        if current_stats["trade_count"] < AUTO_TUNER_MIN_TRADES_FLOOR:
            return {
                "rule": "too_few_trades",
                "changes": {
                    "min_edge_score":          _step(self.thresholds["min_edge_score"], -1, "min_edge_score"),
                    "min_predictability_score": _step(self.thresholds["min_predictability_score"], -1, "min_predictability_score"),
                    "min_risk_reward":         _step(self.thresholds["min_risk_reward"], -1, "min_risk_reward"),
                    "max_noise_index":         _step(self.thresholds["max_noise_index"], +1, "max_noise_index"),
                },
            }

        # Only compare trends when we actually have a prior window to compare to.
        if previous_stats["trade_count"] > 0:
            win_rate_delta = current_stats["win_rate"] - previous_stats["win_rate"]
            avg_r_delta    = current_stats["avg_r"] - previous_stats["avg_r"]

            if win_rate_delta < 0:
                return {
                    "rule": "win_rate_decreased",
                    "changes": {
                        "min_edge_score": _step(self.thresholds["min_edge_score"], +1, "min_edge_score"),
                    },
                }

            if win_rate_delta > 0 and avg_r_delta < 0:
                return {
                    "rule": "win_rate_up_avg_r_down",
                    "changes": {
                        "min_risk_reward": _step(self.thresholds["min_risk_reward"], +1, "min_risk_reward"),
                    },
                }

        return None

    def apply(self, adjustment: dict[str, Any]) -> None:
        """Mutate self.thresholds IN PLACE with the (already-clamped)
        changes from decide_adjustment()."""
        for key, value in adjustment["changes"].items():
            self.thresholds[key] = _clamp(value, key)


def maybe_tune(window: int = AUTO_TUNER_INTERVAL_TRADES) -> dict[str, Any] | None:
    """
    Main entry point, called after every trade-evaluator decision is logged.
    Checks whether AUTO_TUNER_INTERVAL_TRADES new resolved decisions have
    accumulated since the last tuning cycle; if so, computes and applies at
    most one threshold adjustment. Returns the applied-adjustment record, or
    None when no tuning happened this call (flag off, shadow mode, not
    enough new trades yet, or the rules decided "no change").

    Never raises — any failure degrades to "no adjustment this cycle" so it
    can never destabilise the live scan loop.
    """
    if not ENABLE_AUTO_TUNER or SHADOW_MODE:
        return None

    try:
        tracker = PerformanceTracker()
        resolved = tracker.resolved_records()
        total = len(resolved)

        state = _load_state()
        last_tuned_count = state.get("last_tuned_count", 0)

        if total - last_tuned_count < window:
            return None

        current_stats  = tracker.rolling_stats(window=window)
        previous_stats = tracker.previous_window_stats(window=window)

        tuner = AutoThresholdTuner()
        adjustment = tuner.decide_adjustment(current_stats, previous_stats)

        state["last_tuned_count"] = total
        _save_state(state)

        if adjustment is None:
            return None

        before = dict(tuner.thresholds)
        tuner.apply(adjustment)
        after = dict(tuner.thresholds)

        record = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "rule":            adjustment["rule"],
            "trade_count":     total,
            "current_stats":   current_stats,
            "previous_stats":  previous_stats,
            "thresholds_before": before,
            "thresholds_after":  after,
        }
        _log_decision(record)
        print(f"  🎛️  Auto-tuner: applied '{adjustment['rule']}' — "
              f"{ {k: v for k, v in after.items() if before.get(k) != v} }")
        return record

    except Exception as e:
        print(f"  ⚠️  auto_tuner: failed safely, no adjustment made ({e})")
        return None
