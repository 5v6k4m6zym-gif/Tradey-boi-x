"""
Self-Optimising Strategy Engine — SAFE MODE (user-supplied spec)
=====================================================================
A third, independent additive wrapper layer stacked between the existing
Evaluation layers (Phase 8 TradeEvaluator / Adaptive Core v4) and Execution:

    Model -> Signal -> Evaluation -> Strategy Optimiser -> Execution

It does NOT touch the prediction model, signal generation, or execution
logic. It only:
  1. Tags each trade with an inferred strategy_type (StrategyProfiler).
  2. Tracks per-(strategy_type, regime) performance (StrategyPerformanceMatrix).
  3. Maintains small, bounded per-strategy weights (StrategyWeightingEngine).
  4. Gates trades whose strategy is disabled/low-edge/wrong-regime
     (StrategyGatingSystem + RegimeStrategyMap).
  5. Logs every decision to JSONL (Part 8).

Hard safety rules (all enforced below):
  - NEVER disable all strategies for a regime via weight adjustment (the one
    deliberate exception is the explicit LOW_LIQUIDITY hard block, which is
    a Part 5 design decision, not a weight-driven disable).
  - NEVER move a weight by more than STRATEGY_WEIGHT_MAX_STEP_PCT per cycle.
  - NEVER let a weight leave [STRATEGY_WEIGHT_FLOOR, STRATEGY_WEIGHT_CAP].
  - process_trade_signal() NEVER modifies model output — it returns the
    trade dict completely unchanged when allowed, or None when blocked.
  - Any internal failure anywhere in this module is caught and logged; the
    fallback is always "allow the original evaluation system's decision"
    (i.e. return the trade unchanged), never a crash and never a block.
"""
from __future__ import annotations

import functools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opportunity.config import (
    ENABLE_STRATEGY_OPTIMIZER,
    SHADOW_MODE,
    STRATEGY_LOG_PATH,
    STRATEGY_WEIGHTS_PATH,
    STRATEGY_WEIGHT_STATE_PATH,
    STRATEGY_TYPES,
    STRATEGY_WEIGHT_FLOOR,
    STRATEGY_WEIGHT_CAP,
    STRATEGY_WEIGHT_MAX_STEP_PCT,
    STRATEGY_WEIGHT_UPDATE_INTERVAL_TRADES,
    STRATEGY_MIN_ACTIVE_WEIGHT,
    STRATEGY_MIN_EXPECTANCY_TRADES,
    STRATEGY_EXPECTANCY_WINDOW,
    REGIME_STRATEGY_MAP,
)
from opportunity.performance_tracker import PerformanceTracker, WIN_OUTCOMES

BASE_DIR        = Path(__file__).resolve().parent.parent
SIGNAL_LOG_FILE = BASE_DIR / "signal_log.json"


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else BASE_DIR / path


def _safe(default: Any = None):
    """Decorator: never let an exception escape. Logs and returns `default`."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"  ⚠️  strategy_optimizer.{fn.__name__}: failed safely ({e})")
                return default
        return wrapper
    return deco


# ─── Part 1: StrategyProfiler ──────────────────────────────────────────────
class StrategyProfiler:
    """
    Infers a strategy_type label for a trade from the signal conditions
    already produced by engine.decide() (the `why` list of human-readable
    reasons, plus row-level indicator fields), without touching or
    re-running any model/signal logic. Purely a classification/tagging
    layer on TOP of an already-generated signal.
    """

    @staticmethod
    @_safe(default="TREND_CONTINUATION")
    def infer_strategy_type(trade: dict[str, Any]) -> str:
        why = trade.get("why") or []
        why_text = " ".join(str(w).lower() for w in why)
        rsi = trade.get("rsi")
        breakout_flag = bool(trade.get("breakout", 0))

        # Priority order: most specific/high-conviction pattern wins first.
        if breakout_flag or "breakout" in why_text or "squeeze" in why_text:
            return "BREAKOUT"

        if "volatility" in why_text or "expansion" in why_text:
            return "VOLATILITY_EXPANSION"

        if rsi is not None:
            try:
                rsi = float(rsi)
                if rsi < 40 and ("pullback" in why_text or "support" in why_text or "dip" in why_text):
                    return "PULLBACK"
                if rsi < 35 or rsi > 68:
                    return "MEAN_REVERSION"
            except (TypeError, ValueError):
                pass

        if "uptrend" in why_text or "macd bullish" in why_text or "ema20 > ema50" in why_text:
            return "TREND_CONTINUATION"

        return "TREND_CONTINUATION"

    @staticmethod
    @_safe(default={})
    def tag_trade(trade: dict[str, Any], regime: str, edge_score: float) -> dict[str, Any]:
        """Returns a profile record for logging — never mutates `trade`."""
        return {
            "strategy_type": StrategyProfiler.infer_strategy_type(trade),
            "regime":        regime,
            "edge_score":    edge_score,
        }


# ─── Shared log/state IO helpers ───────────────────────────────────────────
@_safe(default=[])
def _load_signal_log() -> list[dict]:
    if not SIGNAL_LOG_FILE.exists():
        return []
    return json.loads(SIGNAL_LOG_FILE.read_text())


@_safe(default=[])
def _load_decisions() -> list[dict]:
    path = _resolve_path(STRATEGY_LOG_PATH)
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _log_decision(record: dict[str, Any]) -> None:
    try:
        path = _resolve_path(STRATEGY_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"  ⚠️  strategy_optimizer: failed to log decision ({e})")


def _r_multiple(entry: dict) -> float | None:
    entry_price = entry.get("entry_price")
    stop_price  = entry.get("stop_price")
    actual_pct  = entry.get("actual_pct")
    if entry_price in (None, 0) or stop_price is None or actual_pct is None:
        return None
    risk_pct = abs(entry_price - stop_price) / entry_price
    if risk_pct <= 0:
        return None
    return round(actual_pct / risk_pct, 4)


@_safe(default=[])
def _resolved_strategy_records() -> list[dict[str, Any]]:
    """
    Joins this module's own decision log with the resolved signal_log
    (ticker + calendar-date match, same pattern as PerformanceTracker) so
    each strategy-tagged decision that has since resolved carries its
    outcome and R multiple.
    """
    signal_log = _load_signal_log()
    by_key: dict[tuple[str, str], dict] = {}
    for e in signal_log:
        if e.get("outcome") is None:
            continue
        key = (e.get("ticker"), e.get("signal_date"))
        by_key.setdefault(key, e)

    joined: list[dict[str, Any]] = []
    for rec in _load_decisions():
        symbol = rec.get("symbol")
        ts = rec.get("timestamp") or ""
        date = ts[:10] if len(ts) >= 10 else None
        if not symbol or not date:
            continue
        resolved = by_key.get((symbol, date))
        if resolved is None:
            continue
        joined.append({
            **rec,
            "outcome":    resolved.get("outcome"),
            "actual_pct": resolved.get("actual_pct"),
            "r_multiple": _r_multiple(resolved),
        })
    return joined


# ─── Part 2: StrategyPerformanceMatrix ─────────────────────────────────────
class StrategyPerformanceMatrix:
    """Strategy x Regime -> {win_rate, expectancy, avg_r, drawdown_contribution}."""

    @staticmethod
    def _cell_stats(records: list[dict]) -> dict[str, Any]:
        n = len(records)
        if n == 0:
            return {"trade_count": 0, "win_rate": 0.0, "avg_r": 0.0,
                    "expectancy_r": 0.0, "drawdown_contribution": 0.0}
        wins = sum(1 for r in records if r.get("outcome") in WIN_OUTCOMES)
        r_values = [r["r_multiple"] for r in records if r.get("r_multiple") is not None]
        win_rate = round(wins / n, 4)
        avg_r = round(sum(r_values) / len(r_values), 4) if r_values else 0.0
        gains = [r for r in r_values if r > 0]
        losses = [r for r in r_values if r <= 0]
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy_r = round((win_rate * avg_gain) - ((1 - win_rate) * avg_loss), 4)

        # Drawdown contribution: max peak-to-trough dip of this cell's own
        # cumulative-R equity curve, in chronological (log) order.
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in r_values:
            cum += r
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        return {
            "trade_count":  n,
            "win_rate":     win_rate,
            "avg_r":        avg_r,
            "expectancy_r": expectancy_r,
            "drawdown_contribution": round(max_dd, 4),
        }

    @_safe(default={})
    def build_matrix(self, records: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        records = records if records is not None else _resolved_strategy_records()
        buckets: dict[str, list[dict]] = {}
        for r in records:
            strategy = r.get("strategy_type")
            regime = r.get("regime")
            if not strategy or not regime:
                continue
            buckets.setdefault(f"{strategy}|{regime}", []).append(r)
        return {key: self._cell_stats(recs) for key, recs in buckets.items()}

    @_safe(default={})
    def strategy_totals(self, records: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        """Per-strategy stats collapsed across all regimes — used by the
        weighting engine, which weights a strategy overall, not per-cell."""
        records = records if records is not None else _resolved_strategy_records()
        buckets: dict[str, list[dict]] = {}
        for r in records:
            strategy = r.get("strategy_type")
            if not strategy:
                continue
            buckets.setdefault(strategy, []).append(r)
        return {strategy: self._cell_stats(recs) for strategy, recs in buckets.items()}


# ─── Weight state persistence ───────────────────────────────────────────────
@_safe(default={})
def _load_weights() -> dict[str, float]:
    path = _resolve_path(STRATEGY_WEIGHTS_PATH)
    if not path.exists():
        return {s: 1.0 for s in STRATEGY_TYPES}
    weights = json.loads(path.read_text())
    # Always fill in any strategy types missing from a stale weights file.
    for s in STRATEGY_TYPES:
        weights.setdefault(s, 1.0)
    return weights


def _save_weights(weights: dict[str, float]) -> None:
    try:
        path = _resolve_path(STRATEGY_WEIGHTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(weights))
        tmp.replace(path)
    except Exception as e:
        print(f"  ⚠️  strategy_optimizer: failed to save weights ({e})")


@_safe(default={"last_updated_count": 0})
def _load_weight_state() -> dict[str, Any]:
    path = _resolve_path(STRATEGY_WEIGHT_STATE_PATH)
    if not path.exists():
        return {"last_updated_count": 0}
    return json.loads(path.read_text())


def _save_weight_state(state: dict[str, Any]) -> None:
    try:
        path = _resolve_path(STRATEGY_WEIGHT_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)
    except Exception as e:
        print(f"  ⚠️  strategy_optimizer: failed to save weight state ({e})")


def _clamp_weight(value: float) -> float:
    return round(min(max(value, STRATEGY_WEIGHT_FLOOR), STRATEGY_WEIGHT_CAP), 4)


# ─── Part 3: StrategyWeightingEngine ────────────────────────────────────────
class StrategyWeightingEngine:
    """
    weight = f(expectancy, stability, drawdown_penalty), adjusted by at most
    STRATEGY_WEIGHT_MAX_STEP_PCT per update cycle, always clamped to
    [STRATEGY_WEIGHT_FLOOR, STRATEGY_WEIGHT_CAP]. Weights persist across
    scanner runs in STRATEGY_WEIGHTS_PATH so a live, running process picks
    up its own gradual adjustments without restart (same in-place pattern
    as AutoThresholdTuner).
    """

    @staticmethod
    def _target_direction(stats: dict[str, Any]) -> int:
        """+1 = nudge weight up, -1 = nudge weight down, 0 = no change.
        Requires a minimum sample size so a strategy is never judged (and
        thus never penalised) on too little evidence."""
        if stats.get("trade_count", 0) < STRATEGY_MIN_EXPECTANCY_TRADES:
            return 0
        expectancy = stats.get("expectancy_r", 0.0)
        drawdown = stats.get("drawdown_contribution", 0.0)
        if expectancy > 0.05 and drawdown < 2.0:
            return 1
        if expectancy < 0:
            return -1
        return 0

    @_safe(default={})
    def compute_new_weights(
        self, current_weights: dict[str, float], totals: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Pure function — does not persist anything. Returns a NEW dict."""
        new_weights = dict(current_weights)
        for strategy in STRATEGY_TYPES:
            current = new_weights.get(strategy, 1.0)
            stats = totals.get(strategy, {})
            direction = self._target_direction(stats)
            if direction == 0:
                continue
            step = current * STRATEGY_WEIGHT_MAX_STEP_PCT * direction
            new_weights[strategy] = _clamp_weight(current + step)
        return new_weights

    @_safe(default=None)
    def maybe_update(self, window: int = STRATEGY_WEIGHT_UPDATE_INTERVAL_TRADES) -> dict[str, Any] | None:
        """
        Runs at most once every `window` new resolved strategy-tagged
        decisions. Never raises — any failure degrades to "no update this
        cycle" so it can never destabilise the live scan loop.
        """
        records = _resolved_strategy_records()
        total = len(records)

        state = _load_weight_state()
        last_updated = state.get("last_updated_count", 0)
        if total - last_updated < window:
            return None

        matrix = StrategyPerformanceMatrix()
        totals = matrix.strategy_totals(records)
        current_weights = _load_weights()
        new_weights = self.compute_new_weights(current_weights, totals)

        state["last_updated_count"] = total
        _save_weight_state(state)

        changed = {k: v for k, v in new_weights.items() if current_weights.get(k) != v}
        if not changed:
            return None

        _save_weights(new_weights)
        record = {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "trade_count":       total,
            "weights_before":    current_weights,
            "weights_after":     new_weights,
            "strategy_totals":   totals,
        }
        try:
            path = _resolve_path(STRATEGY_LOG_PATH).parent / "strategy_weight_updates.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"  ⚠️  strategy_optimizer: failed to log weight update ({e})")
        print(f"  ⚖️  Strategy optimiser: weights updated -> {changed}")
        return record


# ─── Part 5: RegimeStrategyMap ───────────────────────────────────────────────
class RegimeStrategyMap:
    """Regime -> allowed strategy list, per REGIME_STRATEGY_MAP config."""

    @staticmethod
    def allowed_strategies(regime: str) -> list[str]:
        return REGIME_STRATEGY_MAP.get(regime, list(STRATEGY_TYPES))

    @staticmethod
    def is_allowed(strategy_type: str, regime: str) -> bool:
        return strategy_type in RegimeStrategyMap.allowed_strategies(regime)


# ─── Part 4: StrategyGatingSystem ────────────────────────────────────────────
class StrategyGatingSystem:
    """
    Gates a trade based on: (1) regime hard-block (LOW_LIQUIDITY), (2) the
    strategy being allowed in the current regime, (3) the strategy's weight
    being above STRATEGY_MIN_ACTIVE_WEIGHT, (4) the strategy's recent
    expectancy not being negative (with a cold-start-safe minimum sample).
    """

    @staticmethod
    @_safe(default=(False, "gating_check_failed"))
    def check(
        strategy_type: str, regime: str,
        weights: dict[str, float], strategy_totals: dict[str, dict[str, Any]],
    ) -> tuple[bool, str]:
        allowed_here = RegimeStrategyMap.allowed_strategies(regime)
        if not allowed_here:
            return False, "strategy_disabled_or_low_edge"

        if strategy_type not in allowed_here:
            return False, "strategy_disabled_or_low_edge"

        weight = weights.get(strategy_type, 1.0)
        if weight < STRATEGY_MIN_ACTIVE_WEIGHT:
            return False, "strategy_disabled_or_low_edge"

        stats = strategy_totals.get(strategy_type, {})
        if stats.get("trade_count", 0) >= STRATEGY_MIN_EXPECTANCY_TRADES:
            if stats.get("expectancy_r", 0.0) < 0:
                return False, "strategy_disabled_or_low_edge"

        return True, "allowed"


# ─── Part 9: process_trade_signal wrapper ────────────────────────────────────
def process_trade_signal(trade: dict[str, Any], market_data: Any) -> dict[str, Any] | None:
    """
    Main entry point, called from scanner.py after the existing evaluation
    layers. Fail-safe: on ANY internal error, falls back to the existing
    evaluation system's decision (returns `trade` unchanged) rather than
    crashing or blocking. Never modifies model output — the returned trade
    is either the exact same object unchanged, or None.
    """
    if not ENABLE_STRATEGY_OPTIMIZER:
        return trade

    try:
        # Regime: reuse Adaptive Core's per-ticker RegimeDetector rather
        # than duplicating regime-classification logic.
        try:
            from opportunity.adaptive_core import RegimeDetector
            regime_result = RegimeDetector().detect(market_data)
            regime = regime_result.regime
        except Exception:
            regime = "CHOP"

        edge_score = float(trade.get("edge_score", trade.get("prob", 0.5)) or 0.5)
        strategy_type = StrategyProfiler.infer_strategy_type(trade)

        weights = _load_weights()
        totals = StrategyPerformanceMatrix().strategy_totals()
        allowed, reason = StrategyGatingSystem.check(strategy_type, regime, weights, totals)

        record = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "symbol":         trade.get("ticker", trade.get("symbol")),
            "strategy_type":  strategy_type,
            "regime":         regime,
            "weight_at_time": weights.get(strategy_type, 1.0),
            "edge_score":     edge_score,
            "allowed":        allowed,
            "reason":         reason,
            "shadow_mode":    SHADOW_MODE,
        }
        _log_decision(record)

        # Periodic weight update — its own try/except in addition to the
        # engine's internal @_safe guards, so it can never break this flow.
        try:
            StrategyWeightingEngine().maybe_update()
        except Exception as e:
            print(f"  ⚠️  strategy_optimizer: weight update skipped ({e})")

        if not allowed and not SHADOW_MODE:
            return None

        return trade

    except Exception as e:
        print(f"  ⚠️  strategy_optimizer.process_trade_signal: failed safely, passing trade through ({e})")
        return trade
