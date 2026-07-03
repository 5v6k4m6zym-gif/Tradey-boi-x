"""
Full Adaptive Trading Core v4
==============================
A SAFE, ADDITIVE intelligence layer stacked ABOVE the existing bot and the
Phase 8 trade-evaluator. It does NOT modify the prediction model, does NOT
change signal generation, and does NOT touch execution — it only decides
pass/reject + sizing + logging metadata, exactly like the Phase 8 layer, but
adds per-ticker regime awareness, execution-quality filtering, confidence
calibration, bounded dynamic position sizing, an expectancy gate, and
post-trade loss classification.

New Flow (only when ENABLE_ADAPTIVE_CORE=true):
    Model -> Trade Signal -> Adaptive Core v4 -> Execution

Deliberately reuses rather than duplicates existing machinery:
  - TradeEvaluator (trade_evaluator.py)  for edge/predictability/noise/RR
  - PerformanceTracker (performance_tracker.py) for system-wide rolling stats
  - TRADE_EVAL_THRESHOLDS (config.py) as the base thresholds this layer nudges

Controlled by two independent switches (opportunity/config.py):
  - ENABLE_ADAPTIVE_CORE: master on/off. False -> process_trade_signal() is a
    complete passthrough (returns trade unchanged); nothing else in this
    module is ever invoked.
  - SHADOW_MODE (shared with Phase 8): True -> decisions are computed and
    logged but the trade is NEVER allowed through (returns None). Set to
    False only once shadow-mode logs have been reviewed and trusted.

FAIL-SAFE RULE (absolute, per spec): if ANY component in this module raises,
process_trade_signal() logs the error and falls back to PASS-THROUGH — it
returns the original `trade` dict UNCHANGED (not None), so a bug in this
optional layer can never itself block or corrupt a trade.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from opportunity.config import (
    ENABLE_ADAPTIVE_CORE,
    SHADOW_MODE,
    TRADE_EVAL_THRESHOLDS,
    AUTO_TUNER_BOUNDS,
    ADAPTIVE_CORE_LOG_PATH,
    ADAPTIVE_REGIME_LOOKBACK,
    ADAPTIVE_MIN_AVG_VOLUME,
    ADAPTIVE_REGIME_ADJUSTMENTS,
    ADAPTIVE_MIN_EXECUTION_QUALITY,
    ADAPTIVE_BASE_RISK_PCT,
    ADAPTIVE_MAX_SIZE_MULTIPLIER,
    ADAPTIVE_MIN_SIZE_MULTIPLIER,
    ADAPTIVE_MAX_RISK_PCT,
    ADAPTIVE_EXPECTANCY_MIN_TRADES,
    ADAPTIVE_EXPECTANCY_WINDOW,
)
from opportunity.trade_evaluator import TradeEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent

REGIMES = ("TRENDING_UP", "TRENDING_DOWN", "CHOP", "VOLATILITY_EXPANSION", "LOW_LIQUIDITY")

LOSS_CLASSES = (
    "FALSE_BREAKOUT", "LATE_ENTRY", "TREND_REVERSAL", "NOISE_ENTRY", "REGIME_MISMATCH",
)


# ─── Part 1/2: RegimeDetector ─────────────────────────────────────────────────
@dataclass
class RegimeResult:
    regime: str
    confidence: float
    metrics: dict[str, Any] = field(default_factory=dict)


class RegimeDetector:
    """
    Per-ticker regime classifier — distinct from opportunity.regime's macro
    ASX-200-index-level detector. This one works off the same OHLCV window
    already passed into process_trade_signal, so it needs no extra API calls.

    Inputs: volatility, price structure/trend strength (efficiency ratio +
    EMA slope, reusing TradeEvaluator's static helpers), and volume behaviour
    when a 'Volume' column is present.
    """

    def detect(self, market_data: pd.DataFrame, lookback: int = ADAPTIVE_REGIME_LOOKBACK) -> RegimeResult:
        recent = market_data.tail(lookback + 1)
        if len(recent) < 5:
            return RegimeResult("CHOP", 0.40, {"reason": "insufficient_data"})

        # Liquidity check first — hard filter regardless of everything else.
        if "Volume" in recent.columns:
            avg_volume = float(recent["Volume"].tail(lookback).mean())
            if avg_volume < ADAPTIVE_MIN_AVG_VOLUME:
                shortfall = 1.0 - (avg_volume / ADAPTIVE_MIN_AVG_VOLUME) if ADAPTIVE_MIN_AVG_VOLUME else 1.0
                confidence = round(min(0.95, 0.55 + max(shortfall, 0.0) * 0.5), 3)
                return RegimeResult("LOW_LIQUIDITY", confidence, {"avg_volume": round(avg_volume, 1)})
        else:
            avg_volume = None

        noise_index, efficiency_ratio = TradeEvaluator.compute_noise_index(market_data, lookback)

        closes = recent["Close"]
        returns = closes.pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 1 else 0.0

        # Historical volatility distribution over a longer window, so we can
        # tell "elevated for THIS ticker" apart from "just naturally volatile".
        hist = market_data.tail(max(lookback * 3, lookback + 1))
        hist_returns = hist["Close"].pct_change().dropna()
        vol_pct_rank = float((hist_returns.rolling(lookback).std().dropna() < volatility).mean()) \
            if len(hist_returns) > lookback else 0.5

        net_change = float(closes.iloc[-1]) - float(closes.iloc[0])
        trend_up = net_change > 0

        metrics = {
            "noise_index": noise_index,
            "efficiency_ratio": efficiency_ratio,
            "volatility": round(volatility, 5),
            "vol_pct_rank": round(vol_pct_rank, 3),
            "avg_volume": round(avg_volume, 1) if avg_volume is not None else None,
        }

        if vol_pct_rank >= 0.85:
            confidence = round(min(0.95, 0.55 + (vol_pct_rank - 0.85) * 3.0), 3)
            return RegimeResult("VOLATILITY_EXPANSION", confidence, metrics)

        if efficiency_ratio >= 0.35:
            confidence = round(min(0.95, 0.45 + efficiency_ratio * 0.5), 3)
            return RegimeResult("TRENDING_UP" if trend_up else "TRENDING_DOWN", confidence, metrics)

        confidence = round(max(0.40, 0.70 - efficiency_ratio), 3)
        return RegimeResult("CHOP", confidence, metrics)


# ─── Regime-aware thresholds ──────────────────────────────────────────────────
def regime_adjusted_thresholds(base: dict[str, float], regime: str) -> dict[str, float]:
    """
    Returns a NEW dict — never mutates `base` (TRADE_EVAL_THRESHOLDS), and
    never mutated/persisted, since these adjustments are transient per-trade.
    Every adjusted value is clamped to the same AUTO_TUNER_BOUNDS already
    trusted as safe for this system, so regime nudges can never push a
    threshold outside a limit the system has already established as safe.
    """
    adjustments = ADAPTIVE_REGIME_ADJUSTMENTS.get(regime, {})
    adjusted = dict(base)
    for key, multiplier in adjustments.items():
        if key not in adjusted:
            continue
        value = adjusted[key] * multiplier
        low, high = AUTO_TUNER_BOUNDS.get(key, (float("-inf"), float("inf")))
        adjusted[key] = round(min(max(value, low), high), 4)
    return adjusted


# ─── Part 3: ExecutionQualityFilter ───────────────────────────────────────────
@dataclass
class ExecutionQualityResult:
    score: float
    cost_pct: float
    expected_edge_pct: float
    timing_quality: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class ExecutionQualityFilter:
    """
    Cheap, model-free proxies for spread cost, slippage risk, and entry
    timing quality — reuses TRADING_COSTS (already-tuned per-market cost
    assumptions from the realistic-backtest upgrade) and the noise index as
    a slippage-risk proxy, rather than inventing new cost estimates.
    """

    def evaluate(self, trade: dict[str, Any], market_data: pd.DataFrame,
                 edge_score: float, noise_index: float) -> ExecutionQualityResult:
        from opportunity.config import TRADING_COSTS

        ticker = str(trade.get("ticker", trade.get("symbol", "")))
        is_asx = ticker.upper().endswith(".AX")
        suffix = "asx" if is_asx else "us"
        cost_pct = (
            TRADING_COSTS.get(f"commission_pct_{suffix}", 0.0)
            + TRADING_COSTS.get(f"slippage_pct_{suffix}", 0.0)
            + TRADING_COSTS.get(f"spread_pct_{suffix}", 0.0)
        )

        # Expected edge in price-% terms — edge_score (0-1) scaled by the
        # trade's own risk/reward-implied reward, so it's on a comparable
        # scale to cost_pct rather than an arbitrary constant.
        entry = float(trade.get("entry", 0) or 0)
        take_profit = float(trade.get("take_profit", 0) or 0)
        reward_pct = abs(take_profit - entry) / entry if entry else 0.0
        expected_edge_pct = round(edge_score * reward_pct, 5)

        # Entry timing: how close is the current close to the recent high
        # (breakout confirmation strength) vs how "late"/extended it looks.
        recent = market_data.tail(20)
        if len(recent) >= 2 and float(recent["High"].max()) > 0:
            prior_high = float(recent["High"].iloc[:-1].max())
            close = float(recent["Close"].iloc[-1])
            extension = (close - prior_high) / prior_high if prior_high else 0.0
            # Small confirmed breakout (0-3% above prior high) = good timing;
            # deeply extended (>8%) = late-entry risk.
            timing_quality = round(max(0.0, 1.0 - max(extension - 0.03, 0.0) * 10.0), 3)
            timing_quality = min(timing_quality, 1.0)
        else:
            timing_quality = 0.5

        slippage_penalty = min(noise_index / 3.0, 1.0)
        cost_penalty = min(cost_pct * 50.0, 1.0)
        score = round(max(0.0, 0.5 * (1 - cost_penalty) + 0.3 * (1 - slippage_penalty) + 0.2 * timing_quality), 3)

        reasons: list[str] = []
        if cost_pct > expected_edge_pct:
            reasons.append(f"execution_cost {cost_pct:.4f} > expected_edge {expected_edge_pct:.4f}")
        if score < ADAPTIVE_MIN_EXECUTION_QUALITY:
            reasons.append(f"execution_quality_score {score:.2f} < {ADAPTIVE_MIN_EXECUTION_QUALITY:.2f}")

        return ExecutionQualityResult(
            score=score, cost_pct=round(cost_pct, 5), expected_edge_pct=expected_edge_pct,
            timing_quality=timing_quality, passed=(len(reasons) == 0), reasons=reasons,
        )


# ─── Part 6: ConfidenceCalibrator ─────────────────────────────────────────────
@dataclass
class CalibrationResult:
    raw_probability: float
    calibrated_probability: float
    bucket: str | None
    calibration_status: str | None


class ConfidenceCalibrator:
    """
    Reuses opportunity.performance.calibration_buckets() (already computes
    actual win-rate per probability bucket from resolved signal history) as
    an INTERPRETATION layer only — it never modifies trade["probability"] or
    the model's own output. Callers use `calibrated_probability` internally
    (e.g. for position sizing) when they want a historically-grounded
    estimate instead of the raw model number.
    """

    MIN_BUCKET_SAMPLE = 10

    def calibrate(self, probability: float) -> CalibrationResult:
        try:
            from opportunity.performance import calibration_buckets
            from opportunity.performance_tracker import _load_signal_log  # reuse existing loader

            entries = [e for e in _load_signal_log() if e.get("outcome") is not None]
            buckets = calibration_buckets(entries) or []
        except Exception:
            buckets = []

        bucket_data = None
        for b in buckets:
            lo = b.get("predicted_min", 0.0)
            hi = b.get("predicted_max", 1.01)
            if lo <= probability < hi:
                bucket_data = b
                break

        bucket_label = bucket_data.get("label") if bucket_data else None

        if bucket_data and bucket_data.get("count", 0) >= self.MIN_BUCKET_SAMPLE:
            actual = bucket_data.get("actual_win_rate")
            if actual is not None:
                return CalibrationResult(
                    raw_probability=probability,
                    calibrated_probability=round(float(actual), 4),
                    bucket=bucket_label,
                    calibration_status=bucket_data.get("calibration_status"),
                )

        # Safe default: insufficient data (or probability below the lowest
        # tracked bucket, e.g. <50%) -> passthrough, unchanged.
        return CalibrationResult(probability, probability, bucket_label, None)


# ─── Part 4: PositionSizer ─────────────────────────────────────────────────────
@dataclass
class SizingResult:
    multiplier: float
    adjusted_risk_pct: float
    quality_score: float


class PositionSizer:
    """
    Dynamic sizing bounded to [ADAPTIVE_MIN_SIZE_MULTIPLIER,
    ADAPTIVE_MAX_SIZE_MULTIPLIER] (default 0.5x-1.5x) of a base risk-per-trade
    percentage, additionally hard-capped at ADAPTIVE_MAX_RISK_PCT regardless
    of what the multiplier computes to. Never increases risk beyond that cap.
    """

    def size(self, edge_score: float, regime_confidence: float, predictability_score: float,
              base_risk_pct: float = ADAPTIVE_BASE_RISK_PCT) -> SizingResult:
        quality_score = round(
            0.5 * edge_score + 0.3 * regime_confidence + 0.2 * predictability_score, 4
        )

        if quality_score >= 0.75:
            span = ADAPTIVE_MAX_SIZE_MULTIPLIER - 1.0
            multiplier = 1.0 + min((quality_score - 0.75) / 0.25, 1.0) * span
        elif quality_score <= 0.45:
            span = 1.0 - ADAPTIVE_MIN_SIZE_MULTIPLIER
            multiplier = 1.0 - min((0.45 - quality_score) / 0.45, 1.0) * span
        else:
            multiplier = 1.0

        multiplier = round(min(max(multiplier, ADAPTIVE_MIN_SIZE_MULTIPLIER), ADAPTIVE_MAX_SIZE_MULTIPLIER), 4)
        adjusted_risk_pct = round(min(base_risk_pct * multiplier, ADAPTIVE_MAX_RISK_PCT), 4)

        return SizingResult(multiplier=multiplier, adjusted_risk_pct=adjusted_risk_pct, quality_score=quality_score)


# ─── Part 7: ExpectancyEngine ─────────────────────────────────────────────────
class ExpectancyEngine:
    """
    System-wide (not per-trade) expectancy gate:
        Expectancy = (Win Rate x Avg Win) - (Loss Rate x Avg Loss)
    Reuses PerformanceTracker.rolling_stats() for the win-rate/avg-R inputs
    instead of recomputing the join logic. Cold-start safe: with fewer than
    ADAPTIVE_EXPECTANCY_MIN_TRADES resolved+joined trades to judge, the gate
    PASSES (fail-open) — this system must never lock itself out of trading
    before it has enough of a track record to judge itself by.
    """

    def compute_expectancy(self, stats: dict[str, Any]) -> float:
        return float(stats.get("expectancy_r", 0.0))

    def gate(self, stats: dict[str, Any], regime_valid: bool,
              execution_quality_passed: bool) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        trade_count = stats.get("trade_count", 0)

        if trade_count < ADAPTIVE_EXPECTANCY_MIN_TRADES:
            expectancy_ok = True   # fail-open on insufficient data
        else:
            expectancy = self.compute_expectancy(stats)
            expectancy_ok = expectancy > 0
            if not expectancy_ok:
                reasons.append(f"system expectancy_r {expectancy:.4f} <= 0 (n={trade_count})")

        if not regime_valid:
            reasons.append("regime invalid (LOW_LIQUIDITY)")
        if not execution_quality_passed:
            reasons.append("execution quality rejected")

        return (expectancy_ok and regime_valid and execution_quality_passed), reasons


# ─── Part 5: LossClassifier ────────────────────────────────────────────────────
class LossClassifier:
    """
    Post-trade, best-effort labeling utility — used for system improvement
    only, never gates or blocks anything. Operates on a resolved signal_log
    entry plus whatever contextual metadata was captured at entry time (via
    the adaptive-core decision log), applying a priority-ordered heuristic.
    Returns "UNCLASSIFIED" (never raises) when there isn't enough signal.
    """

    WIN_OUTCOMES = {"WIN", "HIT_TARGET", "EXPIRED_GAIN", "TARGET_HIT"}

    def classify(self, resolved_entry: dict[str, Any],
                 entry_context: dict[str, Any] | None = None) -> str | None:
        try:
            outcome = resolved_entry.get("outcome")
            if outcome in self.WIN_OUTCOMES:
                return None   # only losses are classified

            entry_context = entry_context or {}
            regime = entry_context.get("regime_type")
            noise_index = entry_context.get("noise_index")
            timing_quality = entry_context.get("timing_quality")

            actual_pct = resolved_entry.get("actual_pct")
            hit_stop_fast = outcome in ("HIT_STOP", "LOSS") and actual_pct is not None and actual_pct < 0

            if regime in ("CHOP", "VOLATILITY_EXPANSION", "LOW_LIQUIDITY"):
                return "REGIME_MISMATCH"
            if timing_quality is not None and timing_quality < 0.35:
                return "LATE_ENTRY"
            if noise_index is not None and noise_index > 1.3 and hit_stop_fast:
                return "NOISE_ENTRY"
            if hit_stop_fast:
                return "FALSE_BREAKOUT"
            if outcome in ("EXPIRED_LOSS",):
                return "TREND_REVERSAL"
            return "UNCLASSIFIED"
        except Exception:
            return "UNCLASSIFIED"


# ─── Logging (append-only JSONL) ───────────────────────────────────────────────
def _log_path() -> Path:
    p = Path(ADAPTIVE_CORE_LOG_PATH)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def _log_decision(record: dict[str, Any]) -> None:
    """Never raises — a logging failure must not be able to break the scan/
    trade flow."""
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"  \u26a0\ufe0f  adaptive_core: failed to log decision ({e})")


# ─── Part 8: process_trade_signal wrapper (entry point) ───────────────────────
_regime_detector = RegimeDetector()
_execution_filter = ExecutionQualityFilter()
_calibrator = ConfidenceCalibrator()
_sizer = PositionSizer()
_expectancy_engine = ExpectancyEngine()


def process_trade_signal(trade: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any] | None:
    """
    The single integration point for the Adaptive Core v4 layer.

    1. If disabled, passes `trade` through completely unchanged.
    2. Otherwise: detect regime -> regime-aware thresholds -> evaluate
       (reusing TradeEvaluator) -> execution quality -> calibration ->
       expectancy gate -> position sizing -> log full decision -> gate.
    3. SHADOW_MODE=True -> always returns None after logging (observe only).
    4. SHADOW_MODE=False -> returns a NEW dict (never mutates the original
       `trade`) with sizing/regime metadata if approved, else None.
    5. FAIL-SAFE: any internal exception -> log the error and return `trade`
       UNCHANGED (pass-through), per the spec's absolute fail-safe rule.
    """
    if not ENABLE_ADAPTIVE_CORE:
        return trade

    try:
        regime_result = _regime_detector.detect(market_data)
        regime_valid = regime_result.regime != "LOW_LIQUIDITY"

        if regime_valid:
            adjusted_thresholds = regime_adjusted_thresholds(TRADE_EVAL_THRESHOLDS, regime_result.regime)
            evaluation = TradeEvaluator(thresholds=adjusted_thresholds).evaluate(trade, market_data)
        else:
            # Hard filter — don't even bother computing thresholds.
            evaluation = TradeEvaluator(thresholds=TRADE_EVAL_THRESHOLDS).evaluate(trade, market_data)

        exec_quality = _execution_filter.evaluate(
            trade, market_data, evaluation.edge_score, evaluation.noise_index
        )
        calibration = _calibrator.calibrate(float(trade.get("probability", trade.get("prob", 0.5))))

        try:
            from opportunity.performance_tracker import PerformanceTracker
            stats = PerformanceTracker().rolling_stats(window=ADAPTIVE_EXPECTANCY_WINDOW)
        except Exception:
            stats = {"trade_count": 0, "win_rate": 0.0, "avg_r": 0.0, "expectancy_r": 0.0}

        expectancy_ok, expectancy_reasons = _expectancy_engine.gate(stats, regime_valid, exec_quality.passed)

        passed = regime_valid and evaluation.passed and exec_quality.passed and expectancy_ok
        rejection_reasons = list(evaluation.rejection_reasons) + list(exec_quality.reasons) + list(expectancy_reasons)
        if not regime_valid:
            rejection_reasons.insert(0, f"regime {regime_result.regime} — no trade")

        sizing = _sizer.size(evaluation.edge_score, regime_result.confidence, evaluation.predictability_score)

        record = {
            "timestamp":                datetime.now(timezone.utc).isoformat(),
            "symbol":                   trade.get("ticker", trade.get("symbol")),
            "direction":                trade.get("direction", "LONG"),
            "model_probability":        trade.get("probability", trade.get("prob")),
            "calibrated_probability":   calibration.calibrated_probability,
            "edge_score":               evaluation.edge_score,
            "regime_type":              regime_result.regime,
            "regime_confidence":        regime_result.confidence,
            "predictability_score":     evaluation.predictability_score,
            "noise_index":              evaluation.noise_index,
            "execution_quality_score":  exec_quality.score,
            "position_size_multiplier": sizing.multiplier,
            "adjusted_risk_pct":        sizing.adjusted_risk_pct,
            "system_expectancy_r":      stats.get("expectancy_r", 0.0),
            "passed":                   passed,
            "rejection_reasons":        rejection_reasons,
            "shadow_mode":              SHADOW_MODE,
        }
        _log_decision(record)

        if SHADOW_MODE:
            return None
        if not passed:
            return None

        approved = dict(trade)
        approved["regime_type"] = regime_result.regime
        approved["regime_confidence"] = regime_result.confidence
        approved["position_size_multiplier"] = sizing.multiplier
        approved["adjusted_risk_pct"] = sizing.adjusted_risk_pct
        approved["calibrated_probability"] = calibration.calibrated_probability
        return approved

    except Exception as e:
        print(f"  \u26a0\ufe0f  adaptive_core: error ({e}) — passing trade through unchanged (fail-safe)")
        try:
            _log_decision({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": trade.get("ticker", trade.get("symbol")),
                "error": str(e),
                "fail_safe_passthrough": True,
            })
        except Exception:
            pass
        return trade
