"""
Opportunity Optimisation Engine — Feature Flags & Configuration
All flags default to False so the existing bot behaviour is unchanged
unless explicitly enabled via environment variables.
"""
from __future__ import annotations
import os


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


ENABLE_MARKET_REGIME         = _flag("ENABLE_MARKET_REGIME", default=True)
ENABLE_OPPORTUNITY_ENGINE    = _flag("ENABLE_OPPORTUNITY_ENGINE", default=True)
ENABLE_ENHANCED_ALERTS       = _flag("ENABLE_ENHANCED_ALERTS", default=True)
ENABLE_ADVANCED_BACKTESTS    = _flag("ENABLE_ADVANCED_BACKTESTS", default=True)
ENABLE_PERFORMANCE_ANALYTICS = _flag("ENABLE_PERFORMANCE_ANALYTICS", default=True)
ENABLE_STRATEGY_CHALLENGER   = _flag("ENABLE_STRATEGY_CHALLENGER", default=True)
ENABLE_SYSTEM_HEALTH         = _flag("ENABLE_SYSTEM_HEALTH", default=True)
ENABLE_DRIFT_MONITORING      = _flag("ENABLE_DRIFT_MONITORING", default=True)

# ── Live vs backtest drift monitoring (institutional upgrade T011) ───────────
# Compares a recent rolling window of resolved live/paper trades against the
# older resolved-trade history (acting as the "validation baseline") using
# opportunity.backtester.compute_metrics(). Purely a reporting/alerting layer
# — never touches signal generation. No-op unless ENABLE_DRIFT_MONITORING=true.
DRIFT_LIVE_WINDOW_DAYS = int(os.getenv("DRIFT_LIVE_WINDOW_DAYS", "30"))
DRIFT_MIN_LIVE_TRADES  = int(os.getenv("DRIFT_MIN_LIVE_TRADES", "10"))
DRIFT_MIN_BASELINE_TRADES = int(os.getenv("DRIFT_MIN_BASELINE_TRADES", "20"))

# Absolute-difference thresholds beyond which live performance is flagged as
# having "drifted" from the baseline (either direction — better or worse).
DRIFT_THRESHOLDS: dict[str, float] = {
    "win_rate":       float(os.getenv("DRIFT_WIN_RATE_DELTA",       "0.15")),
    "expectancy_r":   float(os.getenv("DRIFT_EXPECTANCY_R_DELTA",   "0.20")),
    "profit_factor":  float(os.getenv("DRIFT_PROFIT_FACTOR_DELTA",  "0.30")),
}

# ── Trade Evaluation & Filtering Layer (Phase 8) ──────────────────────────────
# Purely additive instrumentation layer — never modifies the prediction model,
# signal generation, or execution logic. Runs in SHADOW_MODE by default, which
# means it only logs pass/fail decisions and never blocks an alert/trade.
ENABLE_TRADE_EVALUATOR = _flag("ENABLE_TRADE_EVALUATOR", default=True)
SHADOW_MODE            = _flag("SHADOW_MODE", default=False)

TRADE_EVAL_THRESHOLDS: dict[str, float] = {
    "min_edge_score":          float(os.getenv("TE_MIN_EDGE_SCORE",          "0.65")),
    "min_predictability_score": float(os.getenv("TE_MIN_PREDICTABILITY_SCORE", "0.60")),
    "min_risk_reward":         float(os.getenv("TE_MIN_RISK_REWARD",         "2.5")),
    "max_noise_index":         float(os.getenv("TE_MAX_NOISE_INDEX",         "1.2")),
}

TRADE_EVAL_LOG_PATH = os.getenv("TE_LOG_PATH", "logs/trade_evaluations.jsonl")

# ── Auto Threshold Tuner (SAFE, constrained) ──────────────────────────────────
# Purely additive: every AUTO_TUNER_INTERVAL_TRADES resolved decisions, nudges
# TRADE_EVAL_THRESHOLDS by at most AUTO_TUNER_MAX_STEP_PCT (5%) per cycle,
# clamped to the safe bounds below, and only ONE threshold family per cycle.
# Never runs while SHADOW_MODE is True (observation-only phase) or when
# ENABLE_AUTO_TUNER is False (default) — complete no-op otherwise.
ENABLE_AUTO_TUNER = _flag("ENABLE_AUTO_TUNER", default=True)

AUTO_TUNER_INTERVAL_TRADES = int(os.getenv("AUTO_TUNER_INTERVAL_TRADES", "50"))
AUTO_TUNER_MAX_STEP_PCT    = float(os.getenv("AUTO_TUNER_MAX_STEP_PCT", "0.05"))
AUTO_TUNER_MIN_TRADES_FLOOR = int(os.getenv("AUTO_TUNER_MIN_TRADES_FLOOR", "5"))

# (low, high) safe bounds — thresholds can never move outside this range,
# regardless of what the adjustment rules compute.
AUTO_TUNER_BOUNDS: dict[str, tuple[float, float]] = {
    "min_edge_score":          (0.55, 0.80),
    "min_predictability_score": (0.50, 0.75),
    "min_risk_reward":         (2.0, 4.0),
    "max_noise_index":         (1.0, 1.5),
}

AUTO_TUNER_STATE_PATH = os.getenv("AUTO_TUNER_STATE_PATH", "logs/auto_tuner_state.json")
AUTO_TUNER_LOG_PATH   = os.getenv("AUTO_TUNER_LOG_PATH", "logs/auto_tuner_decisions.jsonl")

# ── Adaptive Trading Core v4 (SAFE, constrained) ──────────────────────────────
# A second, independent additive wrapper layer, stacked ABOVE the Phase 8
# trade-evaluator: per-ticker regime detection, regime-aware threshold nudges,
# execution-quality filtering, confidence calibration, bounded position
# sizing, expectancy gating, and loss classification. Reuses TradeEvaluator's
# edge/predictability/noise/RR computation and PerformanceTracker's rolling
# stats rather than duplicating them. Off by default; when off, scanner.py's
# existing flow (Phase 8 evaluator or none) is completely unaffected.
ENABLE_ADAPTIVE_CORE = _flag("ENABLE_ADAPTIVE_CORE", default=True)

ADAPTIVE_CORE_LOG_PATH = os.getenv("ADAPTIVE_CORE_LOG_PATH", "logs/adaptive_core_decisions.jsonl")

# Per-ticker regime classification inputs (distinct from the macro
# ASX-200-level regime.py detector — this one looks at the individual
# ticker's own OHLCV window passed into process_trade_signal).
ADAPTIVE_REGIME_LOOKBACK = int(os.getenv("ADAPTIVE_REGIME_LOOKBACK", "20"))
ADAPTIVE_MIN_AVG_VOLUME  = float(os.getenv("ADAPTIVE_MIN_AVG_VOLUME", "50000"))

# Regime-aware threshold nudges — small, bounded multipliers applied to the
# SAME TRADE_EVAL_THRESHOLDS base values, clamped to AUTO_TUNER_BOUNDS so
# they can never exceed the limits already established as "safe" for this
# system. Never mutates TRADE_EVAL_THRESHOLDS itself — computed fresh
# per-trade and discarded.
ADAPTIVE_REGIME_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "TRENDING_UP":          {"min_edge_score": 0.95, "max_noise_index": 1.05},
    "TRENDING_DOWN":        {"min_edge_score": 0.95, "max_noise_index": 1.05},
    "CHOP":                 {"min_edge_score": 1.10, "min_predictability_score": 1.10},
    "VOLATILITY_EXPANSION": {"max_noise_index": 0.90, "min_risk_reward": 1.15},
    "LOW_LIQUIDITY":        {},   # hard-rejected before thresholds are even applied
}

# Execution Quality Filter — bounded 0-1 score; trade rejected if the score
# falls below this floor OR estimated execution cost exceeds estimated edge.
ADAPTIVE_MIN_EXECUTION_QUALITY = float(os.getenv("ADAPTIVE_MIN_EXECUTION_QUALITY", "0.40"))

# Position Sizing Engine — hard caps, cannot be exceeded regardless of what
# the quality-weighted multiplier computes to.
ADAPTIVE_BASE_RISK_PCT       = float(os.getenv("ADAPTIVE_BASE_RISK_PCT", "1.0"))
ADAPTIVE_MAX_SIZE_MULTIPLIER = float(os.getenv("ADAPTIVE_MAX_SIZE_MULTIPLIER", "1.5"))
ADAPTIVE_MIN_SIZE_MULTIPLIER = float(os.getenv("ADAPTIVE_MIN_SIZE_MULTIPLIER", "0.5"))
ADAPTIVE_MAX_RISK_PCT        = float(os.getenv("ADAPTIVE_MAX_RISK_PCT", "2.0"))

# Expectancy Engine — system-wide (not per-trade) gate using PerformanceTracker's
# rolling stats. Cold-start safe default: with too few resolved trades to judge,
# the gate PASSES (fail-open on insufficient data) so the system can never
# permanently lock itself out before it has any track record.
ADAPTIVE_EXPECTANCY_MIN_TRADES = int(os.getenv("ADAPTIVE_EXPECTANCY_MIN_TRADES", "20"))
ADAPTIVE_EXPECTANCY_WINDOW     = int(os.getenv("ADAPTIVE_EXPECTANCY_WINDOW", "100"))

# ── Realistic Backtesting — Trading Costs (institutional upgrade T003) ────────
# Applied only to backtest/report metrics (opportunity.backtester.compute_metrics)
# so validation reflects real-world execution costs. Never touches signal
# generation, decide(), or the live scanner/alert pipeline.
ENABLE_REALISTIC_COSTS = _flag("ENABLE_REALISTIC_COSTS", default=True)

# Per-side costs, expressed as a fraction of trade value. ASX (esp. small/mid
# caps, the bulk of the 408-ticker watchlist) carries materially wider spreads
# and slippage than large-cap US names.
TRADING_COSTS: dict[str, float] = {
    "commission_pct_us":  float(os.getenv("COST_COMMISSION_PCT_US",  "0.0005")),
    "slippage_pct_us":    float(os.getenv("COST_SLIPPAGE_PCT_US",    "0.0008")),
    "spread_pct_us":      float(os.getenv("COST_SPREAD_PCT_US",      "0.0005")),
    "commission_pct_asx": float(os.getenv("COST_COMMISSION_PCT_ASX", "0.0005")),
    "slippage_pct_asx":   float(os.getenv("COST_SLIPPAGE_PCT_ASX",   "0.0020")),
    "spread_pct_asx":     float(os.getenv("COST_SPREAD_PCT_ASX",     "0.0015")),
}

# ── Full System Audit Suite (BacktestEngine/ForwardValidator/PerformanceAnalytics/
#    SystemAudit/BugDetector) ───────────────────────────────────────────────────
# A read-only, parallel observability layer alongside the live bot. NEVER
# modifies the prediction model, signal generation, or execution logic — it
# only observes, logs, and evaluates. The `audit_trade()` wrapper never blocks
# execution, never mutates the trade object, and on ANY internal failure logs
# the error and returns a safe empty result rather than raising. Off by
# default (complete no-op) via ENABLE_AUDIT_ENGINE.
ENABLE_AUDIT_ENGINE = _flag("ENABLE_AUDIT_ENGINE", default=True)

AUDIT_LOG_PATH     = os.getenv("AUDIT_LOG_PATH", "logs/audit_trades.jsonl")
AUDIT_REPORTS_DIR  = os.getenv("AUDIT_REPORTS_DIR", "reports/audit")
AUDIT_STATE_PATH   = os.getenv("AUDIT_STATE_PATH", "logs/audit_state.json")

# BacktestEngine — bar-by-bar historical simulation (entry/stop/target/
# time-exit against real OHLCV bars), independent of the already-resolved
# signal_log outcomes computed by engine.resolve_outcomes(). Reuses the same
# slippage/commission/spread cost model as opportunity.costs.
AUDIT_BACKTEST_MAX_HOLD_DAYS = int(os.getenv("AUDIT_BACKTEST_MAX_HOLD_DAYS", "14"))

# SystemAudit anomaly-detection thresholds — advisory only, never block/stop
# the system; a detected anomaly is logged with a suggested likely cause.
AUDIT_REJECTION_RATE_SPIKE_DELTA = float(os.getenv("AUDIT_REJECTION_RATE_SPIKE_DELTA", "0.25"))
AUDIT_FREQUENCY_DROP_PCT         = float(os.getenv("AUDIT_FREQUENCY_DROP_PCT",         "0.50"))
AUDIT_CALIBRATION_DRIFT_DELTA    = float(os.getenv("AUDIT_CALIBRATION_DRIFT_DELTA",    "0.15"))
AUDIT_MIN_TRADES_FOR_CHECKS      = int(os.getenv("AUDIT_MIN_TRADES_FOR_CHECKS",        "20"))
AUDIT_RECENT_WINDOW_TRADES       = int(os.getenv("AUDIT_RECENT_WINDOW_TRADES",         "50"))

# PerformanceAnalytics rolling windows + signal-quality (edge-score) buckets
AUDIT_ROLLING_WINDOWS: tuple[int, ...] = (50, 100, 200)

AUDIT_EDGE_SCORE_BUCKETS: list[tuple[str, float, float]] = [
    ("0.0-0.4", 0.0, 0.4), ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8), ("0.8-1.0", 0.8, 1.01),
]

# ── Self-Optimising Strategy Engine (SAFE MODE) ───────────────────────────────
# A third, independent additive wrapper layer stacked between the Phase 8
# evaluator/Adaptive Core and execution: tags each trade with an inferred
# strategy_type, tracks per-(strategy, regime) performance, and gates/weights
# trades by that strategy's own track record. NEVER modifies the prediction
# model, signal generation, or execution logic — only gates/weights the
# existing decision. Off by default (complete no-op) via
# ENABLE_STRATEGY_OPTIMIZER. Reuses opportunity.adaptive_core.RegimeDetector
# for regime classification and opportunity.performance_tracker-style
# signal_log joins rather than duplicating them.
ENABLE_STRATEGY_OPTIMIZER = _flag("ENABLE_STRATEGY_OPTIMIZER", default=True)

STRATEGY_LOG_PATH         = os.getenv("STRATEGY_LOG_PATH",         "logs/strategy_optimizer_decisions.jsonl")
STRATEGY_WEIGHTS_PATH     = os.getenv("STRATEGY_WEIGHTS_PATH",     "logs/strategy_weights.json")
STRATEGY_WEIGHT_STATE_PATH = os.getenv("STRATEGY_WEIGHT_STATE_PATH", "logs/strategy_weight_state.json")

STRATEGY_TYPES: tuple[str, ...] = (
    "BREAKOUT", "PULLBACK", "TREND_CONTINUATION",
    "MEAN_REVERSION", "VOLATILITY_EXPANSION",
)

# Strategy Weighting Engine — bounded, gradual-only adjustments (Part 3/6).
# Every strategy starts at weight 1.0. Weights can NEVER leave [FLOOR, CAP]
# and can move by at most WEIGHT_MAX_STEP_PCT (5-10%) per update cycle.
STRATEGY_WEIGHT_FLOOR       = float(os.getenv("STRATEGY_WEIGHT_FLOOR", "0.2"))
STRATEGY_WEIGHT_CAP         = float(os.getenv("STRATEGY_WEIGHT_CAP",   "1.5"))
STRATEGY_WEIGHT_MAX_STEP_PCT = float(os.getenv("STRATEGY_WEIGHT_MAX_STEP_PCT", "0.08"))
STRATEGY_WEIGHT_UPDATE_INTERVAL_TRADES = int(os.getenv("STRATEGY_WEIGHT_UPDATE_INTERVAL_TRADES", "50"))

# Gating System (Part 4) — a strategy below this weight is treated as
# "disabled" for new trades (existing floor still applies so it can recover).
STRATEGY_MIN_ACTIVE_WEIGHT = float(os.getenv("STRATEGY_MIN_ACTIVE_WEIGHT", "0.3"))

# Minimum resolved trades for a strategy before its recent expectancy is
# allowed to gate new trades — fails OPEN (passes) below this, same
# cold-start-safe pattern as ADAPTIVE_EXPECTANCY_MIN_TRADES.
STRATEGY_MIN_EXPECTANCY_TRADES = int(os.getenv("STRATEGY_MIN_EXPECTANCY_TRADES", "20"))
STRATEGY_EXPECTANCY_WINDOW     = int(os.getenv("STRATEGY_EXPECTANCY_WINDOW", "100"))

# Regime → Allowed Strategies map (Part 5). LOW_LIQUIDITY is an explicit
# hard block of ALL strategies by design (mirrors the existing hard
# liquidity filter in adaptive_core) — this is the one deliberate exception
# to the "always keep >=1 active strategy per regime" rule, which otherwise
# governs weight-based disabling, not this explicit regime block.
REGIME_STRATEGY_MAP: dict[str, list[str]] = {
    "TRENDING_UP":          ["BREAKOUT", "PULLBACK", "TREND_CONTINUATION"],
    "TRENDING_DOWN":        ["PULLBACK", "MEAN_REVERSION"],
    "CHOP":                 ["MEAN_REVERSION"],
    "VOLATILITY_EXPANSION": ["BREAKOUT", "VOLATILITY_EXPANSION"],
    "LOW_LIQUIDITY":        [],   # hard block — no strategy is allowed to trade
}

# ── Opportunity scoring weights (must sum to 1.0) ─────────────────────────────
WEIGHTS: dict[str, float] = {
    "expected_return":    float(os.getenv("OPP_W_EXPECTED_RETURN",   "0.35")),
    "technical_strength": float(os.getenv("OPP_W_TECHNICAL",         "0.20")),
    "volume_expansion":   float(os.getenv("OPP_W_VOLUME",            "0.15")),
    "momentum":           float(os.getenv("OPP_W_MOMENTUM",          "0.10")),
    "news_catalyst":      float(os.getenv("OPP_W_NEWS",              "0.10")),
    "institutional":      float(os.getenv("OPP_W_INSTITUTIONAL",     "0.05")),
    "risk_reward":        float(os.getenv("OPP_W_RISK_REWARD",       "0.05")),
}

# ── Opportunity filter thresholds ─────────────────────────────────────────────
FILTERS: dict[str, float] = {
    "min_opportunity_score": float(os.getenv("OPP_MIN_SCORE",      "60")),
    "min_confidence":        float(os.getenv("OPP_MIN_CONFIDENCE",  "0.60")),
    "min_expected_upside":   float(os.getenv("OPP_MIN_UPSIDE",     "0.10")),
    "min_avg_daily_volume":  float(os.getenv("OPP_MIN_VOLUME",     "500000")),
    "min_rr_ratio":          float(os.getenv("OPP_MIN_RR",         "2.0")),
    "max_downside":          float(os.getenv("OPP_MAX_DOWNSIDE",   "0.08")),
}
