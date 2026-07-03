"""
Tradey Boi X — Opportunity Optimisation Engine (3.0)
=====================================================
All features are optional and controlled by feature flags in config.py.
When all flags are False (the default), this package is a complete no-op
and the existing strategy operates exactly as it does today.

Public API
----------
run_opportunity_pass(scan_data)   — second-pass analysis on collected scan data
get_regime()                      — cached regime classification
refresh_regime()                  — force-refresh regime cache
regime_label(regime_dict)         — one-line Discord-friendly regime string
run_backtest(mode, ...)           — Phase 4: walk-forward / out-of-sample etc.
run_performance_analytics()       — Phase 5: calibration + weekly Discord report
run_challenger(candidate_weights) — Phase 6: shadow strategy comparison
run_health_check()                — Phase 7: memory + log summary
wrap_run_scan(fn)                 — Phase 7: wraps scanner with health monitoring
run_drift_monitor()               — Phase 9: live vs backtest drift alerting
PerformanceTracker                — trade-eval decisions joined with outcomes
run_auto_tuner()                  — SAFE constrained threshold auto-tuning
"""
from __future__ import annotations
import pandas as pd

from opportunity.config  import (
    ENABLE_OPPORTUNITY_ENGINE,
    ENABLE_ENHANCED_ALERTS,
    ENABLE_ADVANCED_BACKTESTS,
    ENABLE_PERFORMANCE_ANALYTICS,
    ENABLE_STRATEGY_CHALLENGER,
    ENABLE_SYSTEM_HEALTH,
)
from opportunity.regime  import detect_regime, regime_label
from opportunity.scoring import score_opportunity
from opportunity.alerts  import send_opportunity_alert, send_outcome_alert
from opportunity.trade_evaluator import TradeEvaluator, process_trade_signal

_cached_regime: dict | None = None


def get_regime() -> dict | None:
    """Return cached regime dict, fetching fresh if not yet loaded."""
    global _cached_regime
    if _cached_regime is None:
        _cached_regime = detect_regime()
    return _cached_regime


def refresh_regime() -> dict | None:
    """Force-refresh the regime cache. Call once at the start of each scan."""
    global _cached_regime
    _cached_regime = detect_regime()
    return _cached_regime


def run_opportunity_pass(
    scan_data: list[tuple[str, pd.DataFrame]],
) -> int:
    """
    Second-pass opportunity analysis using data already fetched by the scanner.

    Parameters
    ----------
    scan_data : list of (ticker, df) tuples
        Data collected during the main scanner loop — no extra API calls made.

    Returns
    -------
    int
        Number of opportunity alerts sent (0 when engine is disabled).
    """
    if not ENABLE_OPPORTUNITY_ENGINE:
        return 0

    regime = get_regime()
    sent   = 0

    for ticker, df in scan_data:
        opp = score_opportunity(ticker, df, regime=regime)
        if opp is None:
            continue

        print(
            f"  🎯 Opportunity: {ticker} — "
            f"Score {opp['opportunity_score']:.0f}/100  "
            f"+{opp['expected_upside_pct']:.1f}% expected  "
            f"Conf {opp['confidence'] * 100:.0f}%"
        )

        if ENABLE_ENHANCED_ALERTS:
            ok = send_opportunity_alert(opp)
            if ok:
                sent += 1
                print(f"     └─ Opportunity alert sent ✅")

    if sent > 0:
        print(f"  Opportunity engine: {sent} high-conviction alert(s) sent.")

    return sent


# ── Lazy imports for heavy optional modules (avoid cost when flags are off) ───

def run_backtest(mode: str = "walk_forward", **kwargs):
    """Phase 4 — Backtesting Expansion. Returns None when flag is off."""
    from opportunity.backtester import run_backtest as _rb
    return _rb(mode=mode, **kwargs)


def run_performance_analytics(lookback_days: int = 7):
    """Phase 5 — Performance Learning. Returns None when flag is off."""
    from opportunity.performance import run_performance_analytics as _rpa
    return _rpa(lookback_days=lookback_days)


def send_weekly_performance_report(lookback_days: int = 7) -> bool:
    """Phase 5 — Weekly Discord performance report."""
    from opportunity.performance import send_weekly_performance_report as _swpr
    return _swpr(lookback_days=lookback_days)


def run_challenger(candidate_weights=None, **kwargs):
    """Phase 6 — Strategy Challenger Sandbox. Returns None when flag is off."""
    from opportunity.challenger import run_challenger as _rc
    return _rc(candidate_weights=candidate_weights, **kwargs)


def run_health_check():
    """Phase 7 — One-off health check. Returns None when flag is off."""
    from opportunity.health import run_health_check as _rhc
    return _rhc()


def wrap_run_scan(run_scan_fn):
    """Phase 7 — Wrap scanner with health monitoring."""
    from opportunity.health import wrap_run_scan as _wrs
    return _wrs(run_scan_fn)


def send_weekly_health_report(lookback_days: int = 7) -> bool:
    """Phase 7 — Weekly Discord health summary."""
    from opportunity.health import send_weekly_health_report as _swhr
    return _swhr(lookback_days=lookback_days)


def run_drift_monitor(live_window_days: int | None = None, notify: bool = True):
    """Phase 9 — Live vs backtest drift monitoring. Returns None when flag is off."""
    from opportunity.drift_monitor import run_drift_monitor as _rdm
    from opportunity.config import DRIFT_LIVE_WINDOW_DAYS
    return _rdm(live_window_days=live_window_days or DRIFT_LIVE_WINDOW_DAYS, notify=notify)


def get_performance_tracker():
    """Trade Evaluation upgrade — joins trade_evaluator decisions with their
    eventual outcomes. Always available (read-only reporting); returns a
    fresh PerformanceTracker instance for the caller to query."""
    from opportunity.performance_tracker import PerformanceTracker as _PT
    return _PT()


def run_auto_tuner(window: int | None = None):
    """Trade Evaluation upgrade — SAFE constrained auto threshold tuning.
    Returns None when ENABLE_AUTO_TUNER is off, SHADOW_MODE is on, or not
    enough new resolved trades have accumulated since the last cycle."""
    from opportunity.auto_tuner import maybe_tune as _mt
    from opportunity.config import AUTO_TUNER_INTERVAL_TRADES
    return _mt(window=window or AUTO_TUNER_INTERVAL_TRADES)


def process_adaptive_trade_signal(trade, market_data):
    """Adaptive Trading Core v4 — regime-aware thresholds, execution-quality
    filtering, confidence calibration, bounded dynamic sizing, and an
    expectancy gate, stacked above the Phase 8 trade evaluator. Returns
    `trade` unchanged when ENABLE_ADAPTIVE_CORE is off (complete no-op)."""
    from opportunity.adaptive_core import process_trade_signal as _pts
    return _pts(trade, market_data)


def classify_loss(resolved_entry, entry_context=None):
    """Adaptive Trading Core v4 — post-trade loss classification (system
    improvement only, never gates or blocks anything). Returns None for
    winning trades, a label string for losses, or "UNCLASSIFIED"."""
    from opportunity.adaptive_core import LossClassifier as _LC
    return _LC().classify(resolved_entry, entry_context=entry_context)


__all__ = [
    # Core
    "run_opportunity_pass",
    "get_regime",
    "refresh_regime",
    "regime_label",
    "score_opportunity",
    "send_opportunity_alert",
    "send_outcome_alert",
    # Phase 4
    "run_backtest",
    # Phase 5
    "run_performance_analytics",
    "send_weekly_performance_report",
    # Phase 6
    "run_challenger",
    # Phase 7
    "run_health_check",
    "wrap_run_scan",
    "send_weekly_health_report",
    # Phase 8
    "TradeEvaluator",
    "process_trade_signal",
    # Phase 9
    "run_drift_monitor",
    # Trade Evaluation upgrade — PerformanceTracker + AutoThresholdTuner
    "get_performance_tracker",
    "run_auto_tuner",
    # Adaptive Trading Core v4
    "process_adaptive_trade_signal",
    "classify_loss",
]
