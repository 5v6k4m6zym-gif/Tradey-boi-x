"""
Opportunity Optimisation Engine — Feature Flags & Configuration
All flags default to False so the existing bot behaviour is unchanged
unless explicitly enabled via environment variables.
"""
from __future__ import annotations
import os


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


ENABLE_MARKET_REGIME         = _flag("ENABLE_MARKET_REGIME")
ENABLE_OPPORTUNITY_ENGINE    = _flag("ENABLE_OPPORTUNITY_ENGINE")
ENABLE_ENHANCED_ALERTS       = _flag("ENABLE_ENHANCED_ALERTS")
ENABLE_ADVANCED_BACKTESTS    = _flag("ENABLE_ADVANCED_BACKTESTS")
ENABLE_PERFORMANCE_ANALYTICS = _flag("ENABLE_PERFORMANCE_ANALYTICS")
ENABLE_STRATEGY_CHALLENGER   = _flag("ENABLE_STRATEGY_CHALLENGER")
ENABLE_SYSTEM_HEALTH         = _flag("ENABLE_SYSTEM_HEALTH")

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
