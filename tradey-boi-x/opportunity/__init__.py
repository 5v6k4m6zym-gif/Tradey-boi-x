"""
Tradey Boi X — Opportunity Optimisation Engine
===============================================
All features are optional and controlled by feature flags in config.py.
When all flags are False (the default), this package is a complete no-op
and the existing strategy operates exactly as it does today.

Public API
----------
run_opportunity_pass(scan_data)   — second-pass analysis on collected scan data
get_regime()                      — cached regime classification
refresh_regime()                  — force-refresh regime cache
regime_label(regime_dict)         — one-line Discord-friendly regime string
"""
from __future__ import annotations
import pandas as pd
from opportunity.config import ENABLE_OPPORTUNITY_ENGINE, ENABLE_ENHANCED_ALERTS
from opportunity.regime  import detect_regime, regime_label
from opportunity.scoring import score_opportunity
from opportunity.alerts  import send_opportunity_alert

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


__all__ = [
    "run_opportunity_pass",
    "get_regime",
    "refresh_regime",
    "regime_label",
    "score_opportunity",
    "send_opportunity_alert",
]
