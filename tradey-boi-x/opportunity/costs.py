"""
Realistic Backtesting — Trading Cost Model (institutional upgrade T003).

Applies commission + slippage + spread assumptions to backtest/report P&L
metrics only. Never imported by engine.py, scanner.py, or decide() — this
module has zero influence on signal generation or live alerts.
"""
from __future__ import annotations

from opportunity.config import ENABLE_REALISTIC_COSTS, TRADING_COSTS


def is_asx_ticker(ticker: str) -> bool:
    return bool(ticker) and ticker.upper().endswith(".AX")


def round_trip_cost_pct(ticker: str) -> float:
    """
    Estimated round-trip (entry + exit) execution cost as a fraction of
    trade value: 2 x (commission + slippage + half-spread) per side.

    ASX (mostly small/mid-cap names on this watchlist) uses wider slippage
    and spread assumptions than large-cap US tickers.
    """
    if not ENABLE_REALISTIC_COSTS:
        return 0.0

    suffix = "asx" if is_asx_ticker(ticker) else "us"
    per_side = (
        TRADING_COSTS[f"commission_pct_{suffix}"]
        + TRADING_COSTS[f"slippage_pct_{suffix}"]
        + TRADING_COSTS[f"spread_pct_{suffix}"]
    )
    return per_side * 2.0


def apply_cost(actual_pct: float, ticker: str) -> float:
    """Net return after subtracting round-trip execution costs."""
    return (actual_pct or 0.0) - round_trip_cost_pct(ticker)
