"""
Settings management for Tradey Boi Pro.
All config persisted in SQLite — user never edits code.
"""
from __future__ import annotations
from db.database import get_setting, set_setting, get_all_settings

DEFAULTS: dict = {
    # Connection
    "mode":                  "PAPER",      # PAPER | LIVE
    "ibkr_host":             "127.0.0.1",
    "ibkr_port":             4002,          # 4002=Gateway paper, 4001=Gateway live, 7497=TWS paper
    "ibkr_client_id":        1,
    "bot_enabled":           False,

    # Risk management
    "max_positions":         5,
    "risk_pct":              2.0,           # % of account per trade (STRONG BUY baseline)
    "risk_pct_elite":        3.0,           # % of account for ELITE signals (higher conviction = larger size)
    "regime_size_scale":     True,          # scale position size by market regime (1.2× BULL, 0.75× NEUTRAL)
    "max_daily_loss_pct":    3.0,
    "max_exposure_pct":      30.0,
    "brokerage":             2.0,           # $ per side

    # Exit parameters — pro-sweep winner: tight ATR stops + 15-day hold + early BE
    # Backtest result: PF=2.248, WR=80%, ROI=+6.4%, 54 trades, MaxDD=1.4%
    "hold_days":             15,
    "sl_mult_hi":            0.8,     # tight stop for high-ATR stocks (≥3% ATR)
    "sl_mult_mid":           0.6,     # tight stop for mid-ATR stocks (1.5-3% ATR)
    "sl_mult_lo":            0.5,     # tight stop for low-ATR stocks (<1.5% ATR)
    "target_hi":             15.0,
    "target_mid":            10.0,
    "target_lo":             7.0,

    # Dynamic stop management — pro-sweep winner: BE=0.5R, Trail=1.5R/0.7R
    "min_hold_days":         2,       # stop cannot trigger in first N days (entry-day noise)
    "be_trigger_r":          0.5,     # slide stop to entry at +0.5R — fast BE protection reduces avg loss
    "trail_trigger_r":       1.5,     # start trailing at +1.5R peak
    "trail_dist_r":          0.7,     # trail 0.7R below peak

    # Signal quality gates — pro-sweep winner: score≥5, prob≥0.50
    # Lowered from 8/0.58: oversold-recovery signals (RSI 35-42, vol>1.5) score 5
    # and have heuristic prob 0.55-0.57 — previously excluded, now included
    "min_prob":              0.55,    # raised from 0.50 — require higher AI confidence
    "min_score":             7,       # raised from 6 — requires breakout OR high-prob setup
    "min_expected_r":        1.5,     # minimum EV in R units (gates low R:R setups)
    "min_composite":         7.5,     # live bot: composite_score threshold (ranker 0-10 scale)

    # Earnings guard — skip entries within N calendar days of a known earnings date.
    # Earnings can gap a stock ±15-30% overnight, bypassing the stop loss entirely.
    # 5 days gives a full trading week of protection. Set 0 to disable.
    "earnings_guard_days":   5,

    # Circuit breaker
    "cb_consecutive_losses": 3,
    "cb_pause_days":         7,

    # Scanner (Pro-specific — much more frequent than X)
    "scan_interval_mins":    15,            # scan every 15 min during market hours
    "enabled_markets":       ["ASX", "US"], # which markets to scan

    # X integration (optional secondary source)
    "signal_log_path":       "../tradey-boi-x/signal_log.json",
}


def get(key: str):
    val = get_setting(key)
    if val is None:
        return DEFAULTS.get(key)
    return val


def set(key: str, value):
    set_setting(key, value)


def all_settings() -> dict:
    stored = get_all_settings()
    return {**DEFAULTS, **stored}


def ensure_defaults():
    stored = get_all_settings()
    for key, val in DEFAULTS.items():
        if key not in stored:
            set_setting(key, val)
