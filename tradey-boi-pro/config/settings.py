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

    # Exit parameters — stops from v3 sweep, targets widened for better R:R
    "hold_days":             15,
    "sl_mult_hi":            1.2,
    "sl_mult_mid":           1.0,
    "sl_mult_lo":            0.8,
    "target_hi":             15.0,    # was 12 — achievable over 15-day hold, improves R:R
    "target_mid":            10.0,    # was 8
    "target_lo":             7.0,     # was 5

    # Dynamic stop management (mirrors backtest/engine.py exit mechanics exactly)
    "min_hold_days":         2,       # stop cannot trigger in first N days (entry-day noise)
    "be_trigger_r":          1.0,     # slide stop to entry when price hits entry+1R — protects against full -1R losses on reversals
    "trail_trigger_r":       2.0,     # start trailing at +2R peak — realistic within 15-day hold
    "trail_dist_r":          1.0,     # trail 1R below peak: at +2R peak, stop locks at +1R (solid partial profit)

    # Signal quality gates — tightened from 7/0.53 to filter marginal signals
    "min_prob":              0.58,    # was 0.53 — AI needs higher confidence
    "min_score":             8,       # was 7 — fewer but better-quality setups
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
