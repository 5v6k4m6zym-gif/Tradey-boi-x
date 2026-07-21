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
    "ibkr_port":             7497,          # 7497=paper, 7496=live
    "ibkr_client_id":        1,
    "bot_enabled":           False,

    # Risk management
    "max_positions":         5,
    "risk_pct":              2.0,           # % of account per trade
    "max_daily_loss_pct":    3.0,
    "max_exposure_pct":      30.0,
    "brokerage":             2.0,           # $ per side

    # Exit parameters (optimal from Tradey Boi X v3 sweep)
    "hold_days":             15,
    "sl_mult_hi":            1.2,
    "sl_mult_mid":           1.0,
    "sl_mult_lo":            0.8,
    "target_hi":             12.0,
    "target_mid":            8.0,
    "target_lo":             5.0,

    # Signal quality gates
    "min_prob":              0.53,
    "min_score":             7,

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
