"""
Risk engine for Tradey Boi Pro.
Account-aware position sizing, exposure limits, daily loss limit, circuit breaker.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import db.database as db
import config.settings as cfg

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient


def position_size(
    account_value:    float,
    entry_price:      float,
    stop_price:       float,
    risk_pct:         float | None = None,
    regime_multiplier: float = 1.0,
) -> float:
    """
    Calculate number of shares to buy given a fixed-fractional risk rule.

    risk_pct          : % of account to risk on this trade (default from settings).
    regime_multiplier : 1.2 in BULL, 0.75 in NEUTRAL, 0.0 in BEAR
                        (from RegimeData.size_multiplier — default 1.0 = no adjustment).

    Returns 0 if the inputs are invalid or regime blocks trading.
    """
    if regime_multiplier <= 0:
        return 0.0   # BEAR regime — no new positions

    risk_pct  = risk_pct or cfg.get("risk_pct") or 2.0
    brokerage = (cfg.get("brokerage") or 2.0) * 2          # entry + exit
    stop_dist = entry_price - stop_price
    if stop_dist <= 0 or entry_price <= 0 or account_value <= 0:
        return 0.0

    # Scale by regime: 1.2× in BULL markets, 0.75× in NEUTRAL markets.
    # Cap at 4.0% to prevent any single trade becoming oversized.
    effective_risk_pct = min(risk_pct * regime_multiplier, 4.0)

    dollar_risk = account_value * (effective_risk_pct / 100)
    # Subtract brokerage from available risk budget
    dollar_risk = max(dollar_risk - brokerage, 0)
    shares = dollar_risk / stop_dist
    return math.floor(shares) if shares >= 1 else 0.0


def sl_and_target(entry: float, atr_pct: float) -> tuple[float, float]:
    """Return (stop_price, target_price) based on ATR and settings."""
    atr = atr_pct / 100 * entry
    if atr_pct >= 3.0:
        sl_mult = cfg.get("sl_mult_hi")  or 1.2
        tp_pct  = cfg.get("target_hi")   or 15.0
    elif atr_pct >= 1.5:
        sl_mult = cfg.get("sl_mult_mid") or 1.0
        tp_pct  = cfg.get("target_mid")  or 10.0
    else:
        sl_mult = cfg.get("sl_mult_lo")  or 0.8
        tp_pct  = cfg.get("target_lo")   or 7.0

    stop   = max(entry - sl_mult * atr, entry * 0.88)
    target = entry * (1 + tp_pct / 100)
    return round(stop, 4), round(target, 4)


def current_exposure(account_value: float) -> float:
    """
    % of account currently in open positions (based on entry values).
    """
    positions   = db.open_positions()
    if not positions or account_value <= 0:
        return 0.0
    total_value = sum(p["entry_price"] * p["quantity"] for p in positions)
    return (total_value / account_value) * 100


def can_open_new_position(account_value: float) -> tuple[bool, str]:
    """Returns (ok, reason). Checks all limits before allowing a new trade."""
    max_pos      = cfg.get("max_positions")      or 5
    max_exp      = cfg.get("max_exposure_pct")   or 30.0
    open_pos     = db.open_positions()

    if len(open_pos) >= max_pos:
        return False, f"Max positions reached ({max_pos})"

    exp = current_exposure(account_value)
    if exp >= max_exp:
        return False, f"Max exposure reached ({exp:.1f}% ≥ {max_exp}%)"

    if circuit_breaker_active():
        return False, "Circuit breaker active (consecutive losses)"

    if daily_loss_limit_hit(account_value):
        return False, "Daily loss limit hit — trading paused"

    return True, ""


def circuit_breaker_active() -> bool:
    """True if the last N trades are all losses within the pause window."""
    threshold    = int(cfg.get("cb_consecutive_losses") or 3)
    pause_days   = int(cfg.get("cb_pause_days") or 7)
    trades       = db.all_trades(limit=threshold * 3)
    if len(trades) < threshold:
        return False

    last_n = trades[:threshold]
    if not all(t["outcome"] in ("LOSS", "STOP", "HIT_STOP") for t in last_n):
        return False

    last_date_str = last_n[0].get("exit_date", "")[:10]
    if not last_date_str:
        return False
    try:
        last_dt  = datetime.strptime(last_date_str, "%Y-%m-%d")
        days_ago = (datetime.utcnow() - last_dt).days
        return days_ago <= pause_days
    except ValueError:
        return False


def daily_loss_limit_hit(account_value: float) -> bool:
    """True if today's realised losses exceed the daily loss limit."""
    limit_pct   = cfg.get("max_daily_loss_pct") or 3.0
    today_str   = datetime.utcnow().strftime("%Y-%m-%d")
    trades      = db.all_trades(limit=100)
    today_pnl   = sum(
        t["pnl"] for t in trades
        if t.get("exit_date", "")[:10] == today_str
    )
    if today_pnl >= 0 or account_value <= 0:
        return False
    loss_pct = abs(today_pnl) / account_value * 100
    return loss_pct >= limit_pct


def performance_metrics() -> dict:
    """
    Compute the full institutional metric set from all closed trades.
    Matches Tradey Boi X's stat suite: Profit Factor, Expectancy R,
    Sharpe, Sortino, Max Drawdown, Win/Loss streaks, Annualised Return.
    """
    _EMPTY = {
        "trade_count": 0, "win_rate": 0, "profit_factor": 0,
        "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
        "avg_gain_pct": 0, "avg_loss_pct": 0,
        "max_drawdown": 0, "sharpe": 0, "sortino": 0,
        "expectancy_r": 0, "win_streak": 0, "loss_streak": 0,
        "avg_hold_days": 0, "annualised_return_pct": 0,
    }
    trades = db.all_trades(limit=1000)
    if not trades:
        return _EMPTY

    wins = [t for t in trades if t["pnl"] >= 0]
    loss = [t for t in trades if t["pnl"] <  0]
    gw   = sum(t["pnl"] for t in wins)
    gl   = abs(sum(t["pnl"] for t in loss))
    pf   = gw / gl if gl > 0 else 99.0
    wr   = len(wins) / len(trades)

    # Return series (pnl_pct stored as decimal, e.g. 0.05 = 5%)
    rets     = [t["pnl_pct"] for t in trades]
    win_rets = [t["pnl_pct"] for t in wins]
    los_rets = [abs(t["pnl_pct"]) for t in loss]
    n        = len(rets)
    mean_r   = sum(rets) / n

    # Sharpe (annualised, assuming ~15-day avg hold)
    std = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / max(n - 1, 1))
    sharpe = (mean_r / std) * math.sqrt(252 / 15) if std > 0 else 0.0

    # Sortino (downside deviation only — penalises losses, not gains)
    neg_rets = [r for r in rets if r < 0]
    if neg_rets:
        downside = math.sqrt(sum(r ** 2 for r in neg_rets) / n)
        sortino  = (mean_r / downside * math.sqrt(252 / 15)) if downside > 0 else 0.0
    else:
        sortino  = 9.99   # no losing trades — cap display at 9.99

    # Expectancy in R-multiples
    avg_win_pct  = sum(win_rets) / len(win_rets)  if win_rets else 0.0
    avg_loss_pct = sum(los_rets) / len(los_rets)  if los_rets else 0.0
    r_unit       = avg_loss_pct if avg_loss_pct > 0 else 1.0
    expectancy_r = (wr * avg_win_pct - (1 - wr) * avg_loss_pct) / r_unit

    # Win / loss streaks (oldest → newest)
    best_win = cur_win = best_loss = cur_loss = 0
    for t in reversed(trades):
        if t["pnl"] >= 0:
            cur_win  += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win  = 0
        best_win  = max(best_win,  cur_win)
        best_loss = max(best_loss, cur_loss)

    # Average hold days
    hold_list = []
    for t in trades:
        hd = t.get("hold_days")
        if hd is not None:
            hold_list.append(float(hd))
        elif t.get("entry_date") and t.get("exit_date"):
            try:
                e  = datetime.strptime(t["entry_date"][:10], "%Y-%m-%d")
                x  = datetime.strptime(t["exit_date"][:10],  "%Y-%m-%d")
                hold_list.append(max((x - e).days, 1))
            except ValueError:
                pass
    avg_hold = sum(hold_list) / len(hold_list) if hold_list else 15.0

    # Annualised return (compound, by hold-day weighting)
    total_hold = max(sum(hold_list), 1)
    total_ret  = sum(rets)
    periods_yr = 252.0 / (total_hold / max(n, 1))
    ann_return = round(total_ret * periods_yr * 100, 2)

    # Max drawdown (cumulative $ PnL series, oldest first)
    cum = peak = max_dd = 0.0
    for t in reversed(trades):
        cum += t["pnl"]
        if cum > peak:
            peak = cum
        dd = (peak - cum) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "trade_count":           len(trades),
        "win_rate":              round(wr,           4),
        "profit_factor":         round(pf,           3),
        "total_pnl":             round(sum(t["pnl"] for t in trades), 2),
        "avg_win":               round(gw / len(wins), 2) if wins else 0,
        "avg_loss":              round(gl / len(loss), 2) if loss else 0,
        "avg_gain_pct":          round(avg_win_pct  * 100, 2),
        "avg_loss_pct":          round(avg_loss_pct * 100, 2),
        "max_drawdown":          round(max_dd,       4),
        "sharpe":                round(sharpe,       3),
        "sortino":               round(min(sortino, 9.99), 3),
        "expectancy_r":          round(expectancy_r, 3),
        "win_streak":            best_win,
        "loss_streak":           best_loss,
        "avg_hold_days":         round(avg_hold,     1),
        "annualised_return_pct": ann_return,
    }
