"""
Phase 4 — Backtesting Expansion
Walk-forward, out-of-sample, historical simulation, and paper-trading modes
against the existing signal log.

Feature flag: ENABLE_ADVANCED_BACKTESTS  (default: false → complete no-op)
Writes JSON reports to tradey-boi-x/reports/. Never modifies engine.py.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from opportunity.config import ENABLE_ADVANCED_BACKTESTS

BASE_DIR    = Path(__file__).parent.parent
LOG_FILE    = BASE_DIR / "signal_log.json"
REPORTS_DIR = BASE_DIR / "reports"

WIN_OUTCOMES: tuple = ("WIN", "HIT_TARGET", "EXPIRED_GAIN")

BACKTEST_MODES = ("walk_forward", "out_of_sample", "historical_sim", "paper")


# ─── Signal log helpers ───────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return []


def _resolved(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("outcome") is not None]


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict]) -> dict[str, Any]:
    """
    Compute standard backtest metrics from a list of resolved trade entries.
    All trades must have an `outcome` field.  Accepts empty list gracefully.

    Returns
    -------
    dict with keys:
        trade_count, win_count, loss_count, win_rate, loss_rate,
        avg_gain_pct, avg_loss_pct, profit_factor, max_drawdown_pct,
        sharpe_ratio, sortino_ratio, expectancy_r, avg_hold_days,
        annualised_return_pct, winning_streak, losing_streak,
        false_positive_rate, false_negative_rate
    """
    if not trades:
        return _empty_metrics()

    wins   = [t for t in trades if t.get("outcome") in WIN_OUTCOMES]
    losses = [t for t in trades if t.get("outcome") not in WIN_OUTCOMES]

    total       = len(trades)
    win_count   = len(wins)
    loss_count  = len(losses)
    win_rate    = win_count / total if total else 0.0

    gain_pcts = [t.get("actual_pct", 0.0) or 0.0 for t in wins]
    loss_pcts = [abs(t.get("actual_pct", 0.0) or 0.0) for t in losses]

    avg_gain = statistics.mean(gain_pcts) if gain_pcts else 0.0
    avg_loss = statistics.mean(loss_pcts) if loss_pcts else 0.0

    gross_profit = sum(gain_pcts)
    gross_loss   = sum(loss_pcts)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Expectancy in R-multiples (simplified: treat avg_gain / avg_loss as 1R)
    r_unit = avg_loss if avg_loss > 0 else 1.0
    expectancy_r = (win_rate * avg_gain - (1 - win_rate) * avg_loss) / r_unit

    # Hold days
    hold_days_list = _compute_hold_days(trades)
    avg_hold_days  = round(statistics.mean(hold_days_list), 1) if hold_days_list else 0.0

    # Returns series for Sharpe / Sortino
    returns = [t.get("actual_pct", 0.0) or 0.0 for t in trades]
    sharpe  = _sharpe(returns)
    sortino = _sortino(returns)

    # Max drawdown
    max_dd = _max_drawdown(returns)

    # Annualised return (assume 252 trading days / year)
    trading_days = max(sum(hold_days_list), 1)
    total_return = sum(returns)
    periods_per_year = 252.0 / (trading_days / max(total, 1))
    ann_return = round(total_return * periods_per_year * 100, 2)

    # Streaks
    win_streak, loss_streak = _streaks(trades)

    # FP/FN: scored as alert where outcome is loss (FP) vs no-alert miss (approx 0)
    fp_rate = round(loss_count / total, 4) if total else 0.0
    fn_rate = 0.0   # can't compute without non-alerted universe — placeholder

    return {
        "trade_count":          total,
        "win_count":            win_count,
        "loss_count":           loss_count,
        "win_rate":             round(win_rate, 4),
        "loss_rate":            round(1 - win_rate, 4),
        "avg_gain_pct":         round(avg_gain * 100, 2),
        "avg_loss_pct":         round(avg_loss * 100, 2),
        "profit_factor":        round(profit_factor, 3),
        "max_drawdown_pct":     round(max_dd * 100, 2),
        "sharpe_ratio":         round(sharpe, 3),
        "sortino_ratio":        round(sortino, 3),
        "expectancy_r":         round(expectancy_r, 3),
        "avg_hold_days":        avg_hold_days,
        "annualised_return_pct": ann_return,
        "winning_streak":       win_streak,
        "losing_streak":        loss_streak,
        "false_positive_rate":  fp_rate,
        "false_negative_rate":  fn_rate,
    }


def _empty_metrics() -> dict[str, Any]:
    return {k: 0 for k in (
        "trade_count", "win_count", "loss_count", "win_rate", "loss_rate",
        "avg_gain_pct", "avg_loss_pct", "profit_factor", "max_drawdown_pct",
        "sharpe_ratio", "sortino_ratio", "expectancy_r", "avg_hold_days",
        "annualised_return_pct", "winning_streak", "losing_streak",
        "false_positive_rate", "false_negative_rate",
    )}


def _compute_hold_days(trades: list[dict]) -> list[float]:
    """Estimate hold days per trade from signal_date and pred_days."""
    out = []
    for t in trades:
        hold = t.get("pred_days") or 14
        out.append(float(hold))
    return out


def _sharpe(returns: list[float], rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns) - rf
    std  = statistics.stdev(returns)
    return (mean / std * math.sqrt(252)) if std > 0 else 0.0


def _sortino(returns: list[float], rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean      = statistics.mean(returns) - rf
    neg_rets  = [r for r in returns if r < rf]
    if not neg_rets:
        return float("inf")
    downside = math.sqrt(sum((r - rf) ** 2 for r in neg_rets) / len(returns))
    return (mean / downside * math.sqrt(252)) if downside > 0 else 0.0


def _max_drawdown(returns: list[float]) -> float:
    peak = 0.0
    equity = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _streaks(trades: list[dict]) -> tuple[int, int]:
    best_win = cur_win = best_loss = cur_loss = 0
    for t in trades:
        if t.get("outcome") in WIN_OUTCOMES:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win  = 0
        best_win  = max(best_win,  cur_win)
        best_loss = max(best_loss, cur_loss)
    return best_win, best_loss


# ─── Backtest modes ───────────────────────────────────────────────────────────

def _walk_forward(
    entries: list[dict],
    window_size: int = 90,
    test_size:   int = 30,
) -> list[dict]:
    """
    Slice resolved entries by signal_date into rolling windows.
    Returns one metrics dict per window.
    """
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: e.get("signal_date", ""))
    results = []
    start   = 0

    while start + window_size < len(sorted_entries):
        train = sorted_entries[start : start + window_size]
        test  = sorted_entries[start + window_size : start + window_size + test_size]
        if not test:
            break
        window_metrics = compute_metrics(test)
        # Label with date range
        window_metrics["window_start"] = train[0].get("signal_date", "")
        window_metrics["window_end"]   = test[-1].get("signal_date",  "")
        window_metrics["train_size"]   = len(train)
        window_metrics["test_size"]    = len(test)
        results.append(window_metrics)
        start += test_size

    return results


def _out_of_sample(entries: list[dict], holdout_pct: float = 0.20) -> dict:
    """Keep the last `holdout_pct` of entries as an unseen test set."""
    if not entries:
        return compute_metrics([])
    sorted_entries = sorted(entries, key=lambda e: e.get("signal_date", ""))
    split = max(1, int(len(sorted_entries) * (1 - holdout_pct)))
    test  = sorted_entries[split:]
    m     = compute_metrics(test)
    m["holdout_pct"] = holdout_pct
    m["test_start"]  = test[0].get("signal_date",  "") if test else ""
    m["test_end"]    = test[-1].get("signal_date", "") if test else ""
    return m


def _historical_simulation(entries: list[dict]) -> dict:
    """Run metrics over ALL resolved entries — full historical record."""
    m = compute_metrics(entries)
    m["mode"] = "historical_sim"
    return m


def _paper_trading_snapshot(entries: list[dict], days: int = 30) -> dict:
    """Metrics for the most recent `days` days of signals (live-track proxy)."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [e for e in entries if (e.get("signal_date") or "") >= cutoff]
    m = compute_metrics(recent)
    m["mode"]           = "paper"
    m["paper_days"]     = days
    m["paper_since"]    = cutoff
    m["open_positions"] = sum(
        1 for e in entries if e.get("outcome") is None
        and (e.get("signal_date") or "") >= cutoff
    )
    return m


# ─── Report helpers ───────────────────────────────────────────────────────────

def _save_backtest_report(results: dict, mode: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = REPORTS_DIR / f"backtest_{mode}_{datetime.utcnow().strftime('%Y%m%d')}.json"
    fname.write_text(json.dumps(results, indent=2))
    return fname


def send_backtest_discord(results: dict, mode: str) -> bool:
    """Post a short backtest summary to Discord. Returns True on success."""
    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook")
    if not webhook:
        return False

    m = results.get("summary", results)
    msg = (
        f"📊 **BACKTEST REPORT — {mode.upper()}** "
        f"({datetime.utcnow().strftime('%d %b %Y')})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Trades:       {m.get('trade_count', 0)}\n"
        f"Win Rate:     {m.get('win_rate', 0)*100:.1f}%\n"
        f"Avg Gain:     +{m.get('avg_gain_pct', 0):.1f}%  "
        f"Avg Loss: -{m.get('avg_loss_pct', 0):.1f}%\n"
        f"Profit Factor: {m.get('profit_factor', 0):.2f}\n"
        f"Max Drawdown: {m.get('max_drawdown_pct', 0):.1f}%\n"
        f"Sharpe:       {m.get('sharpe_ratio', 0):.2f}  "
        f"Sortino: {m.get('sortino_ratio', 0):.2f}\n"
        f"Expectancy:   {m.get('expectancy_r', 0):+.2f}R  "
        f"Ann. Return: {m.get('annualised_return_pct', 0):+.1f}%\n"
        f"Avg Hold:     {m.get('avg_hold_days', 0):.1f} days  "
        f"Best streak: {m.get('winning_streak', 0)}W / {m.get('losing_streak', 0)}L"
    )
    try:
        import urllib.request
        data = json.dumps({"content": msg}).encode()
        req  = urllib.request.Request(webhook, data=data,
                                       headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10):
            pass
        return True
    except Exception:
        return False


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    mode:        str = "walk_forward",
    window_size: int = 90,
    test_size:   int = 30,
    holdout_pct: float = 0.20,
    paper_days:  int = 30,
    notify:      bool = True,
) -> dict | None:
    """
    Run a backtest against the existing signal log.

    Parameters
    ----------
    mode         : one of 'walk_forward', 'out_of_sample', 'historical_sim', 'paper'
    window_size  : (walk_forward) training window in trades
    test_size    : (walk_forward) test window in trades
    holdout_pct  : (out_of_sample) fraction held out as unseen test
    paper_days   : (paper) look-back window in calendar days
    notify       : send Discord summary if True

    Returns None when flag is off or log is empty.
    """
    if not ENABLE_ADVANCED_BACKTESTS:
        return None

    entries = _resolved(_load_log())
    if not entries:
        return None

    if mode == "walk_forward":
        windows  = _walk_forward(entries, window_size, test_size)
        if not windows:
            return None
        all_m    = [w for w in windows]
        # Aggregate across windows
        def _avg(key: str) -> float:
            vals = [w.get(key, 0) for w in all_m if isinstance(w.get(key), (int, float))]
            return round(statistics.mean(vals), 4) if vals else 0.0
        summary = {k: _avg(k) for k in _empty_metrics()}
        summary["trade_count"] = sum(w.get("trade_count", 0) for w in all_m)
        summary["windows"]     = len(windows)
        result = {"mode": mode, "summary": summary, "windows": windows}

    elif mode == "out_of_sample":
        result = {"mode": mode, "summary": _out_of_sample(entries, holdout_pct)}

    elif mode == "historical_sim":
        result = {"mode": mode, "summary": _historical_simulation(entries)}

    elif mode == "paper":
        result = {"mode": mode, "summary": _paper_trading_snapshot(entries, paper_days)}

    else:
        raise ValueError(f"Unknown backtest mode: {mode!r}. "
                         f"Choose from {BACKTEST_MODES}")

    result["generated_at"] = datetime.utcnow().isoformat()
    report_path = _save_backtest_report(result, mode)
    print(f"  📊 Backtest report saved: {report_path}")

    if notify:
        send_backtest_discord(result, mode)

    return result
