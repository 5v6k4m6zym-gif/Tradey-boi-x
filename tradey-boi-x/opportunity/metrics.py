"""
Multi-metric performance scorer — v4.0 Adaptive Gate Validation.

Priority order (spec §2):
  1. Profit Factor     6. Sortino Ratio      11. Average Hold Time
  2. Expectancy        7. Calmar Ratio        12. Risk/Reward Ratio
  3. CAGR              8. Win Rate            13. Trade Frequency
  4. Max Drawdown      9. Average Winner      14. Equity Curve Stability
  5. Sharpe Ratio     10. Average Loser
"""
from __future__ import annotations
import math
import statistics
from datetime import datetime
from typing import Sequence

RISK_FREE_ANNUAL = 0.04
TRADING_DAYS     = 252

WIN_OUTCOMES = frozenset(
    {"HIT_TARGET", "EXPIRED_WIN", "WIN", "TARGET", "PARTIAL_WIN"}
)


def compute_metrics(trades: Sequence[dict], *, account_start: float = 10_000) -> dict:
    """
    Compute the full 14-metric suite from a sequence of resolved signal_log entries.

    Required fields per entry:
        actual_pct  — realised % return (0.05 = +5%)
        outcome     — outcome string
        signal_date — YYYY-MM-DD  (used for CAGR span)

    Optional:
        stop_price, entry_price — for R-multiple computation
        hold_days               — holding period
    """
    resolved = [t for t in trades if t.get("actual_pct") is not None]
    if not resolved:
        return _empty()

    n    = len(resolved)
    pcts = [float(t["actual_pct"]) for t in resolved]
    wins  = [p for p in pcts if p >= 0]
    loses = [p for p in pcts if p <  0]

    wr       = len(wins)  / n
    avg_win  = sum(wins)  / len(wins)  if wins  else 0.0
    avg_loss = sum(loses) / len(loses) if loses else 0.0

    gross_win  = sum(wins)
    gross_loss = abs(sum(loses))
    pf         = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    expectancy = wr * avg_win + (1 - wr) * avg_loss
    rr         = abs(avg_win / avg_loss) if avg_loss else 99.0

    # R-multiple expectancy — matches engine.py _expectancy_stats() formula exactly.
    # Uses same 1.3× asymmetric loss penalty so this value is directly comparable
    # to the adaptive-gate tighten/ease thresholds (−0.30R / +0.80R).
    def _r(t: dict) -> float:
        try:
            actual   = float(t.get("actual_pct") or 0.0)
            entry_px = float(t.get("entry_price") or 0.0)
            stop_px  = t.get("stop_price")
            if entry_px > 0 and stop_px and float(stop_px) > 0:
                risk_pct = abs(entry_px - float(stop_px)) / entry_px
            else:
                risk_pct = max(abs(float(t.get("target_pct") or 0.06)), 0.001)
            return max(-5.0, min(5.0, actual / max(risk_pct, 0.001)))
        except Exception:
            return 0.0

    win_Rs  = [_r(t) for t in resolved if float(t.get("actual_pct", 0)) >= 0]
    loss_Rs = [abs(_r(t)) for t in resolved if float(t.get("actual_pct", 0)) < 0]
    avg_win_R  = sum(win_Rs)  / len(win_Rs)  if win_Rs  else 0.0
    avg_loss_R = sum(loss_Rs) / len(loss_Rs) if loss_Rs else 0.0
    expectancy_r = avg_win_R * wr - avg_loss_R * 1.3 * (1 - wr)

    dates: list[datetime] = []
    for t in resolved:
        ds = t.get("signal_date", "")
        try:
            dates.append(datetime.strptime(str(ds)[:10], "%Y-%m-%d"))
        except Exception:
            pass

    span_days = max((max(dates) - min(dates)).days, 1) if len(dates) >= 2 else max(n * 10, 1)
    years     = span_days / 365.25
    avg_hold  = sum(t.get("hold_days", span_days / n) for t in resolved) / n

    equity    = account_start
    eq_curve  = [equity]
    for p in pcts:
        equity = max(equity * (1 + p * 0.02), 0.01)
        eq_curve.append(equity)

    peak   = eq_curve[0]
    max_dd = 0.0
    for e in eq_curve:
        peak   = max(peak, e)
        dd     = (e - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

    final_eq = eq_curve[-1]
    cagr     = (final_eq / account_start) ** (1 / years) - 1 if years > 0 else 0.0

    if n > 1:
        mean_r  = statistics.mean(pcts)
        std_r   = statistics.stdev(pcts)
        neg     = [p for p in pcts if p < 0]
        down_r  = statistics.stdev(neg) if len(neg) > 1 else std_r
        ann_f   = math.sqrt(TRADING_DAYS / max(avg_hold, 1))
        rf_pt   = (RISK_FREE_ANNUAL / TRADING_DAYS) * avg_hold
        sharpe  = ((mean_r - rf_pt) / std_r)  * ann_f if std_r  > 0 else 0.0
        sortino = ((mean_r - rf_pt) / down_r) * ann_f if down_r > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    calmar      = cagr / abs(max_dd) if max_dd < 0 else (99.0 if cagr > 0 else 0.0)
    trade_freq  = (n / span_days * 30) if span_days > 0 else 0.0

    if len(eq_curve) > 3:
        xs    = list(range(len(eq_curve)))
        xm    = sum(xs) / len(xs)
        ym    = sum(eq_curve) / len(eq_curve)
        sxx   = sum((x - xm) ** 2 for x in xs)
        sxy   = sum((x - xm) * (y - ym) for x, y in zip(xs, eq_curve))
        slope = sxy / sxx if sxx > 0 else 0.0
        icept = ym - slope * xm
        sstot = sum((y - ym) ** 2 for y in eq_curve)
        ssres = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, eq_curve))
        r2    = max(0.0, 1.0 - ssres / sstot) if sstot > 0 else 0.0
    else:
        r2 = 0.0

    return {
        "n":                 n,
        "win_rate":          round(wr,             4),
        "profit_factor":     round(pf,             3),
        "expectancy":        round(expectancy,     4),
        "expectancy_r":      round(expectancy_r,   4),
        "cagr":              round(cagr,           4),
        "max_drawdown":      round(max_dd,         4),
        "sharpe":            round(sharpe,         3),
        "sortino":           round(sortino,        3),
        "calmar":            min(round(calmar,     3), 99.0),
        "avg_win":           round(avg_win,        4),
        "avg_loss":          round(avg_loss,       4),
        "avg_hold_days":     round(avg_hold,       1),
        "rr_ratio":          min(round(rr,         3), 99.0),
        "trade_freq_month":  round(trade_freq,     2),
        "equity_stability":  round(r2,             4),
        "final_equity":      round(final_eq,       2),
    }


def _empty() -> dict:
    return {
        "n": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "expectancy": 0.0, "expectancy_r": 0.0,
        "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_hold_days": 0.0,
        "rr_ratio": 0.0, "trade_freq_month": 0.0, "equity_stability": 0.0,
        "final_equity": 0.0,
    }
