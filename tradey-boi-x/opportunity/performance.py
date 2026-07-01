"""
Phase 5 — Performance Learning & Calibration
Reads the existing signal_log.json, builds confidence-calibration tables,
and sends a weekly Discord report. Never writes to engine.py.

Feature flag: ENABLE_PERFORMANCE_ANALYTICS  (default: false → complete no-op)
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from opportunity.config import ENABLE_PERFORMANCE_ANALYTICS

BASE_DIR = Path(__file__).parent.parent
LOG_FILE = BASE_DIR / "signal_log.json"

WIN_OUTCOMES: tuple = ("WIN", "HIT_TARGET", "EXPIRED_GAIN")

# Confidence calibration buckets: (label, min_inclusive, max_exclusive)
CALIB_BUCKETS: list[tuple[str, float, float]] = [
    ("50–60%", 0.50, 0.60),
    ("60–70%", 0.60, 0.70),
    ("70–80%", 0.70, 0.80),
    ("80%+",   0.80, 1.01),
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return []


def _resolved_entries() -> list[dict]:
    return [e for e in _load_log() if e.get("outcome") is not None]


# ─── Calibration ──────────────────────────────────────────────────────────────

def calibration_buckets(entries: list[dict]) -> list[dict]:
    """
    Group resolved trades by predicted confidence (prob field) and compute
    actual win rate per bucket.

    Returns
    -------
    list of dicts: [{"label", "predicted_min", "predicted_max",
                     "count", "actual_win_rate", "calibration_status"}]
    """
    results = []
    for label, lo, hi in CALIB_BUCKETS:
        bucket = [
            e for e in entries
            if lo <= float(e.get("prob", 0) or 0) < hi
        ]
        count = len(bucket)
        if count == 0:
            results.append({
                "label": label, "predicted_min": lo, "predicted_max": hi,
                "count": 0, "actual_win_rate": None,
                "calibration_status": "NO_DATA",
            })
            continue

        wins        = sum(1 for e in bucket if e.get("outcome") in WIN_OUTCOMES)
        actual_rate = wins / count
        mid         = (lo + hi) / 2.0

        if abs(actual_rate - mid) <= 0.05:
            status = "WELL_CALIBRATED"
        elif actual_rate > mid + 0.05:
            status = "UNDERCONFIDENT"
        else:
            status = "OVERCONFIDENT"

        results.append({
            "label":              label,
            "predicted_min":      lo,
            "predicted_max":      hi,
            "count":              count,
            "actual_win_rate":    round(actual_rate, 4),
            "calibration_status": status,
        })

    return results


# ─── Sector aggregation ───────────────────────────────────────────────────────

_SECTOR_MAP: dict[str, str] = {
    "BHP.AX": "Resources", "RIO.AX": "Resources", "FMG.AX": "Resources",
    "NCM.AX": "Gold",      "NST.AX": "Gold",       "EVN.AX": "Gold",
    "CBA.AX": "Banks",     "ANZ.AX": "Banks",      "NAB.AX": "Banks",     "WBC.AX": "Banks",
    "CSL.AX": "Healthcare","RMD.AX": "Healthcare",
    "WES.AX": "Consumer",  "WOW.AX": "Consumer",   "COL.AX": "Consumer",
    "XRO.AX": "Tech",      "WTC.AX": "Tech",       "CAR.AX": "Tech",
    "LTR.AX": "Lithium",   "PLS.AX": "Lithium",    "AKE.AX": "Lithium",
    "PDN.AX": "Uranium",   "BOE.AX": "Uranium",
}

def _ticker_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker, "Other")


def sector_performance(entries: list[dict]) -> list[dict]:
    """Aggregate P&L by sector. Returns list of {sector, count, avg_pct, win_rate}."""
    sector_data: dict[str, list] = {}
    for e in entries:
        sector = _ticker_sector(e.get("ticker", ""))
        sector_data.setdefault(sector, []).append(e)

    results = []
    for sector, trades in sorted(sector_data.items()):
        count = len(trades)
        wins  = sum(1 for t in trades if t.get("outcome") in WIN_OUTCOMES)
        pcts  = [t.get("actual_pct", 0.0) or 0.0 for t in trades]
        results.append({
            "sector":      sector,
            "count":       count,
            "win_rate":    round(wins / count, 4) if count else 0,
            "avg_pct":     round(statistics.mean(pcts) * 100, 2) if pcts else 0,
            "total_pct":   round(sum(pcts) * 100, 2),
        })

    results.sort(key=lambda x: x["avg_pct"], reverse=True)
    return results


# ─── Summary ─────────────────────────────────────────────────────────────────

def performance_summary(entries: list[dict], lookback_days: int = 7) -> dict[str, Any]:
    """
    Produce a summary dict for the last `lookback_days` days of resolved trades.
    """
    cutoff   = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent   = [e for e in entries if (e.get("signal_date") or "") >= cutoff]
    total    = len(recent)
    wins     = sum(1 for e in recent if e.get("outcome") in WIN_OUTCOMES)
    pcts     = [e.get("actual_pct", 0.0) or 0.0 for e in recent]
    holds    = [float(e.get("pred_days", 14) or 14) for e in recent]

    win_rate = wins / total if total else 0.0

    # Expectancy in R-units
    gains = [p for p in pcts if p >= 0]
    lossv = [abs(p) for p in pcts if p < 0]
    avg_g = statistics.mean(gains) if gains else 0.0
    avg_l = statistics.mean(lossv) if lossv else 0.0
    r_unit = avg_l if avg_l > 0 else 1.0
    expectancy = (win_rate * avg_g - (1 - win_rate) * avg_l) / r_unit

    calibration = calibration_buckets(entries)   # use full history for calibration
    sectors     = sector_performance(recent)

    best_sector  = sectors[0]  if sectors else None
    worst_sector = sectors[-1] if sectors else None

    return {
        "period_days":      lookback_days,
        "cutoff_date":      cutoff,
        "resolved_count":   total,
        "win_count":        wins,
        "loss_count":       total - wins,
        "win_rate":         round(win_rate, 4),
        "expectancy_r":     round(expectancy, 3),
        "avg_hold_days":    round(statistics.mean(holds), 1) if holds else 0.0,
        "calibration":      calibration,
        "sector_breakdown": sectors,
        "best_sector":      best_sector,
        "worst_sector":     worst_sector,
    }


# ─── Discord report ───────────────────────────────────────────────────────────

def send_weekly_performance_report(lookback_days: int = 7) -> bool:
    """Build and post the weekly performance report to Discord."""
    if not ENABLE_PERFORMANCE_ANALYTICS:
        return False

    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook")
    if not webhook:
        return False

    entries = _resolved_entries()
    s = performance_summary(entries, lookback_days)

    date_str = datetime.utcnow().strftime("%d %b %Y")
    lines = [
        f"📊 **WEEKLY PERFORMANCE — {date_str}**",
        f"Resolved trades: {s['resolved_count']}  |  "
        f"Win rate: {s['win_rate']*100:.0f}%",
        f"Expectancy: {s['expectancy_r']:+.2f}R  |  "
        f"Avg hold: {s['avg_hold_days']:.1f} days",
        "",
        "**Confidence Calibration:**",
    ]

    for b in s["calibration"]:
        if b["actual_win_rate"] is None:
            lines.append(f"  {b['label']} — no data")
            continue
        icon = {"WELL_CALIBRATED": "✅", "UNDERCONFIDENT": "✅",
                "OVERCONFIDENT":   "⚠️"}.get(b["calibration_status"], "❓")
        lines.append(
            f"  {b['label']} predicted → "
            f"{b['actual_win_rate']*100:.0f}% actual  "
            f"{icon} {b['calibration_status'].replace('_', ' ').lower()}"
        )

    if s["best_sector"]:
        lines.append(
            f"\nBest sector:  {s['best_sector']['sector']}  "
            f"({s['best_sector']['avg_pct']:+.1f}% avg)"
        )
    if s["worst_sector"] and s["worst_sector"] != s["best_sector"]:
        lines.append(
            f"Worst sector: {s['worst_sector']['sector']}  "
            f"({s['worst_sector']['avg_pct']:+.1f}% avg)"
        )

    if s["resolved_count"] == 0:
        lines = [f"📊 **WEEKLY PERFORMANCE — {date_str}**",
                 "No resolved trades in the last 7 days."]

    msg = "\n".join(lines)

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

def run_performance_analytics(lookback_days: int = 7) -> dict | None:
    """
    Main entry point. Returns summary dict or None if flag is off / no data.
    Also sends Discord report if the webhook is configured.
    """
    if not ENABLE_PERFORMANCE_ANALYTICS:
        return None

    entries = _resolved_entries()
    if not entries:
        return None

    summary = performance_summary(entries, lookback_days)
    print(
        f"  📊 Performance: {summary['resolved_count']} resolved trades  "
        f"| Win rate {summary['win_rate']*100:.0f}%  "
        f"| Expectancy {summary['expectancy_r']:+.2f}R"
    )
    return summary
