"""
Phase 9 — Live vs Backtest Drift Monitoring (institutional upgrade T011)
Compares a recent rolling window of resolved live/paper trades against the
older resolved-trade history (acting as the validation baseline) and flags
meaningful divergence in win_rate / expectancy_r / profit_factor.

Feature flag: ENABLE_DRIFT_MONITORING  (default: false -> complete no-op)
Reads the existing signal_log.json. Never writes to engine.py or scanner.py,
never affects signal generation — purely a reporting/alerting layer.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from opportunity.backtester import compute_metrics
from opportunity.config import (
    ENABLE_DRIFT_MONITORING,
    DRIFT_LIVE_WINDOW_DAYS,
    DRIFT_MIN_LIVE_TRADES,
    DRIFT_MIN_BASELINE_TRADES,
    DRIFT_THRESHOLDS,
)

BASE_DIR = Path(__file__).parent.parent
LOG_FILE = BASE_DIR / "signal_log.json"


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


def _split_baseline_live(
    entries: list[dict], live_window_days: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split resolved entries into an older "baseline" set and a recent "live"
    set based on signal_date. Entries with no/unparseable signal_date are
    excluded from both (can't be time-bucketed).
    """
    cutoff = (datetime.utcnow() - timedelta(days=live_window_days)).strftime("%Y-%m-%d")
    baseline, live = [], []
    for e in entries:
        date = e.get("signal_date")
        if not date:
            continue
        if date >= cutoff:
            live.append(e)
        else:
            baseline.append(e)
    return baseline, live


# ─── Drift detection ──────────────────────────────────────────────────────────

def detect_drift(
    entries: list[dict],
    live_window_days: int = DRIFT_LIVE_WINDOW_DAYS,
) -> dict[str, Any]:
    """
    Compare recent live performance against the older baseline.

    Returns a dict with baseline metrics, live metrics, per-metric deltas,
    and a list of drift flags for any metric whose absolute delta exceeds
    its configured threshold. `sufficient_data` is False (and drift_flags
    empty) when either window doesn't have enough resolved trades to draw
    a meaningful conclusion.
    """
    resolved = _resolved(entries)
    baseline_entries, live_entries = _split_baseline_live(resolved, live_window_days)

    baseline_metrics = compute_metrics(baseline_entries)
    live_metrics     = compute_metrics(live_entries)

    sufficient_data = (
        len(baseline_entries) >= DRIFT_MIN_BASELINE_TRADES
        and len(live_entries) >= DRIFT_MIN_LIVE_TRADES
    )

    deltas: dict[str, float] = {}
    drift_flags: list[dict[str, Any]] = []

    for metric, threshold in DRIFT_THRESHOLDS.items():
        b = baseline_metrics.get(metric, 0.0)
        l = live_metrics.get(metric, 0.0)
        if not (isinstance(b, (int, float)) and isinstance(l, (int, float))):
            continue
        if b in (float("inf"), float("-inf")) or l in (float("inf"), float("-inf")):
            continue
        delta = round(l - b, 4)
        deltas[metric] = delta
        if sufficient_data and abs(delta) > threshold:
            drift_flags.append({
                "metric":    metric,
                "baseline":  b,
                "live":      l,
                "delta":     delta,
                "threshold": threshold,
                "direction": "improved" if delta > 0 else "degraded",
            })

    return {
        "generated_at":       datetime.utcnow().isoformat(),
        "live_window_days":   live_window_days,
        "baseline_trade_count": len(baseline_entries),
        "live_trade_count":   len(live_entries),
        "sufficient_data":    sufficient_data,
        "baseline_metrics":   baseline_metrics,
        "live_metrics":       live_metrics,
        "deltas":             deltas,
        "drift_flags":        drift_flags,
    }


# ─── Discord alert ────────────────────────────────────────────────────────────

def send_drift_alert(report: dict[str, Any]) -> bool:
    """Post a drift alert to Discord when at least one metric has drifted."""
    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook")
    if not webhook or not report.get("drift_flags"):
        return False

    date_str = datetime.utcnow().strftime("%d %b %Y")
    lines = [
        f"⚠️ **PERFORMANCE DRIFT DETECTED — {date_str}**",
        f"Live window: last {report['live_window_days']} days "
        f"({report['live_trade_count']} trades) vs baseline "
        f"({report['baseline_trade_count']} trades)",
        "",
    ]
    for f in report["drift_flags"]:
        icon = "📉" if f["direction"] == "degraded" else "📈"
        lines.append(
            f"{icon} {f['metric']}: baseline {f['baseline']} → live {f['live']} "
            f"(Δ{f['delta']:+.3f}, threshold {f['threshold']})"
        )

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

def run_drift_monitor(
    live_window_days: int = DRIFT_LIVE_WINDOW_DAYS,
    notify: bool = True,
) -> dict | None:
    """
    Main entry point. Returns the drift report dict, or None when the flag
    is off or there are no resolved entries at all.
    """
    if not ENABLE_DRIFT_MONITORING:
        return None

    entries = _load_log()
    if not _resolved(entries):
        return None

    report = detect_drift(entries, live_window_days)

    if report["drift_flags"]:
        print(f"  ⚠️  Drift monitor: {len(report['drift_flags'])} metric(s) drifted "
              f"beyond threshold vs baseline.")
    elif report["sufficient_data"]:
        print("  ✅ Drift monitor: live performance within expected range of baseline.")

    if notify and report["drift_flags"]:
        send_drift_alert(report)

    return report
