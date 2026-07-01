"""
Phase 7 — System Health Monitor
Tracks scan duration, memory, failed requests, duplicate alerts,
and Discord delivery. Logs to logs/health.log. Weekly Discord summary.

Feature flag: ENABLE_SYSTEM_HEALTH  (default: false → complete no-op)
Wraps scanner.run_scan() non-invasively from scanner.py.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from opportunity.config import ENABLE_SYSTEM_HEALTH

BASE_DIR  = Path(__file__).parent.parent
LOGS_DIR  = BASE_DIR / "logs"
HEALTH_LOG = LOGS_DIR / "health.log"

# Alert thresholds
SCAN_DURATION_WARN_SEC = 600     # 10 minutes
MEMORY_WARN_PCT        = 80.0    # 80 % RAM
MAX_DUPE_WINDOW_SEC    = 3600    # duplicates within 1 hour flag as dupes


# ─── Low-level log helpers ────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def log_health_event(event_type: str, **kwargs: Any) -> None:
    """Append a single JSON line to health.log. No-op when flag is off."""
    if not ENABLE_SYSTEM_HEALTH:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": _ts(), "event": event_type, **kwargs}
    with HEALTH_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _load_health_log(since: datetime | None = None) -> list[dict]:
    """Read health log, optionally filtering by timestamp."""
    if not HEALTH_LOG.exists():
        return []
    records = []
    try:
        for line in HEALTH_LOG.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if since:
                ts_str = r.get("ts", "")
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                    if ts < since:
                        continue
                except Exception:
                    pass
            records.append(r)
    except Exception:
        pass
    return records


# ─── Memory check ─────────────────────────────────────────────────────────────

def check_memory() -> dict[str, Any]:
    """
    Return memory usage as {used_pct, used_mb, total_mb, warning}.
    Falls back gracefully if psutil is not installed.
    """
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        used_pct = mem.percent
        used_mb  = round(mem.used  / 1024 / 1024, 1)
        total_mb = round(mem.total / 1024 / 1024, 1)
    except ImportError:
        used_pct = 0.0
        used_mb  = 0.0
        total_mb = 0.0

    warning = used_pct >= MEMORY_WARN_PCT
    if warning:
        log_health_event("MEMORY_WARNING", used_pct=used_pct, used_mb=used_mb)

    return {"used_pct": used_pct, "used_mb": used_mb,
            "total_mb": total_mb, "warning": warning}


# ─── Duplicate alert detection ────────────────────────────────────────────────

_alert_registry: dict[str, float] = {}   # ticker → last alert unix timestamp


def check_duplicate(ticker: str) -> bool:
    """
    Return True if `ticker` was already alerted within DUPE_WINDOW.
    Registers the ticker if not a duplicate.
    """
    now = time.monotonic()
    last = _alert_registry.get(ticker, float("-inf"))   # -inf → never seen
    if now - last < MAX_DUPE_WINDOW_SEC:
        log_health_event("DUPLICATE_ALERT", ticker=ticker,
                         seconds_since_last=round(now - last, 0))
        return True
    _alert_registry[ticker] = now
    return False


# ─── Context-manager scan wrapper ─────────────────────────────────────────────

class HealthMonitor:
    """
    Context manager that times a scan, checks memory, and logs the result.

    Usage (in scanner.py — additive):
        with HealthMonitor("run_scan"):
            result = run_scan(...)
    """

    def __init__(self, label: str = "scan"):
        self.label      = label
        self._start: float = 0.0

    def __enter__(self) -> "HealthMonitor":
        if not ENABLE_SYSTEM_HEALTH:
            return self
        self._start = time.monotonic()
        log_health_event("SCAN_START", label=self.label)
        check_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not ENABLE_SYSTEM_HEALTH:
            return False   # do not suppress exceptions
        duration = round(time.monotonic() - self._start, 2)
        status   = "ERROR" if exc_type else "OK"
        warning  = duration > SCAN_DURATION_WARN_SEC
        log_health_event(
            "SCAN_END",
            label    = self.label,
            duration_sec = duration,
            status   = status,
            slow     = warning,
            error    = str(exc_val) if exc_val else None,
        )
        if warning:
            print(f"  ⚠️  Health: scan took {duration:.0f}s — above {SCAN_DURATION_WARN_SEC}s threshold")
        check_memory()
        return False   # never suppress exceptions


def wrap_run_scan(run_scan_fn: Callable) -> Callable:
    """
    Return a wrapped version of run_scan that uses HealthMonitor.
    Drop-in replacement when ENABLE_SYSTEM_HEALTH is True.
    """
    if not ENABLE_SYSTEM_HEALTH:
        return run_scan_fn

    def _wrapped(*args, **kwargs):
        with HealthMonitor("run_scan"):
            return run_scan_fn(*args, **kwargs)

    return _wrapped


# ─── Discord health summary ───────────────────────────────────────────────────

def send_weekly_health_report(lookback_days: int = 7) -> bool:
    """Post a weekly health summary to Discord. Returns True on success."""
    if not ENABLE_SYSTEM_HEALTH:
        return False

    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook")
    if not webhook:
        return False

    since   = datetime.utcnow() - timedelta(days=lookback_days)
    records = _load_health_log(since=since)

    scans       = [r for r in records if r.get("event") == "SCAN_END"]
    slow_scans  = [r for r in scans   if r.get("slow")]
    errors      = [r for r in scans   if r.get("status") == "ERROR"]
    mem_warns   = [r for r in records if r.get("event") == "MEMORY_WARNING"]
    dupes       = [r for r in records if r.get("event") == "DUPLICATE_ALERT"]

    durations   = [r.get("duration_sec", 0) for r in scans if r.get("duration_sec")]
    avg_dur     = round(sum(durations) / len(durations), 0) if durations else 0
    max_dur     = round(max(durations), 0)                  if durations else 0

    date_str = datetime.utcnow().strftime("%d %b %Y")
    status   = "🟢 Healthy" if not (slow_scans or errors or mem_warns) else "🟡 Warnings"

    msg = (
        f"🔧 **SYSTEM HEALTH — {date_str}** ({status})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Scans completed:  {len(scans)}\n"
        f"Avg duration:     {avg_dur:.0f}s  |  Max: {max_dur:.0f}s\n"
        f"Slow scans (>{SCAN_DURATION_WARN_SEC}s): {len(slow_scans)}\n"
        f"Scan errors:      {len(errors)}\n"
        f"Memory warnings:  {len(mem_warns)}\n"
        f"Duplicate alerts: {len(dupes)}"
    )

    if errors:
        last_err = errors[-1]
        msg += f"\n\n⚠️ Last error: {last_err.get('error', 'unknown')} @ {last_err.get('ts', '')}"

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

def run_health_check() -> dict | None:
    """
    Perform a one-off health check: memory + recent log summary.
    Returns summary dict or None when flag is off.
    """
    if not ENABLE_SYSTEM_HEALTH:
        return None

    mem    = check_memory()
    since  = datetime.utcnow() - timedelta(hours=24)
    recent = _load_health_log(since=since)

    scan_ends = [r for r in recent if r.get("event") == "SCAN_END"]
    summary   = {
        "timestamp":         _ts(),
        "memory":            mem,
        "scans_24h":         len(scan_ends),
        "errors_24h":        sum(1 for r in scan_ends if r.get("status") == "ERROR"),
        "slow_scans_24h":    sum(1 for r in scan_ends if r.get("slow")),
        "duplicate_alerts":  sum(1 for r in recent   if r.get("event") == "DUPLICATE_ALERT"),
    }

    log_health_event("HEALTH_CHECK", **{k: v for k, v in summary.items()
                                        if k != "timestamp"})
    return summary
