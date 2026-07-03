"""Part 4 — Full System Trace Logger.

Append-only, fail-safe JSONL trace writer. Never raises. Never mutates
anything outside its own log file. Intended to be called from scanner.py at
key scheduling decision points (morning-brief trigger, scan-cycle start,
regime refresh) to build an auditable record of *when* and *why* each
morning-evaluation-related event fired.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diagnostics import config

_lock = threading.Lock()


def log_trace(event: str, **fields: Any) -> bool:
    """Append one trace record. Returns True on success, False on any
    failure (never raises)."""
    if not config.ENABLE_TRACE_LOGGER:
        return False
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        config.TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(config.TRACE_LOG_PATH, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return True
    except Exception as e:  # fail-safe: never let tracing break the scanner
        try:
            print(f"  ⚠️  [trace_logger] failed to write trace: {e}")
        except Exception:
            pass
        return False


def read_traces(path: Path | None = None) -> list[dict]:
    """Read all trace records. Returns [] on any failure."""
    p = path or config.TRACE_LOG_PATH
    try:
        if not p.exists():
            return []
        out = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
    except Exception:
        return []


def load_scanner_state() -> dict:
    """Load persisted scheduler state (e.g. brief_sent_date). Returns {} on
    any failure — callers must treat missing/corrupt state as 'no prior
    state' rather than crashing."""
    try:
        if config.SCANNER_STATE_PATH.exists():
            with open(config.SCANNER_STATE_PATH) as f:
                return json.load(f)
    except Exception as e:
        try:
            print(f"  ⚠️  [trace_logger] failed to read scanner_state.json: {e}")
        except Exception:
            pass
    return {}


def save_scanner_state(state: dict) -> bool:
    """Persist scheduler state. Fail-safe: returns False on error, never
    raises. This does NOT touch trading/signal logic — it only persists
    scheduling bookkeeping (e.g. which calendar date the morning brief was
    already sent for), mirroring the existing cooldowns.json pattern."""
    try:
        config.SCANNER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.SCANNER_STATE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(config.SCANNER_STATE_PATH)
        return True
    except Exception as e:
        try:
            print(f"  ⚠️  [trace_logger] failed to save scanner_state.json: {e}")
        except Exception:
            pass
        return False
