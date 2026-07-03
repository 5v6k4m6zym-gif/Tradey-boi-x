"""Part 3 — Filter Impact Analysis.

Reads each opportunity-layer's own append-only JSONL log and reports, per
layer: activation count, output count, duplication rate, suppression rate.
Read-only. Never raises.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Any

from diagnostics import config

LAYERS = {
    "trade_evaluator": config.TRADE_EVAL_LOG_PATH,
    "adaptive_core": config.ADAPTIVE_CORE_LOG_PATH,
    "strategy_optimizer": config.STRATEGY_OPTIMIZER_LOG_PATH,
    "audit_engine": config.AUDIT_TRADES_LOG_PATH,
}


def _read_jsonl(path: Path) -> list[dict]:
    try:
        if not path.exists():
            return []
        out = []
        with open(path) as f:
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


def _signal_key(rec: dict) -> str:
    symbol = rec.get("symbol") or rec.get("ticker") or "UNKNOWN"
    ts = str(rec.get("timestamp", ""))[:10]  # date-granularity key
    return f"{symbol}|{ts}"


def analyze_layer(name: str, path: Path | None = None) -> dict[str, Any]:
    """Returns activation/output/duplication/suppression stats for one layer.
    Never raises — returns a dict with an 'error' field on failure."""
    try:
        p = path or LAYERS.get(name)
        if p is None:
            return {"layer": name, "error": f"unknown layer '{name}'"}
        records = _read_jsonl(p)
        activations = len(records)
        keys = [_signal_key(r) for r in records]
        counts = Counter(keys)
        duplicated = sum(1 for c in counts.values() if c > 1)
        passed = sum(1 for r in records if r.get("passed") is True
                     or r.get("approved") is True
                     or (r.get("passed") is None and r.get("error") is None
                         and r.get("rejection_reasons") in (None, [])))
        suppressed = activations - passed if activations else 0
        return {
            "layer": name,
            "log_path": str(p),
            "activations": activations,
            "unique_signals": len(counts),
            "outputs_passed": passed,
            "suppressed": max(suppressed, 0),
            "duplicated_signal_keys": duplicated,
            "duplication_rate": round(duplicated / len(counts), 4) if counts else 0.0,
            "suppression_rate": round(max(suppressed, 0) / activations, 4) if activations else 0.0,
        }
    except Exception as e:
        return {"layer": name, "error": str(e)}


class FilterImpactAnalyzer:
    """Part 3 entry point — analyzes all known filter layers."""

    def analyze_all(self) -> dict[str, Any]:
        results = {name: analyze_layer(name) for name in LAYERS}
        flagged = [
            name for name, stats in results.items()
            if stats.get("duplication_rate", 0) > 0
        ]
        return {
            "layers": results,
            "layers_with_duplication": flagged,
        }
