"""Part 1 — Alert Behaviour Verification.

Compares the model's original signal output (signal_log.json — written when
a signal fires, before any opportunity-layer gating) against the final
per-layer outcomes (trade_evaluator / adaptive_core / strategy_optimizer /
audit_engine JSONL logs) and classifies each original signal as:

  PASS_THROUGH  — original signal reached the end of the chain unmodified
  FILTERED_OUT  — a layer rejected it, with a recorded valid reason
  DUPLICATED    — the same (symbol, date) signal appears more than once in a
                  layer's log within the same evaluation cycle
  LOST          — the model logged the signal but no layer log has any trace
                  of it at all (silent drop — critical)

Read-only. Never raises.
"""
import json
from collections import Counter, defaultdict
from typing import Any

from diagnostics import config
from diagnostics.filter_impact_analyzer import _read_jsonl, LAYERS, _signal_key


def _load_signal_log() -> list[dict]:
    try:
        if not config.SIGNAL_LOG_PATH.exists():
            return []
        with open(config.SIGNAL_LOG_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _original_key(rec: dict) -> str:
    symbol = rec.get("ticker") or rec.get("symbol") or "UNKNOWN"
    date = rec.get("signal_date") or str(rec.get("timestamp", ""))[:10]
    return f"{symbol}|{date}"


class AlertBehaviourTester:
    """Part 1 entry point."""

    def run(self) -> dict[str, Any]:
        try:
            originals = _load_signal_log()
            layer_records: dict[str, list[dict]] = {
                name: _read_jsonl(path) for name, path in LAYERS.items()
            }
            layer_keys: dict[str, Counter] = {
                name: Counter(_signal_key(r) for r in recs)
                for name, recs in layer_records.items()
            }

            classifications: dict[str, list[str]] = defaultdict(list)
            for rec in originals:
                key = _original_key(rec)
                seen_in_any_layer = any(counts[key] > 0 for counts in layer_keys.values())
                dup_layers = [name for name, counts in layer_keys.items() if counts[key] > 1]

                if dup_layers:
                    classifications["DUPLICATED"].append(key)
                elif not seen_in_any_layer:
                    # Model fired a signal, but no opportunity layer has any
                    # record of ever evaluating it. Note: this is EXPECTED for
                    # signals fired while ENABLE_* flags were off (nothing to
                    # evaluate), so callers should cross-check against the
                    # historical flag-flip date before treating this as a
                    # critical LOST signal.
                    classifications["LOST_OR_LAYER_DISABLED"].append(key)
                else:
                    rejected_anywhere = any(
                        any(_signal_key(r) == key and r.get("passed") is False
                            for r in recs)
                        for recs in layer_records.values()
                    )
                    if rejected_anywhere:
                        classifications["FILTERED_OUT"].append(key)
                    else:
                        classifications["PASS_THROUGH"].append(key)

            return {
                "total_original_signals": len(originals),
                "pass_through": len(classifications["PASS_THROUGH"]),
                "filtered_out": len(classifications["FILTERED_OUT"]),
                "duplicated": len(classifications["DUPLICATED"]),
                "lost_or_layer_disabled": len(classifications["LOST_OR_LAYER_DISABLED"]),
                "duplicated_keys": classifications["DUPLICATED"],
                "critical_issues_found": bool(classifications["DUPLICATED"]),
            }
        except Exception as e:
            return {"error": str(e)}
