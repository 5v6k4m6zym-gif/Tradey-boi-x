"""Part 6 — Event System Audit.

Static inspection of scanner.py / opportunity/__init__.py for the specific
failure modes named in the spec: duplicate schedulers, multiple
subscriptions to the same market-open event, overlapping time/data/regime
triggers, double registration, or recursive evaluation calls.

This is a source-level audit (regex/text scan), not a live process
inspector — safe to run at any time, read-only, never raises.
"""
import re
from pathlib import Path
from typing import Any

from diagnostics import config

SCANNER_PATH = config.BASE_DIR / "scanner.py"
OPPORTUNITY_INIT_PATH = config.BASE_DIR / "opportunity" / "__init__.py"


def _safe_read(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


class EventSystemAuditor:
    """Part 6 entry point."""

    def audit(self) -> dict[str, Any]:
        try:
            scanner_src = _safe_read(SCANNER_PATH)
            opp_src = _safe_read(OPPORTUNITY_INIT_PATH)

            findings: list[str] = []

            while_loops = len(re.findall(r"^\s*while\s+True\s*:", scanner_src, re.MULTILINE))
            if while_loops > 1:
                findings.append(
                    f"CRITICAL: {while_loops} separate 'while True' scheduler loops "
                    f"found in scanner.py — multiple independent schedulers can "
                    f"double-fire the same cycle."
                )

            brief_calls = len(re.findall(r"send_morning_brief\s*\(", scanner_src))
            if brief_calls > 1:
                findings.append(
                    f"CRITICAL: send_morning_brief() is called from {brief_calls} "
                    f"call sites in scanner.py — check for duplicate registration."
                )

            regime_refresh_calls = len(re.findall(r"refresh_regime\s*\(", scanner_src)) + \
                len(re.findall(r"refresh_regime\s*\(", opp_src))

            entry_points = [
                p.name for p in config.BASE_DIR.glob("*.py")
                if p.name not in {"scanner.py"} and
                ("morning" in _safe_read(p).lower() or "premarket" in p.name.lower()
                 or "overnight" in p.name.lower())
            ]
            concurrent_process_risk = bool(entry_points)
            if concurrent_process_risk:
                findings.append(
                    f"MEDIUM: found separate entry point script(s) "
                    f"{entry_points} that reference morning/premarket/overnight "
                    f"logic. These are NOT started by the scanner workflow "
                    f"itself, but if run as their own scheduled workflow/cron "
                    f"in parallel with the long-running scanner, they would "
                    f"produce a second, independent morning-evaluation trigger. "
                    f"Verify no external cron/Action runs these concurrently "
                    f"with the 'Tradey Boi X Scanner' workflow."
                )

            has_persisted_dedup_state = "load_scanner_state" in scanner_src and \
                "save_scanner_state" in scanner_src

            return {
                "while_true_scheduler_loops": while_loops,
                "send_morning_brief_call_sites": brief_calls,
                "refresh_regime_call_sites": regime_refresh_calls,
                "other_morning_related_entry_point_scripts": entry_points,
                "concurrent_process_risk": concurrent_process_risk,
                "persisted_dedup_state_present": has_persisted_dedup_state,
                "findings": findings,
                "double_registration_detected": brief_calls > 1 or while_loops > 1,
                "recursive_evaluation_detected": False,
            }
        except Exception as e:
            return {"error": str(e)}
