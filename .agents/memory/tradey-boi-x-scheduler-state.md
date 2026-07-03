---
name: Tradey Boi X scheduler dedup state must be persisted
description: In-memory-only "once per day/cycle" gates in scanner.py's long-running loop get defeated by any process restart, causing duplicate scheduled events (e.g. morning brief re-sent).
---

Any "run once per calendar day" or "run once per cycle" dedup gate inside `scanner.py`'s long-running `main()` loop must persist its marker to disk (a small JSON state file), not just hold it in a local variable.

**Why:** The scanner workflow gets restarted routinely (config changes, redeploys, crashes). An in-memory local var like `_brief_sent_date` resets to its initial value on every restart, silently defeating the "once per day" gate — the next loop iteration re-fires the event even though it already ran earlier that day in the previous process. This was the confirmed root cause of a real "multiple morning market evaluations" incident.

**How to apply:** Follow the existing `cooldowns.json` pattern — write/read a small state file (e.g. `scanner_state.json`) at the point the gate is checked/updated. Keep this strictly to scheduling bookkeeping; never let it touch prediction/signal/execution logic. `tradey-boi-x/diagnostics/trace_logger.py` provides `load_scanner_state()`/`save_scanner_state()` helpers plus `log_trace()` for auditing these events going forward — reuse them for any new scheduled/dedup logic instead of inventing a new mechanism.
