---
name: Overnight scanner activation
description: overnight_scanner.py exists to scan a second ticker universe while both US/ASX markets are closed
---

`tradey-boi-x/overnight_scanner.py` covers a separate ~238-ticker universe
(distinct from the main `engine.WATCHLIST` of 408) — stocks NOT already
scanned by the market-hours `scanner.py`. It reuses the same `decide()` /
`send_alert()` / `log_signal()` pipeline, just on a rotating cursor-based
batch (250 tickers/run, wraps around) so the whole universe gets covered
across a night of closed-market hours.

**Why this matters:** the file was fully built (correct logic, market-closed
check, persistent cursor) but was never registered as a workflow and never
had a run-loop — it only processed one batch and exited, so it had *never
actually executed* despite looking complete. A user-reported "BIIB alert"
turned out to be untraceable to this system for exactly that reason (BIIB
only exists in this dormant file, no cursor file existed, no matching
process was running, no log entry anywhere).

**How to apply:** if a user reports an alert for a ticker that isn't in
`engine.WATCHLIST`, check `overnight_scanner.py`'s `OVERNIGHT_UNIVERSE` next
before assuming it's external — but also verify which execution path is
actually live. This project's real overnight scanner is a **GitHub Actions
cron job** (`.github/workflows/overnight_scan.yml`, hourly on weekdays, own
Discord webhook secret, commits cursor/log state back to git), NOT a Replit
workflow — a Replit process/workflow existing (or not) tells you nothing
about whether GitHub Actions is running it. Do not add a second Replit
workflow that also runs `overnight_scanner.py`: it duplicates alerts and, if
it makes the script loop forever, breaks the GitHub Actions job (which
expects one-shot-then-exit within its 45-min timeout so its "save state"
commit step can run).

As of 2026-07-03, `run_overnight_scan()` scans `WATCHLIST + OVERNIGHT_UNIVERSE`
merged/deduped (553 unique tickers) rather than just the OVERNIGHT_UNIVERSE
extras — so the main 408-ticker watchlist itself now also gets scanned
overnight, not just the separate extras list. Cycles fully every ~3 hourly
cron runs at BATCH_SIZE=250.

**Two entry points, one shared model — but NOT one shared decision pipeline.**
`scanner.py` and `overnight_scanner.py` both call the same `engine.decide()`/
`train_model()`, so the core prediction model is always identical. But any
*additive gating layer* built under `opportunity/` (trade evaluator, adaptive
core, strategy optimizer, audit engine, regime/opportunity second-pass) is
wired individually into each entry point's per-ticker loop — it is NOT
automatically inherited just because both scripts import `engine`. Check
`overnight_scanner.py`'s per-ticker loop whenever a new opportunity-layer
gate is added to `scanner.py`, or overnight alerts will silently skip it.
As of 2026-07-03 both scripts call the same four gates (trade_evaluator,
adaptive_core, strategy_optimizer, audit_engine) in the same order before
`send_alert()`.
