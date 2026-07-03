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
before assuming it's external — but also verify the workflow is actually
registered and running (`ps aux`, check for `overnight_cursor.json`), since
code existing in the repo does not mean it is wired up to run. Fixed
2026-07-03 by adding a `main()` while-loop (mirroring `scanner.py`'s
open/closed polling pattern) and registering it as its own workflow,
`Tradey Boi X Overnight Scanner`, running `python overnight_scanner.py`.
