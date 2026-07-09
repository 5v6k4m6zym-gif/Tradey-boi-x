---
name: High-frequency GitHub Actions cron unreliability
description: Hourly (or more frequent) GH Actions schedules get silently throttled to a small fraction of intended runs; use one self-looping job per session instead.
---

Observed directly on the Tradey Boi X repo: `scanner.yml` and
`overnight_scan.yml` were both scheduled hourly (`cron: "9 * * * 1-5"` /
`"6 * * * 1-5"`, weekdays), which should produce ~24 runs/day each. Actual
run history showed only ~2-4 runs/day — GitHub Actions silently drops most
executions of high-frequency scheduled workflows under normal platform
load. This is a known GitHub Actions limitation (not a bug in the repo),
and it was the real root cause of "why are there no ASX suggestions" —
the scanner was barely running, not that the signal logic was broken.

**Why:** GitHub's docs note scheduled workflows can be delayed/dropped
during high load, and this is worse the more frequently a cron fires.
A once-daily or twice-daily cron is comparatively reliable; an hourly one
was reduced by ~85-90% in practice.

**How to apply:** don't rely on a frequent cron to repeat a scan/check.
Instead, trigger a single job once per session (e.g. once before ASX
open, once before US open) and have the script loop internally for the
whole session using its own sleep timer — bounded by a `--minutes N`
budget argument so the process returns normally before GitHub's hard job
timeout, letting any trailing steps (e.g. git-commit-the-log step) still
run. GitHub Actions hosted runners hard-cap a single job at 360 minutes,
so a session longer than that (e.g. the ~6.5h US trading session) needs
either a slightly-trimmed budget or a second follow-up job to cover the
tail.

Diagnosing this: compare "intended runs per day" (from the cron
expression) against actual `GET /repos/{owner}/{repo}/actions/workflows/{id}/runs`
history grouped by day — a large gap is the tell, not scanner/signal logs
themselves (which will look totally normal, just infrequent).
