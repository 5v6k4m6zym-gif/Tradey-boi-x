---
name: Tradey Boi X market report labeling and multi-schedule workflows
description: Why time-of-day inference for report labeling is unreliable on GitHub Actions, and the fix pattern.
---

A single workflow with two `schedule:` cron entries plus an in-script
`datetime.now().hour` check to decide "which market this run is for" is
fragile: GitHub Actions scheduled runs can be delayed by hours under
runner queue pressure, so a late-firing ASX-slot run can drift past the
hour window and get mislabeled as the US report (or vice versa).

**Why:** confirmed on Tradey Boi X — `market_open.yml` runs were landing
at wildly inconsistent UTC times (observed delays of several hours),
and the report content is generated from an hour-based branch, so a
delayed run silently sends the wrong market's report with the wrong
label.

**How to apply:** when a script's job is "generate report A or B
depending on which scheduled trigger fired," split into two separate
workflow files, each with its own single cron schedule, and pass the
variant explicitly as a CLI arg/env var to the script (never infer it
from wall-clock time at run time). Also: GitHub Actions CI log
timestamps can appear to cluster/jump because Python's stdout is
block-buffered when not attached to a TTY — don't infer "this loop ran
instantly" from adjacent log timestamps; the real wall-clock time is
the job's total duration, not the gap between print statements.
