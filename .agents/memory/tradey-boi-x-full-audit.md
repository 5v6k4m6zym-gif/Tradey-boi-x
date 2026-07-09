---
name: Tradey Boi X full system audit method
description: How to audit all GitHub Actions workflows and scripts for this repo in one pass, and what it caught the first time it was run.
---

To audit the whole GH-Actions-based system in one pass (no local server, no
cron access), pull everything through the GitHub REST API with the
`GITHUB_WORKFLOW_PAT` secret, in this order:

1. `GET /actions/workflows` — the authoritative list of currently-registered
   workflows and their state. A workflow file that was deleted from the
   default branch stops appearing here, even if an old run for it still
   shows up in run history (that run used the workflow definition from
   before the deletion — it's a ghost of a stale trigger, not a live bug).
2. For each workflow, `GET /actions/workflows/{id}/runs` — compare the
   *count* of runs against the cron's *intended* frequency (see the
   cron-throttling note). This is the fastest way to catch scheduling drift
   without reading a single log line.
3. For any `failure` conclusion, `GET /actions/runs/{id}/jobs` then
   `GET /actions/jobs/{job_id}/logs` via `curl -L` (not raw `urllib` —
   the logs endpoint 302-redirects to blob storage and urllib's
   `Authorization` header on the redirect triggers a 401; curl handles it
   fine, or drop the header before following the redirect).
4. Diff every locally-edited file against its GitHub Contents API copy
   (`base64.b64decode` and byte-compare) to catch push-verification drift.
5. Locally: `python -m py_compile` across the whole package, then the
   existing `pytest` suite, as a cheap correctness gate before/after edits.

First real run of this method found: a stale duplicate workflow
(`eod_report.yml`) that had already been deleted from the branch but still
had one queued/in-flight run fail on a missing `vaderSentiment` dependency
— a ghost, not a live bug, confirmed by workflow-list absence; and a second,
real bug — `overnight_scan.yml`'s hourly cron had the exact same GH
throttling problem as `scanner.yml` (~3 actual runs/day vs ~24 intended),
just not yet fixed because its cursor-based design made it seem lower
priority. Converted it to the same self-looping-job-per-window pattern.
