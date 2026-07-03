---
name: Persistent ticker-history cache for backtest scripts
description: How to run the heavy 407-ticker backtest/live-gating validation reliably in this sandbox
---

The 407-ticker backtest scripts (`full_watchlist_backtest.py`,
`full_pipeline_live_gating_validation.py`) originally cached 2y price history
and trained-model checkpoints under `/tmp/...`, which is wiped on environment
restart, forcing a full re-download/re-train every time — this repeatedly
crashed/died silently in this sandbox.

**Fix applied:** both scripts now point their cache/checkpoint dirs at
`tradey-boi-x/.cache/` (gitignored, but persists across restarts since it's
inside the project directory, not /tmp). A new resumable script,
`tests/populate_ticker_cache.py --limit N`, fetches N new tickers per
invocation and skips already-cached ones — safe to call repeatedly in small
foreground chunks.

**Why this matters:** background/nohup/setsid processes get killed the moment
the shell command that launched them returns in this sandbox (not an OOM
issue — confirmed plenty of free memory at the time) — long jobs must run in
the foreground within a single tool call's timeout, relying on per-item
checkpointing to resume across multiple sequential foreground calls.

**How to apply:** to (re)run a full historical validation, first run
`python3 tests/populate_ticker_cache.py --limit 100` a few times until it
reports 0 remaining, then run the target backtest script in foreground
chunks (~100s each) — it will resume from its own per-ticker checkpoint each
time. Never background these jobs and walk away; always chunk them
synchronously. 5/412 tickers are confirmed delisted (fail every run) —
expected, not a bug.

**Fresh full-repeat result (2026-07-03, post-cache-fix):** re-ran the
trade-evaluator/adaptive_core/strategy_optimizer live-gating chain against
freshly-downloaded data: baseline (gates off) win_rate 38.8%/PF 0.801/
expectancy -0.122R (358 trades) -> live-gated (~31% approval) win_rate 45.1%/
PF 1.216/expectancy +0.119R (111 trades). Consistent with the original
validation's direction and magnitude (~31% approval rate both times); exact
figures drift run-to-run because the model retrains on the latest available
data each time — treat this as directionally stable, not bit-for-bit
reproducible.
