---
name: Historical backtest methodology for trading signal engines
description: Why train/test time-splitting matters when backtesting a stock alert scoring model
---

When backtesting a signal-detection/scoring model (e.g. `engine.decide()`'s AI
probability + technical rules) against historical price data, training the model
on the full historical window and then evaluating signals anywhere in that same
window is invalid — the model has already seen the exact rows and target labels
it's being "tested" against.

**Why:** An initial backtest attempt trained on 2 years of data and evaluated
signals across that same 2-year window, producing an impossible-looking 100% win
rate over 11 trades. That result was a data-leakage artifact, not evidence the
strategy works. A corrected version split each ticker's history by time (first
~70% train, last ~30% held-out test, model never sees test-period rows/labels)
and got 0 qualifying alert-tier signals in the small sample — consistent with
production also being very selective (only ~8 signals logged over months live).

**How to apply:** Any future backtest of this system must (1) use a strict
chronological train/test split per ticker, never train on data that overlaps the
evaluation window, and (2) if the strict alert tier is too rare to yield a
statistically meaningful sample, fall back to a probability-calibration check
(bucket ALL evaluated days by model probability, not just alert-tier days, and
check whether win rate/forward return increases with probability) rather than
force more alerts by loosening thresholds.
