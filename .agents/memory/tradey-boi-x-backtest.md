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

**Threshold-tuning result (2026-07-03):** For the ELITE/STRONG BUY score gates
in `engine.decide()`, sweeping cutoffs downward on a 407-ticker, ~50k-row
out-of-sample set found `score>=8` (ELITE) / `score>=6 & prob>=0.50` (STRONG
BUY) as the best tradeoff — meaningfully revives alert volume while keeping win
rate above the unfiltered baseline. For the big-mover scanner
(`_large_move_check`/`_breakout_setup_check`), every tested relaxation of the
vol/return/ATR/OBV/ADX gates made win rate and expectancy monotonically WORSE
than the current production gates, with no config catching more than 1 of 11
known historical winners — loosening gates is not a good lever here; a
different approach (new features/signals) would be needed to meaningfully
improve mover-catch rate. Reusable trick: cache the trained model/cutoffs and
per-ticker price data to disk, then batch `predict_proba` across all tickers
in one call instead of per-ticker — turns a ~0.5s/ticker sweep into a
near-instant one, making iterative threshold sweeps practical.

**Small-sample vs full-watchlist divergence (2026-07-03):** A change to
PREDICTION_DAYS/TARGET_RETURN (10d/3% → 5d/2%) looked like a clear win on a
diverse 42-ticker sample (win rate 45%→49%, AUC 0.492→0.510) but FAILED on the
full 407-ticker watchlist (win rate 37.7%→34.6%, expectancy +0.002R→-0.143R)
and was reverted. Any optimisation must be validated on the full watchlist,
never kept on sample-only evidence — small samples can show false positives
that don't generalize.

**Expected-value gate result (2026-07-03):** Adding an ATR-implied
reward:risk-based expected-R filter (`engine.expected_value_r`, reject
otherwise-qualifying ELITE/STRONG BUY setups when prob-weighted expected R ≤ 0)
on top of the existing score/prob gates improved every metric on the full
407-ticker validation: win rate 37.7%→40.0%, expectancy +0.002R→+0.034R,
profit factor 1.002→1.057, while cutting trade count ~754→398 (fewer, higher-
quality signals) — kept in production. This confirms probability alone is a
weaker filter than probability × reward:risk for this system.

**Regime-adaptive probability floor result (2026-07-03):** Classifying market
regime (strong/weak bull, sideways, weak/strong bear, high-vol from
price/ema50/ema200/mom20/vix) and raising the ELITE/STRONG BUY probability
floor above 0.50 in unfavorable/volatile regimes REGRESSED every metric vs the
flat-0.50-floor + expected-value-gate baseline on the full 407-ticker
validation: win rate 40.0%→38.8%, expectancy +0.034R→+0.007R, profit factor
1.057→1.012. Reverted the floor back to flat 0.50; kept regime classification
itself as an informational/dashboard-only field on `decide()`'s return dict
(does not gate tier). Lesson: for this system, a point-in-time regime label
computed from broad index/vix signals is too coarse/noisy to safely adapt a
per-ticker probability threshold — it filters out real setups indiscriminately
rather than just bad ones. If regime-based adaptation is revisited, prefer
using it for position sizing or stop distance (impact on trade management)
rather than gating whether a signal fires at all.
