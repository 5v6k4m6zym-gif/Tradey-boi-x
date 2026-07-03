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

**Multi-timeframe confirmation (T005) was already implemented (2026-07-03):**
Before adding new higher-timeframe gating, check whether it already exists —
this system already had `weekly_trend_ok()` (weekly EMA20>EMA50 hard gate) and
`multitimeframe_signal()` (1h vs daily EMA/MACD agreement) wired into
`decide()`. Always grep/read the target function fully before implementing a
spec item; avoid duplicating logic that's already there.

**Institutional liquidity gate (T007) result (2026-07-03):** Added a 20-day
avg-dollar-volume gate (`dollar_vol` in `get_data()`, `MIN_DOLLAR_VOLUME` in
`decide()`'s filters, fails open on NaN/insufficient history) to filter out
thin/illiquid names. Tested $200k/$500k/$2M thresholds on the full 407-ticker
backtest — all three were byte-identical to the pre-change baseline
(profit_factor 0.931, expectancy_r -0.025). Confirmed via a direct
`engine.get_data()` check that the column computes correctly (not a silent
bug) — this ASX-primary 408-ticker watchlist is simply already curated to
liquid names (e.g. BHP.AX ~$600M/day dollar volume), so every signal that
already fires clears even a $2M bar. **Why this matters:** a liquidity/quality
gate can be a legitimate no-op on a well-curated watchlist; don't mistake
"identical backtest metrics" for "broken code" — verify the underlying data
computes as expected before concluding a filter has no effect. Kept at $500k
as a forward-looking safety net (guards against future watchlist expansion
into thinner names) since it caused no regression.

**Smart position sizing (T008) result (2026-07-03):** Added ATR-based fixed-
fractional position sizing (`engine.position_size_pct()`, reuses the same
ATR-tier stop-loss logic as `expected_value_r()`) as an informational field
only (no gating change). On the full 407-ticker backtest, size-weighting
trade returns by this sizing scheme produced a LOWER average return
(+0.411%) than plain equal-weighting (+0.554%) — this system's tighter-stop
setups (favoured with larger size by fixed-fractional sizing) performed
slightly worse than its wider-stop setups historically. **Why this matters:**
"smart" position sizing is a risk-management practice (protects equity from
oversized bets on volatile names), not automatically a return-improving one
— don't assume volatility-based sizing will lift backtest expectancy; it can
legitimately do the opposite if low-vol setups underperform high-vol ones in
the data. Keep it for the equity-protection rationale, but report the
size-weighted vs plain comparison honestly rather than implying it "improved"
results.

**Drift monitoring (T011) + compute_metrics() R-unit quirk (2026-07-03):**
Added `opportunity/drift_monitor.py` comparing a recent rolling live window
of resolved trades against the older resolved-trade baseline (both scored
via `compute_metrics()`), flagging win_rate/expectancy_r/profit_factor
deltas beyond a threshold. **Why this matters:** `compute_metrics()`'s
`expectancy_r` normalizes by that window's OWN avg_loss (the "R-unit"), so
two windows with different loss compositions are not on the same scale —
a 100%-win-rate window (avg_loss=0, R-unit falls back to 1.0) produces an
expectancy_r that looks like a huge regression/improvement vs a window that
has real losses, even though nothing about win quality actually changed.
**How to apply:** any code comparing `expectancy_r` across two different
trade samples (drift monitoring, A/B strategy comparison, before/after
threshold sweeps) should sanity-check that both samples have a non-zero
loss count, or prefer comparing profit_factor/win_rate instead when one
side might be all-wins.

**Validation framework (T009) result (2026-07-03):** Before implementing a
"walk-forward validation" spec item, checked `opportunity/backtester.py` and
found `_walk_forward()`/`_out_of_sample()`/`_historical_simulation()`/
`_paper_trading_snapshot()` already existed and were wired into
`run_backtest(mode=...)` behind `ENABLE_ADVANCED_BACKTESTS` (default off) —
only the Monte Carlo resampling piece was missing, so only that was added
(`_monte_carlo()`: resample resolved trades with replacement, report
profit_factor/expectancy_r/max_drawdown percentile bands + empirical risk-
of-ruin %). **Why this matters:** this is the second spec item (after T005)
where reading the existing code first avoided duplicating already-built
functionality — always check for pre-existing implementations of a spec
item before writing new code, especially in a codebase that has already
been iterated on across many tasks.

**Realistic costs (T003) + adaptive exits (T004) results (2026-07-03):** Adding
a round-trip commission+slippage+spread cost model to `compute_metrics()`
(applied only to reported P&L, never to win/loss classification) revealed the
model's raw statistical edge does NOT survive realistic execution costs on
this ASX-heavy watchlist (profit_factor 1.057→0.925, expectancy_r flipped
positive→-0.045) — a genuine finding, not a bug. Layering adaptive exits
(partial profit-take at halfway-to-target + breakeven/trailing ATR stop,
`engine.simulate_adaptive_exit()`) on top of that improved profit_factor
(0.925→0.931) and expectancy_r (-0.045→-0.025) but did not close the gap to
breakeven. **Why this matters for future work:** whenever the exit mechanism
changes (fixed stop/target vs adaptive/trailing), the win/loss *definition*
itself changes — a fixed-horizon "did forward return clear the target" test
is not comparable to "was the blended adaptive-exit P&L positive." Never
compare win_rate across a methodology change; only compare cost-adjusted
profit_factor/expectancy_r, and call out the methodology shift explicitly.
