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

**Safety-critical specs: check for existing infra before building (2026-07-03):**
When given a hard-constrained "SAFE wrapper layer" spec (TradeEvaluator,
TradeFilter, shadow mode, JSONL logging, fail-safe fallback), most of it
already existed from an earlier phase under different task numbering. Only
the pieces the spec explicitly named as "NEW" (PerformanceTracker,
AutoThresholdTuner) were actually missing. **Why this matters:** re-reading
a safety-constrained spec against the current codebase before writing code
avoids duplicate/conflicting implementations of the same guardrails (e.g. a
second shadow-mode flag or a second logging path) which would be far worse
than merely wasted effort. **How to apply:** for any "build X layer with
these named components" request, grep for each named class/function first;
implement only what's genuinely missing, and wire new pieces onto the
existing flag/logging/mutation patterns rather than parallel ones.

**Auto-tuning mutable config in place (2026-07-03):** When a module needs to
adjust a runtime threshold that another already-instantiated object reads
(e.g. `TradeEvaluator.thresholds`), mutate the shared dict's keys in place
rather than reassigning the module-level constant — reassignment breaks the
reference for anything that captured the dict at construction time.

**Stacking multiple additive wrapper layers with the same-named entry point
(2026-07-03):** Two separate specs each asked for a `process_trade_signal(
trade, market_data)` function. Resolved by giving each its own module
(`trade_evaluator.py`, `adaptive_core.py`) and calling both independently
from the scanner behind separate feature flags, rather than merging them
into one function or letting the newer one silently shadow the older one.
**Why:** each layer has its own on/off flag and its own JSONL log; merging
them would make it impossible to disable one without the other, and a
same-named import shadowing the older module would be a silent, hard-to-spot
regression. **How to apply:** when a new spec's requested entry point name
collides with an existing one, keep them as distinct functions in distinct
modules, wire the call sites to invoke both in sequence, and give the second
layer its own independent enable flag so either can be toggled alone.

**Fail-safe direction differs from shadow-mode/reject-mode (2026-07-03):**
A spec's "fail-safe" rule ("if any component fails, pass the trade through
unchanged") is NOT the same behavior as SHADOW_MODE's "always return None".
**Why:** shadow mode is intentional observation-only (block execution to
gather data); an internal *bug/exception* in an optional add-on layer should
never be able to block a trade that would otherwise have gone through —
blocking on error would let an add-on layer silently suppress real signals.
**How to apply:** in any wrapper's top-level try/except, return the original
input object unchanged on exception (not `None`), even though the normal
rejection path elsewhere in the same function returns `None`.

**Reuse vs. build-new for regime/calibration layers (2026-07-03):** When a
new spec's regime taxonomy or calibration inputs don't match an existing
module's labels/scope (e.g. existing regime.py is macro-index-level with
5 different labels; new spec wants per-ticker regime with a LOW_LIQUIDITY
label), don't force-fit a mapping onto the old module — build a new,
narrowly-scoped module for the new taxonomy while still reusing shared
low-level computations (e.g. efficiency-ratio/noise-index math) from
existing helpers so the underlying signal logic isn't duplicated.

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

**`patch.object` bypasses a method's own decorator (2026-07-03):** A test
using `patch.object(SomeClass, "some_method", side_effect=RuntimeError)` to
simulate an internal failure replaces the ENTIRE bound method, including any
`@_safe`-style fail-safe decorator wrapping it — so the raised exception is
not caught by that method's own guard, it propagates to whatever calls the
method. **Why this matters:** if the caller (e.g. `run_audit()`) has its own
outer `@_safe` wrapper, the realistic result is the caller's fallback value
(e.g. `{}`), not a partial report as if only the one check had silently
failed — the individual check's decorator provides zero protection once
`patch.object` has replaced it. **How to apply:** when writing "never raises"
tests for a fail-safe-decorated method by patching a sub-call, assert on
what the OUTER wrapper actually returns on total failure, not on an
idealized "graceful partial degradation" that the patch itself makes
impossible to observe.

**Feature-flag-off changes need no full-watchlist re-validation (recurring
pattern, confirmed again at T014 and T015):** Any purely additive module
that is (a) gated by a config flag defaulting to False and (b) never
imported into the live decision path in engine.py, only needs pytest + an
import/scanner-boot sanity check before being kept — not a full 407/408-
ticker backtest re-validation, since default behavior is provably
unchanged. Reserve full watchlist validation for changes that alter signal
generation, gating, or values used in a default-on code path.

**Regime-scoped hard blocks are a deliberate exception to "never disable
all strategies" (T015, 2026-07-03):** When a spec defines a regime→allowed-
strategies map and one regime (e.g. LOW_LIQUIDITY) maps to an empty list,
that's an intentional total block for that specific regime, not a violation
of a general "don't let one gate turn off everything" safety principle.
**Why:** the general principle guards against a single global weight/config
bug silently killing all trading; a regime-scoped block only fires when
that specific market condition is detected and is exactly what the spec
asked for. **How to apply:** when implementing a regime/condition-gated
map, treat an explicitly empty allow-list for one key as valid and
intentional — don't add a fallback that force-allows at least one strategy,
as that would contradict the spec.

**Flipping SHADOW_MODE off requires validating actual approval rate, not just
pytest (2026-07-03):** When every opportunity/* gate was switched to live
mode (`SHADOW_MODE=False`, all `ENABLE_*=True`) as requested, pytest passed
585/585 (tests patch flags directly, so they can't catch this), but a full-
pipeline replay of real historical ELITE/STRONG BUY signals through the
three-layer chain (trade_evaluator → adaptive_core → strategy_optimizer)
found `trade_evaluator`'s shipped default thresholds (`min_edge_score=0.65`,
`min_risk_reward=2.5`) rejected ~100% of this system's real signals — the
system's actual edge_score/risk_reward distributions (median ~0.14 / ~1.1)
never came close to those theoretical defaults. **Why this matters:** a
threshold that looks reasonable in isolation can be silently incompatible
with what upstream signal generation actually produces; unit tests that
patch flags/inputs directly cannot catch this class of bug, only an
end-to-end replay against real historical signal data can. **How to apply:**
whenever a gate/threshold layer's SHADOW_MODE or enable-flag flips from
observe-only to live (able to reject), before trusting it in production,
replay real historical qualifying signals through the exact live code path
and check the resulting approval rate is non-degenerate (neither ~0% nor
~100%) — recalibrate thresholds against the actual observed metric
distribution (e.g. 10th/90th percentile of the real data) rather than
generic assumed values, and remember to also update any paired
tuning-bounds table (e.g. `AUTO_TUNER_BOUNDS`) in lockstep, since stale
bounds will silently clamp new realistic thresholds back to the old
broken range the first time a regime adjustment or auto-tuner step runs.
Recalibrated result on the full 408-ticker watchlist: baseline (gates off)
win_rate 40.0%/profit_factor 0.925/expectancy_r -0.045 → live-gated (~31%
approval rate) win_rate 45.1%/profit_factor 1.181/expectancy_r +0.10 —
confirms the gates add real value once thresholds match the system's
actual signal distribution.

**Contextual adjusters (news/short-interest/VWAP/commodity) cannot be
backtested at all (2026-07-04):** `engine.py`'s `news_sentiment()`,
`short_interest_signal()`, `vwap_signal()`, `commodity_signal()` all call
`yf.Ticker(...).news`/`.info`/`.history(period="1d", interval="1h")` with
no date parameter — they only ever reflect TODAY's live data; no
point-in-time history for these exists anywhere in the codebase. Confirmed
the historical baseline generator (`manual_historical_backtest.py`'s
`score_row`/`score_active_mover`/`score_setup_mover`, used by
`full_watchlist_backtest.py`/`full_pipeline_live_gating_validation.py`)
does NOT call any of these adjusters — not an oversight, it's why
backtesting works at all for the rest of the model. **Why this matters:**
any proposed re-weighting of these specific adjusters cannot be validated
via backtest/rescan — don't spend time on an "expensive full rescan"
expecting it to reflect the change. **How to apply:** the only valid
validation path for these adjusters is forward/live paper-trading over
real time; say this up front if asked to tune/test them via backtest.

**Live-gating threshold tuning (validated 2026-07-04, NOT implemented —
user deferred):** Sweeping `TRADE_EVAL_THRESHOLDS` against the cached
live-gated baseline (`tests/candidate_sweep.py`, reuses cached per-ticker
signal pickles + reruns only the cheap 3-layer gating chain) and checking
Monte Carlo resampling (`opportunity/backtester.py::_monte_carlo`) is a
fast, valid way to test threshold changes without a full rescan — raising
`min_edge_score` 0.10→0.14 combined with `max_noise_index` 2.03→1.9 took
profit_factor 1.216→1.399 with consistently lower Monte Carlo
risk-of-ruin (47.9%→30.6%) on the full watchlist baseline (84 trades).
Also confirmed via the same harness: adaptive exit
(`engine.simulate_adaptive_exit`) is a clear regression when retested
against this current baseline (PF 1.399→0.635, expectancy went negative —
worse than the earlier 2026-07-03 result above, which was against an
older/different baseline, so don't compare the two numbers directly); the
existing live-scanner correlation guard (`CORRELATION_GROUPS`) had zero
overlap with this trade set when applied retroactively (safe no-op).
User said "forget it all together for now" — nothing here was merged into
`opportunity/config.py`; re-validate against the current baseline before
reusing if revisited later, since underlying data/model may have shifted.
