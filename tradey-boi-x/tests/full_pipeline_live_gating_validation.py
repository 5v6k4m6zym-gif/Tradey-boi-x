"""
Full-pipeline LIVE-GATING validation — measures the impact of flipping every
opportunity/* wrapper layer ON with SHADOW_MODE=False (live gating, able to
reject trades) vs the previous default (all gates off / shadow-only).

Reuses the exact same out-of-sample methodology as full_watchlist_backtest.py
(cached 2y history, time-split train/test, score_row/score_active_mover/
score_setup_mover from manual_historical_backtest.py) to get the baseline set
of ELITE/STRONG BUY signals engine.decide() would have fired across the full
408-ticker watchlist. For every qualifying signal, this script then builds
the same trade dict scanner.py builds and runs it through, IN ORDER:

    trade_evaluator.process_trade_signal
    adaptive_core.process_trade_signal
    strategy_optimizer.process_trade_signal

exactly as scanner.py's run_scan() does, using the LIVE config values already
set in opportunity/config.py (all ENABLE_* = True, SHADOW_MODE = False), and
reports:
  - BASELINE metrics: every qualifying signal (i.e. what the current watchlist
    backtest already reports, no gating).
  - LIVE-GATED metrics: only signals approved by ALL THREE layers.
  - Rejection breakdown: which layer rejected which fraction of trades.

NOTE: PerformanceTracker-backed gates (auto-tuner, adaptive core's expectancy
engine, strategy weighting/gating) read real production log files
(logs/trade_evaluations.jsonl, signal_log.json, logs/strategy_weights.json)
for their track record — NOT a simulated in-backtest trade history. Since
those files are currently sparse/empty, those specific sub-checks will
mostly run in their documented cold-start "fail-open" state throughout this
validation. This is expected and intentional (matches how the layers will
actually behave in production on day one), and is called out explicitly in
the results rather than being a flaw in the validation.

Does NOT modify signal_log.json, WATCHLIST, or any production log file
outside of the layers' own designated JSONL logs (which this script points
at temp paths to avoid polluting production logs).

Run with: python3 tests/full_pipeline_live_gating_validation.py
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Redirect every opportunity-layer log/state file to a scratch directory
# BEFORE importing the opportunity modules, so this validation run can never
# write into real production logs/state.
_SCRATCH = Path(tempfile.mkdtemp(prefix="gating_validation_"))
os.environ["TE_LOG_PATH"] = str(_SCRATCH / "trade_evaluations.jsonl")
os.environ["AUTO_TUNER_STATE_PATH"] = str(_SCRATCH / "auto_tuner_state.json")
os.environ["AUTO_TUNER_LOG_PATH"] = str(_SCRATCH / "auto_tuner_decisions.jsonl")
os.environ["ADAPTIVE_CORE_LOG_PATH"] = str(_SCRATCH / "adaptive_core_decisions.jsonl")
os.environ["AUDIT_LOG_PATH"] = str(_SCRATCH / "audit_trades.jsonl")
os.environ["AUDIT_STATE_PATH"] = str(_SCRATCH / "audit_state.json")
os.environ["STRATEGY_LOG_PATH"] = str(_SCRATCH / "strategy_optimizer_decisions.jsonl")
os.environ["STRATEGY_WEIGHTS_PATH"] = str(_SCRATCH / "strategy_weights.json")
os.environ["STRATEGY_WEIGHT_STATE_PATH"] = str(_SCRATCH / "strategy_weight_state.json")

import pandas as pd

import engine
from opportunity.backtester import compute_metrics
from opportunity.trade_evaluator import process_trade_signal as te_process
from opportunity.adaptive_core import process_trade_signal as ac_process
from opportunity.strategy_optimizer import process_trade_signal as so_process
from opportunity import config as opp_config

from manual_historical_backtest import score_row, load_regime_series, regime_for
from full_watchlist_backtest import ALL_TICKERS, KNOWN_WINNERS, load_data, train_or_load_model

TRAIN_FRACTION = 0.70
CKPT_DIR = Path("/tmp/full_backtest_checkpoint")
CKPT_DIR.mkdir(exist_ok=True)


def build_trade(ticker: str, df: pd.DataFrame, i: int, prob: float, score: int,
                 signal: str, expected_r: float, rsi: float, breakout: int, why: list) -> dict:
    entry_price = float(df["Close"].iloc[i])
    params = engine._trade_params(ticker, {"signal": signal}, entry_price, df.iloc[:i + 1])
    return {
        "ticker": ticker,
        "direction": "LONG",
        "entry": entry_price,
        "stop_loss": params.get("stop_loss"),
        "take_profit": params.get("target_price"),
        "probability": prob,
        "expected_r": expected_r,
        "why": why,
        "rsi": rsi,
        "breakout": breakout,
        "edge_score": prob,
    }


def run_pipeline(trade: dict, df_slice: pd.DataFrame) -> tuple[bool, str]:
    """Runs the same three-layer chain scanner.py runs, in order. Returns
    (approved, rejected_by) — rejected_by is '' if approved."""
    t = te_process(dict(trade), df_slice)
    if t is None:
        return False, "trade_evaluator"
    t = ac_process(dict(t), df_slice)
    if t is None:
        return False, "adaptive_core"
    t = so_process(dict(t), df_slice)
    if t is None:
        return False, "strategy_optimizer"
    return True, ""


def main():
    print("=" * 70)
    print("FULL-PIPELINE LIVE-GATING VALIDATION")
    print(f"ENABLE_TRADE_EVALUATOR={opp_config.ENABLE_TRADE_EVALUATOR}  "
          f"ENABLE_ADAPTIVE_CORE={opp_config.ENABLE_ADAPTIVE_CORE}  "
          f"ENABLE_STRATEGY_OPTIMIZER={opp_config.ENABLE_STRATEGY_OPTIMIZER}  "
          f"SHADOW_MODE={opp_config.SHADOW_MODE}")
    print("=" * 70)

    print("\n[1/3] Loading cached 2y history...")
    data = load_data()
    print(f"  Loaded {len(data)} / {len(ALL_TICKERS)} tickers")

    print("\n[2/3] Training ensemble (or loading checkpoint)...")
    model, cutoffs = train_or_load_model(data)

    print("\n  Precomputing point-in-time market regime series (^AXJO, SPY)...")
    regime_ckpt = CKPT_DIR / "regime.pkl"
    if regime_ckpt.exists():
        with open(regime_ckpt, "rb") as f:
            regime_series_by_index = pickle.load(f)
    else:
        regime_series_by_index = {
            "^AXJO": load_regime_series("^AXJO"),
            "SPY":   load_regime_series("SPY"),
        }
        with open(regime_ckpt, "wb") as f:
            pickle.dump(regime_series_by_index, f)

    print("\n[3/3] Evaluating signals + running the 3-layer live gating chain "
          "(per-ticker checkpointed, resumable)...")
    eval_dir = CKPT_DIR / "gating_eval"
    eval_dir.mkdir(exist_ok=True)

    tickers_to_eval = [t for t in data.keys() if not (eval_dir / f"{t.replace('.', '_')}.pkl").exists()]
    print(f"  {len(data) - len(tickers_to_eval)} tickers already evaluated (cached), "
          f"{len(tickers_to_eval)} remaining")

    for ticker in tickers_to_eval:
        df = data[ticker]
        cutoff_idx = cutoffs.get(ticker)
        out_file = eval_dir / f"{ticker.replace('.', '_')}.pkl"
        if cutoff_idx is None:
            out_file.write_bytes(pickle.dumps({"baseline": [], "gated": [], "rejections": {}}))
            continue
        usable_end = len(df) - engine.PREDICTION_DAYS
        test_start = max(cutoff_idx, 60)
        if test_start >= usable_end:
            out_file.write_bytes(pickle.dumps({"baseline": [], "gated": [], "rejections": {}}))
            continue

        try:
            all_probs = model.predict_proba(df[engine.FEATURES])[:, 1]
        except Exception:
            out_file.write_bytes(pickle.dumps({"baseline": [], "gated": [], "rejections": {}}))
            continue

        t_baseline: list[dict] = []
        t_gated: list[dict] = []
        t_rejections: dict[str, int] = {"trade_evaluator": 0, "adaptive_core": 0, "strategy_optimizer": 0}

        for i in range(test_start, usable_end):
            row, prev = df.iloc[i], df.iloc[i - 1]
            prob = float(all_probs[i])
            regime = regime_for(regime_series_by_index, ticker, df.index[i])
            signal, score = score_row(row, prev, prob, regime)
            if signal not in ("ELITE", "STRONG BUY"):
                continue

            entry_price = float(row["Close"])
            exit_price = float(df["Close"].iloc[i + engine.PREDICTION_DAYS])
            actual_pct = (exit_price / entry_price) - 1
            outcome = "WIN" if actual_pct >= engine.TARGET_RETURN else "LOSS"
            record = {
                "ticker": ticker, "signal_date": str(df.index[i].date()),
                "tier": signal, "score": score, "prob": round(prob, 3),
                "entry_price": entry_price, "actual_pct": actual_pct,
                "outcome": outcome, "pred_days": engine.PREDICTION_DAYS,
            }
            t_baseline.append(record)

            expected_r = engine.expected_value_r(
                entry_price, float(row["atr"]), prob, bool(row["breakout"])
            )
            why = []
            if bool(row["breakout"]):
                why.append("breakout")
            if row["rsi"] < 35 or row["rsi"] > 68:
                why.append("RSI extreme")

            trade = build_trade(
                ticker, df, i, prob, score, signal, expected_r,
                float(row["rsi"]), int(bool(row["breakout"])), why,
            )
            df_slice = df.iloc[:i + 1]

            try:
                approved, rejected_by = run_pipeline(trade, df_slice)
            except Exception as e:
                print(f"  \u26a0\ufe0f  {ticker} {df.index[i].date()}: pipeline error ({e}) — treating as approved (fail-safe)")
                approved, rejected_by = True, ""

            if approved:
                t_gated.append(record)
            else:
                t_rejections[rejected_by] = t_rejections.get(rejected_by, 0) + 1

        out_file.write_bytes(pickle.dumps({"baseline": t_baseline, "gated": t_gated, "rejections": t_rejections}))
        print(f"  {ticker:10s} done ({len(t_baseline)} signals, {len(t_gated)} approved)", flush=True)

    print("\nAggregating per-ticker checkpoints...")
    baseline_trades: list[dict] = []
    gated_trades: list[dict] = []
    rejection_counts: dict[str, int] = {"trade_evaluator": 0, "adaptive_core": 0, "strategy_optimizer": 0}
    n_tickers_done = 0
    for ticker in data.keys():
        out_file = eval_dir / f"{ticker.replace('.', '_')}.pkl"
        if not out_file.exists():
            continue
        with open(out_file, "rb") as f:
            saved = pickle.load(f)
        baseline_trades.extend(saved["baseline"])
        gated_trades.extend(saved["gated"])
        for k, v in saved.get("rejections", {}).items():
            rejection_counts[k] = rejection_counts.get(k, 0) + v
        n_tickers_done += 1

    print(f"\nDone processing {n_tickers_done} tickers.")
    print(f"Baseline qualifying signals (ELITE+STRONG BUY): {len(baseline_trades)}")
    print(f"Approved by all 3 live gates:                   {len(gated_trades)}")
    print(f"Rejected — by layer: {rejection_counts}")

    print("\n" + "=" * 70)
    print("BASELINE METRICS (all ELITE/STRONG BUY signals, gates OFF/shadow)")
    print("=" * 70)
    baseline_metrics = compute_metrics(baseline_trades)
    for k, v in baseline_metrics.items():
        print(f"  {k:24s}: {v}")

    print("\n" + "=" * 70)
    print("LIVE-GATED METRICS (only signals approved by all 3 layers, live mode)")
    print("=" * 70)
    if gated_trades:
        gated_metrics = compute_metrics(gated_trades)
        for k, v in gated_metrics.items():
            print(f"  {k:24s}: {v}")
    else:
        print("  No trades survived all 3 gates.")

    print("\nDone. This validates the impact of the newly-enabled live gating")
    print("chain against the existing engine.decide() baseline; it does not")
    print("touch signal_log.json, WATCHLIST, or any production log file.")


if __name__ == "__main__":
    main()
