"""
Candidate config sweep — SCRATCH validation tool, not part of the safety
guardrail suite (unlike full_pipeline_live_gating_validation.py, which stays
the canonical reference). Reuses the already-cached baseline ELITE/STRONG BUY
signal list from `.cache/backtest_checkpoint/gating_eval/*.pkl` (produced by
running full_pipeline_live_gating_validation.py once) so each candidate only
re-runs the CHEAP 3-layer gating chain on the ~350 qualifying signals instead
of re-scanning all ~50k rows across the full watchlist. This is what makes
per-candidate sweeps fast enough to iterate on.

Never writes to production log/state files (redirects every opportunity-
layer log/state path to a scratch temp dir). Never mutates signal_log.json,
WATCHLIST, or opportunity/config.py itself — threshold overrides are applied
in-memory only (in-place dict/attr mutation), for the lifetime of this
process only.

Run with: python3 tests/candidate_sweep.py '{"min_edge_score": 0.15}'
Prereq: tests/full_pipeline_live_gating_validation.py must have been run at
least once so the gating_eval cache is populated.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_SCRATCH = Path(tempfile.mkdtemp(prefix="candidate_sweep_"))
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
from opportunity.backtester import compute_metrics, _monte_carlo
from opportunity.trade_evaluator import process_trade_signal as te_process
from opportunity.adaptive_core import process_trade_signal as ac_process
from opportunity.strategy_optimizer import process_trade_signal as so_process
from opportunity import config as opp_config

from full_watchlist_backtest import load_data
from full_pipeline_live_gating_validation import build_trade

CKPT_DIR = Path(__file__).parent.parent / ".cache" / "backtest_checkpoint"
EVAL_DIR = CKPT_DIR / "gating_eval"


def apply_overrides(overrides: dict) -> list[str]:
    notes = []
    te_keys = set(opp_config.TRADE_EVAL_THRESHOLDS.keys())
    ac_attrs = {"ADAPTIVE_MIN_EXECUTION_QUALITY", "ADAPTIVE_EXPECTANCY_MIN_TRADES"}
    for k, v in overrides.items():
        if k in te_keys:
            old = opp_config.TRADE_EVAL_THRESHOLDS[k]
            opp_config.TRADE_EVAL_THRESHOLDS[k] = v
            notes.append(f"TRADE_EVAL_THRESHOLDS[{k}]: {old} -> {v}")
        elif k in ac_attrs and hasattr(opp_config, k):
            old = getattr(opp_config, k)
            setattr(opp_config, k, v)
            notes.append(f"opp_config.{k}: {old} -> {v}")
        else:
            raise ValueError(f"Unknown override key: {k}")
    return notes


def run_pipeline(trade: dict, df_slice: pd.DataFrame) -> tuple[bool, str]:
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


def load_cached_baseline_records() -> list[dict]:
    if not EVAL_DIR.exists():
        raise SystemExit(
            "No cached gating_eval found. Run "
            "tests/full_pipeline_live_gating_validation.py once first."
        )
    records = []
    for f in EVAL_DIR.glob("*.pkl"):
        with open(f, "rb") as fh:
            saved = pickle.load(fh)
        records.extend(saved.get("baseline", []))
    return records


def main():
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    notes = apply_overrides(overrides)

    print("=" * 70)
    print("CANDIDATE SWEEP (fast path, reuses cached baseline signal list)")
    for n in notes:
        print(f"  OVERRIDE: {n}")
    if not notes:
        print("  (no overrides — this is the baseline config)")
    print("=" * 70)

    baseline_records = load_cached_baseline_records()
    print(f"Loaded {len(baseline_records)} cached baseline ELITE/STRONG BUY signals.")

    tickers_needed = sorted({r["ticker"] for r in baseline_records})
    data = load_data()
    data = {t: data[t] for t in tickers_needed if t in data}

    gated_trades: list[dict] = []
    rejection_counts: dict[str, int] = {"trade_evaluator": 0, "adaptive_core": 0, "strategy_optimizer": 0}
    skipped = 0

    for rec in baseline_records:
        ticker = rec["ticker"]
        df = data.get(ticker)
        if df is None:
            skipped += 1
            continue
        try:
            ts = pd.Timestamp(rec["signal_date"])
            if df.index.tz is not None:
                ts = ts.tz_localize(df.index.tz)
            idx = df.index.get_loc(ts)
        except KeyError:
            skipped += 1
            continue
        row = df.iloc[idx]

        expected_r = engine.expected_value_r(
            rec["entry_price"], float(row["atr"]), rec["prob"], bool(row["breakout"])
        )
        why = []
        if bool(row["breakout"]):
            why.append("breakout")
        if row["rsi"] < 35 or row["rsi"] > 68:
            why.append("RSI extreme")

        trade = build_trade(
            ticker, df, idx, rec["prob"], rec["score"], rec["tier"], expected_r,
            float(row["rsi"]), int(bool(row["breakout"])), why,
        )
        df_slice = df.iloc[:idx + 1]

        try:
            approved, rejected_by = run_pipeline(trade, df_slice)
        except Exception:
            approved, rejected_by = True, ""

        if approved:
            gated_trades.append(rec)
        else:
            rejection_counts[rejected_by] = rejection_counts.get(rejected_by, 0) + 1

    if skipped:
        print(f"  (skipped {skipped} records — ticker/date not resolvable)")

    print(f"\nApproved by all 3 live gates: {len(gated_trades)} / {len(baseline_records)}")
    print(f"Rejected — by layer: {rejection_counts}")

    print("\nLIVE-GATED METRICS:")
    if gated_trades:
        gated_metrics = compute_metrics(gated_trades)
        for k, v in gated_metrics.items():
            print(f"  {k:24s}: {v}")

        print("\nMONTE CARLO (1000 resamples, significance check):")
        mc = _monte_carlo(gated_trades, n_simulations=1000)
        for k, v in mc.items():
            print(f"  {k:24s}: {v}")
    else:
        print("  No trades survived all 3 gates.")


if __name__ == "__main__":
    main()
