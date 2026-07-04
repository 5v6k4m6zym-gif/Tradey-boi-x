"""
Candidate C — adaptive exit retest, SCRATCH validation tool.

Retests engine.simulate_adaptive_exit() (partial profit-take + trailing
stop) against the CURRENT live-gated baseline (1.216 PF), not the
pre-live-gating baseline it was last tested against. Reuses the exact same
gated-trade set candidate_sweep.py produces for a given threshold override
(defaults to the config candidate being considered for Candidate A), and
recomputes each trade's outcome via the adaptive-exit simulation instead of
the fixed-horizon PREDICTION_DAYS exit, using the same cached price data.

Per project memory: never compare win_rate across an exit-methodology
change (the win/loss *definition* itself changes) — only compare
cost-adjusted profit_factor/expectancy_r, and call out the shift.

Run with: python3 tests/candidate_exit_sweep.py '{"min_edge_score": 0.14, "max_noise_index": 1.9}'
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_SCRATCH = Path(tempfile.mkdtemp(prefix="candidate_exit_"))
for var, name in [
    ("TE_LOG_PATH", "trade_evaluations.jsonl"),
    ("AUTO_TUNER_STATE_PATH", "auto_tuner_state.json"),
    ("AUTO_TUNER_LOG_PATH", "auto_tuner_decisions.jsonl"),
    ("ADAPTIVE_CORE_LOG_PATH", "adaptive_core_decisions.jsonl"),
    ("AUDIT_LOG_PATH", "audit_trades.jsonl"),
    ("AUDIT_STATE_PATH", "audit_state.json"),
    ("STRATEGY_LOG_PATH", "strategy_optimizer_decisions.jsonl"),
    ("STRATEGY_WEIGHTS_PATH", "strategy_weights.json"),
    ("STRATEGY_WEIGHT_STATE_PATH", "strategy_weight_state.json"),
]:
    os.environ[var] = str(_SCRATCH / name)

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


def apply_overrides(overrides: dict):
    te_keys = set(opp_config.TRADE_EVAL_THRESHOLDS.keys())
    for k, v in overrides.items():
        if k in te_keys:
            opp_config.TRADE_EVAL_THRESHOLDS[k] = v
        elif hasattr(opp_config, k):
            setattr(opp_config, k, v)


def run_pipeline(trade, df_slice):
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


def load_cached_baseline_records():
    records = []
    for f in EVAL_DIR.glob("*.pkl"):
        with open(f, "rb") as fh:
            saved = pickle.load(fh)
        records.extend(saved.get("baseline", []))
    return records


def main():
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    apply_overrides(overrides)

    baseline_records = load_cached_baseline_records()
    tickers_needed = sorted({r["ticker"] for r in baseline_records})
    data = load_data()
    data = {t: data[t] for t in tickers_needed if t in data}

    gated_fixed: list[dict] = []
    gated_adaptive: list[dict] = []

    for rec in baseline_records:
        ticker = rec["ticker"]
        df = data.get(ticker)
        if df is None:
            continue
        try:
            ts = pd.Timestamp(rec["signal_date"])
            if df.index.tz is not None:
                ts = ts.tz_localize(df.index.tz)
            idx = df.index.get_loc(ts)
        except KeyError:
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
            approved, _ = run_pipeline(trade, df_slice)
        except Exception:
            approved = True
        if not approved:
            continue

        gated_fixed.append(rec)

        hold_slice = df.iloc[idx + 1: idx + 1 + engine.PREDICTION_DAYS]
        sim = engine.simulate_adaptive_exit(
            hold_slice, rec["entry_price"], trade.get("stop_loss"), trade.get("take_profit"),
        )
        adaptive_rec = dict(rec)
        adaptive_rec["actual_pct"] = sim["actual_pct"]
        adaptive_rec["outcome"] = "WIN" if sim["actual_pct"] >= engine.TARGET_RETURN else "LOSS"
        gated_adaptive.append(adaptive_rec)

    print(f"Gated trade count: {len(gated_fixed)}")
    print("\n=== FIXED-HORIZON EXIT (current production exit mechanism) ===")
    fm = compute_metrics(gated_fixed)
    for k in ("trade_count", "win_rate", "profit_factor", "expectancy_r", "max_drawdown_pct"):
        print(f"  {k:20s}: {fm[k]}")

    print("\n=== ADAPTIVE EXIT (partial profit-take + trailing stop) ===")
    am = compute_metrics(gated_adaptive)
    for k in ("trade_count", "win_rate", "profit_factor", "expectancy_r", "max_drawdown_pct"):
        print(f"  {k:20s}: {am[k]}")

    print("\nNOTE per project memory: win_rate is not directly comparable across this "
          "exit-methodology change (the win/loss definition itself changes). Compare "
          "profit_factor/expectancy_r only.")

    print("\n=== MONTE CARLO: FIXED-HORIZON ===")
    mc_f = _monte_carlo(gated_fixed, n_simulations=1000)
    for k in ("profit_factor_median", "profit_factor_p5", "risk_of_ruin_pct"):
        print(f"  {k:24s}: {mc_f[k]}")

    print("\n=== MONTE CARLO: ADAPTIVE EXIT ===")
    mc_a = _monte_carlo(gated_adaptive, n_simulations=1000)
    for k in ("profit_factor_median", "profit_factor_p5", "risk_of_ruin_pct"):
        print(f"  {k:24s}: {mc_a[k]}")


if __name__ == "__main__":
    main()
