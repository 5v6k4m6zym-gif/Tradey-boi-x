"""
Candidate D — trade correlation / diversification cap, SCRATCH validation tool.

The live scanner (scanner.py/overnight_scanner.py) already enforces a
same-scan correlation guard: only ONE ticker per CORRELATION_GROUPS group can
alert per scan cycle. This guard is NOT modeled in the historical backtest's
baseline_records — the backtest allows multiple correlated tickers (e.g.
BHP.AX + FMG.AX + RIO.AX) to count as separate trades on the same signal
date, which live trading never would. This checks whether retroactively
applying the same guard to the gated backtest trade set changes profit
factor (if correlated names cluster their losses together, dropping the
extras should reduce variance/tail risk without hurting PF much).

Run with: python3 tests/candidate_correlation_check.py '{"min_edge_score": 0.14, "max_noise_index": 1.9}'
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_SCRATCH = Path(tempfile.mkdtemp(prefix="candidate_corr_"))
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
        return False
    t = ac_process(dict(t), df_slice)
    if t is None:
        return False
    t = so_process(dict(t), df_slice)
    if t is None:
        return False
    return True


def corr_group(ticker: str) -> int | None:
    for i, group in enumerate(engine.CORRELATION_GROUPS):
        if ticker in group:
            return i
    return None


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

    gated: list[dict] = []
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
            approved = run_pipeline(trade, df_slice)
        except Exception:
            approved = True
        if approved:
            gated.append(rec)

    print(f"Gated trade count (no correlation guard): {len(gated)}")

    by_date: dict = {}
    for rec in gated:
        by_date.setdefault(rec["signal_date"], []).append(rec)

    deduped = []
    dropped = 0
    for date, recs in by_date.items():
        kept_groups: dict = {}
        ungrouped = []
        for rec in recs:
            g = corr_group(rec["ticker"])
            if g is None:
                ungrouped.append(rec)
                continue
            if g not in kept_groups or rec["score"] > kept_groups[g]["score"]:
                if g in kept_groups:
                    dropped += 1
                kept_groups[g] = rec
            else:
                dropped += 1
        deduped.extend(ungrouped)
        deduped.extend(kept_groups.values())

    print(f"Gated trade count (WITH correlation guard applied retroactively): {len(deduped)}  (dropped {dropped})")

    print("\n=== WITHOUT correlation guard (current backtest methodology) ===")
    m1 = compute_metrics(gated)
    for k in ("trade_count", "win_rate", "profit_factor", "expectancy_r", "max_drawdown_pct"):
        print(f"  {k:20s}: {m1[k]}")

    print("\n=== WITH correlation guard (matches live scanner behavior) ===")
    m2 = compute_metrics(deduped)
    for k in ("trade_count", "win_rate", "profit_factor", "expectancy_r", "max_drawdown_pct"):
        print(f"  {k:20s}: {m2[k]}")

    print("\n=== MONTE CARLO: WITH correlation guard ===")
    mc = _monte_carlo(deduped, n_simulations=1000)
    for k in ("profit_factor_median", "profit_factor_p5", "risk_of_ruin_pct"):
        print(f"  {k:24s}: {mc[k]}")


if __name__ == "__main__":
    main()
