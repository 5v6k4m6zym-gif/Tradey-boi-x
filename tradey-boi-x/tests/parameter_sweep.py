"""
Parameter sweep — find optimal stop/hold/threshold combinations.

Uses the 358 already-identified ELITE/STRONG BUY signals from the backtest
eval cache. Only loads price data for the 103 unique tickers that had signals
(no re-running predict_proba on 406 tickers).

Phase 1: Load signals from eval cache, load price data for 103 tickers only.
Phase 2: For each signal compute exit outcomes for all (config, hold_window)
         combos — stored as plain floats, not DataFrames.
Phase 3: Sweep 60 parameter combos in < 1s.
Phase 4: Report + recommend.

Run: python3 tests/parameter_sweep.py
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import engine
from opportunity.backtester import compute_metrics

CACHE_DIR    = Path(__file__).parent.parent / ".cache" / "ticker_history"
CKPT_DIR     = Path(__file__).parent.parent / ".cache" / "backtest_checkpoint"
EVAL_DIR     = CKPT_DIR / "eval"
SWEEP_CACHE  = CKPT_DIR / "sweep_v3.pkl"

HOLD_WINDOWS = [10, 15, 20]

# ATR stop/target configs — (sl_hi, sl_mid, sl_lo, tgt_hi, tgt_mid, tgt_lo)
SL_TARGET_CONFIGS: dict[str, tuple] = {
    "current  (2.0/1.5/1.2sl | 8/5/3%tgt)":  (2.0, 1.5, 1.2, 0.08, 0.05, 0.03),
    "tighter  (1.5/1.2/1.0sl | 8/5/3%tgt)":  (1.5, 1.2, 1.0, 0.08, 0.05, 0.03),
    "tightest (1.2/1.0/0.8sl | 8/5/3%tgt)":  (1.2, 1.0, 0.8, 0.08, 0.05, 0.03),
    "wide-tgt (1.5/1.2/1.0sl |12/8/5%tgt)":  (1.5, 1.2, 1.0, 0.12, 0.08, 0.05),
    "balanced (1.5/1.2/1.0sl |10/6/4%tgt)":  (1.5, 1.2, 1.0, 0.10, 0.06, 0.04),
}
CFG_KEYS = list(SL_TARGET_CONFIGS.keys())

ELITE_THRESHOLDS = [8, 9, 10]
MIN_EXPECTED_R   = [0.0, 0.3]


# ─────────────────────────────────────────────────────────────────────────────
def _compute_stop_target(cfg: tuple, atr_pct: float, entry: float,
                          tier: str, breakout: bool) -> tuple[float, float]:
    sl_hi, sl_mid, sl_lo, tgt_hi, tgt_mid, tgt_lo = cfg
    atr_abs = atr_pct / 100.0 * entry
    if atr_pct >= 3.0:
        sl = sl_hi; tgt = tgt_hi
    elif atr_pct >= 1.5:
        sl = sl_mid; tgt = tgt_mid
    else:
        sl = sl_lo; tgt = tgt_lo
    t = 1.25 if tier == "ELITE" else 1.0
    if breakout: t *= 1.15
    stop   = max(entry - sl * atr_abs, entry * 0.85)
    target = entry * (1 + min(tgt * t, 0.20))
    return stop, target


def _exit_pct(lo: np.ndarray, hi: np.ndarray, close: np.ndarray,
               entry: float, stop: float, target: float) -> float:
    for i in range(len(lo)):
        if lo[i] <= stop:
            return (stop - entry) / entry
        if hi[i] >= target:
            return (target - entry) / entry
    return (close[-1] - entry) / entry if len(close) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def build_enriched_signals(force: bool = False) -> list[dict]:
    """
    Load the 358 ELITE/STRONG BUY signals from the eval cache, then load
    price data for the 103 unique tickers and compute exit outcomes for every
    (config, hold_window) combo. Returns a flat list of enriched records.
    """
    if SWEEP_CACHE.exists() and not force:
        print(f"  Loading sweep cache …")
        with open(SWEEP_CACHE, "rb") as f:
            return pickle.load(f)

    # 1. Load all signals from eval cache
    raw_signals: list[dict] = []
    for f in sorted(EVAL_DIR.glob("*.pkl")):
        try:
            with open(f, "rb") as fh:
                saved = pickle.load(fh)
        except Exception:
            continue
        for rec in saved.get("all", []):
            if rec["tier"] in ("ELITE", "STRONG BUY"):
                tr = rec.pop("_trade_record", None)
                rec["_trade_record"] = tr
                raw_signals.append(rec)

    print(f"  {len(raw_signals)} qualifying signals across "
          f"{len(set(r['ticker'] for r in raw_signals))} tickers")

    # 2. Load price data for unique tickers only
    unique_tickers = list(set(r["ticker"] for r in raw_signals))
    price_data: dict[str, pd.DataFrame] = {}
    for ticker in unique_tickers:
        pkl = CACHE_DIR / f"{ticker.replace('.AX', '_AX').replace('.', '_')}.pkl"
        if not pkl.exists():
            # Try alternate naming
            alts = list(CACHE_DIR.glob(f"{ticker.split('.')[0]}*.pkl"))
            if not alts:
                continue
            pkl = alts[0]
        try:
            price_data[ticker] = pd.read_pickle(pkl)
        except Exception:
            pass

    print(f"  Price data loaded for {len(price_data)} / {len(unique_tickers)} tickers")

    max_hold = max(HOLD_WINDOWS)

    # 3. For each signal, compute exits for every (cfg, hold_window)
    enriched: list[dict] = []
    missing = 0
    for rec in raw_signals:
        ticker = rec["ticker"]
        if ticker not in price_data:
            missing += 1
            continue

        df    = price_data[ticker]
        entry = float(rec["entry_price"])
        atr_pct = float(rec["atr_pct"])
        breakout = bool(rec.get("breakout", False))

        # Find the index for signal_date
        try:
            date_idx = df.index.get_loc(rec["signal_date"])
        except KeyError:
            # Try as Timestamp
            try:
                ts = pd.Timestamp(rec["signal_date"])
                date_idx = df.index.get_loc(ts)
            except Exception:
                missing += 1
                continue

        # Pre-extract numpy arrays
        lo_arr  = df["Low"].to_numpy(float)
        hi_arr  = df["High"].to_numpy(float)
        cl_arr  = df["Close"].to_numpy(float)

        # expected_r from stored prob + atr_pct
        prob    = float(rec["prob"])
        atr_abs = atr_pct / 100.0 * entry
        exp_r   = engine.expected_value_r(entry, atr_abs, prob, breakout)

        exits: dict[tuple, float] = {}
        for cfg_key, cfg_tuple in SL_TARGET_CONFIGS.items():
            stop, target = _compute_stop_target(
                cfg_tuple, atr_pct, entry, rec["tier"], breakout
            )
            for hw in HOLD_WINDOWS:
                start = date_idx + 1
                end   = start + hw
                if end > len(lo_arr):
                    continue
                lo = lo_arr[start:end]
                hi = hi_arr[start:end]
                cl = cl_arr[start:end]
                pct = _exit_pct(lo, hi, cl, entry, stop, target)
                exits[(cfg_key, hw)] = pct

        enriched.append({
            "tier":      rec["tier"],
            "score":     rec["score"],
            "prob":      prob,
            "atr_pct":   atr_pct,
            "breakout":  breakout,
            "expected_r": exp_r,
            "exits":     exits,
        })

    if missing:
        print(f"  ⚠  {missing} signals skipped (no price data or date not found)")
    print(f"  {len(enriched)} signals enriched with exit outcomes")

    with open(SWEEP_CACHE, "wb") as f:
        pickle.dump(enriched, f)
    print(f"  Cached to {SWEEP_CACHE}")
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
def sweep(records: list[dict]) -> list[dict]:
    combos = list(itertools.product(
        HOLD_WINDOWS, ELITE_THRESHOLDS, MIN_EXPECTED_R, CFG_KEYS
    ))
    results = []
    for hold_days, elite_thr, min_er, cfg_key in combos:
        trades = []
        for r in records:
            sig   = r["tier"]
            score = r["score"]
            # Apply parametric ELITE threshold
            if sig == "ELITE" and score < elite_thr:
                sig = "STRONG BUY"
            if sig == "STRONG BUY" and score < 6:
                continue
            if r["expected_r"] < min_er:
                continue
            key = (cfg_key, hold_days)
            if key not in r["exits"]:
                continue
            pct = r["exits"][key]
            trades.append({
                "tier":       sig,
                "actual_pct": pct,
                "outcome":    "WIN" if pct >= 0.0 else "LOSS",
                "pred_days":  hold_days,
                "atr_pct":    r["atr_pct"],
            })
        if not trades:
            continue
        m = compute_metrics(trades)
        results.append({
            "hold_days": hold_days, "elite_thr": elite_thr,
            "min_er":    min_er,    "config":    cfg_key.strip(),
            "n":  m["trade_count"],  "wr":       m["win_rate"],
            "avg_gain":  m["avg_gain_pct"],  "avg_loss": m["avg_loss_pct"],
            "pf":        m["profit_factor"], "exp_r":    m["expectancy_r"],
            "sharpe":    m["sharpe_ratio"],
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
def report(results: list[dict]) -> dict | None:
    valid  = [r for r in results if r["n"] >= 50]
    if not valid: valid = results

    pf_inf = lambda r: r["pf"] if r["pf"] != float("inf") else 999.0
    by_pf  = sorted(valid, key=pf_inf, reverse=True)
    by_exp = sorted(valid, key=lambda x: x["exp_r"], reverse=True)

    baseline = next(
        (r for r in results if r["hold_days"] == 10 and r["elite_thr"] == 8
         and r["min_er"] == 0.0 and "current" in r["config"]), None
    )

    HDR = (f"{'Config':<44} {'Hold':>4} {'EThr':>4} {'mER':>4} "
           f"{'N':>5} {'WR%':>6} {'AvgG%':>7} {'AvgL%':>7} {'PF':>6} {'ExpR':>6}")
    SEP = "─" * 95

    print(f"\n{'='*95}")
    print("TOP 20 BY PROFIT FACTOR  (≥50 signals)")
    print('='*95)
    print(HDR); print(SEP)
    for r in by_pf[:20]:
        print(f"{r['config']:<44} {r['hold_days']:>4} {r['elite_thr']:>4} {r['min_er']:>4.1f} "
              f"{r['n']:>5} {r['wr']*100:>6.1f} {r['avg_gain']:>+7.2f} {r['avg_loss']:>7.2f} "
              f"{pf_inf(r):>6.3f} {r['exp_r']:>+6.3f}")

    if baseline:
        print(f"\n{'BASELINE (current, 10d, thr=8, mER=0)':<44} {baseline['hold_days']:>4} "
              f"{baseline['elite_thr']:>4} {baseline['min_er']:>4.1f} {baseline['n']:>5} "
              f"{baseline['wr']*100:>6.1f} {baseline['avg_gain']:>+7.2f} {baseline['avg_loss']:>7.2f} "
              f"{pf_inf(baseline):>6.3f} {baseline['exp_r']:>+6.3f}")

    print(f"\n{'='*95}")
    print("TOP 10 BY EXPECTANCY")
    print('='*95)
    print(HDR); print(SEP)
    for r in by_exp[:10]:
        print(f"{r['config']:<44} {r['hold_days']:>4} {r['elite_thr']:>4} {r['min_er']:>4.1f} "
              f"{r['n']:>5} {r['wr']*100:>6.1f} {r['avg_gain']:>+7.2f} {r['avg_loss']:>7.2f} "
              f"{pf_inf(r):>6.3f} {r['exp_r']:>+6.3f}")

    # Recommend: best PF with at least 80 signals (statistically robust)
    high_n  = [r for r in by_pf if r["n"] >= 80]
    winner  = high_n[0] if high_n else (by_pf[0] if by_pf else None)
    if winner:
        print(f"\n{'='*95}")
        print("RECOMMENDED IMPLEMENTATION")
        print('='*95)
        pf = pf_inf(winner)
        print(f"  Config       : {winner['config']}")
        print(f"  Hold window  : {winner['hold_days']} days  →  PREDICTION_DAYS")
        print(f"  ELITE thresh : score ≥ {winner['elite_thr']}")
        print(f"  Min Exp(R)   : {winner['min_er']}")
        print(f"  Result       : n={winner['n']}  WR={winner['wr']*100:.1f}%  "
              f"PF={pf:.3f}  ExpR={winner['exp_r']:+.3f}R")
        if baseline:
            bpf = pf_inf(baseline)
            delta_pf  = (pf - bpf) / max(bpf, 0.001) * 100
            delta_exp = winner["exp_r"] - baseline["exp_r"]
            print(f"  vs baseline  : PF {delta_pf:+.0f}%  |  ExpR {delta_exp:+.3f}R")

    # Also show per-tier breakdown for winner
    if winner:
        print(f"\n  Per-tier in winning config:")
        for tier_label in ("ELITE", "STRONG BUY"):
            subset = []
            for r in records_global:
                sig   = r["tier"]
                score = r["score"]
                if sig == "ELITE" and score < winner["elite_thr"]:
                    sig = "STRONG BUY"
                if sig == "STRONG BUY" and score < 6: continue
                if r["expected_r"] < winner["min_er"]: continue
                key = (winner["config_raw"], winner["hold_days"])
                if key not in r["exits"]: continue
                if sig != tier_label: continue
                pct = r["exits"][key]
                subset.append({"actual_pct": pct, "outcome": "WIN" if pct >= 0 else "LOSS",
                                "pred_days": winner["hold_days"], "atr_pct": r["atr_pct"]})
            if subset:
                m = compute_metrics(subset)
                print(f"    {tier_label:<12}: n={m['trade_count']:4d}  WR={m['win_rate']*100:.1f}%  "
                      f"AvgG={m['avg_gain_pct']:+.2f}%  AvgL={m['avg_loss_pct']:.2f}%  "
                      f"PF={pf_inf(m) if m['profit_factor']!=float('inf') else 'm[profit_factor]':.3f}")

    return winner


records_global: list[dict] = []

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force-recompute", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print("PARAMETER SWEEP — Tradey Boi X v3")
    print("=" * 70)

    print("\n[1/3] Loading signals + computing exits for all configs …")
    records_global = build_enriched_signals(force=args.force_recompute)

    print(f"\n[2/3] Sweeping {len(HOLD_WINDOWS)*len(ELITE_THRESHOLDS)*len(MIN_EXPECTED_R)*len(CFG_KEYS)} combos …")
    results = sweep(records_global)

    print("\n[3/3] Results:")
    winner = report(results)
    if winner:
        print(f"\n  → config_raw key: {[k for k in CFG_KEYS if k.strip() == winner['config']]}")

    print("\nDone — no production files modified.")
