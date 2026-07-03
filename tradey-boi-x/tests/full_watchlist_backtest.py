"""Full-watchlist historical backtest — same out-of-sample methodology as
manual_historical_backtest.py (see that file's docstring for full details),
but run across the ENTIRE 408-ticker WATCHLIST plus a handful of known big
historical winners not on the watchlist, to check two things:
  1. Does the win rate / AUC picture change with far more signal instances?
  2. Would the system have actually flagged well-known past big winners
     during their run-up (a qualitative "does it catch obvious winners" check)?

Reuses score_row/score_active_mover/score_setup_mover from
manual_historical_backtest.py unchanged — no scoring-logic duplication.

Checkpoints the trained model + per-ticker probability arrays to
/tmp/full_backtest_checkpoint/ so this can be safely re-run/resumed within
short foreground timeouts without re-fetching or re-training every time.

Does NOT modify signal_log.json, WATCHLIST, or any production file/constant.
Run with: python3 tests/full_watchlist_backtest.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import engine
from opportunity.backtester import compute_metrics
from manual_historical_backtest import (
    score_row, score_active_mover, score_setup_mover,
    load_regime_series, regime_for,
)

CACHE_DIR = Path("/tmp/full_ticker_cache")
CKPT_DIR = Path("/tmp/full_backtest_checkpoint")
CKPT_DIR.mkdir(exist_ok=True)
TRAIN_FRACTION = 0.70

# Known big historical winners (past ~2 years) not already in WATCHLIST —
# spotlighted below to see if the system would have flagged them, regardless
# of whether they clear the ELITE/STRONG BUY bar in the aggregate stats.
KNOWN_WINNERS = [
    "PLTR", "SMCI", "PME.AX", "WEB.AX", "PRU.AX", "NXT.AX", "WGX.AX",
    "MSTR", "VST", "ANET", "COIN",
]

ALL_TICKERS = list(dict.fromkeys(list(engine.WATCHLIST) + KNOWN_WINNERS))


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in ALL_TICKERS:
        cache_file = CACHE_DIR / f"{ticker.replace('.', '_')}.pkl"
        if not cache_file.exists() or cache_file.stat().st_size == 0:
            continue
        try:
            df = pd.read_pickle(cache_file)
        except Exception:
            continue
        if len(df) < 150:
            continue
        data[ticker] = df
    return data


def train_or_load_model(data: dict[str, pd.DataFrame]):
    model_ckpt = CKPT_DIR / "model.pkl"
    cutoffs_ckpt = CKPT_DIR / "cutoffs.pkl"
    if model_ckpt.exists() and cutoffs_ckpt.exists():
        print("  (loading cached trained model from checkpoint)")
        with open(model_ckpt, "rb") as f:
            model = pickle.load(f)
        with open(cutoffs_ckpt, "rb") as f:
            cutoffs = pickle.load(f)
        return model, cutoffs

    train_frames = []
    cutoffs: dict[str, int] = {}
    for ticker, df in data.items():
        cutoff_idx = int(len(df) * TRAIN_FRACTION)
        cutoffs[ticker] = cutoff_idx
        train_df = df.iloc[:cutoff_idx].copy()
        train_df["target"] = (
            train_df["Close"].shift(-engine.PREDICTION_DAYS) / train_df["Close"] - 1
            > engine.TARGET_RETURN
        ).astype(int)
        train_frames.append(train_df.dropna())

    combined = pd.concat(train_frames, ignore_index=True)
    X_train, y_train = combined[engine.FEATURES], combined["target"]
    print(f"  Train set: {len(combined):,} rows "
          f"({int(y_train.sum())} buy / {int((y_train == 0).sum())} no-buy)")

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    neg = int((y_train == 0).sum())
    pos = int(y_train.sum())
    spw = round(neg / pos, 2) if pos > 0 else 1.0

    xgb_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, verbosity=0,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)

    rf_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)

    model = engine.EnsembleModel(xgb_pipe, rf_pipe)
    with open(model_ckpt, "wb") as f:
        pickle.dump(model, f)
    with open(cutoffs_ckpt, "wb") as f:
        pickle.dump(cutoffs, f)
    return model, cutoffs


def main():
    print("=" * 70)
    print(f"FULL-WATCHLIST HISTORICAL BACKTEST — {len(ALL_TICKERS)} tickers "
          f"(408 watchlist + {len(KNOWN_WINNERS)} known winners)")
    print("=" * 70)

    print("\n[1/4] Loading cached 2y history...")
    data = load_data()
    print(f"  Loaded {len(data)} / {len(ALL_TICKERS)} tickers")

    print("\n[2/4] Training ensemble (or loading checkpoint)...")
    model, cutoffs = train_or_load_model(data)

    print("\n" + "=" * 70)
    print("AI MODEL DIAGNOSTICS (held-out test-period rows, all tickers combined)")
    print("=" * 70)
    test_frames = []
    for ticker, df in data.items():
        cutoff_idx = cutoffs.get(ticker)
        if cutoff_idx is None:
            continue
        test_df = df.iloc[cutoff_idx:].copy()
        test_df["target"] = (
            test_df["Close"].shift(-engine.PREDICTION_DAYS) / test_df["Close"] - 1
            > engine.TARGET_RETURN
        ).astype(int)
        test_frames.append(test_df.dropna(subset=engine.FEATURES + ["target"]))
    test_combined = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    if len(test_combined):
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

        X_test = test_combined[engine.FEATURES]
        y_test = test_combined["target"]
        p_test = model.predict_proba(X_test)[:, 1]
        base_rate = float(y_test.mean())
        try:
            auc = roc_auc_score(y_test, p_test)
        except Exception:
            auc = float("nan")
        try:
            ll = log_loss(y_test, p_test)
            ll_baseline = log_loss(y_test, [base_rate] * len(y_test))
        except Exception:
            ll, ll_baseline = float("nan"), float("nan")
        brier = brier_score_loss(y_test, p_test)

        print(f"  Held-out rows: {len(test_combined):,}  |  base rate (actual buy%): {base_rate*100:.1f}%")
        print(f"  ROC AUC:        {auc:.3f}   (0.50 = coin flip, 1.00 = perfect)")
        print(f"  Log loss:       {ll:.4f}   (baseline: {ll_baseline:.4f})")
        print(f"  Brier score:    {brier:.4f}")
    else:
        print("  ⚠️  No held-out rows available for diagnostics.")

    print(f"\n[3/4] Evaluating signals on held-out test period only (per-ticker checkpointed)...")
    eval_dir = CKPT_DIR / "eval"
    eval_dir.mkdir(exist_ok=True)

    regime_ckpt = CKPT_DIR / "regime.pkl"
    if regime_ckpt.exists():
        with open(regime_ckpt, "rb") as f:
            regime_series_by_index = pickle.load(f)
    else:
        print("  Precomputing point-in-time market regime series (^AXJO, SPY)...")
        regime_series_by_index = {
            "^AXJO": load_regime_series("^AXJO"),
            "SPY":   load_regime_series("SPY"),
        }
        with open(regime_ckpt, "wb") as f:
            pickle.dump(regime_series_by_index, f)

    tickers_to_eval = [t for t in data.keys() if not (eval_dir / f"{t.replace('.', '_')}.pkl").exists()]
    print(f"  {len(data) - len(tickers_to_eval)} tickers already evaluated (cached), "
          f"{len(tickers_to_eval)} remaining")

    for ticker in tickers_to_eval:
        df = data[ticker]
        cutoff_idx = cutoffs.get(ticker)
        out_file = eval_dir / f"{ticker.replace('.', '_')}.pkl"
        if cutoff_idx is None:
            out_file.write_bytes(pickle.dumps({"all": [], "moves": []}))
            continue
        usable_end = len(df) - engine.PREDICTION_DAYS
        test_start = max(cutoff_idx, 60)
        if test_start >= usable_end:
            out_file.write_bytes(pickle.dumps({"all": [], "moves": []}))
            continue

        try:
            all_probs = model.predict_proba(df[engine.FEATURES])[:, 1]
        except Exception as e:
            print(f"  ⚠️  {ticker}: predict_proba failed ({e})")
            out_file.write_bytes(pickle.dumps({"all": [], "moves": []}))
            continue

        ticker_evaluated: list[dict] = []
        ticker_moves: list[dict] = []

        for i in range(test_start, usable_end):
            row, prev = df.iloc[i], df.iloc[i - 1]
            prob = float(all_probs[i])
            regime = regime_for(regime_series_by_index, ticker, df.index[i])

            signal, score = score_row(row, prev, prob, regime)

            entry_price = float(row["Close"])
            exit_price = float(df["Close"].iloc[i + engine.PREDICTION_DAYS])
            actual_pct = (exit_price / entry_price) - 1
            outcome = "WIN" if actual_pct >= engine.TARGET_RETURN else "LOSS"

            record = {
                "ticker": ticker,
                "signal_date": str(df.index[i].date()),
                "tier": signal,
                "score": score,
                "prob": round(prob, 3),
                "entry_price": entry_price,
                "actual_pct": actual_pct,
                "outcome": outcome,
                "pred_days": engine.PREDICTION_DAYS,
            }
            ticker_evaluated.append(record)

            fired, mv_score = score_active_mover(df, i, prob)
            mv_tier = "ACTIVE" if fired else None
            if not fired:
                fired, mv_score = score_setup_mover(df, i, prob)
                mv_tier = "SETUP" if fired else None

            if fired:
                ticker_moves.append({**record, "tier": mv_tier, "score": mv_score})

        out_file.write_bytes(pickle.dumps({"all": ticker_evaluated, "moves": ticker_moves}))
        print(f"  {ticker:10s} done", flush=True)

    print("\n[3b/4] Aggregating per-ticker checkpoints...")
    trades: list[dict] = []
    all_evaluated: list[dict] = []
    tier_counts: dict[str, int] = {"ELITE": 0, "STRONG BUY": 0, "WATCH": 0, "IGNORE": 0, "GATED": 0}
    mover_trades: list[dict] = []
    mover_counts: dict[str, int] = {"ACTIVE": 0, "SETUP": 0}
    winner_hits: dict[str, list[dict]] = {t: [] for t in KNOWN_WINNERS}

    for ticker in data.keys():
        out_file = eval_dir / f"{ticker.replace('.', '_')}.pkl"
        if not out_file.exists():
            print(f"  ⚠️  {ticker}: not yet evaluated (checkpoint missing), skipping in aggregation")
            continue
        with open(out_file, "rb") as f:
            saved = pickle.load(f)
        for record in saved["all"]:
            all_evaluated.append(record)
            tier_counts[record["tier"]] = tier_counts.get(record["tier"], 0) + 1
            if record["tier"] in ("ELITE", "STRONG BUY"):
                trades.append(record)
                if ticker in winner_hits:
                    winner_hits[ticker].append({**record, "kind": record["tier"]})
        for mv_record in saved["moves"]:
            mover_counts[mv_record["tier"]] = mover_counts.get(mv_record["tier"], 0) + 1
            mover_trades.append(mv_record)
            if ticker in winner_hits:
                winner_hits[ticker].append({**mv_record, "kind": f"MOVER-{mv_record['tier']}"})

    print(f"\nComputing metrics over {len(trades)} out-of-sample historical signals "
          f"(across {len(data)} tickers, {len(all_evaluated):,} evaluated days)...")
    print("\nTier distribution across all evaluated days:")
    for tier, count in tier_counts.items():
        print(f"  {tier:12s}: {count}")

    metrics = compute_metrics(trades)
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (ELITE + STRONG BUY signals only, full watchlist)")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:24s}: {v}")

    if trades:
        elite = [t for t in trades if t["tier"] == "ELITE"]
        strong = [t for t in trades if t["tier"] == "STRONG BUY"]
        print("\nPer-tier breakdown:")
        for label, subset in (("ELITE", elite), ("STRONG BUY", strong)):
            if subset:
                m = compute_metrics(subset)
                print(f"  {label:12s}: n={m['trade_count']:4d}  win_rate={m['win_rate']*100:5.1f}%  "
                      f"avg_gain={m['avg_gain_pct']:+.2f}%  avg_loss={m['avg_loss_pct']:.2f}%  "
                      f"profit_factor={m['profit_factor']}")

    print("\n" + "=" * 70)
    print("PROBABILITY CALIBRATION (all out-of-sample days, by prob bucket)")
    print("=" * 70)
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        bucket_rows = [e for e in all_evaluated if lo <= e["prob"] < hi]
        if not bucket_rows:
            print(f"  prob [{lo:.1f}-{hi:.1f}): n=0")
            continue
        wins = sum(1 for e in bucket_rows if e["outcome"] == "WIN")
        avg_ret = sum(e["actual_pct"] for e in bucket_rows) / len(bucket_rows)
        print(f"  prob [{lo:.1f}-{hi:.1f}): n={len(bucket_rows):5d}  "
              f"win_rate={wins / len(bucket_rows) * 100:5.1f}%  "
              f"avg_fwd_return={avg_ret * 100:+.2f}%")

    print("\n" + "=" * 70)
    print("BIG MOVER ALERT RESULTS (engine.big_mover_check: ACTIVE + SETUP)")
    print("=" * 70)
    print(f"Qualifying signals: ACTIVE={mover_counts['ACTIVE']}  SETUP={mover_counts['SETUP']}  "
          f"(out of {len(all_evaluated)} evaluated days)")
    if mover_trades:
        mover_metrics = compute_metrics(mover_trades)
        print("\nCombined mover metrics:")
        for k, v in mover_metrics.items():
            print(f"  {k:24s}: {v}")
        for label in ("ACTIVE", "SETUP"):
            subset = [t for t in mover_trades if t["tier"] == label]
            if subset:
                m = compute_metrics(subset)
                print(f"\n  {label:8s}: n={m['trade_count']:4d}  win_rate={m['win_rate']*100:5.1f}%  "
                      f"avg_gain={m['avg_gain_pct']:+.2f}%  avg_loss={m['avg_loss_pct']:.2f}%  "
                      f"profit_factor={m['profit_factor']}")
    else:
        print("\nNo qualifying ACTIVE/SETUP mover signals in this sample/period.")

    print("\n" + "=" * 70)
    print("[4/4] KNOWN BIG-WINNER SPOTLIGHT — did the system flag these historically?")
    print("=" * 70)
    for ticker in KNOWN_WINNERS:
        if ticker not in data:
            print(f"  {ticker:10s}: no cached data (fetch failed or delisted symbol)")
            continue
        df = data[ticker]
        cutoff_idx = cutoffs.get(ticker, 0)
        total_ret = (df["Close"].iloc[-1] / df["Close"].iloc[cutoff_idx] - 1) * 100
        hits = winner_hits.get(ticker, [])
        print(f"\n  {ticker:10s} test-period return: {total_ret:+.1f}%  |  "
              f"signals fired: {len(hits)}")
        for h in hits[:8]:
            print(f"      {h['signal_date']}  {h['kind']:12s} prob={h['prob']:.2f}  "
                  f"fwd_return={h['actual_pct']*100:+.1f}%  ({h['outcome']})")
        if len(hits) > 8:
            print(f"      ... and {len(hits) - 8} more")
        if not hits:
            print("      (no ELITE/STRONG BUY/mover signal fired during its held-out run-up)")

    print("\nDone. This validates the model+scoring logic against real historical")
    print("outcomes across the full watchlist; it does not touch signal_log.json,")
    print("WATCHLIST, or any production state/constants.")


if __name__ == "__main__":
    main()
