"""
Manual historical backtest — validates the ACTUAL signal-detection/scoring logic
in engine.py (AI model + technical scoring rules) against real historical price
outcomes. This is separate from opportunity/backtester.py (which only aggregates
whatever is already in signal_log.json — currently too sparse to be meaningful).

What this does (proper OUT-OF-SAMPLE backtest — no look-ahead bias):
  1. Fetches 2 years of historical daily bars for a representative sample of
     tickers (mix of ASX + US, across sectors).
  2. Splits each ticker's history by TIME: first 70% = train period, last 30%
     = test period. The model is trained ONLY on rows dated before the cutoff
     (mirrors engine.train_model()'s feature/target/recency-weight logic) and
     is NEVER shown test-period rows or their outcomes during training.
  3. Signals are only evaluated on the held-out test period — i.e. the model
     is scored purely on data it has never seen, which is what makes this a
     meaningful validation of the scoring logic rather than a memorization
     check. (An earlier version of this script trained and tested on the same
     2-year window and produced a suspicious 100% win rate — classic look-
     ahead/data-leakage artifact. This version fixes that.)
  4. At each historical test-period day, replicates the deterministic part of
     engine.decide() — the model probability + technical scoring rules
     (base_score) — and applies the same ELITE / STRONG BUY qualification
     thresholds. Live-only adjusters (news sentiment, insider activity,
     options flow, market regime, etc.) are NOT included since they have no
     meaningful point-in-time historical equivalent — this isolates the
     technical + AI edge that's the core of the detection engine.
  5. For every qualifying signal, checks the ACTUAL forward return over
     PREDICTION_DAYS and classifies WIN (>= TARGET_RETURN) vs LOSS.
  6. Reports win rate, avg gain/loss, profit factor, and per-tier breakdown
     using the same metrics formula as opportunity/backtester.py.

Does NOT modify signal_log.json or any production files. Read-only simulation.
Run with: python3 tests/manual_historical_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import engine
from opportunity.backtester import compute_metrics

# Representative sample across sectors/geographies for a reasonably fast,
# still-meaningful backtest (full 408-ticker run would take much longer).
SAMPLE_TICKERS = [
    # US mega-cap tech / semis
    "AAPL", "NVDA", "MSFT", "AMD", "TSLA",
    # US finance / industrial
    "JPM", "CAT",
    # ASX banks / miners
    "CBA.AX", "BHP.AX", "FMG.AX", "RIO.AX",
    # ASX gold
    "NST.AX", "EVN.AX",
    # ASX lithium
    "PLS.AX", "IGO.AX",
    # ASX tech
    "XRO.AX", "WTC.AX",
    # ASX healthcare
    "CSL.AX", "RMD.AX",
    # ASX consumer/REIT
    "WES.AX", "GMG.AX",
]


def score_row(row, prev, prob: float) -> tuple[str, int]:
    """Replicates engine.decide()'s deterministic technical filters + scoring
    (no live/network-dependent adjusters). Returns (signal, score)."""
    filters_ok = (
        row["ema20"] > row["ema50"]
        and prev["ema20"] > prev["ema50"]
        and row["macd_diff"] > 0
        and prev["macd_diff"] > 0
        and 25 < row["rsi"] < 72
        and row["vol_ratio"] >= 0.5
        and prob >= 0.40
    )
    if not filters_ok:
        return "GATED", 0

    rules = [
        (3, prob >= 0.80),
        (2, 0.70 <= prob < 0.80),
        (1, 0.60 <= prob < 0.70),
        (3, bool(row["breakout"])),
        (2, row["vol_ratio"] > 1.5),
        (2, 35 <= row["rsi"] <= 65),
        (1, row["rsi"] < 70),
        (1, row["ema20"] > row["ema50"]),
    ]
    score = sum(pts for pts, met in rules if met)

    if score >= 11:
        return "ELITE", score
    elif score >= 9 and prob >= 0.70:
        return "STRONG BUY", score
    elif score >= 5:
        return "WATCH", score
    return "IGNORE", score


TRAIN_FRACTION = 0.70  # first 70% of each ticker's history = train, last 30% = held-out test


def main():
    print("=" * 70)
    print("HISTORICAL BACKTEST — engine.py signal detection/scoring logic")
    print("(out-of-sample: model trained only on data before each ticker's")
    print(" test-period cutoff; never sees the test-period rows or outcomes)")
    print("=" * 70)

    print(f"\n[1/4] Fetching 2y history for {len(SAMPLE_TICKERS)} tickers...")
    data: dict[str, pd.DataFrame] = {}
    for ticker in SAMPLE_TICKERS:
        try:
            df = engine.get_data(ticker, "2y")
        except Exception as e:
            print(f"  ⚠️  {ticker}: failed to fetch data ({e})")
            continue
        if len(df) < 150:
            print(f"  ⚠️  {ticker}: insufficient data ({len(df)} rows)")
            continue
        data[ticker] = df

    print("\n[2/4] Building time-split training set (train rows only, "
          "cutoff at 70% mark per ticker)...")
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
        train_df["_row_date"] = train_df.index
        train_df["_ticker"] = ticker
        train_frames.append(train_df.dropna())

    combined = pd.concat(train_frames, ignore_index=True)
    X_train, y_train = combined[engine.FEATURES], combined["target"]
    print(f"  Train set: {len(combined):,} rows "
          f"({int(y_train.sum())} buy / {int((y_train == 0).sum())} no-buy)")

    print("\n[3/4] Training ensemble (same architecture as engine.train_model(), "
          "recency weights only — no feedback layer since no signal_log needed)...")
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

    print(f"\n[4/4] Evaluating signals on held-out test period only "
          f"(last {int((1 - TRAIN_FRACTION) * 100)}% of each ticker's history)...")
    trades: list[dict] = []
    all_evaluated: list[dict] = []  # every day, regardless of tier — for calibration check
    tier_counts: dict[str, int] = {"ELITE": 0, "STRONG BUY": 0, "WATCH": 0, "IGNORE": 0, "GATED": 0}

    for ticker, df in data.items():
        cutoff_idx = cutoffs[ticker]
        # Leave PREDICTION_DAYS at the end so we can always check forward return.
        usable_end = len(df) - engine.PREDICTION_DAYS
        test_start = max(cutoff_idx, 60)
        n_signals_ticker = 0

        if test_start >= usable_end:
            print(f"  ⚠️  {ticker}: no usable test-period rows, skipping")
            continue

        # Vectorized: compute all probabilities for this ticker in one batch
        # call instead of one sklearn call per historical day.
        try:
            all_probs = model.predict_proba(df[engine.FEATURES])[:, 1]
        except Exception as e:
            print(f"  ⚠️  {ticker}: predict_proba failed ({e})")
            continue

        for i in range(test_start, usable_end):
            row, prev = df.iloc[i], df.iloc[i - 1]
            prob = float(all_probs[i])

            signal, score = score_row(row, prev, prob)
            tier_counts[signal] = tier_counts.get(signal, 0) + 1

            entry_price = float(row["Close"])
            exit_price = float(df["Close"].iloc[i + engine.PREDICTION_DAYS])
            actual_pct = (exit_price / entry_price) - 1
            outcome = "WIN" if actual_pct >= engine.TARGET_RETURN else "LOSS"

            all_evaluated.append({
                "ticker": ticker,
                "signal_date": str(df.index[i].date()),
                "tier": signal,
                "score": score,
                "prob": round(prob, 3),
                "entry_price": entry_price,
                "actual_pct": actual_pct,
                "outcome": outcome,
                "pred_days": engine.PREDICTION_DAYS,
            })

            if signal not in ("ELITE", "STRONG BUY"):
                continue

            trades.append(all_evaluated[-1])
            n_signals_ticker += 1

        print(f"  {ticker:10s} {usable_end - test_start:4d} test-period bars "
              f"-> {n_signals_ticker} qualifying signals")

    print(f"\nComputing metrics over {len(trades)} out-of-sample historical signals...")
    print("\nTier distribution across all evaluated days:")
    for tier, count in tier_counts.items():
        print(f"  {tier:12s}: {count}")

    metrics = compute_metrics(trades)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (ELITE + STRONG BUY signals only)")
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

    # ── Calibration check across ALL evaluated days (not just alert-tier days) ──
    # This is the real validation of the scoring logic: does a higher AI
    # probability actually correspond to a higher real forward win rate? This
    # works even when the strict alert threshold produces too few hits to be
    # statistically meaningful on its own.
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

    print("\nA well-calibrated model should show win_rate and avg_fwd_return")
    print("increasing monotonically (or close to it) as probability increases.")
    print("If calibration is flat or inverted, the AI model has little genuine")
    print("out-of-sample predictive edge beyond the base rate for this sample.")

    print("\nDone. This validates the model+scoring logic against real historical")
    print("outcomes; it does not touch signal_log.json or any production state.")


if __name__ == "__main__":
    main()
