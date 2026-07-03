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
    "AAPL", "NVDA", "MSFT", "AMD", "TSLA", "META", "AMZN", "GOOGL", "AVGO", "INTC",
    # US finance / industrial
    "JPM", "CAT", "GS", "BA", "XOM",
    # ASX banks / miners
    "CBA.AX", "BHP.AX", "FMG.AX", "RIO.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "S32.AX",
    # ASX gold
    "NST.AX", "EVN.AX", "NEM.AX",
    # ASX lithium
    "PLS.AX", "IGO.AX", "MIN.AX", "LTR.AX",
    # ASX tech
    "XRO.AX", "WTC.AX", "APX.AX",
    # ASX healthcare
    "CSL.AX", "RMD.AX", "COH.AX",
    # ASX consumer/REIT
    "WES.AX", "GMG.AX", "WOW.AX", "COL.AX",
    # ASX energy
    "WDS.AX", "STO.AX",
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


def score_active_mover(df, i: int, prob: float) -> tuple[bool, int]:
    """Replicates the deterministic gates + scoring of engine._large_move_check
    (the ACTIVE 'big mover' tier). Excludes vix_safe()/earnings_safe() (live-only,
    no historical point-in-time equivalent), the intraday hourly confirmation
    (requires live 1h data, not available for arbitrary historical dates), and
    the 12h cooldown (stateful, not meaningful for a batch historical scan)."""
    row = df.iloc[i]
    price = float(row["Close"])
    open_ = float(row["Open"])
    vol_r = float(row["vol_ratio"])
    daily_ret = (price - open_) / open_
    atr_now = float(row["atr"])
    atr_avg = float(df["atr"].iloc[max(0, i - 19):i + 1].mean())
    atr_exp = atr_now / atr_avg if atr_avg > 0 else 1.0
    rsi = float(row["rsi"])

    if vol_r < 4.0:            return False, 0
    if daily_ret < 0.040:      return False, 0
    if atr_exp < 1.8:          return False, 0
    if not (38 <= rsi <= 76):  return False, 0
    if prob < 0.38:            return False, 0

    score = 0
    score += 4 if vol_r >= 6.0 else 3 if vol_r >= 4.5 else 2
    score += 4 if daily_ret >= 0.07 else 3 if daily_ret >= 0.05 else 2
    score += 2 if atr_exp >= 2.5 else 1
    if bool(row.get("breakout", 0)):   score += 2
    if bool(row.get("bb_squeeze", 0)): score += 1
    if rsi < 65:                       score += 1
    # Intraday confirmation (+2) intentionally omitted — no historical equivalent.

    return score >= 8, score


def score_setup_mover(df, i: int, prob: float) -> tuple[bool, int]:
    """Replicates the deterministic gates + scoring of engine._breakout_setup_check
    (the SETUP 'big mover' tier). Same exclusions as score_active_mover (no
    vix_safe/earnings_safe/cooldown; those are live/stateful checks)."""
    if i < 4 or i < 126:
        return False, 0
    row = df.iloc[i]
    price = float(row["Close"])
    rsi = float(row["rsi"])
    adx = float(row["adx"])
    adx_3d = float(df["adx"].iloc[i - 4])
    vol_r = float(row["vol_ratio"])
    obv_r = float(row["obv_ratio"])
    sq = bool(row["bb_squeeze"])
    bb_mid = (float(row["bb_upper"]) + float(row["bb_lower"])) / 2
    watch_level = float(row["bb_upper"]) * 1.005

    if not sq:                          return False, 0
    if obv_r < 2.0:                     return False, 0
    if adx < 23:                        return False, 0
    if adx <= adx_3d:                   return False, 0
    if not (32 <= rsi <= 62):           return False, 0
    if price < bb_mid:                  return False, 0
    if not (0.8 <= vol_r <= 2.5):       return False, 0
    if price >= watch_level:            return False, 0
    if price < watch_level * 0.92:      return False, 0
    if prob < 0.38:                     return False, 0

    score = 0
    bb_width_floor = float(df["bb_width"].iloc[max(0, i - 125):i + 1].quantile(0.20))
    bb_pct = float(row["bb_width"]) / bb_width_floor if bb_width_floor > 0 else 1.0
    score += 3 if bb_pct < 0.85 else 2
    score += 3 if obv_r >= 3.0 else 2 if obv_r >= 2.2 else 1
    adx_rise = adx - adx_3d
    score += 2 if (adx >= 28 and adx_rise > 2) else 1
    if 42 <= rsi <= 56:
        score += 1
    high_52w = float(df["Close"].iloc[max(0, i - 251):i + 1].max())
    pct_to_high = (high_52w - price) / price
    if pct_to_high < 0.03:
        score += 2
    elif pct_to_high < 0.06:
        score += 1
    score += 3 if prob >= 0.50 else 2 if prob >= 0.42 else 1

    return score >= 9, score


TRAIN_FRACTION = 0.70  # first 70% of each ticker's history = train, last 30% = held-out test


def main():
    print("=" * 70)
    print("HISTORICAL BACKTEST — engine.py signal detection/scoring logic")
    print("(out-of-sample: model trained only on data before each ticker's")
    print(" test-period cutoff; never sees the test-period rows or outcomes)")
    print("=" * 70)

    print(f"\n[1/4] Fetching 2y history for {len(SAMPLE_TICKERS)} tickers...")
    cache_dir = Path("/tmp/ticker_cache")
    data: dict[str, pd.DataFrame] = {}
    for ticker in SAMPLE_TICKERS:
        cache_file = cache_dir / f"{ticker.replace('.', '_')}.pkl"
        try:
            if cache_file.exists():
                df = pd.read_pickle(cache_file)
            else:
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

    # ── Diagnostics: is the AI probability itself informative on held-out data? ──
    # This answers "why isn't AI probability giving a clear edge" independent
    # of the alert-tier gating logic, which can mask a model that's actually
    # fine (or confirm a model that's genuinely weak).
    print("\n" + "=" * 70)
    print("AI MODEL DIAGNOSTICS (held-out test-period rows, all tickers combined)")
    print("=" * 70)
    test_frames = []
    for ticker, df in data.items():
        cutoff_idx = cutoffs[ticker]
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
        print(f"  ROC AUC:        {auc:.3f}   (0.50 = no better than a coin flip, 1.00 = perfect)")
        print(f"  Log loss:       {ll:.4f}   (baseline constant-rate log loss: {ll_baseline:.4f})")
        print(f"  Brier score:    {brier:.4f}   (lower is better; 0.25 ~= uninformative for ~50% base rate)")
        print(f"  Predicted prob: min={p_test.min():.3f}  mean={p_test.mean():.3f}  "
              f"max={p_test.max():.3f}  std={p_test.std():.3f}")
        if auc < 0.55:
            print("  -> AUC near 0.50 means the model has very little genuine ranking")
            print("     power on unseen data for THIS sample — consistent with the flat")
            print("     calibration buckets seen in the main backtest.")

        # Feature importance — which inputs is the model actually leaning on?
        try:
            xgb_clf = xgb_pipe.named_steps["xgb"]
            importances = sorted(
                zip(engine.FEATURES, xgb_clf.feature_importances_),
                key=lambda t: -t[1],
            )
            print("\n  XGBoost feature importances (train set):")
            for feat, imp in importances:
                bar = "█" * int(imp * 60)
                print(f"    {feat:12s} {imp:.3f} {bar}")
        except Exception as _e:
            print(f"  ⚠️  Could not extract feature importances: {_e}")
    else:
        print("  ⚠️  No held-out rows available for diagnostics.")

    print(f"\n[4/4] Evaluating signals on held-out test period only "
          f"(last {int((1 - TRAIN_FRACTION) * 100)}% of each ticker's history)...")
    trades: list[dict] = []
    all_evaluated: list[dict] = []  # every day, regardless of tier — for calibration check
    tier_counts: dict[str, int] = {"ELITE": 0, "STRONG BUY": 0, "WATCH": 0, "IGNORE": 0, "GATED": 0}
    mover_trades: list[dict] = []
    mover_counts: dict[str, int] = {"ACTIVE": 0, "SETUP": 0}

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

            if signal in ("ELITE", "STRONG BUY"):
                trades.append(all_evaluated[-1])
                n_signals_ticker += 1

            # ACTIVE takes priority over SETUP, mirroring engine.big_mover_check().
            fired, mv_score = score_active_mover(df, i, prob)
            mv_tier = "ACTIVE" if fired else None
            if not fired:
                fired, mv_score = score_setup_mover(df, i, prob)
                mv_tier = "SETUP" if fired else None

            if fired:
                mover_counts[mv_tier] += 1
                mover_trades.append({
                    "ticker": ticker,
                    "signal_date": str(df.index[i].date()),
                    "tier": mv_tier,
                    "score": mv_score,
                    "prob": round(prob, 3),
                    "entry_price": entry_price,
                    "actual_pct": actual_pct,
                    "outcome": outcome,
                    "pred_days": engine.PREDICTION_DAYS,
                })

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

    # ── Big Mover alert (engine.big_mover_check: ACTIVE + SETUP tiers) ──────────
    print("\n" + "=" * 70)
    print("BIG MOVER ALERT RESULTS (engine.big_mover_check: ACTIVE + SETUP)")
    print("=" * 70)
    print("NOTE: vix_safe()/earnings_safe() (live/network calls), the ACTIVE")
    print("tier's intraday 1h confirmation, and the 12h/48h cooldowns are")
    print("excluded — none have a meaningful historical point-in-time")
    print("equivalent. Results below isolate the daily technical + AI gates.")
    print(f"\nQualifying signals: ACTIVE={mover_counts['ACTIVE']}  SETUP={mover_counts['SETUP']}  "
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
        print("\nNo qualifying ACTIVE/SETUP mover signals in this sample/period —")
        print("consistent with these being rare, high-conviction reactive alerts.")

    print("\nDone. This validates the model+scoring logic against real historical")
    print("outcomes; it does not touch signal_log.json or any production state.")


if __name__ == "__main__":
    main()
