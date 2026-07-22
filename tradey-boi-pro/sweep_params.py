"""
Parameter sweep for Tradey Boi Pro backtest.

Strategy: Pre-compute ALL indicator values + ML predictions in ONE pass
(once per ticker, not once per ticker×day). The sweep then runs simulation
loops using cached signals — no indicator recomputation, no ML calls inside
the loop. Fast enough to test 50+ combos in under 2 minutes.

Usage:  python sweep_params.py
Output: ranked table + sweep_winner.json
"""
import sys, os, json, math, time, logging, contextlib, io, warnings as _w
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)

from datetime import date, timedelta
import pandas as pd
import yfinance as yf

from backtest.engine import run_backtest
from scanner.market_scanner import (
    _compute_x_features,
    _load_x_model,
    _expected_value_r,
    _normalize_columns as _norm_cols,
    FEATURES,
)

# ── Test window & tickers ─────────────────────────────────────────────────────
TEST_START     = date(2024, 1, 2)
TEST_END       = date(2024, 6, 28)
INITIAL_CAP    = 10_000.0
DOWNLOAD_EXTRA = 120   # warm-up days for indicators

TICKERS = [
    # ASX
    "BHP.AX","RIO.AX","CBA.AX","ANZ.AX","WBC.AX","NAB.AX","MQG.AX",
    "CSL.AX","WES.AX","WOW.AX","FMG.AX","TLS.AX","RMD.AX","ALL.AX",
    "GMG.AX","REA.AX","COL.AX","STO.AX","AGL.AX","ORG.AX","QBE.AX",
    "SHL.AX","ALX.AX","ASX.AX","CPU.AX","SOL.AX","MIN.AX","TWE.AX",
    "NXT.AX","JBH.AX","EVN.AX","NST.AX","IGO.AX",
    # US
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","JPM","V","UNH",
    "XOM","JNJ","PG","HD","MA","BAC","CVX","ABBV","MRK","PEP",
    "KO","AVGO","COST","TMO","ACN","MCD","ABT","CRM","LLY","NFLX",
]

# ── Parameter grid ────────────────────────────────────────────────────────────
MIN_SCORES = [6, 7, 8, 9]
HOLD_DAYS  = [10, 15, 20, 25]

TARGET_PROFILES = {
    "tight":    {"target_hi": 8.0,  "target_mid": 5.0,  "target_lo": 3.0},
    "standard": {"target_hi": 12.0, "target_mid": 8.0,  "target_lo": 5.0},
    "wide":     {"target_hi": 18.0, "target_mid": 12.0, "target_lo": 8.0},
}

SL_PROFILES = {
    "tight":    {"sl_mult_hi": 0.8, "sl_mult_mid": 0.6, "sl_mult_lo": 0.5},
    "standard": {"sl_mult_hi": 1.2, "sl_mult_mid": 1.0, "sl_mult_lo": 0.8},
    "loose":    {"sl_mult_hi": 1.8, "sl_mult_mid": 1.4, "sl_mult_lo": 1.1},
}

BASE_PARAMS = {
    "min_prob":              0.50,
    "max_positions":         5,
    "risk_pct":              2.0,
    "brokerage":             2.0,
    "cb_consecutive_losses": 3,
    "cb_pause_days":         7,
    "use_regime_filter":     True,
    "backtest_mode":         True,
}

def build_grid():
    combos = []
    for ms in MIN_SCORES:
        for hd in HOLD_DAYS:
            for tp_name, tp in TARGET_PROFILES.items():
                for sl_name, sl in SL_PROFILES.items():
                    label = f"score={ms}  hold={hd:2d}d  tgt={tp_name:<8}  sl={sl_name}"
                    p = dict(BASE_PARAMS, min_score=ms, hold_days=hd, **tp, **sl)
                    combos.append((label, p))
    return combos

# ── Data download ─────────────────────────────────────────────────────────────

def download_all():
    dl_start = TEST_START - timedelta(days=DOWNLOAD_EXTRA)
    dl_end   = TEST_END   + timedelta(days=2)
    print(f"\nStep 1/3 — Downloading {len(TICKERS)} tickers  ({dl_start} → {dl_end}) …", flush=True)
    t0 = time.time()

    all_data = {}
    for i, batch in enumerate([TICKERS[j:j+20] for j in range(0, len(TICKERS), 20)]):
        try:
            sink = io.StringIO()
            with contextlib.redirect_stderr(sink), _w.catch_warnings():
                _w.simplefilter("ignore")
                raw = yf.download(
                    " ".join(batch),
                    start=dl_start.isoformat(), end=dl_end.isoformat(),
                    interval="1d", auto_adjust=True,
                    progress=False, group_by="ticker", threads=False,
                )
            if len(batch) == 1:
                df = _norm_cols(raw.dropna(how="all"))
                if not df.empty:
                    all_data[batch[0]] = df
            else:
                for t in batch:
                    try:
                        df = _norm_cols(raw[t].dropna(how="all"))
                        if not df.empty and len(df) >= 30:
                            all_data[t] = df
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"  batch {i+1} error: {e}")

    print(f"  Got data for {len(all_data)}/{len(TICKERS)} tickers  ({time.time()-t0:.1f}s)")

    # Regime indices
    print("  Downloading regime indices …", flush=True)
    asx_regime, us_regime = {}, {}
    for sym, rdict in [("^AXJO", asx_regime), ("^GSPC", us_regime)]:
        try:
            sink = io.StringIO()
            with contextlib.redirect_stderr(sink), _w.catch_warnings():
                _w.simplefilter("ignore")
                raw = yf.download(sym, start=dl_start.isoformat(), end=dl_end.isoformat(),
                                  interval="1d", auto_adjust=True, progress=False, threads=False)
            idx_df = _norm_cols(raw.dropna(how="all"))
            if not idx_df.empty:
                idx_df["ma50"] = idx_df["Close"].rolling(50).mean()
                for d_ts, row in idx_df.iterrows():
                    dk = d_ts.date() if hasattr(d_ts, "date") else d_ts
                    ma = row.get("ma50", float("nan"))
                    if not math.isnan(float(ma)):
                        rdict[dk] = float(row["Close"]) > float(ma)
            print(f"    {sym}: {len(rdict)} regime days")
        except Exception as e:
            print(f"    {sym} failed: {e}")

    return all_data, (asx_regime, us_regime)


# ── Pre-compute signal cache ──────────────────────────────────────────────────

def _weekly_trend_ok_vectorized(df: pd.DataFrame) -> dict:
    """
    Compute weekly EMA20 > EMA50 for EVERY date in df in ONE resample pass.
    Returns {date: bool}. Much faster than calling _weekly_trend_ok per slice.
    """
    try:
        idx   = pd.to_datetime(df.index)
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        close = pd.Series(df["Close"].squeeze().values, index=idx)
        weekly = close.resample("W").last().dropna()
        if len(weekly) < 50:
            return {}   # not enough history — treat as "always OK" (fail open)
        w_ema20 = weekly.ewm(span=20, adjust=False).mean()
        w_ema50 = weekly.ewm(span=50, adjust=False).mean()
        trend_ok = (w_ema20 > w_ema50)   # bool Series indexed by week-end dates
        # Forward-fill to daily: each day inherits the most recent week's status
        result = {}
        for d_ts, df_row in zip(idx, close):
            d = d_ts.date()
            # Find the most recent Sunday ≤ d
            week_ts = trend_ok.index[trend_ok.index <= d_ts]
            if len(week_ts) == 0:
                result[d] = True   # no weekly data yet — fail open
            else:
                result[d] = bool(trend_ok.loc[week_ts[-1]])
        return result
    except Exception:
        return {}


def precompute_signal_cache(all_data: dict) -> dict:
    """
    One-pass precomputation for all tickers:
      - _compute_x_features() ONCE per ticker (not per day)
      - Batch ML predict_proba across all rows of each ticker
      - Hard filters applied row by row (pure Python, no yfinance calls)
      - Weekly trend computed with one resample per ticker

    Returns {(ticker, date): raw_signal_dict}
    """
    print(f"Step 2/3 — Pre-computing signals for {len(all_data)} tickers …", flush=True)
    t0 = time.time()

    model   = _load_x_model()
    cache   = {}
    ok_cnt  = 0

    for ticker, df in all_data.items():
        try:
            feat_df = _compute_x_features(df)
            if feat_df is None or len(feat_df) < 62:
                continue

            # ── Batch ML prediction across entire ticker history ──────────────
            if model is not None:
                try:
                    feat_matrix = feat_df[FEATURES].fillna(0)
                    all_probs   = model.predict_proba(feat_matrix)[:, 1]
                    prob_arr    = pd.Series(all_probs, index=feat_df.index)
                except Exception:
                    prob_arr = None
            else:
                prob_arr = None

            # ── Weekly trend per date (vectorized, one resample) ─────────────
            weekly_ok = _weekly_trend_ok_vectorized(df)

            exchange = "ASX" if ticker.endswith(".AX") else "SMART"

            # ── Row-by-row hard filters for test period ───────────────────────
            for i in range(1, len(feat_df)):
                ts  = feat_df.index[i]
                dt  = ts.date() if hasattr(ts, "date") else ts
                if dt < TEST_START or dt > TEST_END:
                    continue

                row  = feat_df.iloc[i]
                prev = feat_df.iloc[i-1]

                # NaN guard on critical columns
                skip = False
                for col in ("ema20", "ema50", "rsi", "macd_diff", "vol_ratio", "atr"):
                    if pd.isna(row.get(col)):
                        skip = True; break
                if skip:
                    continue

                # Hard filter: EMA trend (today and previous day)
                if float(row["ema20"])  <= float(row["ema50"]):  continue
                if float(prev["ema20"]) <= float(prev["ema50"]): continue
                # Hard filter: MACD positive both days
                if not pd.isna(row.get("macd_diff"))  and float(row["macd_diff"])  <= 0: continue
                if not pd.isna(prev.get("macd_diff")) and float(prev["macd_diff"]) <= 0: continue
                # RSI range
                rsi = float(row["rsi"])
                if rsi >= 72 or rsi <= 25: continue
                # Volume ratio
                vr = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 0.0
                if vr < 1.2: continue
                # Momentum: price and EMA20 must be rising
                if float(row["Close"])  <= float(prev["Close"]):  continue
                if float(row["ema20"])  <= float(prev["ema20"]):  continue
                # Weekly trend
                if weekly_ok and not weekly_ok.get(dt, True):
                    continue

                # Probability
                if prob_arr is not None:
                    prob = float(prob_arr.get(ts, 0.0))
                else:
                    rsi_c = max(0.0, (rsi - 40) / 120)
                    vr_c  = min((max(vr - 0.5, 0)) / 20, 0.15)
                    prob  = min(0.52 + rsi_c + vr_c, 0.82)
                    prob  = max(prob, 0.40)

                if prob < 0.40: continue

                # Scoring (mirrors _score_signal exactly)
                is_breakout = bool(int(row.get("breakout", 0)))
                score = 0
                if   prob >= 0.80: score += 3
                elif prob >= 0.70: score += 2
                elif prob >= 0.60: score += 1
                if is_breakout:    score += 3
                if vr > 1.5:       score += 2
                if 35 <= rsi <= 65:  score += 2
                elif rsi < 70:       score += 1
                if float(row["ema20"]) > float(row["ema50"]): score += 1

                # Signals with score < 5 would never pass any min_score threshold
                if score < 5:
                    continue

                # ATR
                curr_price = float(row["Close"])
                atr     = float(row["atr"]) if not pd.isna(row["atr"]) else curr_price * 0.015
                atr_pct = atr / curr_price * 100 if curr_price > 0 else 0.0

                # Expected value (sweep will filter expected_r <= 0)
                expected_r = _expected_value_r(curr_price, atr, prob, is_breakout)

                cache[(ticker, dt)] = {
                    "score":       score,
                    "prob":        round(prob,      3),
                    "atr_pct":     round(atr_pct,   2),
                    "atr":         round(atr,        4),
                    "entry_price": round(curr_price, 4),
                    "rsi":         round(rsi,        1),
                    "vol_ratio":   round(vr,         1),
                    "breakout":    is_breakout,
                    "expected_r":  round(expected_r, 3),
                    "exchange":    exchange,
                }
                ok_cnt += 1

        except Exception as e:
            print(f"  Pre-compute error {ticker}: {e}")

    elapsed = time.time() - t0
    print(f"  {ok_cnt} signal candidates across {len(all_data)} tickers  ({elapsed:.1f}s)")
    return cache


# ── Score/rank combos ─────────────────────────────────────────────────────────

def composite_score(m):
    pf  = m.get("profit_factor", 0)
    wr  = m.get("win_rate", 0)
    dd  = m.get("max_drawdown", 1)
    n   = m.get("trade_count", 0)
    roi = m.get("roi_pct", 0)
    if n < 3:
        return -999   # too few trades — statistically meaningless
    sign = 1 if roi >= 0 else -1
    return sign * pf * wr * (1 - dd) * math.log(max(n, 1))


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    all_data, regimes = download_all()
    if not all_data:
        print("No data — aborting."); sys.exit(1)

    sig_cache = precompute_signal_cache(all_data)
    if not sig_cache:
        print("No signals pre-computed — aborting."); sys.exit(1)

    combos = build_grid()
    print(f"\nStep 3/3 — Running {len(combos)} parameter combinations …\n", flush=True)

    LW = 46   # label column width
    hdr = f"{'Config':<{LW}}  {'PF':>5}  {'WR%':>5}  {'#':>4}  {'ROI%':>6}  {'DD%':>5}  {'Score':>7}"
    print(hdr)
    print("─" * len(hdr))

    rows = []
    t_sweep = time.time()

    for label, p in combos:
        try:
            res = run_backtest(
                tickers              = TICKERS,
                test_start           = TEST_START,
                test_end             = TEST_END,
                initial_capital      = INITIAL_CAP,
                params               = p,
                preloaded_data       = all_data,
                preloaded_regimes    = regimes,
                precomputed_signals  = sig_cache,
            )
            m   = res["metrics"]
            pf  = m.get("profit_factor", 0)
            wr  = m.get("win_rate", 0) * 100
            n   = m.get("trade_count", 0)
            roi = m.get("roi_pct", 0)
            dd  = m.get("max_drawdown", 0) * 100
            sc  = composite_score(m)
            rows.append({"label": label, "params": p, "metrics": m, "score": sc})
            print(f"{label:<{LW}}  {pf:>5.2f}  {wr:>5.1f}  {n:>4d}  {roi:>+6.1f}  {dd:>5.1f}  {sc:>7.3f}")
        except Exception as e:
            print(f"{label:<{LW}}  ERROR: {e}")

    elapsed_sweep = time.time() - t_sweep
    if not rows:
        print("No results."); sys.exit(1)

    print(f"\n  ({elapsed_sweep:.1f}s for {len(combos)} runs = {elapsed_sweep/len(combos):.2f}s/run)")

    # ── Rankings ──────────────────────────────────────────────────────────────
    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n" + "═" * len(hdr))
    print("\n  TOP 10 CONFIGURATIONS  (composite: PF × WR × (1−DD) × log(trades))\n")
    for i, r in enumerate(rows[:10], 1):
        m = r["metrics"]
        flag = "  ← WINNER" if i == 1 else ""
        print(f"  #{i:>2}  {r['label']}{flag}")
        print(f"        PF={m.get('profit_factor',0):.3f}  "
              f"WR={m.get('win_rate',0)*100:.1f}%  "
              f"Trades={m.get('trade_count',0)}  "
              f"ROI={m.get('roi_pct',0):+.1f}%  "
              f"MaxDD={m.get('max_drawdown',0)*100:.1f}%  "
              f"Sharpe={m.get('sharpe',0):.2f}  "
              f"Score={r['score']:.3f}")
        print()

    winner = rows[0]
    print(f"  ✅  WINNER:  {winner['label']}")

    out      = {"label": winner["label"], "params": winner["params"], "metrics": winner["metrics"]}
    out_path = os.path.join(os.path.dirname(__file__), "sweep_winner.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Winner saved → {out_path}")
    print("  Run: python apply_winner.py  to set these as new defaults.\n")


if __name__ == "__main__":
    main()
