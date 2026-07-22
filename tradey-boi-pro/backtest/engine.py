"""
Tradey Boi Pro — Walk-Forward Backtester

Simulates the Pro scanner day-by-day on real historical data.
No lookahead bias: for each day T, only data up to and including T is visible.
Entry: next day's open price.
Exit: stop/target checked intraday via High/Low, else close. Max hold exits at close.
Brokerage deducted both sides.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("Backtest")

# Import the real scanner so the backtest tests the actual ML strategy
try:
    from scanner.market_scanner import (
        _score_signal as _real_score,
        _load_x_model,
        _normalize_columns as _norm_cols,
        _compute_x_features,
        _expected_value_r,
        _weekly_trend_ok,
        _regime_score_thresholds,
        FEATURES,
    )
    _USE_REAL_SCANNER = True
except Exception as _e:
    log.warning(f"Could not import real scanner — falling back to rule-based: {_e}")
    _USE_REAL_SCANNER = False

    # Inline fallback so _norm_cols is always available even when scanner import fails
    def _norm_cols(df):  # type: ignore[override]
        import pandas as _pd
        df = df.copy()
        if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
            _ohlcv = {"open","high","low","close","volume","Open","High","Low","Close","Volume"}
            lvl0 = [str(c) for c in df.columns.get_level_values(0)]
            lvl1 = [str(c) for c in df.columns.get_level_values(1)]
            use1 = sum(1 for c in lvl1 if c in _ohlcv) > sum(1 for c in lvl0 if c in _ohlcv)
            df.columns = df.columns.get_level_values(1 if use1 else 0)
        df.columns = [str(c).title() for c in df.columns]
        return df


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BtPosition:
    ticker:         str
    entry_date:     date
    entry_price:    float
    stop_price:     float
    target_price:   float
    quantity:       float
    max_hold:       int
    score:          float = 0
    prob:           float = 0
    orig_stop_dist: float = 0.0   # initial risk distance: entry - stop (fixed at open)
    peak_close:     float = 0.0   # rolling highest close since entry (for trailing)

@dataclass
class BtTrade:
    ticker:      str
    entry_date:  date
    exit_date:   date
    entry_price: float
    exit_price:  float
    quantity:    float
    stop_price:  float
    target_price: float
    exit_reason: str
    score:       float = 0
    prob:        float = 0

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def hold_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def outcome(self) -> str:
        return "WIN" if self.pnl >= 0 else "LOSS"


# ── Scanner signal detection (stateless, uses a data slice) ───────────────────

def _detect_signal(df_slice: pd.DataFrame, ticker: str, params: dict) -> dict | None:
    """
    Evaluate one ticker using the real X EnsembleModel (same as live bot).
    Falls back to simple rule-based scoring if the scanner module is unavailable.
    No lookahead: df_slice contains only data up to and including the signal day.
    """
    if df_slice is None or len(df_slice) < 30:
        return None

    # ── Use the real scanner — same logic as the live bot ────────────────────
    if _USE_REAL_SCANNER:
        # Inject backtest_mode so the scorer uses the slider threshold instead
        # of the adaptive file, and skips per-ticker learning (no lookahead bias)
        bt_params = dict(params, backtest_mode=True)
        try:
            return _real_score(df_slice, ticker, bt_params)
        except Exception as e:
            log.debug(f"Real scorer error for {ticker}: {e}")
            return None

    # ── Scanner unavailable (import failed at startup) ────────────────────────
    # Falls back to a simple rule check. Results will not match live bot exactly.
    log.debug(f"Real scanner unavailable — skipping {ticker}")
    return None


# ── Fast pre-scan: compute features ONCE per ticker, read rows for each date ──

def _prescan_all(
    all_data:    dict,
    available:   list,
    date_idx:    dict,
    ts_start:    "pd.Timestamp",
    ts_end:      "pd.Timestamp",
    p:           dict,
    progress_cb: "Callable | None",
    total_units: int,
) -> dict:
    """
    Pre-compute signals for all tickers in one vectorised pass.

    Key correctness property: all technical indicators (EMA, RSI, MACD, ATR, …)
    are purely backward-looking — they depend only on data up to and including
    the current row.  Therefore, computing them on the FULL DataFrame and reading
    row i gives the exact same result as computing on df.iloc[:i+1] and reading
    the last row.  This lets us call _compute_x_features() ONCE per ticker
    instead of once per (ticker, day), giving a ~250× speedup.

    Returns: {(ticker, date): signal_dict}  — ready for the fast-path simulation.
    """
    if not _USE_REAL_SCANNER:
        return {}

    model      = _load_x_model()
    sb_base    = int(p.get("min_score",  5))
    prob_floor = float(p.get("min_prob", 0.50))
    elite_min, sb_min = _regime_score_thresholds(sb_base)

    signals: dict = {}

    for t_idx, ticker in enumerate(available):
        if progress_cb and t_idx % 10 == 0:
            pct = int(t_idx / max(len(available), 1) * total_units)
            progress_cb(pct, total_units,
                        f"Pre-scanning {t_idx+1}/{len(available)}: {ticker}…")
        try:
            df = all_data[ticker]
            feat_df = _compute_x_features(df)   # ← ONE call per ticker
            if feat_df is None or len(feat_df) < 2:
                continue

            for i in range(1, len(df)):
                ts = df.index[i]
                if ts < ts_start or ts > ts_end:
                    continue

                row  = feat_df.iloc[i]
                prev = feat_df.iloc[i - 1]

                # ── Cheap pre-filters (avoid model call on majority of rows) ──
                ema20 = row.get("ema20", float("nan"))
                ema50 = row.get("ema50", float("nan"))
                if pd.isna(ema20) or pd.isna(ema50):
                    continue
                if float(ema20) <= float(ema50):
                    continue
                prev_e20 = prev.get("ema20", float("nan"))
                prev_e50 = prev.get("ema50", float("nan"))
                if pd.isna(prev_e20) or pd.isna(prev_e50):
                    continue
                if float(prev_e20) <= float(prev_e50):
                    continue
                macd = row.get("macd_diff", float("nan"))
                if not pd.isna(macd) and float(macd) <= 0:
                    continue
                prev_macd = prev.get("macd_diff", float("nan"))
                if not pd.isna(prev_macd) and float(prev_macd) <= 0:
                    continue
                rsi_v = row.get("rsi", float("nan"))
                if pd.isna(rsi_v):
                    continue
                rsi = float(rsi_v)
                if rsi >= 72 or rsi <= 25:
                    continue
                vr_v = row.get("vol_ratio", float("nan"))
                vr   = float(vr_v) if not pd.isna(vr_v) else 0
                if vr < 1.2:
                    continue
                # Price must be rising
                close_now  = row.get("Close",  float("nan"))
                close_prev = prev.get("Close", float("nan"))
                if pd.isna(close_now) or pd.isna(close_prev):
                    continue
                if float(close_now) <= float(close_prev):
                    continue
                # EMA20 must be rising
                if float(ema20) <= float(prev_e20):
                    continue

                # ── ML probability ────────────────────────────────────────────
                prob = None
                if model is not None:
                    try:
                        feat_row = pd.DataFrame([{f: row.get(f, 0) for f in FEATURES}])
                        prob = float(model.predict_proba(feat_row)[0][1])
                    except Exception:
                        prob = None
                if prob is None:
                    prob = min(
                        0.52 + max(0.0, (rsi - 40) / 120)
                             + min(max(vr - 0.5, 0) / 20, 0.15),
                        0.82,
                    )
                    prob = max(prob, 0.40)
                if prob < 0.40:
                    continue

                # ── Score ─────────────────────────────────────────────────────
                is_breakout = bool(int(row.get("breakout", 0)))
                score = 0
                if   prob >= 0.80: score += 3
                elif prob >= 0.70: score += 2
                elif prob >= 0.60: score += 1
                if is_breakout:    score += 3
                if vr > 1.5:       score += 2
                if 35 <= rsi <= 65: score += 2
                elif rsi < 70:      score += 1
                score += 1   # ema20 > ema50 always True at this point

                # ── Tier ──────────────────────────────────────────────────────
                curr_price = float(close_now)
                atr_v  = row.get("atr", float("nan"))
                atr    = float(atr_v) if not pd.isna(atr_v) else curr_price * 0.015
                atr_pct = atr / curr_price * 100 if curr_price > 0 else 0
                expected_r = _expected_value_r(curr_price, atr, prob, is_breakout)

                if   score >= elite_min and prob >= prob_floor and expected_r > 0:
                    tier = "ELITE"
                elif score >= sb_min    and prob >= prob_floor and expected_r > 0:
                    tier = "STRONG BUY"
                elif score >= 5:
                    tier = "BUY"
                else:
                    continue

                # ── Stop / target ─────────────────────────────────────────────
                if atr_pct >= 3.0:
                    sl_mult = p.get("sl_mult_hi",  0.8); tp_pct = p.get("target_hi",  12.0)
                elif atr_pct >= 1.5:
                    sl_mult = p.get("sl_mult_mid", 0.6); tp_pct = p.get("target_mid",  8.0)
                else:
                    sl_mult = p.get("sl_mult_lo",  0.5); tp_pct = p.get("target_lo",   5.0)

                stop_price   = max(curr_price - sl_mult * atr, curr_price * 0.88)
                target_price = curr_price * (1 + tp_pct / 100)

                signals[(ticker, ts.date())] = {
                    "ticker":       ticker,
                    "entry_price":  round(curr_price, 4),
                    "stop_price":   round(stop_price, 4),
                    "target_price": round(target_price, 4),
                    "atr_pct":      round(atr_pct, 2),
                    "atr":          round(atr, 4),
                    "score":        score,
                    "prob":         round(prob, 3),
                    "tier":         tier,
                    "exchange":     "ASX" if ticker.endswith(".AX") else "SMART",
                    "expected_r":   expected_r,
                }

        except Exception as _pe:
            log.debug(f"Pre-scan error for {ticker}: {_pe}")
            continue

    log.info(f"Pre-scan complete: {len(signals)} signal candidates across {len(available)} tickers")
    return signals


# ── Main backtest engine ───────────────────────────────────────────────────────

def run_backtest(
    tickers:             list[str],
    test_start:          date,
    test_end:            date,
    initial_capital:     float = 10_000.0,
    params:              dict  | None = None,
    progress_cb:         Callable[[int, int, str], None] | None = None,
    preloaded_data:      dict  | None = None,
    preloaded_regimes:   tuple[dict, dict] | None = None,
    precomputed_signals: dict  | None = None,
) -> dict:
    """
    Walk-forward backtest of the Pro scanner.

    params keys (all optional, defaults match Tradey Boi Pro settings):
        min_score, min_prob, max_positions, risk_pct, brokerage,
        hold_days, sl_mult_hi/mid/lo, target_hi/mid/lo

    Returns:
        {trades, equity_curve, metrics, params_used}
    """
    if params is None:
        params = {}

    # Shared mutable dict for rejection reason tracking.
    # _score_signal increments counters here; returned in the result dict
    # so the dashboard can show exactly which filter is blocking trades.
    reasons: dict[str, int] = {}

    p = {
        "min_score":         params.get("min_score",         6),
        "min_prob":          params.get("min_prob",          0.53),
        "max_positions":     params.get("max_positions",     5),
        "risk_pct":          params.get("risk_pct",          2.0),
        "brokerage":         params.get("brokerage",         2.0),
        "hold_days":         params.get("hold_days",         10),
        # sl_mult defaults widened: 0.5-0.8× ATR sits inside daily noise;
        # use 1.5-2.0× so the breakeven/trailing stop mechanics have room to work.
        "sl_mult_hi":        params.get("sl_mult_hi",        2.0),
        "sl_mult_mid":       params.get("sl_mult_mid",       1.5),
        "sl_mult_lo":        params.get("sl_mult_lo",        1.0),
        "target_hi":         params.get("target_hi",         15.0),
        "target_mid":        params.get("target_mid",        10.0),
        "target_lo":         params.get("target_lo",         7.0),
        # min_hold_days: stop cannot trigger during the first N days after entry.
        # Prevents entry-day noise (gap opens, spread) from immediately stopping out trades.
        "min_hold_days":     params.get("min_hold_days",     2),
        "cb_losses":         params.get("cb_consecutive_losses", 3),
        "cb_pause_days":     params.get("cb_pause_days",     7),
        "use_regime_filter": params.get("use_regime_filter", True),
        "_reasons":          reasons,
    }

    # Pre-warm the ML model so it loads once, not per-ticker
    if _USE_REAL_SCANNER:
        try:
            _load_x_model()
        except Exception:
            pass

    import contextlib, io, warnings as _warnings

    # ── Download historical data (skipped when preloaded_data is supplied) ───
    download_start = test_start - timedelta(days=120)

    if preloaded_data is not None:
        all_data = preloaded_data
        log.info(f"Using preloaded data for {len(all_data)} tickers — skipping download")
    else:
        period_label = f"{download_start.isoformat()} → {test_end.isoformat()}"
        log.info(f"Downloading data: {period_label}  ({len(tickers)} tickers)")

        if progress_cb:
            progress_cb(0, len(tickers), "Downloading historical data…")

        import socket as _socket

        all_data: dict[str, pd.DataFrame] = {}
        batch_size   = 20
        batches      = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        _total_units = len(tickers) * 2

        _dl_start_str = download_start.isoformat()
        _dl_end_str   = (test_end + timedelta(days=2)).isoformat()

        # Socket timeout — cuts the actual TCP connection if Yahoo Finance hangs.
        # ThreadPoolExecutor cannot cancel hung threads in Python; socket timeout can.
        _SOCK_TIMEOUT = 30   # seconds before a stalled connection raises socket.timeout

        for b_idx, batch in enumerate(batches):
            if progress_cb:
                progress_cb(b_idx * batch_size, _total_units,
                            f"Downloading batch {b_idx+1}/{len(batches)}…")

            _prev_timeout = _socket.getdefaulttimeout()
            try:
                _socket.setdefaulttimeout(_SOCK_TIMEOUT)
                _sink = io.StringIO()
                with contextlib.redirect_stderr(_sink), _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    raw = yf.download(
                        " ".join(batch),
                        start=_dl_start_str,
                        end=_dl_end_str,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=False,
                    )
                if len(batch) == 1:
                    df = _norm_cols(raw.dropna(how="all"))
                    if not df.empty and len(df) >= 30:
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
                log.warning(
                    f"Batch {b_idx+1}/{len(batches)} failed ({type(e).__name__}: {e}) "
                    f"— retrying one ticker at a time"
                )
                # Per-ticker fallback — each gets its own socket timeout window
                for sym in batch:
                    try:
                        _sink2 = io.StringIO()
                        with contextlib.redirect_stderr(_sink2), _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            raw1 = yf.download(
                                sym,
                                start=_dl_start_str,
                                end=_dl_end_str,
                                interval="1d",
                                auto_adjust=True,
                                progress=False,
                                threads=False,
                            )
                        df = _norm_cols(raw1.dropna(how="all"))
                        if not df.empty and len(df) >= 30:
                            all_data[sym] = df
                    except Exception as e2:
                        log.debug(f"  {sym} skipped: {type(e2).__name__}: {e2}")
            finally:
                _socket.setdefaulttimeout(_prev_timeout)

    available     = list(all_data.keys())
    skipped       = len(tickers) - len(available)
    _total_units  = len(available) * 2   # used for progress_cb (download 50% + sim 50%)
    log.info(f"Got data for {len(available)}/{len(tickers)} tickers  ({skipped} skipped)")

    if not available:
        return {"trades": [], "equity_curve": [], "metrics": _empty_metrics(), "params_used": p}

    # ── Normalise all DataFrames to tz-naive DatetimeIndex + build O(1) date lookup ──
    # Replaces all slow Python list-comprehension date comparisons in the simulation loop.
    # _date_idx[ticker][Timestamp] → iloc position  (O(1) instead of O(N) per lookup)
    if progress_cb:
        progress_cb(len(available), _total_units, "Building date index…")
    for ticker in list(all_data.keys()):
        df = all_data[ticker]
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.DatetimeIndex(df.index)
            except Exception:
                pass
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        all_data[ticker] = df

    _date_idx: dict[str, dict] = {
        ticker: {ts: i for i, ts in enumerate(df.index)}
        for ticker, df in all_data.items()
    }

    # ── Market regime (skipped when preloaded_regimes is supplied) ──────────
    if preloaded_regimes is not None:
        asx_regime, us_regime = preloaded_regimes
        log.info("Using preloaded regime data — skipping regime download")
    else:
        asx_regime: dict[date, bool] = {}
        us_regime:  dict[date, bool] = {}
        if p["use_regime_filter"]:
            for sym, regime_dict in [("^AXJO", asx_regime), ("^GSPC", us_regime)]:
                try:
                    _sink2 = io.StringIO()
                    with contextlib.redirect_stderr(_sink2), _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        idx_raw = yf.download(
                            sym,
                            start=download_start.isoformat(),
                            end=(test_end + timedelta(days=2)).isoformat(),
                            interval="1d",
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )
                    idx_df = _norm_cols(idx_raw.dropna(how="all"))
                    if not idx_df.empty:
                        idx_df["ma50"] = idx_df["Close"].rolling(50).mean()
                        for d_ts, idx_row in idx_df.iterrows():
                            d_key = d_ts.date() if hasattr(d_ts, "date") else d_ts
                            ma    = idx_row.get("ma50", float("nan"))
                            if not math.isnan(float(ma)):
                                regime_dict[d_key] = float(idx_row["Close"]) > float(ma)
                    log.info(f"Regime data OK for {sym}: {len(regime_dict)} days")
                except Exception as _re:
                    log.warning(f"Regime index {sym} failed: {_re}")

    # ── Build trading calendar from all available data ─────────────────────
    _ts_start = pd.Timestamp(test_start)
    _ts_end   = pd.Timestamp(test_end)
    all_dates: set[date] = set()
    for _idx in _date_idx.values():
        for ts in _idx:
            if _ts_start <= ts <= _ts_end:
                all_dates.add(ts.date())
    trading_days = sorted(all_dates)

    if not trading_days:
        return {"trades": [], "equity_curve": [], "metrics": _empty_metrics(), "params_used": p}

    # ── Pre-scan signals (skipped when caller supplies precomputed_signals) ─
    # Calls _compute_x_features ONCE per ticker instead of once per (ticker×day).
    # The slow path inside the simulation loop is used only when this fails.
    if precomputed_signals is None:
        if progress_cb:
            progress_cb(len(available), _total_units, "Pre-scanning signals (fast path)…")
        try:
            precomputed_signals = _prescan_all(
                all_data, available, _date_idx,
                _ts_start, _ts_end, p, progress_cb, _total_units,
            )
        except Exception as _pse:
            log.warning(f"Pre-scan failed ({_pse}) — falling back to slow path per-day scoring")
            precomputed_signals = None   # keep slow path as fallback

    # ── Simulation state ───────────────────────────────────────────────────
    account     = initial_capital
    open_pos:   list[BtPosition] = []
    closed:     list[BtTrade]    = []
    equity_crv: list[dict]       = []
    cb_trips    = 0          # consecutive losses for circuit breaker
    cb_paused_until: date | None = None

    total_days  = len(trading_days)

    for day_idx, sim_date in enumerate(trading_days):
        if progress_cb and day_idx % 5 == 0:
            # Second half of progress: len(tickers) → len(tickers)*2
            sim_done = len(tickers) + int(day_idx / max(total_days, 1) * len(tickers))
            progress_cb(sim_done, _total_units,
                        f"Simulating day {day_idx+1}/{total_days}  ({sim_date})…")

        # ── 1. Close positions that stop/target/expire today ────────────────
        still_open = []
        for pos in open_pos:
            df = all_data.get(pos.ticker)
            if df is None:
                still_open.append(pos)
                continue

            # Get today's bar — O(1) dict lookup via pre-built date index
            _ts   = pd.Timestamp(sim_date)
            _iloc = _date_idx.get(pos.ticker, {}).get(_ts)
            if _iloc is None:
                still_open.append(pos)
                continue
            _row  = df.iloc[_iloc]
            day_high  = float(_row["High"])
            day_low   = float(_row["Low"])
            day_close = float(_row["Close"])
            days_held = (sim_date - pos.entry_date).days

            # ── Dynamic stop management ────────────────────────────────────
            # Use the original entry-to-stop distance (fixed at open) as 1R.
            one_r = pos.orig_stop_dist if pos.orig_stop_dist > 0 \
                    else max(pos.entry_price - pos.stop_price, 0.0001)

            # Update rolling peak close (needed for trailing stop)
            if day_close > pos.peak_close:
                pos.peak_close = day_close

            # Break-even stop: once day's high hits entry+1R, slide stop to entry.
            # Converts potential losses on round-trips into flat scratches.
            be_trigger = pos.entry_price + one_r
            if day_high >= be_trigger and pos.stop_price < pos.entry_price:
                pos.stop_price = pos.entry_price

            # Trailing stop: once peak close exceeds entry+1.5R, trail 1R below peak.
            # Lets big winners run while locking in gains above break-even.
            if pos.peak_close >= pos.entry_price + 1.5 * one_r:
                trail_stop = round(pos.peak_close - one_r, 4)
                if trail_stop > pos.stop_price:
                    pos.stop_price = trail_stop
            # ─────────────────────────────────────────────────────────────

            exit_price  = None
            exit_reason = None

            # Conservative: stop checked before target (worst-case)
            # min_hold_days guard: don't allow a stop exit in the first N calendar
            # days after entry — protects against entry-day spread / gap noise.
            _past_min_hold = days_held >= p.get("min_hold_days", 2)
            if day_low <= pos.stop_price and _past_min_hold:
                exit_price  = pos.stop_price
                exit_reason = "STOP_HIT"
            elif day_high >= pos.target_price:
                exit_price  = pos.target_price
                # If close is still above target the stock kept running past it
                exit_reason = "ABOVE_TARGET" if day_close > pos.target_price else "TARGET_HIT"
            elif days_held >= pos.max_hold:
                exit_price  = day_close
                exit_reason = "MAX_HOLD"

            if exit_price is not None:
                brok = p["brokerage"] * 2
                trade = BtTrade(
                    ticker       = pos.ticker,
                    entry_date   = pos.entry_date,
                    exit_date    = sim_date,
                    entry_price  = pos.entry_price,
                    exit_price   = exit_price,
                    quantity     = pos.quantity,
                    stop_price   = pos.stop_price,
                    target_price = pos.target_price,
                    exit_reason  = exit_reason,
                    score        = pos.score,
                    prob         = pos.prob,
                )
                account += trade.pnl - brok
                closed.append(trade)

                # Circuit breaker tracking
                if trade.outcome == "LOSS":
                    cb_trips += 1
                    if cb_trips >= p["cb_losses"]:
                        cb_paused_until = sim_date + timedelta(days=p["cb_pause_days"])
                        log.info(f"Circuit breaker tripped on {sim_date} — paused until {cb_paused_until}")
                else:
                    cb_trips = 0
            else:
                still_open.append(pos)

        open_pos = still_open

        # ── 2. Record equity snapshot ────────────────────────────────────────
        unrealised = 0.0
        _ts_eq = pd.Timestamp(sim_date)
        for pos in open_pos:
            _i = _date_idx.get(pos.ticker, {}).get(_ts_eq)
            if _i is not None:
                curr = float(all_data[pos.ticker].iloc[_i]["Close"])
                unrealised += (curr - pos.entry_price) * pos.quantity

        equity_crv.append({
            "date":         sim_date.isoformat(),
            "account":      round(account, 2),
            "unrealised":   round(unrealised, 2),
            "equity":       round(account + unrealised, 2),
            "open_positions": len(open_pos),
        })

        # ── 3. Scan for new signals ──────────────────────────────────────────
        if cb_paused_until and sim_date < cb_paused_until:
            continue

        at_capacity = len(open_pos) >= p["max_positions"]

        open_tickers = {pos.ticker for pos in open_pos}
        new_signals  = []

        if precomputed_signals is not None:
            # ── Fast path: use pre-computed signal cache (no indicator recomputation) ──
            elite_min = p["min_score"] + 2
            sb_min    = p["min_score"]
            for ticker in available:
                if ticker in open_tickers:
                    continue
                raw = precomputed_signals.get((ticker, sim_date))
                if raw is None:
                    continue
                score = raw["score"]
                prob  = raw["prob"]
                if score < p["min_score"] or prob < p["min_prob"]:
                    continue
                if raw.get("expected_r", 1.0) <= 0:
                    continue
                # Determine tier from current min_score
                if score >= elite_min:
                    tier = "ELITE"
                elif score >= sb_min:
                    tier = "STRONG BUY"
                else:
                    continue
                # Recompute stop/target from stored atr and current sl/target params
                atr_pct = raw["atr_pct"]
                atr     = raw["atr"]
                ep      = raw["entry_price"]
                if atr_pct >= 3.0:
                    sl_mult = p["sl_mult_hi"];  tp_pct = p["target_hi"]
                elif atr_pct >= 1.5:
                    sl_mult = p["sl_mult_mid"]; tp_pct = p["target_mid"]
                else:
                    sl_mult = p["sl_mult_lo"];  tp_pct = p["target_lo"]
                stop_price   = max(ep - sl_mult * atr, ep * 0.88)
                target_price = ep * (1 + tp_pct / 100)
                new_signals.append({
                    "ticker":       ticker,
                    "score":        score,
                    "prob":         prob,
                    "tier":         tier,
                    "entry_price":  ep,
                    "stop_price":   stop_price,
                    "target_price": target_price,
                    "atr_pct":      atr_pct,
                    "exchange":     raw.get("exchange", "SMART"),
                })
        else:
            # ── Slow path: full indicator recomputation per (ticker, day) ────────────
            _ts_le = pd.Timestamp(sim_date)
            for ticker, df in all_data.items():
                if ticker in open_tickers:
                    continue
                # Use pandas binary search on sorted DatetimeIndex — O(log N)
                df_slice = df.loc[:_ts_le]
                sig = _detect_signal(df_slice, ticker, p)
                if sig and sig.get("score", 0) >= p["min_score"]:
                    new_signals.append(sig)

        # Sort by score + probability descending
        new_signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)

        if at_capacity:
            # ── ELITE-only bump: swap out worst losing position ───────────────
            # Rules: only ELITE tier qualifies; only bumps positions currently
            # underwater (negative P&L vs entry price) — never closes a winner.
            elite_sigs = [s for s in new_signals if s.get("tier") == "ELITE"]
            if not elite_sigs:
                continue   # no ELITE available — hold current positions

            # Find the open position with the worst current loss
            worst_pos  = None
            worst_pnl  = 0.0   # threshold: must be negative (losing money)
            _ts_bump   = pd.Timestamp(sim_date)
            for pos in open_pos:
                _bi = _date_idx.get(pos.ticker, {}).get(_ts_bump)
                if _bi is None:
                    continue
                curr    = float(all_data[pos.ticker].iloc[_bi]["Close"])
                pnl_pct = (curr - pos.entry_price) / pos.entry_price
                if pnl_pct < worst_pnl:
                    worst_pnl = pnl_pct
                    worst_pos = pos

            if worst_pos is None:
                continue   # all positions are profitable — never bump a winner

            # Close the losing position at today's close
            _wi       = _date_idx.get(worst_pos.ticker, {}).get(_ts_bump)
            bump_exit = float(all_data[worst_pos.ticker].iloc[_wi]["Close"]) if _wi is not None else worst_pos.entry_price
            bump_trade = BtTrade(
                ticker       = worst_pos.ticker,
                entry_date   = worst_pos.entry_date,
                exit_date    = sim_date,
                entry_price  = worst_pos.entry_price,
                exit_price   = bump_exit,
                quantity     = worst_pos.quantity,
                stop_price   = worst_pos.stop_price,
                target_price = worst_pos.target_price,
                exit_reason  = "ELITE_BUMP",
                score        = worst_pos.score,
                prob         = worst_pos.prob,
            )
            account += bump_trade.pnl - p["brokerage"] * 2
            closed.append(bump_trade)
            open_pos.remove(worst_pos)
            new_signals = [elite_sigs[0]]   # only enter the single best ELITE

        slots = p["max_positions"] - len(open_pos)

        for sig in new_signals[:slots]:
            # Market regime gate: skip if the broad market is in a downtrend
            if p["use_regime_filter"]:
                is_asx    = sig["ticker"].upper().endswith(".AX")
                regime    = asx_regime if is_asx else us_regime
                # Only block if we actually have data for this date; default=allow
                if regime and not regime.get(sim_date, True):
                    continue

            # Entry = next trading day's open — vectorized index comparison
            df     = all_data[sig["ticker"]]
            future = df[df.index > pd.Timestamp(sim_date)]

            if future.empty:
                continue

            entry_price = float(future["Open"].iloc[0])
            entry_date  = future.index[0].date()

            if entry_price <= 0:
                continue

            # Position size
            stop_dist   = entry_price - sig["stop_price"]
            if stop_dist <= 0:
                continue
            dollar_risk = account * (p["risk_pct"] / 100) - p["brokerage"] * 2
            qty         = math.floor(dollar_risk / stop_dist)
            if qty < 1:
                continue

            trade_value = qty * entry_price
            if trade_value > account * 0.95:
                continue

            # Adjust stop/target relative to actual entry (in case entry gaps up)
            stop_price   = sig["stop_price"]
            target_price = sig["target_price"]
            # Re-anchor stop AND target to the actual entry price whenever
            # the stock has gapped significantly from the signal-day close
            # (handles BOTH gap-up and gap-down — previously only gap-up was adjusted).
            _gap_pct = abs(entry_price - sig["entry_price"]) / max(sig["entry_price"], 0.001)
            if _gap_pct > 0.005:
                _sl_pct  = (sig["entry_price"] - stop_price)   / max(sig["entry_price"], 0.001)
                _tp_pct  = (target_price        - sig["entry_price"]) / max(sig["entry_price"], 0.001)
                stop_price   = entry_price * (1 - _sl_pct)
                target_price = entry_price * (1 + _tp_pct)
                # If gap-down pushed stop below entry, skip — risk is undefined
                if stop_price >= entry_price:
                    continue

            orig_dist = round(entry_price - stop_price, 4)
            open_pos.append(BtPosition(
                ticker         = sig["ticker"],
                entry_date     = entry_date,
                entry_price    = entry_price,
                stop_price     = round(stop_price,   4),
                target_price   = round(target_price, 4),
                quantity       = qty,
                max_hold       = p["hold_days"],
                score          = sig["score"],
                prob           = sig["prob"],
                orig_stop_dist = orig_dist,
                peak_close     = entry_price,
            ))
            account -= p["brokerage"]   # entry commission

    # ── Close any remaining open positions at final close ────────────────────
    last_day = trading_days[-1]
    for pos in open_pos:
        df = all_data.get(pos.ticker)
        if df is None:
            continue
        rows = df.loc[:pd.Timestamp(last_day)]
        if rows.empty:
            continue
        exit_price = float(rows["Close"].iloc[-1])
        trade = BtTrade(
            ticker       = pos.ticker,
            entry_date   = pos.entry_date,
            exit_date    = last_day,
            entry_price  = pos.entry_price,
            exit_price   = exit_price,
            quantity     = pos.quantity,
            stop_price   = pos.stop_price,
            target_price = pos.target_price,
            exit_reason  = "END_OF_TEST",
            score        = pos.score,
            prob         = pos.prob,
        )
        account += trade.pnl - p["brokerage"] * 2
        closed.append(trade)

    return {
        "trades":             closed,
        "equity_curve":       equity_crv,
        "metrics":            _calc_metrics(closed, initial_capital, account),
        "params_used":        p,
        "tickers_scanned":    len(available),
        "tickers_skipped":    skipped,
        "trading_days":       len(trading_days),
        "rejection_reasons":  dict(sorted(reasons.items(), key=lambda x: -x[1])),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def _calc_metrics(trades: list[BtTrade], initial_capital: float, final_capital: float) -> dict:
    if not trades:
        return _empty_metrics()

    wins  = [t for t in trades if t.pnl >= 0]
    loss  = [t for t in trades if t.pnl <  0]
    gw    = sum(t.pnl for t in wins)
    gl    = abs(sum(t.pnl for t in loss))
    pf    = gw / gl if gl > 0 else 99.0

    rets  = [t.pnl_pct for t in trades]
    n     = len(rets)
    mean  = sum(rets) / n
    std   = math.sqrt(sum((r - mean)**2 for r in rets) / max(n-1, 1))
    holds = [t.hold_days for t in trades]
    avg_hold = sum(holds) / n

    # Annualised Sharpe (assume 252 trading days, avg_hold days per trade)
    trades_per_year = 252 / max(avg_hold, 1)
    sharpe = (mean / std) * math.sqrt(trades_per_year) if std > 0 else 0.0

    # Max drawdown on equity curve, measured against account value
    # (not peak P&L gain — avoids the absurdly large % when early P&L peak is tiny)
    equity = initial_capital; peak_equity = initial_capital; max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_date):
        equity     += t.pnl
        peak_equity = max(peak_equity, equity)
        dd          = (peak_equity - equity) / peak_equity
        max_dd      = max(max_dd, dd)

    total_pnl = sum(t.pnl for t in trades)
    roi       = total_pnl / initial_capital * 100

    # Exits breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        "trade_count":     n,
        "win_count":       len(wins),
        "loss_count":      len(loss),
        "win_rate":        round(len(wins) / n, 4),
        "profit_factor":   round(pf, 3),
        "total_pnl":       round(total_pnl, 2),
        "roi_pct":         round(roi, 2),
        "initial_capital": round(initial_capital, 2),
        "final_capital":   round(final_capital, 2),
        "avg_win":         round(gw / len(wins), 2) if wins else 0,
        "avg_loss":        round(gl / len(loss), 2) if loss else 0,
        "avg_hold_days":   round(avg_hold, 1),
        "max_drawdown":    round(max_dd, 4),
        "sharpe":          round(sharpe, 3),
        "exit_reasons":    reasons,
    }


def _empty_metrics() -> dict:
    return {
        "trade_count": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0, "profit_factor": 0, "total_pnl": 0,
        "roi_pct": 0, "initial_capital": 0, "final_capital": 0,
        "avg_win": 0, "avg_loss": 0, "avg_hold_days": 0,
        "max_drawdown": 0, "sharpe": 0, "exit_reasons": {},
    }


# ── Parameter sweep helper ─────────────────────────────────────────────────────

def parameter_sweep(
    tickers:     list[str],
    test_start:  date,
    test_end:    date,
    sweep:       list[dict],
    initial_capital: float = 10_000.0,
    progress_cb: Callable | None = None,
) -> list[dict]:
    """
    Run backtest for each param set in `sweep`.
    Returns list of {params, metrics} sorted by profit_factor desc.
    """
    results = []
    for i, param_set in enumerate(sweep):
        if progress_cb:
            progress_cb(i, len(sweep), f"Sweep {i+1}/{len(sweep)}…")
        res = run_backtest(tickers, test_start, test_end, initial_capital, param_set)
        results.append({"params": param_set, "metrics": res["metrics"]})
    results.sort(key=lambda r: r["metrics"]["profit_factor"], reverse=True)
    return results
