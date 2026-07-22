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
    from scanner.market_scanner import _score_signal as _real_score, _load_x_model
    _USE_REAL_SCANNER = True
except Exception as _e:
    log.warning(f"Could not import real scanner — falling back to rule-based: {_e}")
    _USE_REAL_SCANNER = False


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BtPosition:
    ticker:       str
    entry_date:   date
    entry_price:  float
    stop_price:   float
    target_price: float
    quantity:     float
    max_hold:     int
    score:        float = 0
    prob:         float = 0

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


# ── Main backtest engine ───────────────────────────────────────────────────────

def run_backtest(
    tickers:         list[str],
    test_start:      date,
    test_end:        date,
    initial_capital: float = 10_000.0,
    params:          dict  | None = None,
    progress_cb:     Callable[[int, int, str], None] | None = None,
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
        "min_score":      params.get("min_score",      7),
        "min_prob":       params.get("min_prob",       0.53),
        "max_positions":  params.get("max_positions",  5),
        "risk_pct":       params.get("risk_pct",       2.0),
        "brokerage":      params.get("brokerage",      2.0),
        "hold_days":      params.get("hold_days",      15),
        "sl_mult_hi":     params.get("sl_mult_hi",     1.2),
        "sl_mult_mid":    params.get("sl_mult_mid",    1.0),
        "sl_mult_lo":     params.get("sl_mult_lo",     0.8),
        "target_hi":      params.get("target_hi",      12.0),
        "target_mid":     params.get("target_mid",     8.0),
        "target_lo":      params.get("target_lo",      5.0),
        "cb_losses":      params.get("cb_consecutive_losses", 3),
        "cb_pause_days":  params.get("cb_pause_days",  7),
        "_reasons":       reasons,   # shared mutable — _score_signal writes to this
    }

    # Pre-warm the ML model so it loads once, not per-ticker
    if _USE_REAL_SCANNER:
        try:
            _load_x_model()
        except Exception:
            pass

    # ── Download historical data ─────────────────────────────────────────────
    # Need 90 extra days before test_start for indicator warm-up
    download_start = test_start - timedelta(days=120)
    period_label   = f"{download_start.isoformat()} → {test_end.isoformat()}"
    log.info(f"Downloading data: {period_label}  ({len(tickers)} tickers)")

    if progress_cb:
        progress_cb(0, len(tickers), "Downloading historical data…")

    # Batch download — 50 tickers at a time
    # Suppress yfinance console noise (delisted/missing tickers print to stderr)
    import contextlib, io, warnings as _warnings
    all_data: dict[str, pd.DataFrame] = {}
    batch_size = 20                     # smaller batches = more reliable on slow connections
    batches    = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    _total_units = len(tickers) * 2   # first half = download, second half = simulation

    for b_idx, batch in enumerate(batches):
        if progress_cb:
            done = b_idx * batch_size          # 0 → len(tickers) covers first 50%
            progress_cb(done, _total_units, f"Downloading batch {b_idx+1}/{len(batches)}…")
        try:
            _sink = io.StringIO()
            with contextlib.redirect_stderr(_sink), _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                raw = yf.download(
                    " ".join(batch),
                    start=download_start.isoformat(),
                    end=(test_end + timedelta(days=2)).isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=False,   # threads=True can deadlock on Windows
                )
            if len(batch) == 1:
                df = raw.dropna(how="all")
                df.columns = [c.title() if isinstance(c, str) else c for c in df.columns]
                if not df.empty:
                    all_data[batch[0]] = df
            else:
                for t in batch:
                    try:
                        df = raw[t].dropna(how="all")
                        df.columns = [c.title() if isinstance(c, str) else c for c in df.columns]
                        if not df.empty and len(df) >= 30:
                            all_data[t] = df
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            log.error(f"Batch download error: {e}")

    available    = list(all_data.keys())
    skipped      = len(tickers) - len(available)
    log.info(f"Got data for {len(available)}/{len(tickers)} tickers  ({skipped} skipped — no data/delisted)")

    if not available:
        return {"trades": [], "equity_curve": [], "metrics": _empty_metrics(), "params_used": p}

    # ── Build trading calendar from all available data ─────────────────────
    all_dates: set[date] = set()
    for df in all_data.values():
        for d in df.index:
            dt = d.date() if hasattr(d, "date") else d
            if test_start <= dt <= test_end:
                all_dates.add(dt)
    trading_days = sorted(all_dates)

    if not trading_days:
        return {"trades": [], "equity_curve": [], "metrics": _empty_metrics(), "params_used": p}

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

            # Get today's bar
            today_rows = df[df.index.date == sim_date] if hasattr(df.index[0], "date") \
                         else df[df.index == pd.Timestamp(sim_date)]
            if today_rows.empty:
                still_open.append(pos)
                continue

            day_high  = float(today_rows["High"].iloc[0])
            day_low   = float(today_rows["Low"].iloc[0])
            day_close = float(today_rows["Close"].iloc[0])
            days_held = (sim_date - pos.entry_date).days

            exit_price  = None
            exit_reason = None

            # Conservative: stop checked before target (worst-case)
            if day_low <= pos.stop_price:
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
        for pos in open_pos:
            df = all_data.get(pos.ticker)
            if df is not None:
                rows = df[df.index.date == sim_date] if hasattr(df.index[0], "date") \
                       else df[df.index == pd.Timestamp(sim_date)]
                if not rows.empty:
                    curr = float(rows["Close"].iloc[0])
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
        if len(open_pos) >= p["max_positions"]:
            continue

        open_tickers = {pos.ticker for pos in open_pos}
        new_signals  = []

        for ticker, df in all_data.items():
            if ticker in open_tickers:
                continue
            # Slice to data available on sim_date
            if hasattr(df.index[0], "date"):
                mask = [d.date() <= sim_date for d in df.index]
            else:
                mask = df.index <= pd.Timestamp(sim_date)
            df_slice = df[mask]
            sig = _detect_signal(df_slice, ticker, p)
            if sig and sig.get("score", 0) >= p["min_score"]:
                new_signals.append(sig)

        # Sort by score desc, take top slots
        new_signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
        slots = p["max_positions"] - len(open_pos)

        for sig in new_signals[:slots]:
            # Entry = next trading day's open (find it)
            df     = all_data[sig["ticker"]]
            if hasattr(df.index[0], "date"):
                future = df[[d.date() > sim_date for d in df.index]]
            else:
                future = df[df.index > pd.Timestamp(sim_date)]

            if future.empty:
                continue

            entry_price = float(future["Open"].iloc[0])
            entry_date  = future.index[0].date() if hasattr(future.index[0], "date") \
                          else future.index[0].to_pydatetime().date()

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
            if entry_price > sig["entry_price"] * 1.02:
                # Entry gapped up — recalculate stop from entry
                sl_pct     = (sig["entry_price"] - stop_price) / sig["entry_price"]
                tp_pct     = (target_price - sig["entry_price"]) / sig["entry_price"]
                stop_price  = entry_price * (1 - sl_pct)
                target_price = entry_price * (1 + tp_pct)

            open_pos.append(BtPosition(
                ticker       = sig["ticker"],
                entry_date   = entry_date,
                entry_price  = entry_price,
                stop_price   = round(stop_price,   4),
                target_price = round(target_price, 4),
                quantity     = qty,
                max_hold     = p["hold_days"],
                score        = sig["score"],
                prob         = sig["prob"],
            ))
            account -= p["brokerage"]   # entry commission

    # ── Close any remaining open positions at final close ────────────────────
    last_day = trading_days[-1]
    for pos in open_pos:
        df = all_data.get(pos.ticker)
        if df is None:
            continue
        if hasattr(df.index[0], "date"):
            rows = df[[d.date() <= last_day for d in df.index]]
        else:
            rows = df[df.index <= pd.Timestamp(last_day)]
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

    # Max drawdown on equity curve from cumulative P&L
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_date):
        cum  += t.pnl
        peak  = max(peak, cum)
        dd    = (peak - cum) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

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
