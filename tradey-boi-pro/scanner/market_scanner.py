"""
Tradey Boi Pro — Core Scanner

Applies Tradey Boi X's proven breakout detection logic to an arbitrary ticker list.
This module is pure signal detection — no scheduling, no ranking, no regime logic.
Those are handled by monitor.py, ranker.py, and market_regime.py respectively.

Preserved from X:
  - 20-day high breakout filter
  - Volume surge ≥ 1.5× average
  - 50-day EMA trend filter
  - ATR-based stop/target calculation
  - 14-day RSI filter (50–80 sweet spot)
  - Score (0–10) construction logic

Upgraded in Pro:
  - Covers any ticker list (not a fixed watchlist)
  - Returns OHLCV cache for ranker to reuse (no double-download)
  - scan_batch() for efficient subset re-scans (Tier 2/3)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pytz

log = logging.getLogger("MarketScanner")

_ASX_TZ = pytz.timezone("Australia/Sydney")
_US_TZ  = pytz.timezone("America/New_York")


# ── Market hours ──────────────────────────────────────────────────────────────

def _is_asx_open() -> bool:
    now = datetime.now(_ASX_TZ)
    return now.weekday() < 5 and 10 <= now.hour < 16


def _is_us_open() -> bool:
    now = datetime.now(_US_TZ)
    return now.weekday() < 5 and (
        (now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16)
    )


def market_is_open() -> bool:
    return _is_asx_open() or _is_us_open()


def next_open_seconds() -> int:
    now_asx = datetime.now(_ASX_TZ)
    now_us  = datetime.now(_US_TZ)
    secs    = []
    for tz, now, oh, om in [(_ASX_TZ, now_asx, 10, 0), (_US_TZ, now_us, 9, 30)]:
        if now.weekday() < 5:
            op = now.replace(hour=oh, minute=om, second=0, microsecond=0)
            if now < op:
                secs.append(int((op - now).total_seconds()))
    return min(secs) if secs else 3600


# ── Core signal detection (Tradey Boi X logic, unchanged) ────────────────────

def _score_signal(df: pd.DataFrame, ticker: str, params: dict) -> Optional[dict]:
    """
    Tradey Boi X breakout detection applied to a single ticker's OHLCV DataFrame.
    Returns a raw signal dict or None if criteria not met.
    All indicator logic preserved from X's strategy.
    """
    if df is None or len(df) < 30:
        return None
    try:
        close  = df["Close"].squeeze().dropna()
        volume = df["Volume"].squeeze().dropna()
        high   = df["High"].squeeze().dropna()
        low    = df["Low"].squeeze().dropna()

        if len(close) < 30:
            return None

        curr_price  = float(close.iloc[-1])
        prev_close  = float(close.iloc[-2])
        if curr_price <= 0:
            return None

        high_20   = float(high.iloc[-21:-1].max())
        avg_vol20 = float(volume.iloc[-21:-1].mean())
        curr_vol  = float(volume.iloc[-1])
        ema50     = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema20     = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

        # ATR-14
        tr_list = []
        for i in range(-15, 0):
            h = float(high.iloc[i]);  l = float(low.iloc[i])
            c = float(close.iloc[i - 1])
            tr_list.append(max(h - l, abs(h - c), abs(l - c)))
        atr     = float(np.mean(tr_list))
        atr_pct = atr / curr_price * 100

        # RSI-14
        delta  = close.diff()
        avg_g  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        avg_l  = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
        rsi    = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100

        # ── Hard filters (X strategy, unchanged) ─────────────────────────────
        if curr_price <= high_20 * 1.001:   return None   # not a breakout
        if curr_price < ema50 * 0.97:        return None   # below trend
        if rsi > 80:                          return None   # overbought
        if curr_vol  < avg_vol20 * 1.1:      return None   # no vol confirmation
        if atr_pct   < 0.3:                  return None   # too thin

        # ── Score 0–10 (X logic preserved) ───────────────────────────────────
        score = 0
        bp    = (curr_price - high_20) / high_20 * 100
        if bp > 3:    score += 2
        elif bp > 1:  score += 1
        vr = curr_vol / avg_vol20 if avg_vol20 > 0 else 1
        if vr > 3:    score += 2
        elif vr > 2:  score += 1
        if curr_price > ema20 > ema50:  score += 2
        elif curr_price > ema50:        score += 1
        if 55 <= rsi <= 70:   score += 2
        elif 50 <= rsi < 55:  score += 1
        dm = (curr_price - prev_close) / prev_close * 100
        if dm > 2:  score += 1
        if dm > 4:  score += 1
        score = min(score, 10)

        # ── Probability estimate (X heuristic) ───────────────────────────────
        prob = min(0.50 + score * 0.025, 0.82)

        min_score = int(params.get("min_score", 7))
        min_prob  = float(params.get("min_prob", 0.53))
        if score < min_score or prob < min_prob:
            return None

        # ── Stop / Target (X ATR-mult logic) ─────────────────────────────────
        if atr_pct >= 3.0:
            sl_mult = params.get("sl_mult_hi",  1.2);  tp_pct = params.get("target_hi",  12.0)
        elif atr_pct >= 1.5:
            sl_mult = params.get("sl_mult_mid", 1.0);  tp_pct = params.get("target_mid",  8.0)
        else:
            sl_mult = params.get("sl_mult_lo",  0.8);  tp_pct = params.get("target_lo",   5.0)

        stop_price   = max(curr_price - sl_mult * atr, curr_price * 0.88)
        target_price = curr_price * (1 + tp_pct / 100)
        exchange     = "ASX" if ticker.endswith(".AX") else "SMART"

        return {
            "ticker":       ticker,
            "entry_price":  round(curr_price,    4),
            "stop_price":   round(stop_price,    4),
            "target_price": round(target_price,  4),
            "atr_pct":      round(atr_pct,       2),
            "score":        score,
            "prob":         round(prob,           3),
            "rsi":          round(float(rsi),     1),
            "vol_ratio":    round(vr,             1),
            "breakout_pct": round(bp,             2),
            "signal_date":  datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "exchange":     exchange,
            "currency":     "AUD" if exchange == "ASX" else "USD",
            "tier":         "STRONG BUY" if score >= 8 else "BUY",   # pre-rank tier
            "source":       "pro_scanner",
        }
    except Exception as e:
        log.debug(f"Score error {ticker}: {e}")
        return None


# ── Batch download ─────────────────────────────────────────────────────────────

def _download_batch(
    tickers: list[str],
    period:  str = "90d",
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    try:
        raw = yf.download(
            " ".join(tickers), period=period, interval="1d",
            auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        if len(tickers) == 1:
            df = raw.dropna(how="all")
            return {tickers[0]: df} if not df.empty else {}
        result = {}
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
                if not df.empty and len(df) >= 30:
                    result[t] = df
            except (KeyError, TypeError):
                pass
        return result
    except Exception as e:
        log.error(f"Batch download error: {e}")
        return {}


# ── Public scan functions ──────────────────────────────────────────────────────

def scan_all(
    tickers:      list[str],
    batch_size:   int = 50,
    progress_cb:  Callable[[int, int, str], None] | None = None,
    return_cache: bool = False,
    params:       dict | None = None,
) -> tuple[list[dict], dict] | list[dict]:
    """
    Full scan of a ticker list. Returns raw signals (pre-ranking).
    If return_cache=True, also returns {ticker: DataFrame} for ranker reuse.
    """
    if params is None:
        params = _default_params()

    signals:  list[dict]              = []
    df_cache: dict[str, pd.DataFrame] = {}
    batches   = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    n_total   = len(tickers)

    for b_idx, batch in enumerate(batches):
        done = b_idx * batch_size
        if progress_cb:
            progress_cb(done, n_total, f"Batch {b_idx+1}/{len(batches)}")

        data = _download_batch(batch)
        df_cache.update(data)

        for ticker, df in data.items():
            sig = _score_signal(df, ticker, params)
            if sig:
                signals.append(sig)
                log.info(f"  RAW SIGNAL: {ticker}  score={sig['score']}  prob={sig['prob']:.2f}")

    signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
    log.info(f"scan_all: {len(signals)} raw signals from {len(tickers)} tickers")

    return (signals, df_cache) if return_cache else signals


def scan_batch(
    tickers:      list[str],
    period:       str  = "90d",
    return_cache: bool = False,
    params:       dict | None = None,
) -> tuple[list[dict], dict] | list[dict]:
    """
    Fast re-scan of a small ticker subset (Tier 2/3 refresh).
    Skips batching overhead for speed.
    """
    if params is None:
        params = _default_params()

    data     = _download_batch(tickers, period=period)
    signals  = []
    for ticker, df in data.items():
        sig = _score_signal(df, ticker, params)
        if sig:
            signals.append(sig)

    signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
    return (signals, data) if return_cache else signals


def _default_params() -> dict:
    try:
        import config.settings as cfg
        return {
            "min_score":    int(cfg.get("min_score")    or 7),
            "min_prob":     float(cfg.get("min_prob")   or 0.53),
            "sl_mult_hi":   float(cfg.get("sl_mult_hi") or 1.2),
            "sl_mult_mid":  float(cfg.get("sl_mult_mid")or 1.0),
            "sl_mult_lo":   float(cfg.get("sl_mult_lo") or 0.8),
            "target_hi":    float(cfg.get("target_hi")  or 12.0),
            "target_mid":   float(cfg.get("target_mid") or 8.0),
            "target_lo":    float(cfg.get("target_lo")  or 5.0),
        }
    except Exception:
        return {
            "min_score": 7, "min_prob": 0.53,
            "sl_mult_hi": 1.2, "sl_mult_mid": 1.0, "sl_mult_lo": 0.8,
            "target_hi": 12.0, "target_mid": 8.0, "target_lo": 5.0,
        }
