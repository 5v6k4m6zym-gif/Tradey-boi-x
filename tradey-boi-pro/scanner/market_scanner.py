"""
Tradey Boi Pro — Market Scanner
Runs locally, scans ASX + US (200+ tickers) every 15–30 min during market hours.
Uses the same proven breakout logic as Tradey Boi X.
Data source: yfinance (free, 15-min delayed).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pytz

import db.database as db
import config.settings as cfg
from scanner.watchlist_manager import get_all_active_tickers

log = logging.getLogger("MarketScanner")

# ── Market hours ──────────────────────────────────────────────────────────────

_ASX_TZ = pytz.timezone("Australia/Sydney")
_US_TZ  = pytz.timezone("America/New_York")


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
    """Seconds until the next market opens (max 24h)."""
    now_asx  = datetime.now(_ASX_TZ)
    now_us   = datetime.now(_US_TZ)
    secs = []
    for tz, now, open_h, open_m in [
        (_ASX_TZ, now_asx, 10, 0),
        (_US_TZ,  now_us,  9, 30),
    ]:
        if now.weekday() < 5:
            open_today = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
            if now < open_today:
                secs.append(int((open_today - now).total_seconds()))
    return min(secs) if secs else 3600


# ── Core breakout scanner ─────────────────────────────────────────────────────

def _score_signal(df: pd.DataFrame, ticker: str) -> Optional[dict]:
    """
    Apply breakout criteria to OHLCV data.
    Returns a signal dict (score 0–10) or None.
    """
    if df is None or len(df) < 30:
        return None

    try:
        close  = df["Close"].squeeze().dropna()
        volume = df["Volume"].squeeze().dropna()
        high   = df["High"].squeeze().dropna()

        if len(close) < 30:
            return None

        curr_price  = float(close.iloc[-1])
        prev_close  = float(close.iloc[-2])
        if curr_price <= 0:
            return None

        # Rolling windows
        high_20   = float(high.iloc[-21:-1].max())   # 20-day high (excl today)
        avg_vol20 = float(volume.iloc[-21:-1].mean())
        curr_vol  = float(volume.iloc[-1])
        ema50     = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema20     = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

        # ATR (14-day)
        tr_list = []
        for i in range(-15, 0):
            h = float(df["High"].squeeze().iloc[i])
            l = float(df["Low"].squeeze().iloc[i])
            c = float(df["Close"].squeeze().iloc[i - 1])
            tr_list.append(max(h - l, abs(h - c), abs(l - c)))
        atr     = float(np.mean(tr_list))
        atr_pct = atr / curr_price * 100

        # RSI (14-day)
        delta   = close.diff()
        gains   = delta.clip(lower=0)
        losses  = (-delta).clip(lower=0)
        avg_g   = gains.rolling(14).mean().iloc[-1]
        avg_l   = losses.rolling(14).mean().iloc[-1]
        rsi     = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100

        # ── Hard filter: must be a breakout ─────────────────────────────────
        if curr_price <= high_20 * 1.001:            # not breaking out
            return None
        if curr_price < ema50 * 0.97:                # below trend
            return None
        if rsi > 80:                                  # overbought
            return None
        if curr_vol < avg_vol20 * 1.1:               # no volume confirmation
            return None
        if atr_pct < 0.3:                            # too thin
            return None

        # ── Score (0–10) ─────────────────────────────────────────────────────
        score = 0

        # Breakout strength
        breakout_pct = (curr_price - high_20) / high_20 * 100
        if breakout_pct > 3:    score += 2
        elif breakout_pct > 1:  score += 1

        # Volume surge
        vol_ratio = curr_vol / avg_vol20 if avg_vol20 > 0 else 1
        if vol_ratio > 3:    score += 2
        elif vol_ratio > 2:  score += 1

        # Trend alignment
        if curr_price > ema20 > ema50:   score += 2
        elif curr_price > ema50:         score += 1

        # RSI sweet spot (55–70)
        if 55 <= rsi <= 70:   score += 2
        elif 50 <= rsi < 55:  score += 1

        # Day's move (positive momentum)
        day_move = (curr_price - prev_close) / prev_close * 100
        if day_move > 2:    score += 1
        if day_move > 4:    score += 1   # extra for big day

        # Clamp
        score = min(score, 10)

        min_score = int(cfg.get("min_score") or 7)
        if score < min_score:
            return None

        # ── Probability estimate (heuristic, no ML) ──────────────────────────
        prob = 0.50 + score * 0.025   # 0.50 → 0.75 range
        prob = min(prob, 0.82)

        min_prob = float(cfg.get("min_prob") or 0.53)
        if prob < min_prob:
            return None

        exchange = "ASX" if ticker.endswith(".AX") else "SMART"
        currency = "AUD" if exchange == "ASX" else "USD"

        return {
            "ticker":      ticker,
            "entry_price": round(curr_price, 4),
            "atr_pct":     round(atr_pct, 2),
            "prob":        round(prob, 3),
            "score":       score,
            "tier":        "STRONG BUY" if score >= 8 else "BUY",
            "rsi":         round(float(rsi), 1),
            "vol_ratio":   round(vol_ratio, 1),
            "breakout_pct": round(breakout_pct, 2),
            "signal_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "exchange":    exchange,
            "currency":    currency,
            "source":      "pro_scanner",
        }

    except Exception as e:
        log.debug(f"Score error for {ticker}: {e}")
        return None


def _download_batch(tickers: list[str], period: str = "90d") -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch of tickers using yfinance multi-ticker."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if len(tickers) == 1:
            return {tickers[0]: raw}
        result = {}
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
                if not df.empty:
                    result[t] = df
            except (KeyError, TypeError):
                pass
        return result
    except Exception as e:
        log.error(f"Batch download error: {e}")
        return {}


def scan_all(batch_size: int = 50, progress_cb=None) -> list[dict]:
    """
    Full watchlist scan. Returns list of signal dicts sorted by score desc.
    progress_cb(done, total) called after each batch if provided.
    """
    tickers = get_all_active_tickers()
    if not tickers:
        return []

    # Only scan relevant markets when open
    if _is_asx_open() and not _is_us_open():
        tickers = [t for t in tickers if t.endswith(".AX")]
    elif _is_us_open() and not _is_asx_open():
        tickers = [t for t in tickers if not t.endswith(".AX")]
    # If both open (overlap) or force-scan: use all

    log.info(f"Scanning {len(tickers)} tickers in batches of {batch_size}…")
    signals = []
    batches  = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for idx, batch in enumerate(batches):
        data = _download_batch(batch)
        for ticker, df in data.items():
            sig = _score_signal(df, ticker)
            if sig:
                signals.append(sig)
                log.info(f"  SIGNAL: {ticker}  score={sig['score']}  prob={sig['prob']:.2f}")
        if progress_cb:
            progress_cb((idx + 1) * batch_size, len(tickers))

    signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
    log.info(f"Scan complete — {len(signals)} signals from {len(tickers)} tickers")
    return signals


# ── Background scanner thread ──────────────────────────────────────────────────

class ContinuousScanner:
    """
    Runs scan_all() on a configurable interval during market hours.
    Caches results; dashboard reads from cache.
    """

    def __init__(self):
        self._thread:       threading.Thread | None = None
        self._stop          = threading.Event()
        self._lock          = threading.Lock()
        self._signals:      list[dict] = []
        self._last_scan:    datetime | None = None
        self._scan_count:   int = 0
        self._scanning:     bool = False
        self._progress:     tuple[int, int] = (0, 0)    # (done, total)
        self._status:       str = "IDLE"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ContinuousScanner"
        )
        self._thread.start()
        log.info("ContinuousScanner started")

    def stop(self):
        self._stop.set()
        self._status = "STOPPED"
        log.info("ContinuousScanner stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def signals(self) -> list[dict]:
        with self._lock:
            return list(self._signals)

    @property
    def last_scan(self) -> datetime | None:
        return self._last_scan

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def progress(self) -> tuple[int, int]:
        return self._progress

    @property
    def status(self) -> str:
        return self._status

    def force_scan(self):
        """Trigger an immediate scan from outside the thread."""
        t = threading.Thread(target=self._do_scan, daemon=True, name="ForceScan")
        t.start()

    def _loop(self):
        # Run an immediate scan on start
        self._do_scan()

        while not self._stop.is_set():
            interval_mins = int(cfg.get("scan_interval_mins") or 15)
            interval_secs = max(interval_mins * 60, 60)

            if market_is_open():
                self._status = f"Waiting {interval_mins}m for next scan…"
                self._stop.wait(interval_secs)
                if not self._stop.is_set():
                    self._do_scan()
            else:
                wait = min(next_open_seconds(), 1800)
                self._status = f"Market closed — sleeping {wait//60}m"
                self._stop.wait(wait)

    def _do_scan(self):
        self._scanning = True
        self._status   = "SCANNING…"
        self._progress = (0, len(get_all_active_tickers()))
        log.info("Starting scan cycle")

        def _progress_cb(done, total):
            self._progress = (done, total)

        try:
            results = scan_all(batch_size=50, progress_cb=_progress_cb)
            with self._lock:
                self._signals   = results
            self._last_scan  = datetime.utcnow()
            self._scan_count += 1
            self._status      = f"Done — {len(results)} signals found"
        except Exception as e:
            log.error(f"Scan error: {e}")
            db.log_error("ContinuousScanner", str(e))
            self._status = f"Error: {e}"
        finally:
            self._scanning = False
