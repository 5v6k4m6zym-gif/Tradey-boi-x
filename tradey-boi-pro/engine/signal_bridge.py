"""
Signal bridge — combines Pro's own live scanner results with optional
Tradey Boi X signal_log.json as a secondary source.

Primary source: ContinuousScanner (local yfinance scan, runs every 15–30 min)
Secondary source: X signal_log.json (GH Actions daily output — optional bonus)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import config.settings as cfg
import db.database as db

log = logging.getLogger("SignalBridge")

X_DIR = Path(__file__).parent.parent.parent / "tradey-boi-x"


# ── Already-handled tickers ────────────────────────────────────────────────────

def _already_handled() -> set[str]:
    open_tickers = {p["ticker"] for p in db.open_positions()}
    recent_trades = {
        t["ticker"] for t in db.all_trades(limit=200)
        if t.get("entry_date", "")[:10] >= (
            datetime.utcnow() - timedelta(days=2)
        ).strftime("%Y-%m-%d")
    }
    return open_tickers | recent_trades


# ── Merge + filter ─────────────────────────────────────────────────────────────

def get_pending_signals(
    scanner_signals: list[dict] | None = None,
    lookback_hours: int = 48,
) -> list[dict]:
    """
    Return actionable signals from both sources, deduplicated, quality-filtered,
    sorted by score desc.

    scanner_signals: pass the ContinuousScanner.signals list directly.
                     If None, only X log is used.
    """
    min_prob  = float(cfg.get("min_prob")  or 0.53)
    min_score = int(cfg.get("min_score")   or 7)
    handled   = _already_handled()

    combined: dict[str, dict] = {}   # ticker → best signal

    # ── 1. Pro local scanner (primary) ──────────────────────────────────────
    for sig in (scanner_signals or []):
        ticker = sig.get("ticker")
        if not ticker or ticker in handled:
            continue
        if float(sig.get("prob", 0)) < min_prob:
            continue
        if float(sig.get("score", 0)) < min_score:
            continue
        # Keep highest score if duplicate
        if ticker not in combined or sig["score"] > combined[ticker]["score"]:
            combined[ticker] = sig

    # ── 2. Tradey Boi X signal_log.json (secondary) ──────────────────────────
    log_path = _x_signal_log_path()
    if log_path and log_path.exists():
        try:
            lines = log_path.read_text().splitlines()
            raw   = [json.loads(l) for l in lines if l.strip()]
        except Exception as e:
            log.warning(f"Failed to read X signal log: {e}")
            raw = []

        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        for rec in raw:
            ticker = rec.get("ticker")
            if not ticker or ticker in handled:
                continue
            if rec.get("tier") not in ("STRONG BUY", "ELITE"):
                continue
            if rec.get("resolved"):
                continue
            if float(rec.get("prob", 0)) < min_prob:
                continue
            if float(rec.get("score", 0)) < min_score:
                continue
            # Recency check
            sig_date = rec.get("date") or rec.get("signal_date") or ""
            if sig_date:
                try:
                    sig_dt = datetime.strptime(sig_date[:16], "%Y-%m-%d %H:%M")
                    if sig_dt < cutoff:
                        continue
                except ValueError:
                    try:
                        sig_dt = datetime.strptime(sig_date[:10], "%Y-%m-%d")
                        if sig_dt.date() < cutoff.date():
                            continue
                    except ValueError:
                        pass

            sig = {
                "ticker":      ticker,
                "entry_price": float(rec.get("entry_price", 0)),
                "atr_pct":     float(rec.get("atr_pct", 2.0)),
                "prob":        float(rec.get("prob", 0)),
                "score":       float(rec.get("score", 0)),
                "tier":        rec.get("tier", "STRONG BUY"),
                "signal_date": sig_date,
                "exchange":    "ASX" if ticker.endswith(".AX") else "SMART",
                "currency":    "AUD" if ticker.endswith(".AX") else "USD",
                "source":      "tradey_boi_x",
            }
            # Prefer Pro scanner signal if already seen; X provides a second vote
            if ticker not in combined or sig["score"] > combined[ticker]["score"]:
                combined[ticker] = sig

    pending = sorted(combined.values(), key=lambda s: (s["score"], s["prob"]), reverse=True)
    return pending


def _x_signal_log_path() -> Path | None:
    custom = cfg.get("signal_log_path")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    default = X_DIR / "signal_log.json"
    return default if default.exists() else None
