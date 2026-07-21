"""
Signal bridge — pulls actionable signals from Tradey Boi X into Pro.

Two modes:
  1. Read-from-log: parse Tradey Boi X signal_log.json (already sent by X)
  2. Live-scan:     import X engine directly and evaluate tickers in real time

Mode 1 is the default (safest, no duplication).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import config.settings as cfg
import db.database as db

log = logging.getLogger("SignalBridge")

X_DIR = Path(__file__).parent.parent.parent / "tradey-boi-x"


def _signal_log_path() -> Path:
    custom = cfg.get("signal_log_path")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    default = X_DIR / "signal_log.json"
    return default


def get_pending_signals(lookback_hours: int = 48) -> list[dict]:
    """
    Return STRONG BUY signals from Tradey Boi X that:
      - Were emitted within the last `lookback_hours`
      - Pass min_prob and min_score filters
      - Have NOT already been traded by Pro (checked against open positions + trades)
    """
    log_path = _signal_log_path()
    if not log_path.exists():
        log.warning(f"Signal log not found: {log_path}")
        return []

    try:
        lines = log_path.read_text().splitlines()
        raw   = [json.loads(l) for l in lines if l.strip()]
    except Exception as e:
        log.error(f"Failed to read signal log: {e}")
        return []

    min_prob  = cfg.get("min_prob")  or 0.53
    min_score = cfg.get("min_score") or 7
    cutoff    = datetime.utcnow() - timedelta(hours=lookback_hours)

    # Tickers already in open positions or recently traded
    open_tickers = {p["ticker"] for p in db.open_positions()}
    recent_trades = {
        t["ticker"] for t in db.all_trades(limit=200)
        if t.get("entry_date", "")[:10] >= (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    }
    already_handled = open_tickers | recent_trades

    pending = []
    for rec in raw:
        # Only unresolved STRONG BUY alerts
        if rec.get("tier") not in ("STRONG BUY", "ELITE"):
            continue
        if rec.get("resolved"):
            continue
        if rec.get("ticker") in already_handled:
            continue

        # Quality gates
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
                    if sig_dt < cutoff.replace(hour=0, minute=0):
                        continue
                except ValueError:
                    pass

        pending.append({
            "ticker":      rec.get("ticker"),
            "entry_price": float(rec.get("entry_price", 0)),
            "atr_pct":     float(rec.get("atr_pct", 2.0)),
            "prob":        float(rec.get("prob", 0)),
            "score":       float(rec.get("score", 0)),
            "tier":        rec.get("tier"),
            "signal_date": sig_date,
            "exchange":    "ASX" if rec.get("ticker", "").endswith(".AX") else "SMART",
            "currency":    "AUD" if rec.get("ticker", "").endswith(".AX") else "USD",
            "source":      "signal_log",
        })

    pending.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
    return pending


def format_signal_display(sig: dict) -> str:
    return (
        f"{sig['ticker']}  "
        f"score={sig['score']:.0f}  prob={sig['prob']*100:.0f}%  "
        f"entry=${sig['entry_price']:.3f}  "
        f"atr={sig['atr_pct']:.1f}%"
    )
