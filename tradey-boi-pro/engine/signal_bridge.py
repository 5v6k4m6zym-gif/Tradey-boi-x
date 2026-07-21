"""
Signal bridge — merges Pro's ranked scanner output with optional
Tradey Boi X signal_log.json as a secondary reference source.

Primary source : TieredMonitor (ranked ELITE/STRONG BUY signals)
Secondary source: X signal_log.json (GH Actions daily output — optional bonus)

Key upgrade from v1:
  - Filters on composite_score (multi-factor, 0-10) not just raw score
  - Only ELITE and STRONG BUY tiers are execution candidates
  - X signals are imported at BUY tier and only promoted if composite > threshold
  - Regime veto: signals from a BEAR-regime market are blocked at execution
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

# Only these tiers are actioned by the executor
ACTIONABLE_TIERS = {"ELITE", "STRONG BUY"}


# ── Already-handled tickers ────────────────────────────────────────────────────

def _already_handled() -> set[str]:
    open_tickers   = {p["ticker"] for p in db.open_positions()}
    recent_trades  = {
        t["ticker"] for t in db.all_trades(limit=200)
        if t.get("entry_date", "")[:10] >= (
            datetime.utcnow() - timedelta(days=2)
        ).strftime("%Y-%m-%d")
    }
    return open_tickers | recent_trades


# ── Main bridge ────────────────────────────────────────────────────────────────

def get_pending_signals(
    scanner_signals: list[dict] | None = None,
    lookback_hours:  int = 48,
) -> list[dict]:
    """
    Return actionable signals (ELITE / STRONG BUY only), deduplicated, sorted by
    composite_score desc then ai_confidence desc.

    scanner_signals: pre-filtered actionable list from TieredMonitor.actionable_signals
    """
    min_prob        = float(cfg.get("min_prob")        or 0.53)
    min_score       = int(cfg.get("min_score")         or 7)
    min_composite   = float(cfg.get("min_composite")   or 7.0)
    handled         = _already_handled()

    combined: dict[str, dict] = {}

    # ── 1. Pro TieredMonitor output (primary, already ranked) ────────────────
    for sig in (scanner_signals or []):
        ticker = sig.get("ticker")
        if not ticker or ticker in handled:
            continue
        if sig.get("tier") not in ACTIONABLE_TIERS:
            continue
        # Regime veto
        if sig.get("regime_alignment") == "BEAR":
            log.debug(f"Regime veto: {ticker} skipped (BEAR)")
            continue
        # Quality gates — use original model prob (ai_confidence is reduced by ranker)
        composite = float(sig.get("composite_score", sig.get("score", 0)))
        raw_prob  = float(sig.get("prob", sig.get("ai_confidence", 0)))
        if composite < min_composite:
            continue
        if raw_prob < min_prob:
            continue
        # Keep highest composite if duplicate
        if ticker not in combined or composite > float(combined[ticker].get("composite_score", 0)):
            combined[ticker] = sig

    # ── 2. Tradey Boi X signal_log.json (secondary reference) ────────────────
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
            # X signals: only pass-through tier STRONG BUY or ELITE from X
            if rec.get("tier") not in ("STRONG BUY", "ELITE"):
                continue
            if rec.get("resolved"):
                continue
            # Quality gates — use X's raw score since no composite available
            x_prob  = float(rec.get("prob",  0))
            x_score = float(rec.get("score", 0))
            if x_prob < min_prob or x_score < min_score:
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

            # Build a compatible signal dict from X data
            sig = {
                "ticker":          ticker,
                "entry_price":     float(rec.get("entry_price", 0)),
                "stop_price":      float(rec.get("stop_price",  0)),
                "target_price":    float(rec.get("target_price", 0)),
                "atr_pct":         float(rec.get("atr_pct",  2.0)),
                "prob":            x_prob,
                "score":           x_score,
                "ai_confidence":   x_prob,
                "composite_score": x_score,   # X doesn't have composite; use raw score
                "tier":            rec.get("tier", "STRONG BUY"),
                "signal_date":     sig_date,
                "exchange":        "ASX" if ticker.endswith(".AX") else "SMART",
                "currency":        "AUD" if ticker.endswith(".AX") else "USD",
                "source":          "tradey_boi_x",
                "regime_alignment": "UNKNOWN",
            }
            # Pro scanner takes precedence if it already has this ticker
            if ticker not in combined:
                combined[ticker] = sig

    # Sort by composite_score desc, then ai_confidence
    pending = sorted(
        combined.values(),
        key=lambda s: (
            float(s.get("composite_score", s.get("score", 0))),
            float(s.get("ai_confidence",   s.get("prob",  0))),
        ),
        reverse=True,
    )
    return pending


def _x_signal_log_path() -> Path | None:
    custom = cfg.get("signal_log_path")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    default = X_DIR / "signal_log.json"
    return default if default.exists() else None
