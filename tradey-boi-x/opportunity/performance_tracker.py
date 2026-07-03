"""
Performance Tracker — Trade Evaluation Layer (SAFE upgrade)
=============================================================
Joins the trade-evaluator's decision log (`logs/trade_evaluations.jsonl` —
setup metrics + pass/fail decision, written by `trade_evaluator.py`) with the
existing resolved-outcome signal log (`signal_log.json`, written by
engine.py's `log_signal()`/`resolve_outcomes()`) so we know, for every trade
that was scored, what actually happened to it.

Purely a read-only reporting layer:
  - never modifies the prediction model, signal generation, or execution
  - never writes to signal_log.json
  - all failures are swallowed and reported as empty/insufficient results
    (never raises) so it can be called safely from any context, including
    the hot scan-loop path via the auto-tuner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opportunity.config import TRADE_EVAL_LOG_PATH

BASE_DIR        = Path(__file__).parent.parent
SIGNAL_LOG_FILE = BASE_DIR / "signal_log.json"

WIN_OUTCOMES = {"WIN", "HIT_TARGET", "TARGET_HIT"}


def _eval_log_path() -> Path:
    p = Path(TRADE_EVAL_LOG_PATH)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def _load_evaluations() -> list[dict]:
    """Load every line of the JSONL evaluation log. Never raises."""
    path = _eval_log_path()
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return records


def _load_signal_log() -> list[dict]:
    if not SIGNAL_LOG_FILE.exists():
        return []
    try:
        return json.loads(SIGNAL_LOG_FILE.read_text())
    except Exception:
        return []


def _r_multiple(entry: dict) -> float | None:
    """Convert a resolved signal_log entry's actual_pct into an R multiple
    (return / initial risk). Returns None when stop/entry data is missing
    or risk is degenerate."""
    entry_price = entry.get("entry_price")
    stop_price  = entry.get("stop_price")
    actual_pct  = entry.get("actual_pct")
    if entry_price in (None, 0) or stop_price is None or actual_pct is None:
        return None
    risk_pct = abs(entry_price - stop_price) / entry_price
    if risk_pct <= 0:
        return None
    return round(actual_pct / risk_pct, 4)


class PerformanceTracker:
    """
    Joins trade-evaluation decisions with their eventual outcomes (from the
    existing signal log) so setup-quality metrics (edge score, predictability,
    noise) can be related to what actually happened.

    Join key: (ticker, calendar date) — the evaluation log's `timestamp` and
    the signal log's `signal_date` are both written the same day a trade is
    proposed, so date-matching on ticker is a reliable-enough link without
    requiring any change to either existing log format.
    """

    def __init__(self):
        self._evaluations = _load_evaluations()
        self._signal_log  = _load_signal_log()

    # ── Joining ────────────────────────────────────────────────────────────
    def resolved_records(self) -> list[dict[str, Any]]:
        """
        Every evaluation-log record that can be matched to a *resolved*
        signal-log entry (outcome is not None), enriched with the outcome,
        actual_pct, and r_multiple. Unresolved/unmatched decisions are
        excluded — there's nothing to learn from them yet.
        """
        # Index resolved signal_log entries by (ticker, date) for O(1) lookup.
        by_key: dict[tuple[str, str], dict] = {}
        for e in self._signal_log:
            if e.get("outcome") is None:
                continue
            key = (e.get("ticker"), e.get("signal_date"))
            by_key.setdefault(key, e)

        joined: list[dict[str, Any]] = []
        for rec in self._evaluations:
            symbol = rec.get("symbol")
            ts = rec.get("timestamp") or ""
            date = ts[:10] if len(ts) >= 10 else None
            if not symbol or not date:
                continue
            resolved = by_key.get((symbol, date))
            if resolved is None:
                continue
            r_mult = _r_multiple(resolved)
            joined.append({
                **rec,
                "outcome":     resolved.get("outcome"),
                "actual_pct":  resolved.get("actual_pct"),
                "r_multiple":  r_mult,
                "regime":      resolved.get("regime"),
            })
        return joined

    # ── Rolling window stats ──────────────────────────────────────────────
    @staticmethod
    def _stats(records: list[dict]) -> dict[str, Any]:
        n = len(records)
        if n == 0:
            return {
                "trade_count": 0, "win_rate": 0.0,
                "avg_r": 0.0, "expectancy_r": 0.0,
            }
        wins = sum(1 for r in records if r.get("outcome") in WIN_OUTCOMES)
        r_values = [r["r_multiple"] for r in records if r.get("r_multiple") is not None]
        avg_r = round(sum(r_values) / len(r_values), 4) if r_values else 0.0
        win_rate = round(wins / n, 4)
        gains = [r for r in r_values if r > 0]
        losses = [r for r in r_values if r <= 0]
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy_r = round((win_rate * avg_gain) - ((1 - win_rate) * avg_loss), 4)
        return {
            "trade_count":  n,
            "win_rate":     win_rate,
            "avg_r":        avg_r,
            "expectancy_r": expectancy_r,
        }

    def rolling_stats(self, window: int = 100) -> dict[str, Any]:
        """Stats over the most recent `window` resolved+joined decisions."""
        records = self.resolved_records()
        recent = records[-window:] if window > 0 else records
        return self._stats(recent)

    def previous_window_stats(self, window: int = 100) -> dict[str, Any]:
        """Stats over the window immediately preceding the current rolling
        window — used by the auto-tuner to detect a trend (improving vs
        degrading) rather than judging a single window in isolation."""
        records = self.resolved_records()
        if len(records) <= window:
            return self._stats([])
        prior = records[-(window * 2):-window]
        return self._stats(prior)

    def regime_buckets(self) -> dict[str, dict[str, Any]]:
        """Optional grouping by market regime, when the joined signal_log
        entries carry a `regime` field (added by the T010 dashboard work).
        Returns {} when no records carry regime data."""
        records = self.resolved_records()
        buckets: dict[str, list[dict]] = {}
        for r in records:
            regime = r.get("regime")
            if not regime:
                continue
            buckets.setdefault(regime, []).append(r)
        return {regime: self._stats(recs) for regime, recs in buckets.items()}

    def passed_vs_rejected(self) -> dict[str, dict[str, Any]]:
        """Compare outcomes of trades the filter passed vs would have
        rejected — the clearest signal of whether the filter is adding
        value (only meaningful in shadow mode, where rejected trades still
        get logged with their would-be outcome)."""
        records = self.resolved_records()
        passed   = [r for r in records if r.get("passed")]
        rejected = [r for r in records if not r.get("passed")]
        return {
            "passed":   self._stats(passed),
            "rejected": self._stats(rejected),
        }
