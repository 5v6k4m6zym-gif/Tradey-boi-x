"""
Full System Audit Suite — Backtesting, Forward Validation, Performance
Analytics, System Audit, and Bug Detection.
=========================================================================
A READ-ONLY, parallel observability layer alongside the existing bot:

    Model -> Signal -> Execution
                 \\
              Audit Engine (this module)
                 \\
        Backtest + Forward Validator + Analytics + SystemAudit + BugDetector

HARD SAFETY RULES (per spec):
  - Never modifies, rewrites, or replaces the prediction model, signal
    generation logic, or execution engine.
  - Every component here is non-invasive: it only reads existing logs
    (signal_log.json, logs/trade_evaluations.jsonl, logs/adaptive_core_
    decisions.jsonl) and existing helpers (opportunity.backtester,
    opportunity.costs, opportunity.performance, opportunity.performance_
    tracker, opportunity.trade_evaluator) rather than duplicating them.
  - The system must never break or block live trading. Every public
    function/class method here is wrapped so that ANY internal exception
    is caught, logged, and a safe *empty* result is returned instead of
    raising.
  - `audit_trade(trade, market_data, outcome_data)` — the Part 8 wrapper —
    NEVER mutates `trade`, NEVER blocks execution, and its return value is
    purely informational (callers must not gate on it).

Feature flag: ENABLE_AUDIT_ENGINE (default False -> `audit_trade()` is a
complete no-op; the on-demand report functions below still work even when
the flag is off, mirroring the existing run_backtest()/run_challenger()
pattern, since they are invoked explicitly rather than from the hot scan
loop).
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from opportunity.config import (
    ENABLE_AUDIT_ENGINE,
    AUDIT_LOG_PATH,
    AUDIT_REPORTS_DIR,
    AUDIT_STATE_PATH,
    AUDIT_BACKTEST_MAX_HOLD_DAYS,
    AUDIT_REJECTION_RATE_SPIKE_DELTA,
    AUDIT_FREQUENCY_DROP_PCT,
    AUDIT_CALIBRATION_DRIFT_DELTA,
    AUDIT_MIN_TRADES_FOR_CHECKS,
    AUDIT_RECENT_WINDOW_TRADES,
    AUDIT_ROLLING_WINDOWS,
    AUDIT_EDGE_SCORE_BUCKETS,
)
from opportunity.costs import apply_cost, round_trip_cost_pct
from opportunity.backtester import (
    compute_metrics,
    _empty_metrics,
    _streaks,
    WIN_OUTCOMES,
    _load_log as _load_signal_log,
    _resolved as _resolved_signal_entries,
)
from opportunity.performance import calibration_buckets
from opportunity.performance_tracker import PerformanceTracker
from opportunity.trade_evaluator import TradeEvaluator

BASE_DIR      = Path(__file__).parent.parent
EVAL_LOG_FILE = BASE_DIR / "logs" / "trade_evaluations.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(default: Any = None):
    """Decorator: catch ANY exception, print a warning, and return `default`
    (or an empty-dict copy of it) instead of raising. This is the fail-safe
    backbone required across every component in this module."""
    def _wrap(fn: Callable) -> Callable:
        def _inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"  \u26a0\ufe0f  audit_engine.{fn.__name__}: failed safely ({e})")
                return dict(default) if isinstance(default, dict) else default
        _inner.__name__ = fn.__name__
        return _inner
    return _wrap


# ══════════════════════════════════════════════════════════════════════════
# PART 1 — BacktestEngine (bar-by-bar historical simulation)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeSimResult:
    ticker: str
    outcome: str            # "WIN" | "LOSS" | "TIME_EXIT_WIN" | "TIME_EXIT_LOSS"
    pnl_r: float
    mfe: float               # max favourable excursion, as a fraction (e.g. 0.08 = +8%)
    mae: float               # max adverse excursion, as a fraction (negative or 0)
    duration_days: int
    setup_type: str | None = None

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker, "outcome": self.outcome,
            "pnl_r": self.pnl_r, "mfe": self.mfe, "mae": self.mae,
            "duration_days": self.duration_days, "setup_type": self.setup_type,
        }


class BacktestEngine:
    """
    Historical simulation engine, independent of engine.py's own
    `resolve_outcomes()` bookkeeping. Given an entry/stop/target and the
    OHLCV bars that follow the entry, replays bar-by-bar to determine the
    realistic outcome (whichever of stop/target is touched first, or a
    time-based exit), tracking MFE/MAE along the way and applying the same
    slippage/commission/spread cost model used elsewhere (opportunity.costs).

    Never fetches data itself — callers supply `future_bars` (OHLCV rows
    strictly after the entry date) so this stays fully offline/testable and
    never risks an extra network call from inside the audit layer.
    """

    def __init__(self, max_hold_days: int = AUDIT_BACKTEST_MAX_HOLD_DAYS,
                 apply_slippage: bool = True):
        self.max_hold_days = max_hold_days
        self.apply_slippage = apply_slippage

    def simulate_trade(
        self,
        ticker: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        future_bars: pd.DataFrame,
        direction: str = "LONG",
        setup_type: str | None = None,
    ) -> TradeSimResult | None:
        """Replay `future_bars` (High/Low/Close) bar by bar. Returns None
        when there isn't enough data to simulate anything."""
        if future_bars is None or future_bars.empty or entry <= 0:
            return None

        bars = future_bars.head(self.max_hold_days)
        if bars.empty:
            return None

        is_long = direction.upper() != "SHORT"
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return None

        mfe = 0.0   # best (favourable) pct move seen so far
        mae = 0.0   # worst (adverse) pct move seen so far
        exit_pct: float | None = None
        outcome: str | None = None
        duration = 0

        for i, (_, bar) in enumerate(bars.iterrows(), start=1):
            duration = i
            high = float(bar["High"])
            low  = float(bar["Low"])

            if is_long:
                fav_pct = (high - entry) / entry
                adv_pct = (low - entry) / entry
            else:
                fav_pct = (entry - low) / entry
                adv_pct = (entry - high) / entry

            mfe = max(mfe, fav_pct)
            mae = min(mae, adv_pct)

            hit_target = (high >= take_profit) if is_long else (low <= take_profit)
            hit_stop   = (low  <= stop_loss)   if is_long else (high >= stop_loss)

            # Conservative: if both touched in the same bar, assume stop hit first.
            if hit_stop:
                exit_pct = (stop_loss - entry) / entry if is_long else (entry - stop_loss) / entry
                outcome = "LOSS"
                break
            if hit_target:
                exit_pct = (take_profit - entry) / entry if is_long else (entry - take_profit) / entry
                outcome = "WIN"
                break

        if exit_pct is None:
            # Time-based exit at the last available close.
            last_close = float(bars.iloc[-1]["Close"])
            exit_pct = (last_close - entry) / entry if is_long else (entry - last_close) / entry
            outcome = "TIME_EXIT_WIN" if exit_pct >= 0 else "TIME_EXIT_LOSS"

        net_pct = apply_cost(exit_pct, ticker) if self.apply_slippage else exit_pct
        risk_pct = risk / entry
        pnl_r = round(net_pct / risk_pct, 4) if risk_pct > 0 else 0.0

        return TradeSimResult(
            ticker=ticker, outcome=outcome, pnl_r=pnl_r,
            mfe=round(mfe, 4), mae=round(mae, 4),
            duration_days=duration, setup_type=setup_type,
        )

    @_safe(default=[])
    def run_batch(
        self,
        trades: list[dict[str, Any]],
        price_data: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """
        trades: list of dicts with ticker/entry/stop_loss/take_profit
                (+ optional direction/setup_type).
        price_data: {ticker: DataFrame of bars strictly AFTER entry}.
        Returns a list of TradeSimResult dicts (skips trades with no
        matching/insufficient price data — never raises).
        """
        results = []
        for t in trades:
            ticker = t.get("ticker") or t.get("symbol")
            bars = price_data.get(ticker)
            r = self.simulate_trade(
                ticker=ticker,
                entry=float(t.get("entry", 0) or 0),
                stop_loss=float(t.get("stop_loss", 0) or 0),
                take_profit=float(t.get("take_profit", 0) or 0),
                future_bars=bars,
                direction=t.get("direction", "LONG"),
                setup_type=t.get("setup_type"),
            )
            if r is not None:
                results.append(r.as_dict())
        return results

    @_safe(default={})
    def rolling_window_backtest(
        self,
        trades: list[dict[str, Any]],
        price_data: dict[str, pd.DataFrame],
        window_size: int = 50,
    ) -> dict[str, Any]:
        """Batch-simulate `trades` then slice the resulting per-trade R
        multiples into rolling windows of `window_size`, reporting summary
        stats per window (win rate / avg R) so drift across time is visible."""
        sims = self.run_batch(trades, price_data)
        if not sims:
            return {"windows": []}

        windows = []
        for start in range(0, len(sims), window_size):
            chunk = sims[start:start + window_size]
            wins = sum(1 for s in chunk if s["outcome"] in ("WIN", "TIME_EXIT_WIN"))
            r_vals = [s["pnl_r"] for s in chunk]
            windows.append({
                "window_start": start, "window_end": start + len(chunk),
                "trade_count": len(chunk),
                "win_rate": round(wins / len(chunk), 4) if chunk else 0.0,
                "avg_pnl_r": round(statistics.mean(r_vals), 4) if r_vals else 0.0,
            })
        return {"windows": windows, "total_trades": len(sims)}


# ══════════════════════════════════════════════════════════════════════════
# PART 2 — ForwardValidator (shadow-mode observation)
# ══════════════════════════════════════════════════════════════════════════

class ForwardValidator:
    """
    Purely observational: joins the existing trade-evaluation decision log
    (predicted probability / edge score, written by trade_evaluator.py) with
    the existing resolved-outcome signal log (written by engine.py) via
    PerformanceTracker — reused rather than re-implemented — and adds a
    "deviation from expectation" figure per trade.

    NO live trading changes of any kind. Read-only reporting.
    """

    def __init__(self):
        self._tracker = PerformanceTracker()

    @staticmethod
    def _deviation(record: dict[str, Any]) -> float | None:
        """actual R multiple minus the model's expected edge (proxied by
        (probability - 0.5) * 2, mapped onto the same R-ish scale via
        risk_reward) — a simple, explainable "did reality match the
        prediction" figure. Returns None when insufficient data."""
        r_multiple = record.get("r_multiple")
        probability = record.get("probability")
        risk_reward = record.get("risk_reward")
        if r_multiple is None or probability is None or risk_reward is None:
            return None
        expected_r = (float(probability) - 0.5) * 2.0 * float(risk_reward)
        return round(float(r_multiple) - expected_r, 4)

    @_safe(default=[])
    def validation_records(self) -> list[dict[str, Any]]:
        """Every resolved, joined decision enriched with `deviation`."""
        records = self._tracker.resolved_records()
        out = []
        for r in records:
            out.append({**r, "deviation": self._deviation(r)})
        return out

    @_safe(default={})
    def summary(self) -> dict[str, Any]:
        """Aggregate view: mean/absolute deviation, and how often the model
        was over- vs under-confident relative to what happened."""
        records = self.validation_records()
        deviations = [r["deviation"] for r in records if r.get("deviation") is not None]
        if not deviations:
            return {"trade_count": len(records), "sufficient_data": False}

        overconfident = sum(1 for d in deviations if d < -0.25)
        underconfident = sum(1 for d in deviations if d > 0.25)
        return {
            "trade_count":        len(records),
            "sufficient_data":    len(deviations) >= AUDIT_MIN_TRADES_FOR_CHECKS,
            "mean_deviation":     round(statistics.mean(deviations), 4),
            "mean_abs_deviation": round(statistics.mean([abs(d) for d in deviations]), 4),
            "overconfident_pct":  round(overconfident / len(deviations), 4),
            "underconfident_pct": round(underconfident / len(deviations), 4),
        }


# ══════════════════════════════════════════════════════════════════════════
# PART 3 — PerformanceAnalytics (extends existing performance.py/backtester.py)
# ══════════════════════════════════════════════════════════════════════════

class PerformanceAnalytics:
    """
    Reuses opportunity.backtester.compute_metrics() and opportunity.
    performance_tracker.PerformanceTracker rather than re-deriving win rate /
    expectancy / profit factor / streaks — this class only ADDS the analysis
    cuts the spec asks for that don't already exist: rolling windows of
    50/100/200 trades, a full drawdown curve (not just the scalar max), setup-
    type expectancy, and edge-score bucket segmentation.
    """

    def __init__(self):
        self._tracker = PerformanceTracker()

    @_safe(default=[])
    def rolling_windows(self, windows: tuple[int, ...] = AUDIT_ROLLING_WINDOWS) -> list[dict]:
        entries = _resolved_signal_entries(_load_signal_log())
        out = []
        for w in windows:
            recent = entries[-w:] if w > 0 else entries
            m = compute_metrics(recent)
            out.append({"window": w, **m})
        return out

    @_safe(default=[])
    def regime_breakdown(self) -> list[dict]:
        """Win-rate/expectancy per regime, reusing PerformanceTracker's
        existing regime_buckets() join (regime comes from the signal_log
        entries written by the T010 dashboard work / adaptive core)."""
        buckets = self._tracker.regime_buckets()
        return [{"regime": k, **v} for k, v in sorted(buckets.items())]

    @_safe(default=[])
    def setup_type_expectancy(self) -> list[dict]:
        """Groups resolved signal_log entries by their `signal` label (the
        closest existing proxy for "setup type" — e.g. BREAKOUT/PULLBACK —
        already written by engine.log_signal()) and computes expectancy per
        group via the same compute_metrics() used everywhere else."""
        entries = _resolved_signal_entries(_load_signal_log())
        groups: dict[str, list[dict]] = {}
        for e in entries:
            label = e.get("signal") or e.get("setup_type") or "UNKNOWN"
            groups.setdefault(label, []).append(e)
        out = []
        for label, trades in sorted(groups.items()):
            m = compute_metrics(trades)
            out.append({"setup_type": label, **m})
        out.sort(key=lambda x: x.get("expectancy_r", 0), reverse=True)
        return out

    @_safe(default=[])
    def edge_score_buckets(self) -> list[dict]:
        """Signal-quality segmentation: buckets resolved+joined trade-
        evaluator decisions (which carry `edge_score`) into quality tiers and
        reports the win rate / expectancy realised in each — the clearest
        signal of whether the edge score itself is predictive."""
        records = self._tracker.resolved_records()
        out = []
        for label, lo, hi in AUDIT_EDGE_SCORE_BUCKETS:
            bucket = [r for r in records
                      if r.get("edge_score") is not None and lo <= float(r["edge_score"]) < hi]
            n = len(bucket)
            if n == 0:
                out.append({"bucket": label, "count": 0, "win_rate": None, "expectancy_r": None})
                continue
            wins = sum(1 for r in bucket if r.get("outcome") in WIN_OUTCOMES)
            r_vals = [r["r_multiple"] for r in bucket if r.get("r_multiple") is not None]
            out.append({
                "bucket": label, "count": n,
                "win_rate": round(wins / n, 4),
                "avg_r": round(statistics.mean(r_vals), 4) if r_vals else 0.0,
            })
        return out

    @_safe(default={"curve": [], "max_drawdown_pct": 0.0})
    def drawdown_curve(self) -> dict[str, Any]:
        """Full equity curve (not just the scalar max_drawdown already
        returned by compute_metrics), net of realistic trading costs, so the
        shape of drawdowns over time can be inspected."""
        entries = _resolved_signal_entries(_load_signal_log())
        equity = 1.0
        peak = 1.0
        curve = []
        for e in entries:
            net = apply_cost(e.get("actual_pct", 0.0) or 0.0, e.get("ticker", ""))
            equity *= (1 + net)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            curve.append({
                "signal_date": e.get("signal_date", ""),
                "equity": round(equity, 4),
                "drawdown_pct": round(dd * 100, 2),
            })
        max_dd = max((c["drawdown_pct"] for c in curve), default=0.0)
        return {"curve": curve, "max_drawdown_pct": max_dd}

    @_safe(default={})
    def full_report(self) -> dict[str, Any]:
        entries = _resolved_signal_entries(_load_signal_log())
        return {
            "generated_at":         _now_iso(),
            "overall":              compute_metrics(entries),
            "rolling_windows":      self.rolling_windows(),
            "regime_breakdown":     self.regime_breakdown(),
            "setup_type_expectancy": self.setup_type_expectancy(),
            "edge_score_buckets":   self.edge_score_buckets(),
            "drawdown":             self.drawdown_curve(),
        }


# ══════════════════════════════════════════════════════════════════════════
# PART 4 — SystemAudit (anomaly detection; advisory only, never blocks)
# ══════════════════════════════════════════════════════════════════════════

def _read_eval_log() -> list[dict]:
    if not EVAL_LOG_FILE.exists():
        return []
    records = []
    try:
        with open(EVAL_LOG_FILE) as f:
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


def _load_audit_state() -> dict:
    p = Path(AUDIT_STATE_PATH)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_audit_state(state: dict) -> None:
    try:
        p = Path(AUDIT_STATE_PATH)
        if not p.is_absolute():
            p = BASE_DIR / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"  \u26a0\ufe0f  audit_engine: failed to persist audit state ({e})")


@dataclass
class AuditWarning:
    category: str          # "LOGIC" | "PERFORMANCE" | "DATA"
    code: str
    message: str
    likely_cause: str

    def as_dict(self) -> dict:
        return {"category": self.category, "code": self.code,
                "message": self.message, "likely_cause": self.likely_cause}


class SystemAudit:
    """
    Detects logic/performance/data anomalies WITHOUT ever stopping the
    system. Every check degrades gracefully to "no warning" when there
    isn't enough data to draw a conclusion — false alarms on a cold start
    are worse than silence.
    """

    def __init__(self):
        self._tracker = PerformanceTracker()

    # ── A) Logic anomalies ─────────────────────────────────────────────
    @_safe(default=None)
    def check_rejection_rate_spike(self) -> AuditWarning | None:
        evals = _read_eval_log()
        if len(evals) < AUDIT_MIN_TRADES_FOR_CHECKS * 2:
            return None
        recent   = evals[-AUDIT_RECENT_WINDOW_TRADES:]
        baseline = evals[:-AUDIT_RECENT_WINDOW_TRADES] or evals
        recent_rate   = 1 - (sum(1 for e in recent if e.get("passed")) / len(recent))
        baseline_rate = 1 - (sum(1 for e in baseline if e.get("passed")) / len(baseline))
        delta = recent_rate - baseline_rate
        if delta > AUDIT_REJECTION_RATE_SPIKE_DELTA:
            return AuditWarning(
                "LOGIC", "REJECTION_RATE_SPIKE",
                f"Recent rejection rate {recent_rate*100:.0f}% vs baseline "
                f"{baseline_rate*100:.0f}% (+{delta*100:.0f}pp).",
                "Thresholds may have tightened, or market conditions shifted "
                "such that fewer setups clear the existing bars — check "
                "TRADE_EVAL_THRESHOLDS and recent regime.",
            )
        return None

    @_safe(default=None)
    def check_frequency_drop(self) -> AuditWarning | None:
        entries = _resolved_signal_entries(_load_signal_log())
        if len(entries) < AUDIT_MIN_TRADES_FOR_CHECKS:
            return None
        dated = [e for e in entries if e.get("signal_date")]
        if len(dated) < AUDIT_MIN_TRADES_FOR_CHECKS:
            return None
        dated.sort(key=lambda e: e["signal_date"])
        midpoint = len(dated) // 2
        older, recent = dated[:midpoint], dated[midpoint:]
        older_span  = _date_span_days(older)  or 1
        recent_span = _date_span_days(recent) or 1
        older_rate  = len(older)  / older_span
        recent_rate = len(recent) / recent_span
        if older_rate > 0 and recent_rate < older_rate * (1 - AUDIT_FREQUENCY_DROP_PCT):
            return AuditWarning(
                "LOGIC", "SIGNAL_FREQUENCY_DROP",
                f"Signal frequency dropped from {older_rate:.2f}/day to "
                f"{recent_rate:.2f}/day.",
                "Could indicate a data-feed issue, an overly strict filter, "
                "or a genuinely quieter market — cross-check with health.py "
                "scan-duration/error logs before assuming a bug.",
            )
        return None

    @_safe(default=None)
    def check_signal_structure(self) -> AuditWarning | None:
        entries = _load_signal_log()
        if not entries:
            return None
        required = ("ticker", "price", "signal_date")
        bad = [e for e in entries if any(e.get(k) in (None, "") for k in required)]
        if bad:
            return AuditWarning(
                "LOGIC", "INCONSISTENT_SIGNAL_STRUCTURE",
                f"{len(bad)} of {len(entries)} signal_log entries are missing "
                f"one of {required}.",
                "A recent change to log_signal()'s call sites may have "
                "dropped a field — check engine.log_signal() callers.",
            )
        return None

    # ── B) Performance anomalies ───────────────────────────────────────
    @_safe(default=None)
    def check_regime_mismatch_degradation(self) -> AuditWarning | None:
        buckets = self._tracker.regime_buckets()
        if not buckets:
            return None
        overall = self._tracker.rolling_stats(window=10_000)
        overall_exp = overall.get("expectancy_r", 0.0)
        worst = None
        for regime, stats in buckets.items():
            if stats.get("trade_count", 0) < max(AUDIT_MIN_TRADES_FOR_CHECKS // 2, 5):
                continue
            gap = overall_exp - stats.get("expectancy_r", 0.0)
            if gap > 0.3 and (worst is None or gap > worst[1]):
                worst = (regime, gap, stats)
        if worst:
            regime, gap, stats = worst
            return AuditWarning(
                "PERFORMANCE", "REGIME_MISMATCH_DEGRADATION",
                f"Regime '{regime}' expectancy {stats.get('expectancy_r', 0):+.2f}R "
                f"is {gap:.2f}R below overall ({overall_exp:+.2f}R) over "
                f"{stats.get('trade_count', 0)} trades.",
                "The strategy may not be well-suited to this regime — "
                "consider tightening regime-specific thresholds (see "
                "adaptive_core.ADAPTIVE_REGIME_ADJUSTMENTS) rather than the "
                "global ones.",
            )
        return None

    @_safe(default=None)
    def check_calibration_drift(self) -> AuditWarning | None:
        entries = _resolved_signal_entries(_load_signal_log())
        if len(entries) < AUDIT_MIN_TRADES_FOR_CHECKS:
            return None
        current = calibration_buckets(entries)
        state = _load_audit_state()
        previous = state.get("last_calibration_buckets")

        drifted = []
        if previous:
            prev_by_label = {b["label"]: b for b in previous}
            for b in current:
                if b.get("actual_win_rate") is None:
                    continue
                p = prev_by_label.get(b["label"])
                if not p or p.get("actual_win_rate") is None:
                    continue
                delta = abs(b["actual_win_rate"] - p["actual_win_rate"])
                if delta > AUDIT_CALIBRATION_DRIFT_DELTA:
                    drifted.append((b["label"], p["actual_win_rate"], b["actual_win_rate"]))

        state["last_calibration_buckets"] = current
        state["last_calibration_check"]   = _now_iso()
        _save_audit_state(state)

        if drifted:
            label, prev_rate, cur_rate = drifted[0]
            return AuditWarning(
                "PERFORMANCE", "CALIBRATION_DRIFT",
                f"Confidence bucket '{label}' actual win rate moved from "
                f"{prev_rate*100:.0f}% to {cur_rate*100:.0f}%.",
                "The model's probability output may no longer map to the "
                "same real-world win rate — consider re-checking training "
                "data recency or the ConfidenceCalibrator's bucket mapping.",
            )
        return None

    # ── C) Data issues ──────────────────────────────────────────────────
    @_safe(default=[])
    def check_data_issues(self) -> list[AuditWarning]:
        warnings: list[AuditWarning] = []
        entries = _load_signal_log()
        numeric_fields = ("price", "score", "prob", "actual_pct")
        bad_count = 0
        for e in entries:
            for f_ in numeric_fields:
                v = e.get(f_)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        bad_count += 1
                except (TypeError, ValueError):
                    bad_count += 1
        if bad_count:
            warnings.append(AuditWarning(
                "DATA", "NAN_OR_INVALID_VALUES",
                f"{bad_count} NaN/invalid numeric field value(s) found across "
                f"signal_log.json numeric fields.",
                "Check the upstream calculation that produced these values — "
                "likely a division by zero or missing market data bar.",
            ))

        evals = _read_eval_log()
        malformed = [e for e in evals if "symbol" not in e or "edge_score" not in e]
        if malformed:
            warnings.append(AuditWarning(
                "DATA", "MALFORMED_EVALUATION_RECORD",
                f"{len(malformed)} trade_evaluations.jsonl record(s) missing "
                f"required fields (symbol/edge_score).",
                "trade_evaluator.log_trade_decision()'s record shape may have "
                "changed without updating downstream readers.",
            ))
        return warnings

    @_safe(default={})
    def run_audit(self) -> dict[str, Any]:
        """Runs every check and aggregates warnings. NEVER raises and NEVER
        stops the system — this is purely advisory logging."""
        warnings: list[dict] = []
        for w in (
            self.check_rejection_rate_spike(),
            self.check_frequency_drop(),
            self.check_signal_structure(),
            self.check_regime_mismatch_degradation(),
            self.check_calibration_drift(),
        ):
            if w is not None:
                warnings.append(w.as_dict())
        for w in self.check_data_issues():
            warnings.append(w.as_dict())

        report = {
            "generated_at": _now_iso(),
            "warning_count": len(warnings),
            "warnings": warnings,
        }
        if warnings:
            print(f"  \U0001f6a8 SystemAudit: {len(warnings)} warning(s) detected "
                  f"(system continues running normally).")
        else:
            print("  \u2705 SystemAudit: no anomalies detected.")
        return report


def _date_span_days(entries: list[dict]) -> int:
    dates = sorted(e["signal_date"] for e in entries if e.get("signal_date"))
    if len(dates) < 2:
        return 0
    try:
        d0 = datetime.strptime(dates[0],  "%Y-%m-%d")
        d1 = datetime.strptime(dates[-1], "%Y-%m-%d")
        return max((d1 - d0).days, 1)
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════
# PART 5 — BugDetector (report + suggest only, never auto-applies)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BugReport:
    source: str
    issue: str
    suggested_fix: str

    def as_dict(self) -> dict:
        return {"source": self.source, "issue": self.issue,
                "suggested_fix": self.suggested_fix}


class BugDetector:
    """
    Scans existing logs for structural/runtime inconsistencies and suggests
    (but never applies) a fix. Complements SystemAudit's performance-focused
    checks with lower-level "is this record even well-formed" checks.
    """

    @_safe(default=[])
    def scan_trade_evaluations(self) -> list[BugReport]:
        bugs = []
        for rec in _read_eval_log():
            edge = rec.get("edge_score")
            if edge is not None and not (0.0 <= float(edge) <= 1.0):
                bugs.append(BugReport(
                    "logs/trade_evaluations.jsonl",
                    f"edge_score={edge} outside expected [0,1] range for "
                    f"{rec.get('symbol')}.",
                    "Check TradeEvaluator.compute_edge_score()'s component "
                    "clamping — a component may be feeding an out-of-range "
                    "input (e.g. probability > 1 or negative risk_reward).",
                ))
            rr = rec.get("risk_reward")
            if rr is not None and float(rr) < 0:
                bugs.append(BugReport(
                    "logs/trade_evaluations.jsonl",
                    f"negative risk_reward={rr} for {rec.get('symbol')}.",
                    "Likely an inverted stop/target pair passed into "
                    "TradeEvaluator.compute_risk_reward() — verify the "
                    "trade dict's stop_loss/take_profit assignment.",
                ))
        return bugs

    @_safe(default=[])
    def scan_signal_log(self) -> list[BugReport]:
        bugs = []
        for e in _load_signal_log():
            if e.get("outcome") is not None and e.get("actual_pct") is None:
                bugs.append(BugReport(
                    "signal_log.json",
                    f"{e.get('ticker')}: outcome={e.get('outcome')!r} set but "
                    f"actual_pct is missing.",
                    "Check engine.resolve_outcomes() — a resolved entry "
                    "should always populate actual_pct alongside outcome.",
                ))
        return bugs

    @_safe(default={})
    def run(self) -> dict[str, Any]:
        bugs = self.scan_trade_evaluations() + self.scan_signal_log()
        report = {
            "generated_at": _now_iso(),
            "bug_count": len(bugs),
            "bugs": [b.as_dict() for b in bugs],
        }
        if bugs:
            print(f"  \U0001f41b BugDetector: {len(bugs)} issue(s) found (report-only, "
                  f"no auto-fix applied).")
        return report


# ══════════════════════════════════════════════════════════════════════════
# PART 6 — Unified JSONL logging
# ══════════════════════════════════════════════════════════════════════════

def _audit_log_path() -> Path:
    p = Path(AUDIT_LOG_PATH)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def _log_audit_record(record: dict[str, Any]) -> None:
    """Append one JSONL line. Never raises — a logging failure must not be
    able to break the scan/trade flow."""
    try:
        path = _audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"  \u26a0\ufe0f  audit_engine: failed to log audit record ({e})")


# ══════════════════════════════════════════════════════════════════════════
# PART 8 — audit_trade() wrapper (single integration point)
# ══════════════════════════════════════════════════════════════════════════

_evaluator = TradeEvaluator()


def audit_trade(
    trade: dict[str, Any],
    market_data: pd.DataFrame,
    outcome_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    The single integration point for this whole layer, per spec Part 8.

    - Never blocks execution: the return value is informational only and
      callers MUST NOT use it to skip alerting/trading.
    - Never modifies `trade`: only reads from it.
    - Never interferes with the signal pipeline: this can be called (or not
      called, or fail) with zero effect on send_alert()/log_signal().
    - On ANY internal failure: logs the error and returns a safe empty
      result — never raises.

    When ENABLE_AUDIT_ENGINE is False, this is a strict no-op that returns
    an empty dict immediately (no log line written, no computation done).
    """
    if not ENABLE_AUDIT_ENGINE:
        return {}

    try:
        evaluation = _evaluator.evaluate(trade, market_data)

        regime_type = None
        try:
            from opportunity.adaptive_core import RegimeDetector
            regime_type = RegimeDetector().detect(market_data).regime
        except Exception:
            regime_type = None

        record: dict[str, Any] = {
            "timestamp":             _now_iso(),
            "symbol":                trade.get("ticker", trade.get("symbol")),
            "trade_id":              trade.get("trade_id"),
            "regime_type":           regime_type,
            "edge_score":            evaluation.edge_score,
            "predicted_probability": trade.get("probability", trade.get("prob")),
            "actual_outcome":        None,
            "pnl_r":                 None,
            "mfe":                   None,
            "mae":                   None,
            "pass_fail":             "PASS" if evaluation.passed else "FAIL",
            "reason":                "; ".join(evaluation.rejection_reasons) or None,
            "system_flags":          [],
        }

        if outcome_data:
            record["actual_outcome"] = outcome_data.get("outcome")
            record["pnl_r"]          = outcome_data.get("pnl_r")
            record["mfe"]            = outcome_data.get("mfe")
            record["mae"]            = outcome_data.get("mae")

        _log_audit_record(record)
        return {"logged": True, "pass_fail": record["pass_fail"]}

    except Exception as e:
        print(f"  \u26a0\ufe0f  audit_trade: failed safely for "
              f"{trade.get('ticker', trade.get('symbol', '?'))} ({e})")
        return {"logged": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# PART 7 — Comparison Engine (old baseline vs new filtered system)
# ══════════════════════════════════════════════════════════════════════════

@_safe(default={})
def generate_comparison_report() -> dict[str, Any]:
    """
    Compares:
      1. Old system baseline    — ALL resolved signal_log entries, i.e. the
         bot's raw signal output with no evaluator/adaptive filtering.
      2. New filtered system    — the subset of trade-evaluator/adaptive-
         core decisions that PASSED, reusing PerformanceTracker's existing
         passed_vs_rejected() join rather than recomputing it.
      3. Regime-specific deltas — via PerformanceTracker.regime_buckets().

    Entirely read-only; writes a JSON report to reports/audit/ (mirrors
    challenger.py's save pattern) and returns the same dict.
    """
    tracker = PerformanceTracker()
    all_entries = _resolved_signal_entries(_load_signal_log())
    baseline_metrics = compute_metrics(all_entries)

    pv = tracker.passed_vs_rejected()
    filtered_stats = pv.get("passed", {})

    def _delta(key: str) -> float:
        b = float(baseline_metrics.get(key, 0) or 0)
        n = float(filtered_stats.get(key, 0) or 0)
        return round(n - b, 4)

    regime_buckets = tracker.regime_buckets()

    which_better = "NEW_FILTERED_SYSTEM" if _delta("win_rate") >= 0 and \
        float(filtered_stats.get("expectancy_r", 0) or 0) >= baseline_metrics.get("expectancy_r", 0) \
        else "OLD_SYSTEM_BASELINE" if filtered_stats.get("trade_count", 0) > 0 else "INSUFFICIENT_DATA"

    report = {
        "generated_at": _now_iso(),
        "old_system_baseline": {
            "description": "All resolved signals, no evaluator/adaptive filtering.",
            "metrics": baseline_metrics,
        },
        "new_filtered_system": {
            "description": "Trades that passed the trade-evaluator/adaptive-core layers.",
            "metrics": filtered_stats,
        },
        "delta_win_rate":     _delta("win_rate"),
        "delta_expectancy_r": round(float(filtered_stats.get("expectancy_r", 0) or 0)
                                     - float(baseline_metrics.get("expectancy_r", 0) or 0), 4),
        "regime_breakdown":   {k: v for k, v in regime_buckets.items()},
        "which_performs_better": which_better,
    }

    try:
        reports_dir = Path(AUDIT_REPORTS_DIR)
        if not reports_dir.is_absolute():
            reports_dir = BASE_DIR / reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        fname = reports_dir / f"comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        fname.write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(fname)
    except Exception as e:
        print(f"  \u26a0\ufe0f  audit_engine: failed to save comparison report ({e})")

    return report


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

def run_system_audit() -> dict[str, Any] | None:
    if not ENABLE_AUDIT_ENGINE:
        return None
    return SystemAudit().run_audit()


def run_bug_detector() -> dict[str, Any] | None:
    if not ENABLE_AUDIT_ENGINE:
        return None
    return BugDetector().run()


def run_performance_analytics_v2() -> dict[str, Any] | None:
    if not ENABLE_AUDIT_ENGINE:
        return None
    return PerformanceAnalytics().full_report()


def run_forward_validation_summary() -> dict[str, Any] | None:
    if not ENABLE_AUDIT_ENGINE:
        return None
    return ForwardValidator().summary()
