"""
Tests for opportunity.audit_engine — Full System Audit & Validation Suite.
Pure computation + local filesystem logging only. No network, no side effects
on engine.py's prediction model, signal generation, or execution logic.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.audit_engine as ae_mod
from opportunity.performance_tracker import PerformanceTracker
from opportunity.audit_engine import (
    BacktestEngine,
    TradeSimResult,
    ForwardValidator,
    PerformanceAnalytics,
    SystemAudit,
    AuditWarning,
    BugDetector,
    BugReport,
    audit_trade,
    generate_comparison_report,
)


def _flat_bars(n=10, start=100.0, drift=0.0):
    closes = [start + i * drift for i in range(n)]
    return pd.DataFrame({
        "Open":  closes,
        "High":  [c + 0.5 for c in closes],
        "Low":   [c - 0.5 for c in closes],
        "Close": closes,
    })


def _ohlcv(n=30):
    np.random.seed(3)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "Open": closes - 0.1, "High": closes + 0.4, "Low": closes - 0.4,
        "Close": closes,
    })


def _sample_trade(entry=100.0):
    return {
        "ticker": "TEST.AX", "direction": "LONG", "entry": entry,
        "stop_loss": entry * 0.95, "take_profit": entry * 1.10,
        "probability": 0.68, "expected_r": 0.5,
    }


class TestBacktestEngineSimulateTrade(unittest.TestCase):
    def test_hits_target(self):
        bars = _flat_bars(n=5, start=100.0, drift=0.0)
        bars.loc[2, "High"] = 112.0   # target 110 touched on bar 3
        result = BacktestEngine(max_hold_days=10).simulate_trade(
            "TEST.AX", entry=100.0, stop_loss=95.0, take_profit=110.0, future_bars=bars,
        )
        self.assertIsInstance(result, TradeSimResult)
        self.assertEqual(result.outcome, "WIN")
        self.assertGreater(result.pnl_r, 0)
        self.assertEqual(result.duration_days, 3)

    def test_hits_stop(self):
        bars = _flat_bars(n=5, start=100.0, drift=0.0)
        bars.loc[1, "Low"] = 90.0   # stop 95 touched on bar 2
        result = BacktestEngine(max_hold_days=10).simulate_trade(
            "TEST.AX", entry=100.0, stop_loss=95.0, take_profit=110.0, future_bars=bars,
        )
        self.assertEqual(result.outcome, "LOSS")
        self.assertLess(result.pnl_r, 0)
        self.assertEqual(result.duration_days, 2)

    def test_time_exit_when_neither_touched(self):
        bars = _flat_bars(n=3, start=100.0, drift=0.2)
        result = BacktestEngine(max_hold_days=3).simulate_trade(
            "TEST.AX", entry=100.0, stop_loss=90.0, take_profit=130.0, future_bars=bars,
        )
        self.assertIn(result.outcome, ("TIME_EXIT_WIN", "TIME_EXIT_LOSS"))
        self.assertEqual(result.duration_days, 3)

    def test_mfe_mae_tracked(self):
        bars = _flat_bars(n=4, start=100.0, drift=0.0)
        bars.loc[0, "High"] = 108.0
        bars.loc[1, "Low"] = 92.0
        result = BacktestEngine(max_hold_days=10).simulate_trade(
            "TEST.AX", entry=100.0, stop_loss=85.0, take_profit=150.0, future_bars=bars,
        )
        self.assertGreaterEqual(result.mfe, 0.07)
        self.assertLessEqual(result.mae, -0.07)

    def test_empty_bars_returns_none(self):
        result = BacktestEngine().simulate_trade(
            "TEST.AX", 100.0, 95.0, 110.0, future_bars=pd.DataFrame(),
        )
        self.assertIsNone(result)

    def test_zero_risk_returns_none(self):
        bars = _flat_bars()
        result = BacktestEngine().simulate_trade(
            "TEST.AX", 100.0, 100.0, 110.0, future_bars=bars,
        )
        self.assertIsNone(result)

    def test_slippage_reduces_pnl(self):
        bars = _flat_bars(n=3)
        bars.loc[1, "High"] = 112.0
        no_slip = BacktestEngine(apply_slippage=False).simulate_trade(
            "BHP.AX", 100.0, 95.0, 110.0, future_bars=bars,
        )
        with_slip = BacktestEngine(apply_slippage=True).simulate_trade(
            "BHP.AX", 100.0, 95.0, 110.0, future_bars=bars,
        )
        self.assertGreaterEqual(no_slip.pnl_r, with_slip.pnl_r)


class TestBacktestEngineBatch(unittest.TestCase):
    def test_run_batch_skips_missing_price_data(self):
        trades = [
            {"ticker": "A.AX", "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0},
            {"ticker": "MISSING.AX", "entry": 50.0, "stop_loss": 45.0, "take_profit": 60.0},
        ]
        price_data = {"A.AX": _flat_bars(n=5)}
        results = BacktestEngine().run_batch(trades, price_data)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ticker"], "A.AX")

    def test_run_batch_never_raises(self):
        results = BacktestEngine().run_batch([{"ticker": None}], {})
        self.assertEqual(results, [])

    def test_rolling_window_backtest_windows(self):
        trades = [
            {"ticker": "A.AX", "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0}
            for _ in range(5)
        ]
        price_data = {"A.AX": _flat_bars(n=5)}
        report = BacktestEngine().rolling_window_backtest(trades, price_data, window_size=2)
        self.assertIn("windows", report)
        self.assertGreaterEqual(len(report["windows"]), 1)


class TestForwardValidator(unittest.TestCase):
    def test_summary_handles_no_data(self):
        with patch.object(ForwardValidator, "validation_records", return_value=[]):
            summary = ForwardValidator().summary()
        self.assertEqual(summary["trade_count"], 0)
        self.assertFalse(summary["sufficient_data"])

    def test_deviation_computation(self):
        record = {"r_multiple": 1.0, "probability": 0.75, "risk_reward": 2.0}
        deviation = ForwardValidator._deviation(record)
        # expected_r = (0.75-0.5)*2*2.0 = 1.0; deviation = 1.0 - 1.0 = 0.0
        self.assertAlmostEqual(deviation, 0.0, places=3)

    def test_deviation_none_on_missing_fields(self):
        self.assertIsNone(ForwardValidator._deviation({"r_multiple": 1.0}))

    def test_summary_never_raises_on_bad_tracker(self):
        with patch.object(ForwardValidator, "validation_records", side_effect=RuntimeError("boom")):
            summary = ForwardValidator().summary()
        self.assertEqual(summary, {})


class TestPerformanceAnalytics(unittest.TestCase):
    def test_rolling_windows_empty_data_safe(self):
        with patch("opportunity.audit_engine._load_signal_log", return_value=[]):
            windows = PerformanceAnalytics().rolling_windows()
        self.assertEqual(len(windows), len(ae_mod.AUDIT_ROLLING_WINDOWS))
        for w in windows:
            self.assertEqual(w["trade_count"], 0)

    def test_setup_type_expectancy_groups_by_signal(self):
        entries = [
            {"ticker": "A.AX", "signal": "BREAKOUT", "outcome": "WIN", "actual_pct": 0.1},
            {"ticker": "B.AX", "signal": "BREAKOUT", "outcome": "LOSS", "actual_pct": -0.05},
            {"ticker": "C.AX", "signal": "PULLBACK", "outcome": "WIN", "actual_pct": 0.08},
        ]
        with patch("opportunity.audit_engine._load_signal_log", return_value=entries), \
             patch("opportunity.audit_engine._resolved_signal_entries", side_effect=lambda e: e):
            out = PerformanceAnalytics().setup_type_expectancy()
        labels = {o["setup_type"] for o in out}
        self.assertEqual(labels, {"BREAKOUT", "PULLBACK"})

    def test_edge_score_buckets_shape(self):
        with patch.object(PerformanceAnalytics, "__init__", lambda self: None):
            pa = PerformanceAnalytics()
            pa._tracker = type("T", (), {"resolved_records": lambda self: [
                {"edge_score": 0.75, "outcome": "WIN", "r_multiple": 1.2},
                {"edge_score": 0.30, "outcome": "LOSS", "r_multiple": -0.8},
            ]})()
        buckets = pa.edge_score_buckets()
        self.assertEqual(len(buckets), len(ae_mod.AUDIT_EDGE_SCORE_BUCKETS))
        nonzero = [b for b in buckets if b["count"] > 0]
        self.assertEqual(len(nonzero), 2)

    def test_drawdown_curve_shape(self):
        entries = [
            {"ticker": "A.AX", "signal_date": "2026-01-01", "actual_pct": 0.05},
            {"ticker": "A.AX", "signal_date": "2026-01-02", "actual_pct": -0.10},
        ]
        with patch("opportunity.audit_engine._load_signal_log", return_value=entries), \
             patch("opportunity.audit_engine._resolved_signal_entries", side_effect=lambda e: e):
            dd = PerformanceAnalytics().drawdown_curve()
        self.assertEqual(len(dd["curve"]), 2)
        self.assertGreaterEqual(dd["max_drawdown_pct"], 0.0)

    def test_full_report_never_raises(self):
        with patch("opportunity.audit_engine._load_signal_log", side_effect=RuntimeError("boom")):
            report = PerformanceAnalytics().full_report()
        self.assertEqual(report, {})


class TestSystemAudit(unittest.TestCase):
    def test_check_rejection_rate_spike_insufficient_data(self):
        with patch("opportunity.audit_engine._read_eval_log", return_value=[]):
            self.assertIsNone(SystemAudit().check_rejection_rate_spike())

    def test_check_rejection_rate_spike_detected(self):
        baseline = [{"passed": True} for _ in range(40)]
        recent = [{"passed": False} for _ in range(50)]
        with patch("opportunity.audit_engine._read_eval_log", return_value=baseline + recent), \
             patch("opportunity.audit_engine.AUDIT_RECENT_WINDOW_TRADES", 50):
            warning = SystemAudit().check_rejection_rate_spike()
        self.assertIsInstance(warning, AuditWarning)
        self.assertEqual(warning.code, "REJECTION_RATE_SPIKE")

    def test_check_signal_structure_detects_missing_fields(self):
        entries = [{"ticker": "A.AX", "price": None, "signal_date": "2026-01-01"}]
        with patch("opportunity.audit_engine._load_signal_log", return_value=entries):
            warning = SystemAudit().check_signal_structure()
        self.assertIsInstance(warning, AuditWarning)
        self.assertEqual(warning.code, "INCONSISTENT_SIGNAL_STRUCTURE")

    def test_check_data_issues_detects_nan(self):
        entries = [{"ticker": "A.AX", "price": float("nan")}]
        with patch("opportunity.audit_engine._load_signal_log", return_value=entries), \
             patch("opportunity.audit_engine._read_eval_log", return_value=[]):
            warnings = SystemAudit().check_data_issues()
        codes = [w.code for w in warnings]
        self.assertIn("NAN_OR_INVALID_VALUES", codes)

    def test_run_audit_never_raises(self):
        # Even if a check somehow raised past its own @_safe guard (e.g. a
        # fully-replaced method in a test double), run_audit()'s own @_safe
        # wrapper must still prevent the exception from propagating.
        with patch.object(SystemAudit, "check_rejection_rate_spike", side_effect=RuntimeError("x")):
            try:
                report = SystemAudit().run_audit()
            except Exception as e:
                self.fail(f"run_audit() raised unexpectedly: {e}")
        self.assertIsInstance(report, dict)

    def test_run_audit_aggregates_zero_warnings_cleanly(self):
        with patch("opportunity.audit_engine._read_eval_log", return_value=[]), \
             patch("opportunity.audit_engine._load_signal_log", return_value=[]), \
             patch.object(SystemAudit, "check_regime_mismatch_degradation", return_value=None):
            report = SystemAudit().run_audit()
        self.assertEqual(report["warning_count"], 0)


class TestBugDetector(unittest.TestCase):
    def test_detects_out_of_range_edge_score(self):
        evals = [{"symbol": "A.AX", "edge_score": 1.5}]
        with patch("opportunity.audit_engine._read_eval_log", return_value=evals):
            bugs = BugDetector().scan_trade_evaluations()
        self.assertEqual(len(bugs), 1)
        self.assertIsInstance(bugs[0], BugReport)

    def test_detects_negative_risk_reward(self):
        evals = [{"symbol": "A.AX", "risk_reward": -1.0}]
        with patch("opportunity.audit_engine._read_eval_log", return_value=evals):
            bugs = BugDetector().scan_trade_evaluations()
        self.assertTrue(any("negative risk_reward" in b.issue for b in bugs))

    def test_scan_signal_log_detects_missing_actual_pct(self):
        entries = [{"ticker": "A.AX", "outcome": "WIN", "actual_pct": None}]
        with patch("opportunity.audit_engine._load_signal_log", return_value=entries):
            bugs = BugDetector().scan_signal_log()
        self.assertEqual(len(bugs), 1)

    def test_run_never_raises(self):
        with patch.object(BugDetector, "scan_trade_evaluations", side_effect=RuntimeError("x")):
            report = BugDetector().run()
        self.assertEqual(report, {})

    def test_run_never_auto_applies_only_reports(self):
        with patch("opportunity.audit_engine._read_eval_log", return_value=[]), \
             patch("opportunity.audit_engine._load_signal_log", return_value=[]):
            report = BugDetector().run()
        self.assertEqual(report["bug_count"], 0)
        self.assertEqual(report["bugs"], [])


class TestAuditTradeWrapper(unittest.TestCase):
    def test_noop_when_disabled(self):
        with patch("opportunity.audit_engine.ENABLE_AUDIT_ENGINE", False):
            result = audit_trade(_sample_trade(), _ohlcv())
        self.assertEqual(result, {})

    def test_never_mutates_trade(self):
        trade = _sample_trade()
        original = dict(trade)
        with patch("opportunity.audit_engine.ENABLE_AUDIT_ENGINE", True), \
             patch("opportunity.audit_engine._log_audit_record"):
            audit_trade(trade, _ohlcv())
        self.assertEqual(trade, original)

    def test_logs_record_when_enabled(self):
        captured = {}
        def _capture(record):
            captured.update(record)
        with patch("opportunity.audit_engine.ENABLE_AUDIT_ENGINE", True), \
             patch("opportunity.audit_engine._log_audit_record", side_effect=_capture):
            result = audit_trade(_sample_trade(), _ohlcv())
        self.assertTrue(result.get("logged"))
        self.assertIn("pass_fail", captured)
        self.assertEqual(captured["symbol"], "TEST.AX")

    def test_never_raises_on_internal_failure(self):
        with patch("opportunity.audit_engine.ENABLE_AUDIT_ENGINE", True), \
             patch.object(ae_mod._evaluator, "evaluate", side_effect=RuntimeError("boom")):
            result = audit_trade(_sample_trade(), _ohlcv())
        self.assertFalse(result.get("logged"))
        self.assertIn("error", result)

    def test_outcome_data_included_when_provided(self):
        captured = {}
        def _capture(record):
            captured.update(record)
        outcome = {"outcome": "WIN", "pnl_r": 1.5, "mfe": 0.08, "mae": -0.02}
        with patch("opportunity.audit_engine.ENABLE_AUDIT_ENGINE", True), \
             patch("opportunity.audit_engine._log_audit_record", side_effect=_capture):
            audit_trade(_sample_trade(), _ohlcv(), outcome_data=outcome)
        self.assertEqual(captured["actual_outcome"], "WIN")
        self.assertEqual(captured["pnl_r"], 1.5)


class TestComparisonEngine(unittest.TestCase):
    def test_generate_comparison_report_never_raises(self):
        with patch("opportunity.audit_engine._load_signal_log", side_effect=RuntimeError("boom")):
            report = generate_comparison_report()
        self.assertEqual(report, {})

    def test_generate_comparison_report_shape(self):
        with patch("opportunity.audit_engine._load_signal_log", return_value=[]), \
             patch("opportunity.audit_engine._resolved_signal_entries", return_value=[]), \
             patch.object(ae_mod.PerformanceTracker, "passed_vs_rejected",
                           return_value={"passed": {}, "rejected": {}}), \
             patch.object(ae_mod.PerformanceTracker, "regime_buckets", return_value={}):
            report = generate_comparison_report()
        self.assertIn("old_system_baseline", report)
        self.assertIn("new_filtered_system", report)
        self.assertIn("which_performs_better", report)


if __name__ == "__main__":
    unittest.main()
