"""
Tests for opportunity.drift_monitor — Phase 9 Drift Monitoring
All file I/O and Discord calls are mocked.
"""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.drift_monitor as dm_mod
from opportunity.drift_monitor import (
    detect_drift,
    _split_baseline_live,
    run_drift_monitor,
    send_drift_alert,
)


def _entry(outcome="WIN", actual_pct=0.12, days_ago=5, pred_days=14) -> dict:
    date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "ticker":       "TST.AX",
        "tier":         "B",
        "score":        70,
        "prob":         0.70,
        "entry_price":  1.00,
        "stop_price":   0.90,
        "target_price": 1.15,
        "signal_date":  date,
        "target_pct":   0.15,
        "pred_days":    pred_days,
        "outcome":      outcome,
        "exit_price":   round(1.00 * (1 + actual_pct), 4),
        "actual_pct":   actual_pct,
    }


def _baseline_trades(n: int = 30) -> list[dict]:
    # Old trades (well beyond the live window), roughly 60% win rate.
    return [
        _entry("WIN" if i % 5 < 3 else "HIT_STOP",
               actual_pct=0.10 if i % 5 < 3 else -0.05,
               days_ago=60 + i)
        for i in range(n)
    ]


def _live_trades(n: int = 15, win_ratio: float = 0.6) -> list[dict]:
    n_win = int(n * win_ratio)
    return [
        _entry("WIN" if i < n_win else "HIT_STOP",
               actual_pct=0.10 if i < n_win else -0.05,
               days_ago=i % 20)
        for i in range(n)
    ]


# ─── _split_baseline_live ──────────────────────────────────────────────────────

class TestSplitBaselineLive(unittest.TestCase):

    def test_splits_by_window(self):
        entries = _baseline_trades(10) + _live_trades(5)
        baseline, live = _split_baseline_live(entries, live_window_days=30)
        self.assertEqual(len(baseline), 10)
        self.assertEqual(len(live), 5)

    def test_excludes_entries_without_signal_date(self):
        entries = [{"outcome": "WIN"}]
        baseline, live = _split_baseline_live(entries, live_window_days=30)
        self.assertEqual(baseline, [])
        self.assertEqual(live, [])

    def test_empty_entries(self):
        baseline, live = _split_baseline_live([], live_window_days=30)
        self.assertEqual(baseline, [])
        self.assertEqual(live, [])


# ─── detect_drift ──────────────────────────────────────────────────────────────

class TestDetectDrift(unittest.TestCase):

    def test_empty_returns_insufficient_data(self):
        report = detect_drift([])
        self.assertFalse(report["sufficient_data"])
        self.assertEqual(report["drift_flags"], [])

    def test_similar_performance_no_drift_flags(self):
        entries = _baseline_trades(30) + _live_trades(15, win_ratio=0.6)
        report = detect_drift(entries, live_window_days=30)
        self.assertTrue(report["sufficient_data"])
        self.assertEqual(report["drift_flags"], [])

    def test_degraded_live_performance_flags_drift(self):
        entries = _baseline_trades(30) + _live_trades(15, win_ratio=0.0)
        report = detect_drift(entries, live_window_days=30)
        self.assertTrue(report["sufficient_data"])
        self.assertTrue(len(report["drift_flags"]) > 0)
        for f in report["drift_flags"]:
            self.assertEqual(f["direction"], "degraded")

    def test_improved_live_performance_flags_drift(self):
        # win_ratio kept < 1.0 (some losses) so expectancy_r's R-unit
        # normalization stays comparable between baseline and live.
        entries = _baseline_trades(30) + _live_trades(15, win_ratio=0.9)
        report = detect_drift(entries, live_window_days=30)
        self.assertTrue(report["sufficient_data"])
        win_rate_flags = [f for f in report["drift_flags"] if f["metric"] == "win_rate"]
        self.assertTrue(len(win_rate_flags) > 0)
        self.assertEqual(win_rate_flags[0]["direction"], "improved")

    def test_insufficient_live_trades_suppresses_flags(self):
        entries = _baseline_trades(30) + _live_trades(2, win_ratio=0.0)
        report = detect_drift(entries, live_window_days=30)
        self.assertFalse(report["sufficient_data"])
        self.assertEqual(report["drift_flags"], [])

    def test_insufficient_baseline_trades_suppresses_flags(self):
        entries = _baseline_trades(3) + _live_trades(15, win_ratio=0.0)
        report = detect_drift(entries, live_window_days=30)
        self.assertFalse(report["sufficient_data"])
        self.assertEqual(report["drift_flags"], [])

    def test_report_contains_trade_counts_and_metrics(self):
        entries = _baseline_trades(30) + _live_trades(15)
        report = detect_drift(entries, live_window_days=30)
        self.assertEqual(report["baseline_trade_count"], 30)
        self.assertEqual(report["live_trade_count"], 15)
        self.assertIn("baseline_metrics", report)
        self.assertIn("live_metrics", report)
        self.assertIn("deltas", report)


# ─── send_drift_alert ──────────────────────────────────────────────────────────

class TestSendDriftAlert(unittest.TestCase):

    def test_no_webhook_returns_false(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(send_drift_alert({"drift_flags": [{"metric": "win_rate"}]}))

    def test_no_drift_flags_returns_false(self):
        with patch.dict("os.environ", {"Discordwebhook": "http://example.com"}):
            self.assertFalse(send_drift_alert({"drift_flags": []}))

    def test_sends_when_webhook_and_flags_present(self):
        report = {
            "live_window_days": 30, "live_trade_count": 15, "baseline_trade_count": 30,
            "drift_flags": [{"metric": "win_rate", "baseline": 0.6, "live": 0.2,
                              "delta": -0.4, "threshold": 0.15, "direction": "degraded"}],
        }
        with patch.dict("os.environ", {"Discordwebhook": "http://example.com"}), \
             patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *a: False
            result = send_drift_alert(report)
        self.assertTrue(result)
        mock_urlopen.assert_called_once()


# ─── run_drift_monitor ─────────────────────────────────────────────────────────

class TestRunDriftMonitor(unittest.TestCase):

    def test_returns_none_when_flag_off(self):
        with patch.object(dm_mod, "ENABLE_DRIFT_MONITORING", False):
            self.assertIsNone(run_drift_monitor())

    def test_returns_none_when_no_resolved_entries(self):
        with patch.object(dm_mod, "ENABLE_DRIFT_MONITORING", True), \
             patch.object(dm_mod, "_load_log", return_value=[]):
            self.assertIsNone(run_drift_monitor())

    def test_returns_report_when_enabled_with_data(self):
        entries = _baseline_trades(30) + _live_trades(15)
        with patch.object(dm_mod, "ENABLE_DRIFT_MONITORING", True), \
             patch.object(dm_mod, "_load_log", return_value=entries), \
             patch.object(dm_mod, "send_drift_alert", return_value=False):
            report = run_drift_monitor(notify=False)
        self.assertIsNotNone(report)
        self.assertIn("drift_flags", report)

    def test_notify_calls_send_drift_alert_when_flagged(self):
        entries = _baseline_trades(30) + _live_trades(15, win_ratio=0.0)
        with patch.object(dm_mod, "ENABLE_DRIFT_MONITORING", True), \
             patch.object(dm_mod, "_load_log", return_value=entries), \
             patch.object(dm_mod, "send_drift_alert", return_value=True) as mock_send:
            run_drift_monitor(notify=True)
        mock_send.assert_called_once()

    def test_no_notify_call_when_no_drift(self):
        entries = _baseline_trades(30) + _live_trades(15, win_ratio=0.6)
        with patch.object(dm_mod, "ENABLE_DRIFT_MONITORING", True), \
             patch.object(dm_mod, "_load_log", return_value=entries), \
             patch.object(dm_mod, "send_drift_alert", return_value=False) as mock_send:
            run_drift_monitor(notify=True)
        mock_send.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
