"""
Tests for opportunity.performance_tracker — joins trade_evaluator decisions
with resolved signal_log outcomes. Pure filesystem fixtures, no network.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.performance_tracker as pt_mod
from opportunity.performance_tracker import PerformanceTracker


def _eval_record(symbol="TST.AX", date="2026-06-01", passed=True, edge=0.7):
    return {
        "timestamp": f"{date}T10:00:00+00:00",
        "symbol": symbol,
        "direction": "LONG",
        "probability": 0.75,
        "edge_score": edge,
        "predictability_score": 0.65,
        "noise_index": 0.8,
        "risk_reward": 3.0,
        "passed": passed,
        "rejection_reasons": [] if passed else ["edge_score too low"],
        "shadow_mode": True,
    }


def _signal_entry(ticker="TST.AX", date="2026-06-01", outcome="WIN",
                   entry_price=1.00, stop_price=0.95, actual_pct=0.10, regime=None):
    return {
        "ticker": ticker, "tier": "B", "score": 70, "prob": 0.75,
        "entry_price": entry_price, "stop_price": stop_price,
        "target_price": 1.15, "signal_date": date, "target_pct": 0.15,
        "pred_days": 14, "outcome": outcome, "exit_price": round(entry_price * (1 + actual_pct), 4),
        "actual_pct": actual_pct, "regime": regime,
    }


class _FixtureMixin:
    def setUp(self):
        self.eval_path = Path("/tmp/_test_perf_tracker_evals.jsonl")
        self.signal_path = Path("/tmp/_test_perf_tracker_signal_log.json")
        self._orig_signal_file = pt_mod.SIGNAL_LOG_FILE
        self._orig_eval_path_fn = pt_mod._eval_log_path
        pt_mod.SIGNAL_LOG_FILE = self.signal_path
        pt_mod._eval_log_path = lambda: self.eval_path

    def tearDown(self):
        pt_mod.SIGNAL_LOG_FILE = self._orig_signal_file
        pt_mod._eval_log_path = self._orig_eval_path_fn
        for p in (self.eval_path, self.signal_path):
            if p.exists():
                p.unlink()

    def _write_evals(self, records):
        with open(self.eval_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _write_signals(self, entries):
        self.signal_path.write_text(json.dumps(entries))


class TestResolvedRecords(_FixtureMixin, unittest.TestCase):

    def test_empty_logs_returns_empty(self):
        self._write_evals([])
        self._write_signals([])
        tracker = PerformanceTracker()
        self.assertEqual(tracker.resolved_records(), [])

    def test_joins_matching_ticker_and_date(self):
        self._write_evals([_eval_record()])
        self._write_signals([_signal_entry()])
        tracker = PerformanceTracker()
        records = tracker.resolved_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["outcome"], "WIN")
        self.assertIsNotNone(records[0]["r_multiple"])

    def test_unresolved_signal_entries_excluded(self):
        self._write_evals([_eval_record()])
        entry = _signal_entry()
        entry["outcome"] = None
        self._write_signals([entry])
        tracker = PerformanceTracker()
        self.assertEqual(tracker.resolved_records(), [])

    def test_mismatched_ticker_excluded(self):
        self._write_evals([_eval_record(symbol="A.AX")])
        self._write_signals([_signal_entry(ticker="B.AX")])
        tracker = PerformanceTracker()
        self.assertEqual(tracker.resolved_records(), [])

    def test_r_multiple_computed_correctly(self):
        # entry 1.00, stop 0.90 -> risk 10%; actual_pct 0.20 -> R = 2.0
        self._write_evals([_eval_record()])
        self._write_signals([_signal_entry(entry_price=1.00, stop_price=0.90, actual_pct=0.20)])
        tracker = PerformanceTracker()
        records = tracker.resolved_records()
        self.assertAlmostEqual(records[0]["r_multiple"], 2.0)


class TestRollingStats(_FixtureMixin, unittest.TestCase):

    def test_empty_stats_are_zeroed(self):
        self._write_evals([])
        self._write_signals([])
        tracker = PerformanceTracker()
        stats = tracker.rolling_stats()
        self.assertEqual(stats["trade_count"], 0)
        self.assertEqual(stats["win_rate"], 0.0)

    def test_win_rate_and_avg_r(self):
        evals, signals = [], []
        for i in range(10):
            date = f"2026-06-{i+1:02d}"
            outcome = "WIN" if i < 6 else "HIT_STOP"
            pct = 0.10 if i < 6 else -0.05
            evals.append(_eval_record(symbol="TST.AX", date=date))
            signals.append(_signal_entry(date=date, outcome=outcome, actual_pct=pct))
        self._write_evals(evals)
        self._write_signals(signals)
        tracker = PerformanceTracker()
        stats = tracker.rolling_stats(window=100)
        self.assertEqual(stats["trade_count"], 10)
        self.assertAlmostEqual(stats["win_rate"], 0.6)

    def test_window_limits_to_recent_n(self):
        evals, signals = [], []
        for i in range(20):
            date = f"2026-0{1 + i // 28}-{(i % 28) + 1:02d}"
            evals.append(_eval_record(symbol="TST.AX", date=date))
            signals.append(_signal_entry(date=date, outcome="WIN", actual_pct=0.1))
        self._write_evals(evals)
        self._write_signals(signals)
        tracker = PerformanceTracker()
        stats = tracker.rolling_stats(window=5)
        self.assertEqual(stats["trade_count"], 5)

    def test_previous_window_stats_when_insufficient_history(self):
        self._write_evals([_eval_record()])
        self._write_signals([_signal_entry()])
        tracker = PerformanceTracker()
        prev = tracker.previous_window_stats(window=100)
        self.assertEqual(prev["trade_count"], 0)


class TestRegimeBucketsAndPassedRejected(_FixtureMixin, unittest.TestCase):

    def test_regime_buckets_empty_without_regime_data(self):
        self._write_evals([_eval_record()])
        self._write_signals([_signal_entry(regime=None)])
        tracker = PerformanceTracker()
        self.assertEqual(tracker.regime_buckets(), {})

    def test_regime_buckets_group_correctly(self):
        evals = [_eval_record(date="2026-06-01"), _eval_record(date="2026-06-02")]
        signals = [
            _signal_entry(date="2026-06-01", outcome="WIN", regime="bull"),
            _signal_entry(date="2026-06-02", outcome="HIT_STOP", actual_pct=-0.05, regime="bear"),
        ]
        self._write_evals(evals)
        self._write_signals(signals)
        tracker = PerformanceTracker()
        buckets = tracker.regime_buckets()
        self.assertIn("bull", buckets)
        self.assertIn("bear", buckets)

    def test_passed_vs_rejected_split(self):
        evals = [
            _eval_record(date="2026-06-01", passed=True),
            _eval_record(date="2026-06-02", passed=False),
        ]
        signals = [
            _signal_entry(date="2026-06-01", outcome="WIN"),
            _signal_entry(date="2026-06-02", outcome="HIT_STOP", actual_pct=-0.05),
        ]
        self._write_evals(evals)
        self._write_signals(signals)
        tracker = PerformanceTracker()
        split = tracker.passed_vs_rejected()
        self.assertEqual(split["passed"]["trade_count"], 1)
        self.assertEqual(split["rejected"]["trade_count"], 1)


if __name__ == "__main__":
    unittest.main()
