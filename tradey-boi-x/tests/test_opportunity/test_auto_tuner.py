"""
Tests for opportunity.auto_tuner — SAFE constrained threshold auto-tuning.
Pure filesystem fixtures + monkeypatched module state, no network.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.auto_tuner as at_mod
from opportunity.auto_tuner import AutoThresholdTuner, maybe_tune


def _bounds():
    return {
        "min_edge_score":          (0.55, 0.80),
        "min_predictability_score": (0.50, 0.75),
        "min_risk_reward":         (2.0, 4.0),
        "max_noise_index":         (1.0, 1.5),
    }


class TestDecideAdjustment(unittest.TestCase):
    def setUp(self):
        self._orig_bounds = at_mod.AUTO_TUNER_BOUNDS
        at_mod.AUTO_TUNER_BOUNDS = _bounds()

    def tearDown(self):
        at_mod.AUTO_TUNER_BOUNDS = self._orig_bounds

    def _thresholds(self):
        return {
            "min_edge_score": 0.65, "min_predictability_score": 0.60,
            "min_risk_reward": 2.5, "max_noise_index": 1.2,
        }

    def test_too_few_trades_loosens_all_four(self):
        tuner = AutoThresholdTuner(thresholds=self._thresholds())
        current = {"trade_count": 2, "win_rate": 0.5, "avg_r": 0.5, "expectancy_r": 0.1}
        previous = {"trade_count": 0, "win_rate": 0.0, "avg_r": 0.0, "expectancy_r": 0.0}
        adj = tuner.decide_adjustment(current, previous)
        self.assertEqual(adj["rule"], "too_few_trades")
        self.assertLess(adj["changes"]["min_edge_score"], 0.65)
        self.assertGreater(adj["changes"]["max_noise_index"], 1.2)

    def test_win_rate_decreased_raises_edge_score(self):
        tuner = AutoThresholdTuner(thresholds=self._thresholds())
        current = {"trade_count": 50, "win_rate": 0.40, "avg_r": 0.3, "expectancy_r": 0.05}
        previous = {"trade_count": 50, "win_rate": 0.55, "avg_r": 0.3, "expectancy_r": 0.10}
        adj = tuner.decide_adjustment(current, previous)
        self.assertEqual(adj["rule"], "win_rate_decreased")
        self.assertGreater(adj["changes"]["min_edge_score"], 0.65)
        self.assertEqual(len(adj["changes"]), 1)

    def test_win_rate_up_avg_r_down_raises_rr(self):
        tuner = AutoThresholdTuner(thresholds=self._thresholds())
        current = {"trade_count": 50, "win_rate": 0.65, "avg_r": 0.10, "expectancy_r": 0.05}
        previous = {"trade_count": 50, "win_rate": 0.55, "avg_r": 0.30, "expectancy_r": 0.10}
        adj = tuner.decide_adjustment(current, previous)
        self.assertEqual(adj["rule"], "win_rate_up_avg_r_down")
        self.assertGreater(adj["changes"]["min_risk_reward"], 2.5)
        self.assertEqual(len(adj["changes"]), 1)

    def test_stable_performance_no_change(self):
        tuner = AutoThresholdTuner(thresholds=self._thresholds())
        current = {"trade_count": 50, "win_rate": 0.60, "avg_r": 0.30, "expectancy_r": 0.10}
        previous = {"trade_count": 50, "win_rate": 0.60, "avg_r": 0.30, "expectancy_r": 0.10}
        adj = tuner.decide_adjustment(current, previous)
        self.assertIsNone(adj)

    def test_adjustment_never_exceeds_max_step_pct(self):
        tuner = AutoThresholdTuner(thresholds=self._thresholds())
        current = {"trade_count": 50, "win_rate": 0.40, "avg_r": 0.3, "expectancy_r": 0.05}
        previous = {"trade_count": 50, "win_rate": 0.55, "avg_r": 0.3, "expectancy_r": 0.10}
        adj = tuner.decide_adjustment(current, previous)
        new_val = adj["changes"]["min_edge_score"]
        max_expected = 0.65 * (1 + at_mod.AUTO_TUNER_MAX_STEP_PCT)
        self.assertLessEqual(new_val, max_expected + 1e-9)

    def test_apply_never_exceeds_bounds_even_at_edge(self):
        thresholds = {
            "min_edge_score": 0.79, "min_predictability_score": 0.60,
            "min_risk_reward": 2.5, "max_noise_index": 1.2,
        }
        tuner = AutoThresholdTuner(thresholds=thresholds)
        current = {"trade_count": 50, "win_rate": 0.40, "avg_r": 0.3, "expectancy_r": 0.05}
        previous = {"trade_count": 50, "win_rate": 0.55, "avg_r": 0.3, "expectancy_r": 0.10}
        adj = tuner.decide_adjustment(current, previous)
        tuner.apply(adj)
        self.assertLessEqual(tuner.thresholds["min_edge_score"], 0.80)

    def test_apply_mutates_in_place(self):
        thresholds = self._thresholds()
        tuner = AutoThresholdTuner(thresholds=thresholds)
        adj = {"rule": "test", "changes": {"min_edge_score": 0.70}}
        tuner.apply(adj)
        self.assertEqual(thresholds["min_edge_score"], 0.70)


class TestMaybeTune(unittest.TestCase):
    def setUp(self):
        self.state_path = Path("/tmp/_test_auto_tuner_state.json")
        self.log_path = Path("/tmp/_test_auto_tuner_log.jsonl")
        self._orig_state_path = at_mod.AUTO_TUNER_STATE_PATH
        self._orig_log_path = at_mod.AUTO_TUNER_LOG_PATH
        at_mod.AUTO_TUNER_STATE_PATH = str(self.state_path)
        at_mod.AUTO_TUNER_LOG_PATH = str(self.log_path)

    def tearDown(self):
        at_mod.AUTO_TUNER_STATE_PATH = self._orig_state_path
        at_mod.AUTO_TUNER_LOG_PATH = self._orig_log_path
        for p in (self.state_path, self.log_path):
            if p.exists():
                p.unlink()

    def test_returns_none_when_flag_off(self):
        with patch.object(at_mod, "ENABLE_AUTO_TUNER", False), \
             patch.object(at_mod, "SHADOW_MODE", False):
            self.assertIsNone(maybe_tune())

    def test_returns_none_when_shadow_mode_on(self):
        with patch.object(at_mod, "ENABLE_AUTO_TUNER", True), \
             patch.object(at_mod, "SHADOW_MODE", True):
            self.assertIsNone(maybe_tune())

    def test_returns_none_when_not_enough_new_trades(self):
        class FakeTracker:
            def resolved_records(self):
                return [{"x": i} for i in range(3)]

        with patch.object(at_mod, "ENABLE_AUTO_TUNER", True), \
             patch.object(at_mod, "SHADOW_MODE", False), \
             patch.object(at_mod, "PerformanceTracker", FakeTracker):
            self.assertIsNone(maybe_tune(window=50))

    def test_never_raises_on_internal_error(self):
        with patch.object(at_mod, "ENABLE_AUTO_TUNER", True), \
             patch.object(at_mod, "SHADOW_MODE", False), \
             patch.object(at_mod, "PerformanceTracker", side_effect=RuntimeError("boom")):
            result = maybe_tune()  # must not raise
            self.assertIsNone(result)

    def test_applies_and_logs_when_enough_trades_and_rule_fires(self):
        class FakeTracker:
            def resolved_records(self):
                return [{"x": i} for i in range(50)]

            def rolling_stats(self, window):
                return {"trade_count": 50, "win_rate": 0.40, "avg_r": 0.3, "expectancy_r": 0.05}

            def previous_window_stats(self, window):
                return {"trade_count": 50, "win_rate": 0.55, "avg_r": 0.3, "expectancy_r": 0.10}

        with patch.object(at_mod, "ENABLE_AUTO_TUNER", True), \
             patch.object(at_mod, "SHADOW_MODE", False), \
             patch.object(at_mod, "PerformanceTracker", FakeTracker):
            record = maybe_tune(window=50)
        self.assertIsNotNone(record)
        self.assertEqual(record["rule"], "win_rate_decreased")
        self.assertTrue(self.log_path.exists())
        self.assertTrue(self.state_path.exists())

    def test_second_call_within_same_window_is_noop(self):
        class FakeTracker:
            def resolved_records(self):
                return [{"x": i} for i in range(50)]

            def rolling_stats(self, window):
                return {"trade_count": 50, "win_rate": 0.60, "avg_r": 0.30, "expectancy_r": 0.10}

            def previous_window_stats(self, window):
                return {"trade_count": 50, "win_rate": 0.60, "avg_r": 0.30, "expectancy_r": 0.10}

        with patch.object(at_mod, "ENABLE_AUTO_TUNER", True), \
             patch.object(at_mod, "SHADOW_MODE", False), \
             patch.object(at_mod, "PerformanceTracker", FakeTracker):
            maybe_tune(window=50)   # consumes the cycle (updates state), no rule fires
            result = maybe_tune(window=50)  # not enough NEW trades since last cycle
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
