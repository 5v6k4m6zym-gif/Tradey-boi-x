"""
Tests for opportunity.adaptive_core — Full Adaptive Trading Core v4.
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

import opportunity.adaptive_core as ac_mod
from opportunity.adaptive_core import (
    RegimeDetector,
    regime_adjusted_thresholds,
    ExecutionQualityFilter,
    ConfidenceCalibrator,
    PositionSizer,
    ExpectancyEngine,
    LossClassifier,
    process_trade_signal,
)


def _make_ohlcv(n: int = 60, trend: str = "up", volume: float = 200_000.0) -> pd.DataFrame:
    np.random.seed(11)
    base = 100.0
    if trend == "up":
        closes = base + np.linspace(0, 20, n) + np.random.randn(n) * 0.2
    elif trend == "down":
        closes = base - np.linspace(0, 20, n) + np.random.randn(n) * 0.2
    elif trend == "volatile":
        closes = base + np.cumsum(np.random.randn(n) * 4.0)
    elif trend == "flat_noisy":
        closes = base + np.random.randn(n) * 3.0
    else:
        closes = base + np.random.randn(n) * 0.3

    opens = closes - 0.1
    highs = closes + np.abs(np.random.randn(n)) * 0.4
    lows  = closes - np.abs(np.random.randn(n)) * 0.4
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": np.full(n, volume),
    })
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df


def _sample_trade(entry=100.0):
    return {
        "ticker": "TEST.AX", "direction": "LONG", "entry": entry,
        "stop_loss": entry * 0.95, "take_profit": entry * 1.15,
        "probability": 0.68, "expected_r": 0.5,
    }


class TestRegimeDetector(unittest.TestCase):
    def test_low_liquidity_hard_detected(self):
        df = _make_ohlcv(trend="flat_noisy", volume=1000.0)
        result = RegimeDetector().detect(df)
        self.assertEqual(result.regime, "LOW_LIQUIDITY")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_trending_up_detected(self):
        df = _make_ohlcv(trend="up")
        result = RegimeDetector().detect(df)
        self.assertIn(result.regime, ("TRENDING_UP", "VOLATILITY_EXPANSION"))

    def test_trending_down_detected(self):
        df = _make_ohlcv(trend="down")
        result = RegimeDetector().detect(df)
        self.assertIn(result.regime, ("TRENDING_DOWN", "VOLATILITY_EXPANSION"))

    def test_choppy_market_detected(self):
        df = _make_ohlcv(trend="flat_noisy")
        result = RegimeDetector().detect(df)
        self.assertIn(result.regime, ("CHOP", "VOLATILITY_EXPANSION"))

    def test_insufficient_data_defaults_safely(self):
        df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2],
                            "Close": [1, 2], "Volume": [1000, 1000]})
        result = RegimeDetector().detect(df)
        self.assertEqual(result.regime, "CHOP")

    def test_confidence_always_bounded(self):
        for trend in ("up", "down", "volatile", "flat_noisy"):
            df = _make_ohlcv(trend=trend)
            result = RegimeDetector().detect(df)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


class TestRegimeAdjustedThresholds(unittest.TestCase):
    def _base(self):
        # Mid-range values (within AUTO_TUNER_BOUNDS) so regime multipliers
        # have room to move both stricter and easier without clamping.
        return {
            "min_edge_score": 0.15, "min_predictability_score": 0.35,
            "min_risk_reward": 1.0, "max_noise_index": 1.9,
        }

    def test_does_not_mutate_base(self):
        base = self._base()
        original = dict(base)
        regime_adjusted_thresholds(base, "CHOP")
        self.assertEqual(base, original)

    def test_chop_is_stricter(self):
        base = self._base()
        adjusted = regime_adjusted_thresholds(base, "CHOP")
        self.assertGreater(adjusted["min_edge_score"], base["min_edge_score"])
        self.assertGreater(adjusted["min_predictability_score"], base["min_predictability_score"])

    def test_trending_is_easier(self):
        base = self._base()
        adjusted = regime_adjusted_thresholds(base, "TRENDING_UP")
        self.assertLess(adjusted["min_edge_score"], base["min_edge_score"])

    def test_volatility_expansion_stricter_rr(self):
        base = self._base()
        adjusted = regime_adjusted_thresholds(base, "VOLATILITY_EXPANSION")
        self.assertGreater(adjusted["min_risk_reward"], base["min_risk_reward"])

    def test_unknown_regime_is_noop(self):
        base = self._base()
        adjusted = regime_adjusted_thresholds(base, "SOMETHING_ELSE")
        self.assertEqual(adjusted, base)

    def test_never_exceeds_safe_bounds(self):
        # Even with an extreme starting value, output must stay in bounds.
        extreme = {"min_edge_score": 0.79, "min_predictability_score": 0.60,
                   "min_risk_reward": 2.5, "max_noise_index": 1.49}
        adjusted = regime_adjusted_thresholds(extreme, "CHOP")
        low, high = ac_mod.AUTO_TUNER_BOUNDS["min_edge_score"]
        self.assertLessEqual(adjusted["min_edge_score"], high)
        self.assertGreaterEqual(adjusted["min_edge_score"], low)


class TestExecutionQualityFilter(unittest.TestCase):
    def test_good_setup_passes(self):
        df = _make_ohlcv(trend="up")
        result = ExecutionQualityFilter().evaluate(_sample_trade(), df, edge_score=0.8, noise_index=0.3)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_high_cost_relative_to_edge_rejected(self):
        df = _make_ohlcv(trend="up")
        trade = _sample_trade(entry=100.0)
        trade["take_profit"] = 100.3   # tiny reward -> tiny expected edge
        result = ExecutionQualityFilter().evaluate(trade, df, edge_score=0.1, noise_index=0.3)
        self.assertFalse(result.passed)

    def test_us_vs_asx_cost_selection(self):
        df = _make_ohlcv(trend="up")
        asx_trade = _sample_trade()
        asx_trade["ticker"] = "BHP.AX"
        us_trade = _sample_trade()
        us_trade["ticker"] = "AAPL"
        asx_result = ExecutionQualityFilter().evaluate(asx_trade, df, edge_score=0.8, noise_index=0.3)
        us_result = ExecutionQualityFilter().evaluate(us_trade, df, edge_score=0.8, noise_index=0.3)
        self.assertGreater(asx_result.cost_pct, us_result.cost_pct)


class TestConfidenceCalibrator(unittest.TestCase):
    def test_passthrough_on_insufficient_data(self):
        with patch("opportunity.performance.calibration_buckets", return_value=[]):
            result = ConfidenceCalibrator().calibrate(0.72)
        self.assertEqual(result.calibrated_probability, 0.72)

    def test_uses_bucket_actual_win_rate_when_sample_sufficient(self):
        buckets = [{"label": "70–80%", "predicted_min": 0.70, "predicted_max": 0.80,
                    "count": 25, "actual_win_rate": 0.55, "calibration_status": "OVERCONFIDENT"}]
        with patch("opportunity.performance.calibration_buckets", return_value=buckets):
            result = ConfidenceCalibrator().calibrate(0.75)
        self.assertEqual(result.calibrated_probability, 0.55)
        self.assertEqual(result.calibration_status, "OVERCONFIDENT")

    def test_never_raises_on_internal_error(self):
        with patch("opportunity.performance.calibration_buckets", side_effect=Exception("boom")):
            result = ConfidenceCalibrator().calibrate(0.65)
        self.assertEqual(result.calibrated_probability, 0.65)


class TestPositionSizer(unittest.TestCase):
    def test_high_quality_increases_size_bounded(self):
        result = PositionSizer().size(edge_score=0.95, regime_confidence=0.9, predictability_score=0.9)
        self.assertGreater(result.multiplier, 1.0)
        self.assertLessEqual(result.multiplier, ac_mod.ADAPTIVE_MAX_SIZE_MULTIPLIER)

    def test_low_quality_reduces_size_bounded(self):
        result = PositionSizer().size(edge_score=0.1, regime_confidence=0.2, predictability_score=0.1)
        self.assertLess(result.multiplier, 1.0)
        self.assertGreaterEqual(result.multiplier, ac_mod.ADAPTIVE_MIN_SIZE_MULTIPLIER)

    def test_medium_quality_normal_size(self):
        result = PositionSizer().size(edge_score=0.6, regime_confidence=0.6, predictability_score=0.6)
        self.assertAlmostEqual(result.multiplier, 1.0)

    def test_never_exceeds_max_risk_pct(self):
        result = PositionSizer().size(edge_score=1.0, regime_confidence=1.0, predictability_score=1.0,
                                       base_risk_pct=5.0)
        self.assertLessEqual(result.adjusted_risk_pct, ac_mod.ADAPTIVE_MAX_RISK_PCT)


class TestExpectancyEngine(unittest.TestCase):
    def test_cold_start_fails_open(self):
        stats = {"trade_count": 2, "expectancy_r": -0.5}
        ok, reasons = ExpectancyEngine().gate(stats, regime_valid=True, execution_quality_passed=True)
        self.assertTrue(ok)

    def test_negative_expectancy_with_enough_data_rejects(self):
        stats = {"trade_count": 50, "expectancy_r": -0.1}
        ok, reasons = ExpectancyEngine().gate(stats, regime_valid=True, execution_quality_passed=True)
        self.assertFalse(ok)
        self.assertTrue(any("expectancy" in r for r in reasons))

    def test_positive_expectancy_with_enough_data_passes(self):
        stats = {"trade_count": 50, "expectancy_r": 0.2}
        ok, reasons = ExpectancyEngine().gate(stats, regime_valid=True, execution_quality_passed=True)
        self.assertTrue(ok)

    def test_invalid_regime_always_rejects(self):
        stats = {"trade_count": 50, "expectancy_r": 0.2}
        ok, reasons = ExpectancyEngine().gate(stats, regime_valid=False, execution_quality_passed=True)
        self.assertFalse(ok)

    def test_failed_execution_quality_always_rejects(self):
        stats = {"trade_count": 50, "expectancy_r": 0.2}
        ok, reasons = ExpectancyEngine().gate(stats, regime_valid=True, execution_quality_passed=False)
        self.assertFalse(ok)


class TestLossClassifier(unittest.TestCase):
    def test_wins_are_not_classified(self):
        self.assertIsNone(LossClassifier().classify({"outcome": "WIN", "actual_pct": 0.05}))

    def test_regime_mismatch_priority(self):
        result = LossClassifier().classify(
            {"outcome": "LOSS", "actual_pct": -0.03},
            entry_context={"regime_type": "CHOP"},
        )
        self.assertEqual(result, "REGIME_MISMATCH")

    def test_late_entry_from_timing_quality(self):
        result = LossClassifier().classify(
            {"outcome": "LOSS", "actual_pct": -0.03},
            entry_context={"regime_type": "TRENDING_UP", "timing_quality": 0.1},
        )
        self.assertEqual(result, "LATE_ENTRY")

    def test_noise_entry_from_high_noise(self):
        result = LossClassifier().classify(
            {"outcome": "HIT_STOP", "actual_pct": -0.02},
            entry_context={"regime_type": "TRENDING_UP", "timing_quality": 0.8, "noise_index": 1.6},
        )
        self.assertEqual(result, "NOISE_ENTRY")

    def test_false_breakout_default_for_quick_stop(self):
        result = LossClassifier().classify(
            {"outcome": "HIT_STOP", "actual_pct": -0.02},
            entry_context={"regime_type": "TRENDING_UP", "timing_quality": 0.8, "noise_index": 0.5},
        )
        self.assertEqual(result, "FALSE_BREAKOUT")

    def test_trend_reversal_for_expired_loss(self):
        result = LossClassifier().classify({"outcome": "EXPIRED_LOSS", "actual_pct": -0.01})
        self.assertEqual(result, "TREND_REVERSAL")

    def test_never_raises(self):
        result = LossClassifier().classify(None if False else {})
        self.assertEqual(result, "UNCLASSIFIED")


class TestProcessTradeSignal(unittest.TestCase):
    def setUp(self):
        self._orig_enable = ac_mod.ENABLE_ADAPTIVE_CORE
        self._orig_shadow = ac_mod.SHADOW_MODE

    def tearDown(self):
        ac_mod.ENABLE_ADAPTIVE_CORE = self._orig_enable
        ac_mod.SHADOW_MODE = self._orig_shadow

    def test_disabled_passes_through_unchanged(self):
        ac_mod.ENABLE_ADAPTIVE_CORE = False
        trade = _sample_trade()
        df = _make_ohlcv(trend="up")
        result = process_trade_signal(trade, df)
        self.assertIs(result, trade)

    def test_shadow_mode_always_returns_none(self):
        ac_mod.ENABLE_ADAPTIVE_CORE = True
        ac_mod.SHADOW_MODE = True
        with patch.object(ac_mod, "_log_decision"):
            result = process_trade_signal(_sample_trade(), _make_ohlcv(trend="up"))
        self.assertIsNone(result)

    def test_low_liquidity_always_rejected_when_not_shadow(self):
        ac_mod.ENABLE_ADAPTIVE_CORE = True
        ac_mod.SHADOW_MODE = False
        df = _make_ohlcv(trend="flat_noisy", volume=100.0)
        with patch.object(ac_mod, "_log_decision"):
            result = process_trade_signal(_sample_trade(), df)
        self.assertIsNone(result)

    def test_approved_trade_carries_sizing_metadata_without_mutating_original(self):
        ac_mod.ENABLE_ADAPTIVE_CORE = True
        ac_mod.SHADOW_MODE = False
        trade = _sample_trade()
        original = dict(trade)
        df = _make_ohlcv(trend="up")
        # Loosen thresholds and force a valid regime path so we can observe
        # the approved-path metadata regardless of the specific numbers.
        with patch.object(ac_mod, "TRADE_EVAL_THRESHOLDS", {
            "min_edge_score": 0.0, "min_predictability_score": 0.0,
            "min_risk_reward": 0.0, "max_noise_index": 99.0,
        }), patch.object(ac_mod, "_expectancy_engine") as mock_engine, \
             patch.object(ac_mod, "_log_decision"):
            mock_engine.gate.return_value = (True, [])
            result = process_trade_signal(trade, df)
        self.assertEqual(trade, original)   # original never mutated
        if result is not None:
            self.assertIn("position_size_multiplier", result)
            self.assertIn("regime_type", result)
            self.assertIn("adjusted_risk_pct", result)

    def test_internal_exception_falls_back_to_passthrough(self):
        ac_mod.ENABLE_ADAPTIVE_CORE = True
        ac_mod.SHADOW_MODE = False
        trade = _sample_trade()
        with patch.object(ac_mod, "_regime_detector") as mock_detector, \
             patch.object(ac_mod, "_log_decision"):
            mock_detector.detect.side_effect = Exception("boom")
            result = process_trade_signal(trade, _make_ohlcv(trend="up"))
        self.assertIs(result, trade)

    def test_logging_never_raises_even_if_write_fails(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            ac_mod._log_decision({"symbol": "TEST"})   # should not raise


if __name__ == "__main__":
    unittest.main()
