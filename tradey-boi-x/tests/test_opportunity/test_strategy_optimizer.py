"""
Tests for opportunity.strategy_optimizer — Self-Optimising Strategy Engine
(SAFE MODE). Pure computation + local filesystem logging only. No network,
no side effects on engine.py's prediction model, signal generation, or
execution logic.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.strategy_optimizer as so_mod
from opportunity.strategy_optimizer import (
    StrategyProfiler,
    StrategyPerformanceMatrix,
    StrategyWeightingEngine,
    StrategyGatingSystem,
    RegimeStrategyMap,
    process_trade_signal,
    _clamp_weight,
)
from opportunity.config import (
    STRATEGY_WEIGHT_FLOOR,
    STRATEGY_WEIGHT_CAP,
    STRATEGY_WEIGHT_MAX_STEP_PCT,
    STRATEGY_MIN_ACTIVE_WEIGHT,
    STRATEGY_MIN_EXPECTANCY_TRADES,
    REGIME_STRATEGY_MAP,
)


def _make_ohlcv(n: int = 60, volume: float = 200_000.0) -> pd.DataFrame:
    np.random.seed(7)
    closes = 100.0 + np.linspace(0, 10, n) + np.random.randn(n) * 0.2
    df = pd.DataFrame({
        "Open": closes - 0.1, "High": closes + 0.3, "Low": closes - 0.3,
        "Close": closes, "Volume": np.full(n, volume),
    })
    return df


def _sample_trade(**overrides):
    trade = {
        "ticker": "TEST.AX", "direction": "LONG", "entry": 100.0,
        "stop_loss": 95.0, "take_profit": 115.0,
        "probability": 0.65, "rsi": 50.0, "breakout": 0, "why": [],
    }
    trade.update(overrides)
    return trade


class TestStrategyProfiler(unittest.TestCase):
    def test_breakout_flag_infers_breakout(self):
        trade = _sample_trade(breakout=1)
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "BREAKOUT")

    def test_breakout_keyword_in_why_infers_breakout(self):
        trade = _sample_trade(why=["52-week breakout"])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "BREAKOUT")

    def test_squeeze_keyword_infers_breakout(self):
        trade = _sample_trade(why=["Breakout from volatility squeeze"])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "BREAKOUT")

    def test_volatility_expansion_keyword(self):
        trade = _sample_trade(rsi=50, why=["Volatility expansion detected"])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "VOLATILITY_EXPANSION")

    def test_low_rsi_with_pullback_keyword_infers_pullback(self):
        trade = _sample_trade(rsi=32, why=["Buy the pullback to support"])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "PULLBACK")

    def test_extreme_low_rsi_without_pullback_keyword_infers_mean_reversion(self):
        trade = _sample_trade(rsi=20, why=["RSI oversold"])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "MEAN_REVERSION")

    def test_extreme_high_rsi_infers_mean_reversion(self):
        trade = _sample_trade(rsi=75, why=[])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "MEAN_REVERSION")

    def test_default_uptrend_infers_trend_continuation(self):
        trade = _sample_trade(rsi=50, why=["Uptrend: EMA20 > EMA50"])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "TREND_CONTINUATION")

    def test_no_signals_falls_back_to_trend_continuation(self):
        trade = _sample_trade(rsi=None, why=[])
        self.assertEqual(StrategyProfiler.infer_strategy_type(trade), "TREND_CONTINUATION")

    def test_infer_never_raises_on_malformed_trade(self):
        with patch.object(so_mod, "_safe", lambda default=None: (lambda f: f)):
            pass  # sanity: importing/patching doesn't blow up
        result = StrategyProfiler.infer_strategy_type({"why": None, "rsi": "not-a-number"})
        self.assertIn(result, ("TREND_CONTINUATION", "MEAN_REVERSION", "BREAKOUT",
                                "PULLBACK", "VOLATILITY_EXPANSION"))

    def test_tag_trade_returns_profile_dict(self):
        trade = _sample_trade(breakout=1)
        profile = StrategyProfiler.tag_trade(trade, "TRENDING_UP", 0.7)
        self.assertEqual(profile["strategy_type"], "BREAKOUT")
        self.assertEqual(profile["regime"], "TRENDING_UP")
        self.assertEqual(profile["edge_score"], 0.7)


class TestStrategyPerformanceMatrix(unittest.TestCase):
    def _records(self):
        return [
            {"strategy_type": "BREAKOUT", "regime": "TRENDING_UP", "outcome": "WIN", "r_multiple": 2.0},
            {"strategy_type": "BREAKOUT", "regime": "TRENDING_UP", "outcome": "LOSS", "r_multiple": -1.0},
            {"strategy_type": "BREAKOUT", "regime": "TRENDING_UP", "outcome": "WIN", "r_multiple": 1.5},
            {"strategy_type": "MEAN_REVERSION", "regime": "CHOP", "outcome": "LOSS", "r_multiple": -0.5},
        ]

    def test_build_matrix_groups_by_strategy_and_regime(self):
        matrix = StrategyPerformanceMatrix().build_matrix(self._records())
        self.assertIn("BREAKOUT|TRENDING_UP", matrix)
        self.assertIn("MEAN_REVERSION|CHOP", matrix)
        cell = matrix["BREAKOUT|TRENDING_UP"]
        self.assertEqual(cell["trade_count"], 3)
        self.assertAlmostEqual(cell["win_rate"], 2 / 3, places=4)

    def test_strategy_totals_collapses_across_regimes(self):
        records = self._records() + [
            {"strategy_type": "BREAKOUT", "regime": "CHOP", "outcome": "WIN", "r_multiple": 1.0},
        ]
        totals = StrategyPerformanceMatrix().strategy_totals(records)
        self.assertEqual(totals["BREAKOUT"]["trade_count"], 4)

    def test_empty_records_returns_empty_matrix(self):
        self.assertEqual(StrategyPerformanceMatrix().build_matrix([]), {})

    def test_drawdown_contribution_computed_from_cumulative_r(self):
        records = [
            {"strategy_type": "BREAKOUT", "regime": "TRENDING_UP", "outcome": "WIN", "r_multiple": 2.0},
            {"strategy_type": "BREAKOUT", "regime": "TRENDING_UP", "outcome": "LOSS", "r_multiple": -3.0},
        ]
        cell = StrategyPerformanceMatrix().build_matrix(records)["BREAKOUT|TRENDING_UP"]
        self.assertEqual(cell["drawdown_contribution"], 3.0)

    def test_matrix_never_raises_on_bad_records(self):
        matrix = StrategyPerformanceMatrix().build_matrix([{"bad": "record"}, None])
        self.assertIsInstance(matrix, dict)


class TestStrategyWeightingEngine(unittest.TestCase):
    def test_clamp_weight_respects_floor_and_cap(self):
        self.assertEqual(_clamp_weight(-5.0), STRATEGY_WEIGHT_FLOOR)
        self.assertEqual(_clamp_weight(99.0), STRATEGY_WEIGHT_CAP)
        self.assertEqual(_clamp_weight(1.0), 1.0)

    def test_positive_expectancy_low_drawdown_increases_weight(self):
        totals = {"BREAKOUT": {"trade_count": 30, "expectancy_r": 0.3, "drawdown_contribution": 0.5}}
        engine = StrategyWeightingEngine()
        new_weights = engine.compute_new_weights({"BREAKOUT": 1.0}, totals)
        self.assertGreater(new_weights["BREAKOUT"], 1.0)

    def test_negative_expectancy_decreases_weight(self):
        totals = {"BREAKOUT": {"trade_count": 30, "expectancy_r": -0.2, "drawdown_contribution": 1.0}}
        engine = StrategyWeightingEngine()
        new_weights = engine.compute_new_weights({"BREAKOUT": 1.0}, totals)
        self.assertLess(new_weights["BREAKOUT"], 1.0)

    def test_too_few_trades_leaves_weight_unchanged(self):
        totals = {"BREAKOUT": {"trade_count": 3, "expectancy_r": -5.0, "drawdown_contribution": 10.0}}
        engine = StrategyWeightingEngine()
        new_weights = engine.compute_new_weights({"BREAKOUT": 1.0}, totals)
        self.assertEqual(new_weights["BREAKOUT"], 1.0)

    def test_weight_step_never_exceeds_max_step_pct(self):
        totals = {"BREAKOUT": {"trade_count": 999, "expectancy_r": 5.0, "drawdown_contribution": 0.0}}
        engine = StrategyWeightingEngine()
        new_weights = engine.compute_new_weights({"BREAKOUT": 1.0}, totals)
        max_expected = 1.0 * (1 + STRATEGY_WEIGHT_MAX_STEP_PCT)
        self.assertAlmostEqual(new_weights["BREAKOUT"], max_expected, places=4)

    def test_weight_never_exceeds_cap_even_at_extreme_starting_point(self):
        totals = {"BREAKOUT": {"trade_count": 999, "expectancy_r": 5.0, "drawdown_contribution": 0.0}}
        engine = StrategyWeightingEngine()
        new_weights = engine.compute_new_weights({"BREAKOUT": STRATEGY_WEIGHT_CAP}, totals)
        self.assertLessEqual(new_weights["BREAKOUT"], STRATEGY_WEIGHT_CAP)

    def test_weight_never_drops_below_floor_even_at_extreme_starting_point(self):
        totals = {"BREAKOUT": {"trade_count": 999, "expectancy_r": -5.0, "drawdown_contribution": 10.0}}
        engine = StrategyWeightingEngine()
        new_weights = engine.compute_new_weights({"BREAKOUT": STRATEGY_WEIGHT_FLOOR}, totals)
        self.assertGreaterEqual(new_weights["BREAKOUT"], STRATEGY_WEIGHT_FLOOR)

    def test_maybe_update_never_raises_on_internal_error(self):
        with patch("opportunity.strategy_optimizer._resolved_strategy_records", side_effect=RuntimeError("x")):
            result = StrategyWeightingEngine().maybe_update()
        self.assertIsNone(result)

    def test_maybe_update_noop_when_not_enough_new_trades(self):
        with patch("opportunity.strategy_optimizer._resolved_strategy_records", return_value=[]), \
             patch("opportunity.strategy_optimizer._load_weight_state", return_value={"last_updated_count": 0}):
            result = StrategyWeightingEngine().maybe_update(window=50)
        self.assertIsNone(result)


class TestRegimeStrategyMap(unittest.TestCase):
    def test_low_liquidity_allows_nothing(self):
        self.assertEqual(RegimeStrategyMap.allowed_strategies("LOW_LIQUIDITY"), [])
        self.assertFalse(RegimeStrategyMap.is_allowed("BREAKOUT", "LOW_LIQUIDITY"))

    def test_trending_up_allows_breakout(self):
        self.assertTrue(RegimeStrategyMap.is_allowed("BREAKOUT", "TRENDING_UP"))

    def test_chop_blocks_breakout(self):
        self.assertFalse(RegimeStrategyMap.is_allowed("BREAKOUT", "CHOP"))

    def test_every_regime_except_low_liquidity_has_at_least_one_strategy(self):
        for regime, strategies in REGIME_STRATEGY_MAP.items():
            if regime == "LOW_LIQUIDITY":
                continue
            self.assertGreaterEqual(len(strategies), 1, f"{regime} has no allowed strategies")

    def test_unknown_regime_falls_back_to_all_strategies(self):
        strategies = RegimeStrategyMap.allowed_strategies("SOME_UNKNOWN_REGIME")
        self.assertGreater(len(strategies), 0)


class TestStrategyGatingSystem(unittest.TestCase):
    def test_low_liquidity_regime_rejects(self):
        allowed, reason = StrategyGatingSystem.check(
            "BREAKOUT", "LOW_LIQUIDITY", {"BREAKOUT": 1.0}, {})
        self.assertFalse(allowed)
        self.assertEqual(reason, "strategy_disabled_or_low_edge")

    def test_disallowed_strategy_in_regime_rejects(self):
        allowed, reason = StrategyGatingSystem.check(
            "BREAKOUT", "CHOP", {"BREAKOUT": 1.0}, {})
        self.assertFalse(allowed)

    def test_low_weight_rejects(self):
        allowed, reason = StrategyGatingSystem.check(
            "BREAKOUT", "TRENDING_UP", {"BREAKOUT": STRATEGY_MIN_ACTIVE_WEIGHT - 0.05}, {})
        self.assertFalse(allowed)

    def test_negative_expectancy_with_enough_trades_rejects(self):
        totals = {"BREAKOUT": {"trade_count": STRATEGY_MIN_EXPECTANCY_TRADES + 5, "expectancy_r": -0.1}}
        allowed, reason = StrategyGatingSystem.check(
            "BREAKOUT", "TRENDING_UP", {"BREAKOUT": 1.0}, totals)
        self.assertFalse(allowed)

    def test_negative_expectancy_below_min_trades_passes_open(self):
        totals = {"BREAKOUT": {"trade_count": 2, "expectancy_r": -5.0}}
        allowed, reason = StrategyGatingSystem.check(
            "BREAKOUT", "TRENDING_UP", {"BREAKOUT": 1.0}, totals)
        self.assertTrue(allowed)

    def test_healthy_strategy_in_allowed_regime_passes(self):
        totals = {"BREAKOUT": {"trade_count": 50, "expectancy_r": 0.2}}
        allowed, reason = StrategyGatingSystem.check(
            "BREAKOUT", "TRENDING_UP", {"BREAKOUT": 1.0}, totals)
        self.assertTrue(allowed)
        self.assertEqual(reason, "allowed")

    def test_gating_check_never_raises(self):
        allowed, reason = StrategyGatingSystem.check(None, None, None, None)
        self.assertFalse(allowed)


class TestProcessTradeSignal(unittest.TestCase):
    def test_disabled_flag_passes_trade_through_unchanged(self):
        with patch.object(so_mod, "ENABLE_STRATEGY_OPTIMIZER", False):
            trade = _sample_trade()
            result = process_trade_signal(trade, _make_ohlcv())
        self.assertIs(result, trade)

    def test_enabled_shadow_mode_logs_but_never_blocks(self):
        with patch.object(so_mod, "ENABLE_STRATEGY_OPTIMIZER", True), \
             patch.object(so_mod, "SHADOW_MODE", True), \
             patch.object(so_mod, "_log_decision") as mock_log, \
             patch.object(so_mod, "_load_weights", return_value={"BREAKOUT": 0.1}), \
             patch.object(so_mod.StrategyPerformanceMatrix, "strategy_totals", return_value={}), \
             patch.object(so_mod.StrategyWeightingEngine, "maybe_update", return_value=None):
            trade = _sample_trade(breakout=1)
            result = process_trade_signal(trade, _make_ohlcv())
        self.assertIs(result, trade)
        mock_log.assert_called_once()
        logged = mock_log.call_args[0][0]
        self.assertFalse(logged["allowed"])

    def test_enabled_live_mode_blocks_rejected_trade(self):
        with patch.object(so_mod, "ENABLE_STRATEGY_OPTIMIZER", True), \
             patch.object(so_mod, "SHADOW_MODE", False), \
             patch.object(so_mod, "_log_decision"), \
             patch.object(so_mod, "_load_weights", return_value={"BREAKOUT": 0.1}), \
             patch.object(so_mod.StrategyPerformanceMatrix, "strategy_totals", return_value={}), \
             patch.object(so_mod.StrategyWeightingEngine, "maybe_update", return_value=None):
            trade = _sample_trade(breakout=1)
            result = process_trade_signal(trade, _make_ohlcv())
        self.assertIsNone(result)

    def test_enabled_live_mode_passes_approved_trade_unchanged(self):
        with patch.object(so_mod, "ENABLE_STRATEGY_OPTIMIZER", True), \
             patch.object(so_mod, "SHADOW_MODE", False), \
             patch.object(so_mod, "_log_decision"), \
             patch.object(so_mod, "_load_weights", return_value={"TREND_CONTINUATION": 1.0}), \
             patch.object(so_mod.StrategyPerformanceMatrix, "strategy_totals", return_value={}), \
             patch.object(so_mod.StrategyWeightingEngine, "maybe_update", return_value=None):
            trade = _sample_trade(rsi=50, why=["Uptrend: EMA20 > EMA50"])
            result = process_trade_signal(trade, _make_ohlcv())
        self.assertIs(result, trade)

    def test_never_raises_and_falls_back_to_passthrough_on_internal_error(self):
        with patch.object(so_mod, "ENABLE_STRATEGY_OPTIMIZER", True), \
             patch.object(so_mod, "_load_weights", side_effect=RuntimeError("boom")):
            trade = _sample_trade()
            try:
                result = process_trade_signal(trade, _make_ohlcv())
            except Exception as e:
                self.fail(f"process_trade_signal raised unexpectedly: {e}")
        self.assertIs(result, trade)

    def test_regime_detector_failure_falls_back_to_chop_and_does_not_raise(self):
        with patch.object(so_mod, "ENABLE_STRATEGY_OPTIMIZER", True), \
             patch.object(so_mod, "SHADOW_MODE", True), \
             patch("opportunity.adaptive_core.RegimeDetector.detect", side_effect=RuntimeError("x")), \
             patch.object(so_mod, "_log_decision") as mock_log, \
             patch.object(so_mod, "_load_weights", return_value={"TREND_CONTINUATION": 1.0}), \
             patch.object(so_mod.StrategyPerformanceMatrix, "strategy_totals", return_value={}), \
             patch.object(so_mod.StrategyWeightingEngine, "maybe_update", return_value=None):
            trade = _sample_trade()
            result = process_trade_signal(trade, _make_ohlcv())
        self.assertIs(result, trade)
        logged = mock_log.call_args[0][0]
        self.assertEqual(logged["regime"], "CHOP")


if __name__ == "__main__":
    unittest.main()
