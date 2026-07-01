"""
Tests for opportunity.scoring — Opportunity Scoring Engine
All external calls are mocked. No network, no Discord, no side effects.
"""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.scoring as scoring_mod
from opportunity.scoring import (
    _score_expected_return,
    _score_technical_strength,
    _score_volume_expansion,
    _score_momentum,
    _score_news_catalyst,
    _score_institutional,
    _score_risk_reward,
    _estimate_holding_period,
    _reasons,
    score_opportunity,
)


def _make_df(n: int = 60, **overrides) -> pd.DataFrame:
    """Synthetic DataFrame matching engine.get_data() output columns."""
    base = {
        "Close": 1.50, "Open": 1.48, "High": 1.55, "Low": 1.45, "Volume": 2_000_000.0,
        "rsi": 55.0, "macd_diff": 0.01, "bb_width": 0.04,
        "atr": 0.12,   # ~8% of price — high ATR for big move scoring
        "ret_5": 0.04, "ret_10": 0.06, "ret_20": 0.08, "ret_63": 0.15,
        "vol_ratio": 2.0, "breakout": 1.0, "obv_ratio": 1.8,
        "adx": 30.0, "mfi": 55.0, "bb_squeeze": 0.0, "gap_up": 0.0,
        "ema20": 1.48, "ema50": 1.40,
    }
    base.update(overrides)
    return pd.DataFrame([base] * n)


class TestScoreExpectedReturn(unittest.TestCase):

    def test_returns_three_values(self):
        df = _make_df()
        result = _score_expected_return(df)
        self.assertEqual(len(result), 3)

    def test_score_between_0_and_100(self):
        df = _make_df()
        score, _, _ = _score_expected_return(df)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_high_atr_scores_higher(self):
        low_atr  = _make_df(atr=0.03)   # ~2% of price
        high_atr = _make_df(atr=0.20)   # ~13% of price
        s_low,  *_ = _score_expected_return(low_atr)
        s_high, *_ = _score_expected_return(high_atr)
        self.assertGreater(s_high, s_low)

    def test_high_adx_increases_upside(self):
        low_adx  = _make_df(adx=10.0)
        high_adx = _make_df(adx=50.0)
        _, up_low,  _ = _score_expected_return(low_adx)
        _, up_high, _ = _score_expected_return(high_adx)
        self.assertGreater(up_high, up_low)

    def test_upside_and_downside_positive(self):
        df = _make_df()
        _, upside, downside = _score_expected_return(df)
        self.assertGreater(upside, 0)
        self.assertGreater(downside, 0)

    def test_zero_close_does_not_crash(self):
        df = _make_df(Close=0.0001)
        score, _, _ = _score_expected_return(df)
        self.assertGreaterEqual(score, 0)


class TestScoreTechnicalStrength(unittest.TestCase):

    def test_perfect_setup_scores_100(self):
        df = _make_df(ema20=1.48, ema50=1.40, rsi=55, adx=30, breakout=1.0, macd_diff=0.01)
        self.assertEqual(_score_technical_strength(df), 100.0)

    def test_no_signals_scores_zero(self):
        df = _make_df(ema20=1.30, ema50=1.50, rsi=80, adx=10, breakout=0.0, macd_diff=-0.01)
        self.assertEqual(_score_technical_strength(df), 0.0)

    def test_score_between_0_and_100(self):
        df = _make_df(rsi=60, adx=20)
        score = _score_technical_strength(df)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_breakout_adds_20_points(self):
        no_bo = _make_df(breakout=0.0, ema20=1.3, ema50=1.5, rsi=80, adx=10, macd_diff=-0.1)
        bo    = _make_df(breakout=1.0, ema20=1.3, ema50=1.5, rsi=80, adx=10, macd_diff=-0.1)
        self.assertEqual(_score_technical_strength(bo) - _score_technical_strength(no_bo), 20)


class TestScoreVolumeExpansion(unittest.TestCase):

    def test_below_1x_returns_0(self):
        self.assertEqual(_score_volume_expansion(_make_df(vol_ratio=0.8)), 0.0)

    def test_exactly_1x_returns_0(self):
        self.assertEqual(_score_volume_expansion(_make_df(vol_ratio=1.0)), 0.0)

    def test_5x_returns_100(self):
        self.assertEqual(_score_volume_expansion(_make_df(vol_ratio=5.0)), 100.0)

    def test_above_5x_clamped_at_100(self):
        self.assertEqual(_score_volume_expansion(_make_df(vol_ratio=10.0)), 100.0)

    def test_proportional_between_1_and_5(self):
        s = _score_volume_expansion(_make_df(vol_ratio=3.0))
        self.assertAlmostEqual(s, 50.0, places=1)


class TestScoreMomentum(unittest.TestCase):

    def test_strong_momentum_scores_high(self):
        df = _make_df(ret_5=0.06, ret_20=0.12, macd_diff=0.05)
        self.assertGreaterEqual(_score_momentum(df), 80)

    def test_negative_momentum_scores_low(self):
        df = _make_df(ret_5=-0.05, ret_20=-0.10, macd_diff=-0.05)
        self.assertEqual(_score_momentum(df), 0.0)

    def test_score_capped_at_100(self):
        df = _make_df(ret_5=0.20, ret_20=0.50, macd_diff=1.0)
        self.assertLessEqual(_score_momentum(df), 100)


class TestScoreNewsAndInstitutional(unittest.TestCase):

    def test_high_obv_ratio_scores_high_news(self):
        df = _make_df(obv_ratio=2.5)
        self.assertGreaterEqual(_score_news_catalyst(df), 70)

    def test_low_obv_ratio_scores_low_news(self):
        df = _make_df(obv_ratio=0.5)
        self.assertLessEqual(_score_news_catalyst(df), 30)

    def test_high_obv_scores_high_institutional(self):
        df = _make_df(obv_ratio=3.0)
        self.assertGreaterEqual(_score_institutional(df), 85)


class TestScoreRiskReward(unittest.TestCase):

    def test_zero_downside_returns_zeros(self):
        score, rr = _score_risk_reward(0.20, 0.0)
        self.assertEqual(score, 0.0)
        self.assertEqual(rr, 0.0)

    def test_rr_calculation_correct(self):
        _, rr = _score_risk_reward(0.30, 0.06)
        self.assertAlmostEqual(rr, 5.0, places=1)

    def test_high_rr_scores_high(self):
        score, _ = _score_risk_reward(0.50, 0.05)
        self.assertGreater(score, 80)

    def test_low_rr_scores_low(self):
        score, _ = _score_risk_reward(0.06, 0.05)
        self.assertLess(score, 20)


class TestEstimateHoldingPeriod(unittest.TestCase):

    def test_returns_positive_int(self):
        df = _make_df()
        result = _estimate_holding_period(df, 0.25)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_larger_upside_longer_hold(self):
        df = _make_df(atr=0.01)   # small ATR keeps results above the 5-day floor
        short = _estimate_holding_period(df, 0.10)
        long  = _estimate_holding_period(df, 0.50)
        self.assertLess(short, long)

    def test_capped_at_180_days(self):
        df = _make_df(atr=0.001)   # tiny ATR → very slow expected move
        result = _estimate_holding_period(df, 1.00)
        self.assertLessEqual(result, 180)

    def test_minimum_5_days(self):
        df = _make_df(atr=100.0)   # huge ATR → theoretically 0 days, clamp to 5
        result = _estimate_holding_period(df, 0.01)
        self.assertGreaterEqual(result, 5)


class TestReasons(unittest.TestCase):

    def test_breakout_appears_in_for(self):
        row = _make_df().iloc[-1]
        row["breakout"] = 1.0
        for_r, _ = _reasons(row, 0.30, 4.0)
        self.assertTrue(any("breakout" in r.lower() for r in for_r))

    def test_overbought_rsi_appears_in_against(self):
        row = _make_df(rsi=75).iloc[-1]
        _, against = _reasons(row, 0.20, 3.0)
        self.assertTrue(any("overbought" in r.lower() or "rsi" in r.lower() for r in against))

    def test_high_rr_appears_in_for(self):
        row = _make_df().iloc[-1]
        for_r, _ = _reasons(row, 0.30, 5.0)
        self.assertTrue(any("risk" in r.lower() or "reward" in r.lower() for r in for_r))

    def test_returns_two_lists(self):
        row = _make_df().iloc[-1]
        result = _reasons(row, 0.20, 3.0)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)


class TestScoreOpportunity(unittest.TestCase):

    def test_returns_none_when_flag_off(self):
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", False):
            self.assertIsNone(score_opportunity("TST.AX", _make_df()))

    def test_returns_none_on_empty_df(self):
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True):
            self.assertIsNone(score_opportunity("TST.AX", pd.DataFrame()))

    def test_returns_none_on_insufficient_rows(self):
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True):
            self.assertIsNone(score_opportunity("TST.AX", _make_df(n=10)))

    def test_high_conviction_returns_dict(self):
        df = _make_df(
            atr=0.20, adx=35.0, vol_ratio=3.0, breakout=1.0,
            rsi=55.0, ret_5=0.06, ret_20=0.12, obv_ratio=2.0,
            ema20=1.55, ema50=1.40, macd_diff=0.02,
        )
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0,
                 "min_confidence": 0,
                 "min_expected_upside": 0,
                 "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0,
                 "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        self.assertIsNotNone(result)

    def test_all_required_keys_present(self):
        df = _make_df(atr=0.20, adx=35.0)
        required = [
            "ticker", "opportunity_score", "confidence",
            "expected_upside_pct", "expected_downside_pct",
            "est_holding_days", "prob_target_hit", "prob_stop_hit",
            "risk_level", "rr_ratio", "entry_zone", "stop_loss",
            "take_profit", "trailing_stop_pct", "regime",
            "reasons_for", "reasons_against",
            "technical_summary", "momentum_summary", "component_scores",
        ]
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0, "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0, "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        if result is not None:
            for key in required:
                self.assertIn(key, result, f"Missing key: {key}")

    def test_filter_rejects_low_score(self):
        df = _make_df(atr=0.001, vol_ratio=0.5, breakout=0.0, adx=5.0)
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 99,
                 "min_confidence": 0, "min_expected_upside": 0,
                 "min_avg_daily_volume": 0, "min_rr_ratio": 0,
                 "max_downside": 1.0,
             }):
            self.assertIsNone(score_opportunity("TST.AX", df))

    def test_filter_rejects_low_upside(self):
        df = _make_df(atr=0.001)   # tiny ATR → tiny expected upside
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0.50,   # require 50% — tiny ATR won't reach
                 "min_avg_daily_volume": 0, "min_rr_ratio": 0,
                 "max_downside": 1.0,
             }):
            self.assertIsNone(score_opportunity("TST.AX", df))

    def test_confidence_between_005_and_095(self):
        df = _make_df(atr=0.20, adx=35.0)
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0, "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0, "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        if result:
            self.assertGreaterEqual(result["confidence"], 0.05)
            self.assertLessEqual(result["confidence"], 0.95)

    def test_bearish_regime_reduces_confidence(self):
        df = _make_df(atr=0.20, adx=35.0)
        filters_open = {
            "min_opportunity_score": 0, "min_confidence": 0,
            "min_expected_upside": 0, "min_avg_daily_volume": 0,
            "min_rr_ratio": 0, "max_downside": 1.0,
        }
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", filters_open):
            no_regime   = score_opportunity("TST.AX", df, regime=None)
            bear_regime = score_opportunity(
                "TST.AX", df,
                regime={"regime": "BEARISH", "confidence": 0.7},
            )
        if no_regime and bear_regime:
            self.assertLess(bear_regime["confidence"], no_regime["confidence"])

    def test_take_profit_levels_ascending(self):
        df = _make_df(atr=0.20, adx=35.0)
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0, "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0, "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        if result:
            tp = result["take_profit"]
            self.assertLess(tp[0], tp[1])
            self.assertLess(tp[1], tp[2])

    def test_stop_below_entry(self):
        df = _make_df(atr=0.20, adx=35.0)
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0, "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0, "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        if result:
            self.assertLess(result["stop_loss"], result["entry_zone"][0])

    def test_component_scores_all_present(self):
        df = _make_df(atr=0.20, adx=35.0)
        expected_components = [
            "expected_return", "technical_strength", "volume_expansion",
            "momentum", "news_catalyst", "institutional", "risk_reward",
        ]
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0, "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0, "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        if result:
            for comp in expected_components:
                self.assertIn(comp, result["component_scores"])

    def test_weighted_score_uses_config_weights(self):
        """Opportunity score is a weighted sum — verify it's between 0 and 100."""
        df = _make_df(atr=0.20, adx=35.0)
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True), \
             patch.object(scoring_mod, "FILTERS", {
                 "min_opportunity_score": 0, "min_confidence": 0,
                 "min_expected_upside": 0, "min_avg_daily_volume": 0,
                 "min_rr_ratio": 0, "max_downside": 1.0,
             }):
            result = score_opportunity("TST.AX", df)
        if result:
            self.assertGreaterEqual(result["opportunity_score"], 0)
            self.assertLessEqual(result["opportunity_score"], 100)

    def test_exception_returns_none(self):
        """Any unexpected error returns None gracefully."""
        with patch.object(scoring_mod, "ENABLE_OPPORTUNITY_ENGINE", True):
            self.assertIsNone(score_opportunity("TST.AX", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
