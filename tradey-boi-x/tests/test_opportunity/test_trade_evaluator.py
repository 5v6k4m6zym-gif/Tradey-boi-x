"""
Tests for opportunity.trade_evaluator — Trade Evaluation & Filtering Layer.
Pure computation + local filesystem logging only. No network, no side effects
on engine.py's prediction model or send_alert().
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opportunity.trade_evaluator import (
    TradeEvaluator,
    EvaluationResult,
    process_trade_signal,
    log_trade_decision,
)
import opportunity.trade_evaluator as te_mod


def _make_ohlcv(n: int = 40, trend: str = "up") -> pd.DataFrame:
    np.random.seed(7)
    base = 100.0
    if trend == "up":
        closes = base + np.linspace(0, 15, n) + np.random.randn(n) * 0.2
    elif trend == "flat_noisy":
        closes = base + np.random.randn(n) * 3.0
    else:
        closes = base + np.random.randn(n) * 0.3

    opens = closes - 0.1
    highs = closes + np.abs(np.random.randn(n)) * 0.4
    lows  = closes - np.abs(np.random.randn(n)) * 0.4
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df


class TestRiskReward(unittest.TestCase):
    def test_basic_ratio(self):
        rr = TradeEvaluator.compute_risk_reward(entry=100, stop_loss=95, take_profit=112.5)
        self.assertAlmostEqual(rr, 2.5)

    def test_zero_risk_returns_zero(self):
        rr = TradeEvaluator.compute_risk_reward(entry=100, stop_loss=100, take_profit=110)
        self.assertEqual(rr, 0.0)


class TestNoiseAndPredictability(unittest.TestCase):
    def test_trending_market_is_low_noise_high_predictability(self):
        df = _make_ohlcv(trend="up")
        noise, er = TradeEvaluator.compute_noise_index(df)
        pred = TradeEvaluator.compute_predictability_score(df, er)
        self.assertLess(noise, 1.2)
        self.assertGreater(pred, 0.6)

    def test_choppy_market_is_high_noise_low_predictability(self):
        df = _make_ohlcv(trend="flat_noisy")
        noise, er = TradeEvaluator.compute_noise_index(df)
        pred = TradeEvaluator.compute_predictability_score(df, er)
        self.assertGreater(noise, 1.2)
        self.assertLess(pred, 0.6)

    def test_insufficient_data_treated_as_high_noise(self):
        df = _make_ohlcv(n=2, trend="up")
        noise, er = TradeEvaluator.compute_noise_index(df)
        self.assertGreaterEqual(noise, 1.2)


class TestEdgeScore(unittest.TestCase):
    def test_higher_probability_and_rr_increase_edge_score(self):
        low  = TradeEvaluator.compute_edge_score(0.55, 1.5, None)
        high = TradeEvaluator.compute_edge_score(0.85, 4.0, 1.5)
        self.assertGreater(high, low)

    def test_edge_score_bounded_0_1(self):
        score = TradeEvaluator.compute_edge_score(1.0, 100.0, 100.0)
        self.assertLessEqual(score, 1.0)
        score2 = TradeEvaluator.compute_edge_score(0.0, 0.0, -10.0)
        self.assertGreaterEqual(score2, 0.0)


class TestEvaluateGating(unittest.TestCase):
    def test_strong_trending_setup_passes_all_gates(self):
        df = _make_ohlcv(trend="up")
        trade = {
            "ticker": "GOOD.AX", "entry": float(df["Close"].iloc[-1]),
            "stop_loss": float(df["Close"].iloc[-1]) * 0.97,
            "take_profit": float(df["Close"].iloc[-1]) * 1.15,
            "probability": 0.85, "expected_r": 1.2,
        }
        result = TradeEvaluator().evaluate(trade, df)
        self.assertIsInstance(result, EvaluationResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.rejection_reasons, [])

    def test_weak_choppy_setup_fails_with_reasons(self):
        df = _make_ohlcv(trend="flat_noisy")
        trade = {
            "ticker": "BAD.AX", "entry": float(df["Close"].iloc[-1]),
            "stop_loss": float(df["Close"].iloc[-1]) * 0.95,
            "take_profit": float(df["Close"].iloc[-1]) * 1.05,
            "probability": 0.55, "expected_r": 0.1,
        }
        result = TradeEvaluator().evaluate(trade, df)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.rejection_reasons), 0)

    def test_custom_thresholds_are_respected(self):
        df = _make_ohlcv(trend="up")
        trade = {
            "ticker": "X.AX", "entry": 100, "stop_loss": 99, "take_profit": 100.5,
            "probability": 0.55, "expected_r": 0.05,
        }
        lenient = TradeEvaluator(thresholds={
            "min_edge_score": 0.0, "min_predictability_score": 0.0,
            "min_risk_reward": 0.0, "max_noise_index": 999.0,
        })
        result = lenient.evaluate(trade, df)
        self.assertTrue(result.passed)


class TestProcessTradeSignal(unittest.TestCase):
    def test_disabled_layer_passes_through_unchanged(self):
        te_mod.ENABLE_TRADE_EVALUATOR = False
        df = _make_ohlcv(trend="up")
        trade = {"ticker": "T.AX", "entry": 100, "stop_loss": 95, "take_profit": 115, "probability": 0.8}
        out = process_trade_signal(trade, df)
        self.assertIs(out, trade)

    def test_shadow_mode_always_returns_none_but_logs(self):
        te_mod.ENABLE_TRADE_EVALUATOR = True
        te_mod.SHADOW_MODE = True
        df = _make_ohlcv(trend="up")
        trade = {
            "ticker": "SHADOW.AX", "entry": float(df["Close"].iloc[-1]),
            "stop_loss": float(df["Close"].iloc[-1]) * 0.97,
            "take_profit": float(df["Close"].iloc[-1]) * 1.15,
            "probability": 0.85, "expected_r": 1.2,
        }
        logged = []
        original_log = te_mod.log_trade_decision
        te_mod.log_trade_decision = lambda t, e: logged.append((t, e))
        try:
            out = process_trade_signal(trade, df)
        finally:
            te_mod.log_trade_decision = original_log
            te_mod.ENABLE_TRADE_EVALUATOR = False
            te_mod.SHADOW_MODE = True
        self.assertIsNone(out)
        self.assertEqual(len(logged), 1)
        self.assertTrue(logged[0][1].passed)

    def test_live_mode_blocks_only_failing_trades(self):
        te_mod.ENABLE_TRADE_EVALUATOR = True
        te_mod.SHADOW_MODE = False
        df = _make_ohlcv(trend="flat_noisy")
        bad_trade = {
            "ticker": "BAD.AX", "entry": float(df["Close"].iloc[-1]),
            "stop_loss": float(df["Close"].iloc[-1]) * 0.95,
            "take_profit": float(df["Close"].iloc[-1]) * 1.05,
            "probability": 0.55, "expected_r": 0.1,
        }
        original_log = te_mod.log_trade_decision
        te_mod.log_trade_decision = lambda t, e: None
        try:
            out = process_trade_signal(bad_trade, df)
        finally:
            te_mod.log_trade_decision = original_log
            te_mod.ENABLE_TRADE_EVALUATOR = False
            te_mod.SHADOW_MODE = True
        self.assertIsNone(out)


class TestLogging(unittest.TestCase):
    def test_log_trade_decision_is_append_only_jsonl(self):
        tmp_path = Path("/tmp/_test_trade_eval_log.jsonl")
        if tmp_path.exists():
            tmp_path.unlink()
        original_path = te_mod.TRADE_EVAL_LOG_PATH
        te_mod.TRADE_EVAL_LOG_PATH = str(tmp_path)
        try:
            result = EvaluationResult(
                edge_score=0.7, predictability_score=0.7, noise_index=0.5,
                risk_reward=3.0, passed=True, rejection_reasons=[],
            )
            log_trade_decision({"ticker": "A.AX", "probability": 0.8}, result)
            log_trade_decision({"ticker": "B.AX", "probability": 0.6}, result)
            lines = tmp_path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            rec = json.loads(lines[0])
            self.assertEqual(rec["symbol"], "A.AX")
            self.assertIn("timestamp", rec)
            self.assertIn("rejection_reasons", rec)
        finally:
            te_mod.TRADE_EVAL_LOG_PATH = original_path
            if tmp_path.exists():
                tmp_path.unlink()

    def test_logging_failure_never_raises(self):
        result = EvaluationResult(
            edge_score=0.1, predictability_score=0.1, noise_index=5.0,
            risk_reward=0.1, passed=False, rejection_reasons=["x"],
        )
        original_path = te_mod.TRADE_EVAL_LOG_PATH
        te_mod.TRADE_EVAL_LOG_PATH = "/proc/1/root/forbidden/impossible.jsonl"
        try:
            log_trade_decision({"ticker": "Z.AX"}, result)  # must not raise
        finally:
            te_mod.TRADE_EVAL_LOG_PATH = original_path


if __name__ == "__main__":
    unittest.main()
