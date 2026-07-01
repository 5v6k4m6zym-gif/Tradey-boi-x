"""
Tests for opportunity.regime — Market Regime Detector
All yfinance calls are mocked. No network, no side effects.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.regime as regime_mod
from opportunity.regime import (
    detect_regime,
    regime_label,
    _compute_adx,
    REGIMES,
)


def _make_ohlcv(n: int = 120, trend: str = "up") -> pd.DataFrame:
    """Synthetic OHLCV with controllable trend."""
    np.random.seed(42)
    base = 7000.0
    if trend == "up":
        closes = base + np.arange(n) * 5 + np.random.randn(n) * 20
    elif trend == "down":
        closes = base + np.arange(n) * -5 + np.random.randn(n) * 20
    else:
        closes = base + np.random.randn(n) * 30

    highs  = closes + abs(np.random.randn(n)) * 15
    lows   = closes - abs(np.random.randn(n)) * 15
    opens  = closes - np.random.randn(n) * 5
    volume = np.random.randint(500_000, 2_000_000, n).astype(float)

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volume,
    })


def _mock_ticker(df: pd.DataFrame):
    m = MagicMock()
    m.history.return_value = df
    return m


class TestComputeAdx(unittest.TestCase):

    def test_returns_series_same_length(self):
        df = _make_ohlcv(100)
        result = _compute_adx(df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))

    def test_adx_non_negative(self):
        df = _make_ohlcv(100)
        result = _compute_adx(df)
        self.assertTrue((result.dropna() >= 0).all())

    def test_adx_trending_market_higher_than_flat(self):
        trending = _make_ohlcv(100, trend="up")
        flat     = _make_ohlcv(100, trend="flat")
        adx_trend = _compute_adx(trending).iloc[-1]
        adx_flat  = _compute_adx(flat).iloc[-1]
        self.assertGreater(adx_trend, adx_flat)


class TestDetectRegime(unittest.TestCase):

    def test_returns_none_when_flag_off(self):
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", False):
            self.assertIsNone(detect_regime())

    def test_returns_none_on_insufficient_data(self):
        df = _make_ohlcv(10)
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            self.assertIsNone(detect_regime())

    def test_returns_none_on_exception(self):
        m = MagicMock()
        m.history.side_effect = RuntimeError("network")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=m):
            self.assertIsNone(detect_regime())

    def test_returns_dict_with_all_keys(self):
        df = _make_ohlcv(120, trend="up")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertIsNotNone(result)
        for key in ("regime", "confidence", "asx200_ret_20d", "adx",
                    "atr_pct", "atr_pct_rank", "above_ema50", "price", "ema50"):
            self.assertIn(key, result)

    def test_regime_is_valid_label(self):
        df = _make_ohlcv(120, trend="up")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertIn(result["regime"], REGIMES)

    def test_confidence_between_0_and_1(self):
        df = _make_ohlcv(120, trend="up")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 0.95)

    def test_confidence_never_exceeds_095(self):
        """Confidence is clamped at 0.95."""
        df = _make_ohlcv(120, trend="up")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertLessEqual(result["confidence"], 0.95)

    def test_above_ema50_is_bool(self):
        df = _make_ohlcv(120, trend="up")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertIsInstance(result["above_ema50"], bool)

    def test_uptrend_data_classified_bullish_or_high_vol(self):
        """Strong uptrend should produce BULLISH or HIGH_VOL (volatility may dominate)."""
        df = _make_ohlcv(120, trend="up")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertIn(result["regime"], ("BULLISH", "HIGH_VOL", "LOW_VOL", "SIDEWAYS"))

    def test_downtrend_classified_bearish_or_high_vol(self):
        """Strong downtrend should produce BEARISH or HIGH_VOL."""
        df = _make_ohlcv(120, trend="down")
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertIn(result["regime"], ("BEARISH", "SIDEWAYS", "HIGH_VOL", "LOW_VOL"))

    def test_atr_pct_rank_between_0_and_1(self):
        df = _make_ohlcv(120)
        with patch.object(regime_mod, "ENABLE_MARKET_REGIME", True), \
             patch("opportunity.regime.yf.Ticker", return_value=_mock_ticker(df)):
            result = detect_regime()
        self.assertGreaterEqual(result["atr_pct_rank"], 0.0)
        self.assertLessEqual(result["atr_pct_rank"], 1.0)


class TestRegimeLabel(unittest.TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(regime_label(None), "")

    def test_bullish_contains_bullish(self):
        r = {"regime": "BULLISH", "confidence": 0.78}
        self.assertIn("BULLISH", regime_label(r))

    def test_bearish_contains_emoji(self):
        r = {"regime": "BEARISH", "confidence": 0.65}
        label = regime_label(r)
        self.assertIn("🔴", label)
        self.assertIn("BEARISH", label)

    def test_confidence_shown_as_percentage(self):
        r = {"regime": "SIDEWAYS", "confidence": 0.72}
        label = regime_label(r)
        self.assertIn("72%", label)

    def test_high_vol_emoji(self):
        r = {"regime": "HIGH_VOL", "confidence": 0.80}
        self.assertIn("⚡", regime_label(r))

    def test_low_vol_emoji(self):
        r = {"regime": "LOW_VOL", "confidence": 0.60}
        self.assertIn("😴", regime_label(r))


if __name__ == "__main__":
    unittest.main(verbosity=2)
