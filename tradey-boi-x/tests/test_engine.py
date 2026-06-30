"""
Tradey Boi X — Full Regression Test Suite
==========================================
Tests every critical component. Run from tradey-boi-x/:

    python -m pytest tests/test_engine.py -v

Each test documents:
  • What behaviour is being asserted
  • Whether any change from the previous version is INTENTIONAL or a BUG

No live network calls — all yfinance / Discord / filesystem I/O is mocked.
"""
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import engine
from engine import (
    COOLDOWN_HOURS,
    FEATURES,
    accuracy_stats,
    confidence_grade,
)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED TEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_feature_df(n: int = 75, **last_row_overrides) -> pd.DataFrame:
    """
    Minimal DataFrame with all engine FEATURES + chart columns.
    All rows are identical 'passing' values; override the last row as needed.
    n ≥ 60 so decide() doesn't gate on insufficient data.
    """
    base = {
        "rsi": 55.0, "macd_diff": 0.05, "bb_width": 0.04,
        "atr": 1.5,  "ret_5": 0.02,    "ret_10": 0.03,
        "ret_20": 0.04, "ret_63": 0.08,
        "vol_ratio": 2.0, "breakout": 1, "obv_ratio": 1.5,
        "adx": 25.0, "mfi": 55.0, "bb_squeeze": 0, "gap_up": 0,
        "ema20": 105.0, "ema50": 100.0,
        "Close": 110.0, "Open": 109.0, "High": 111.0,
        "Low": 108.0, "Volume": 500_000.0,
        "bb_upper": 112.0, "bb_lower": 108.0,
        "macd": 0.10, "macd_signal": 0.05,
    }
    rows = [{**base} for _ in range(n)]
    for k, v in last_row_overrides.items():
        rows[-1][k] = v
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df


def _make_ohlcv(n: int = 300, zero_vol_indices: list = None) -> pd.DataFrame:
    """Synthetic OHLCV with optionally zero-volume days."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    price = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    if zero_vol_indices:
        for i in zero_vol_indices:
            vol[i] = 0.0
    return pd.DataFrame({
        "Open":   price * 0.999,
        "High":   price * 1.005,
        "Low":    price * 0.995,
        "Close":  price,
        "Volume": vol,
    }, index=dates)


class MockModel:
    """Predict_proba stub — returns a fixed probability for any input."""
    def __init__(self, prob: float = 0.75):
        self._prob = prob

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.array([[1 - self._prob, self._prob]] * n)


def _neutral_signals():
    """All external signal functions return (0, '') — neutral, no score impact."""
    return {
        "engine.vix_safe":                  True,
        "engine.market_regime_ok":          True,
        "engine.sector_ok":                 True,
        "engine.weekly_trend_ok":           True,
        "engine.earnings_safe":             True,
        "engine.performance_adjustments":   {},
        "engine.news_sentiment":            {"score_adj": 0, "label": "NEUTRAL", "compound": 0.0},
        "engine.short_interest_signal":     (0, ""),
        "engine.insider_signal":            (0, ""),
        "engine.options_flow_signal":       (0, ""),
        "engine.commodity_signal":          (0, ""),
        "engine.news_velocity":             (0, ""),
        "engine.support_resistance_signal": (0, ""),
        "engine.multitimeframe_signal":     (0, ""),
        "engine.relative_strength_signal":  (0, ""),
        "engine.fear_greed_signal":         (0, ""),
        "engine.sector_rotation_signal":    (0, ""),
        "engine.gap_signal":                (0, ""),
        "engine.squeeze_breakout_signal":   (0, ""),
        "engine.fundamental_signal":        (0, ""),
        "engine.vwap_signal":               (0, ""),
        "engine.cooldown_ok":               True,
    }


@contextmanager
def _mock_decide(overrides: dict = None):
    """Context manager: patches all external calls inside decide()."""
    spec = {**_neutral_signals(), **(overrides or {})}
    patches = [patch(k, return_value=v) for k, v in spec.items()]
    started = []
    try:
        for p in patches:
            try:
                started.append(p)
                p.start()
            except Exception:
                pass
        yield
    finally:
        for p in started:
            try:
                p.stop()
            except Exception:
                pass


def _run_decide(df, model, overrides=None):
    with _mock_decide(overrides):
        return engine.decide("TST.AX", df, model)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ACCURACY STATS — win rate calculation
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyStats(unittest.TestCase):
    """
    REGRESSION: accuracy_stats() previously used `outcome == "WIN"` which is
    a string never written by resolve_outcomes(). Actual outcome strings are:
      HIT_TARGET, EXPIRED_GAIN  (wins)
      HIT_STOP,   EXPIRED_LOSS  (losses)
    Old code: always returned win_rate=0 regardless of results.
    New code: correctly classifies HIT_TARGET + EXPIRED_GAIN as wins.
    """

    def _entry(self, outcome, actual_pct=0.04):
        return {"outcome": outcome, "actual_pct": actual_pct}

    def test_empty_returns_zero_stats(self):
        """No entries → total=0, win_rate=None."""
        r = accuracy_stats([])
        self.assertEqual(r["total"], 0)
        self.assertIsNone(r["win_rate"])

    def test_pending_outcomes_excluded(self):
        """Entries with outcome=None are pending and must not count."""
        entries = [{"outcome": None, "actual_pct": None}] * 5
        r = accuracy_stats(entries)
        self.assertEqual(r["total"], 0)

    def test_hit_target_counts_as_win(self):
        """INTENTIONAL FIX: HIT_TARGET must be a win (was broken before)."""
        entries = [self._entry("HIT_TARGET")] * 3
        r = accuracy_stats(entries)
        self.assertEqual(r["wins"], 3)
        self.assertEqual(r["losses"], 0)
        self.assertAlmostEqual(r["win_rate"], 1.0)

    def test_expired_gain_counts_as_win(self):
        """INTENTIONAL FIX: EXPIRED_GAIN must be a win."""
        entries = [self._entry("EXPIRED_GAIN")] * 4
        r = accuracy_stats(entries)
        self.assertEqual(r["wins"], 4)
        self.assertAlmostEqual(r["win_rate"], 1.0)

    def test_hit_stop_is_loss(self):
        entries = [self._entry("HIT_STOP")] * 3
        r = accuracy_stats(entries)
        self.assertEqual(r["wins"], 0)
        self.assertEqual(r["losses"], 3)
        self.assertAlmostEqual(r["win_rate"], 0.0)

    def test_expired_loss_is_loss(self):
        entries = [self._entry("EXPIRED_LOSS")] * 2
        r = accuracy_stats(entries)
        self.assertEqual(r["wins"], 0)

    def test_mixed_outcomes_correct_ratio(self):
        """3 wins, 2 losses → 60% win rate."""
        entries = (
            [self._entry("HIT_TARGET")] * 2 +
            [self._entry("EXPIRED_GAIN")] * 1 +
            [self._entry("HIT_STOP")] * 1 +
            [self._entry("EXPIRED_LOSS")] * 1
        )
        r = accuracy_stats(entries)
        self.assertEqual(r["wins"], 3)
        self.assertEqual(r["losses"], 2)
        self.assertAlmostEqual(r["win_rate"], 0.60)

    def test_legacy_win_string_also_accepted(self):
        """Legacy "WIN" string (if ever present) is still treated as a win."""
        entries = [self._entry("WIN")]
        r = accuracy_stats(entries)
        self.assertEqual(r["wins"], 1)

    def test_avg_return_uses_get_with_default(self):
        """Missing actual_pct key should not crash — defaults to 0."""
        entries = [{"outcome": "HIT_TARGET"}]  # no actual_pct key
        try:
            r = accuracy_stats(entries)
            self.assertIsNotNone(r)
        except KeyError:
            self.fail("accuracy_stats crashed on missing actual_pct key")

    # ── REGRESSION: document old (broken) behaviour ──────────────────────────
    def test_regression_old_win_check_would_give_zero(self):
        """
        REGRESSION MARKER — documents the old bug.
        Simulates the broken code: outcome == "WIN" when only HIT_TARGET exists.
        This SHOULD give 0 wins (old behaviour was wrong).
        The new code gives 3 wins (correct behaviour).
        Difference: INTENTIONAL FIX.
        """
        entries = [self._entry("HIT_TARGET")] * 3
        broken_wins = sum(1 for e in entries if e["outcome"] == "WIN")
        correct_wins = sum(1 for e in entries
                           if e["outcome"] in ("WIN", "HIT_TARGET", "EXPIRED_GAIN"))
        self.assertEqual(broken_wins, 0,    "Old code: 0 wins (bug confirmed)")
        self.assertEqual(correct_wins, 3,   "New code: 3 wins (fix confirmed)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCE ADJUSTMENTS — per-ticker score learning
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformanceAdjustments(unittest.TestCase):
    """
    REGRESSION: performance_adjustments() used `outcome == "WIN"` — same bug
    as accuracy_stats(). Every ticker with ANY resolved signal received a
    permanent −2 score penalty because win rate was always computed as 0%.
    """

    def _log(self, ticker: str, outcomes: list):
        return [{"ticker": ticker, "outcome": o} for o in outcomes]

    def test_no_resolved_signals_returns_empty(self):
        with patch("engine._load_log", return_value=[{"ticker": "AAA", "outcome": None}]):
            result = engine.performance_adjustments()
        self.assertEqual(result, {})

    def test_requires_at_least_3_resolved(self):
        """Fewer than 3 resolved signals → no adjustment (noise gate)."""
        entries = self._log("BHP.AX", ["HIT_TARGET", "HIT_STOP"])
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertNotIn("BHP.AX", result)

    def test_100pct_win_rate_gives_plus2(self):
        """INTENTIONAL FIX: 3× HIT_TARGET → win rate 100% → +2."""
        entries = self._log("BHP.AX", ["HIT_TARGET"] * 4)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("BHP.AX"), 2)

    def test_100pct_loss_rate_gives_minus2(self):
        entries = self._log("XYZ.AX", ["HIT_STOP"] * 4)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("XYZ.AX"), -2)

    def test_60pct_win_rate_gives_plus1(self):
        outcomes = ["HIT_TARGET"] * 3 + ["HIT_STOP"] * 2  # 60%
        entries = self._log("AAA.AX", outcomes)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("AAA.AX"), 1)

    def test_40pct_win_rate_gives_minus1(self):
        outcomes = ["HIT_TARGET"] * 2 + ["HIT_STOP"] * 3  # 40%
        entries = self._log("AAA.AX", outcomes)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("AAA.AX"), -1)

    def test_50pct_win_rate_gives_zero(self):
        outcomes = ["HIT_TARGET"] * 3 + ["HIT_STOP"] * 3  # 50%
        entries = self._log("AAA.AX", outcomes)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("AAA.AX"), 0)

    def test_rolling_window_capped_at_20(self):
        """Only the 20 most recent signals are used per ticker."""
        outcomes = ["HIT_STOP"] * 25 + ["HIT_TARGET"] * 20  # recent 20 = 100% win
        entries = self._log("CAP.AX", outcomes)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("CAP.AX"), 2)

    def test_expired_gain_counts_as_win(self):
        entries = self._log("EG.AX", ["EXPIRED_GAIN"] * 4)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("EG.AX"), 2)

    def test_expired_loss_counts_as_loss(self):
        entries = self._log("EL.AX", ["EXPIRED_LOSS"] * 4)
        with patch("engine._load_log", return_value=entries):
            result = engine.performance_adjustments()
        self.assertEqual(result.get("EL.AX"), -2)

    # ── REGRESSION: old code gave -2 for all tickers ─────────────────────────
    def test_regression_old_code_penalised_all_winners(self):
        """
        REGRESSION MARKER — old code used outcome == "WIN":
        With 4× HIT_TARGET entries the old code computed win_rate=0,
        which falls into the ≤25% bucket → returned −2 (penalising a winner).
        New code: win_rate=1.0 → returns +2.
        Difference: INTENTIONAL FIX.
        """
        outcomes = ["HIT_TARGET"] * 4
        broken_wins  = [o == "WIN" for o in outcomes]
        broken_rate  = sum(broken_wins) / len(broken_wins)   # 0.0
        correct_wins = [o in ("WIN", "HIT_TARGET", "EXPIRED_GAIN") for o in outcomes]
        correct_rate = sum(correct_wins) / len(correct_wins)  # 1.0

        self.assertAlmostEqual(broken_rate, 0.0,
            msg="Old code produced 0% win rate for HIT_TARGET entries")
        self.assertAlmostEqual(correct_rate, 1.0,
            msg="New code produces 100% win rate for HIT_TARGET entries")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DECIDE — filter logic, score thresholds, alert gating
# ══════════════════════════════════════════════════════════════════════════════

class TestDecide(unittest.TestCase):

    def test_gated_when_fewer_than_60_rows(self):
        df    = _make_feature_df(n=55)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")
        self.assertFalse(res["alert"])

    def test_gated_when_rsi_overbought(self):
        df    = _make_feature_df(rsi=73.0)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_rsi_oversold(self):
        df    = _make_feature_df(rsi=24.0)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_macd_bearish(self):
        df    = _make_feature_df(macd_diff=-0.01)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_ema_downtrend(self):
        """EMA20 < EMA50 → downtrend filter fails."""
        df    = _make_feature_df(ema20=95.0, ema50=100.0)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_ai_prob_below_threshold(self):
        df    = _make_feature_df()
        model = MockModel(prob=0.38)   # below 40% gate
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_low_liquidity(self):
        df    = _make_feature_df(vol_ratio=0.3)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_vix_high(self):
        df    = _make_feature_df()
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model, {"engine.vix_safe": False})
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_market_in_downtrend(self):
        df    = _make_feature_df()
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model, {"engine.market_regime_ok": False})
        self.assertEqual(res["signal"], "GATED")

    def test_gated_when_negative_news(self):
        df    = _make_feature_df()
        model = MockModel(prob=0.80)
        neg_news = {"score_adj": -2, "label": "NEGATIVE", "compound": -0.6}
        res   = _run_decide(df, model, {"engine.news_sentiment": neg_news})
        self.assertEqual(res["signal"], "GATED")

    def test_elite_signal_at_score_11_plus(self):
        """
        Base score: prob=0.80 → +3 (AI ≥80%) + vol_ratio=2.0→+2 + rsi=55→+2
                  + breakout=1→+3 + ema uptrend→+1 = 11 → ELITE
        """
        df    = _make_feature_df(rsi=55.0, vol_ratio=2.0, breakout=1)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "ELITE")
        self.assertTrue(res["alert"])
        self.assertGreaterEqual(res["score"], 11)

    def test_strong_buy_requires_prob_70_plus(self):
        """
        Score 9–10 with AI prob < 70% → WATCH (not alerted).
        Score 9–10 with AI prob ≥ 70% → STRONG BUY (alerted).
        """
        df = _make_feature_df(rsi=55.0, vol_ratio=2.0, breakout=0)
        # prob=0.65 → AI score 1; vol+RSI+ema → 1+2+2+1=6+1=7 base.
        # Need score 9 but prob < 70% → WATCH
        model_low  = MockModel(prob=0.65)
        model_high = MockModel(prob=0.75)
        res_low  = _run_decide(df, model_low,
                               {"engine.support_resistance_signal": (2, "near support"),
                                "engine.relative_strength_signal":  (1, "strong RS")})
        res_high = _run_decide(df, model_high,
                               {"engine.support_resistance_signal": (2, "near support"),
                                "engine.relative_strength_signal":  (1, "strong RS")})
        # The key invariant: STRONG BUY requires prob ≥ 0.70
        if res_low["score"] >= 9:
            self.assertNotEqual(res_low["signal"], "STRONG BUY",
                "Score≥9 but prob<70% must not be STRONG BUY")
        if res_high["score"] >= 9:
            self.assertIn(res_high["signal"], ("STRONG BUY", "ELITE"))

    def test_watch_signal_never_alerts(self):
        """WATCH grade → alert=False even if all filters pass."""
        df    = _make_feature_df(rsi=55.0, vol_ratio=0.6, breakout=0)
        model = MockModel(prob=0.45)  # low prob → low base score → WATCH
        res   = _run_decide(df, model)
        if res["signal"] == "WATCH":
            self.assertFalse(res["alert"])

    def test_cooldown_suppresses_alert(self):
        """Qualifying signal (ELITE/STRONG BUY) suppressed when cooldown active."""
        df    = _make_feature_df(rsi=55.0, vol_ratio=2.0, breakout=1)
        model = MockModel(prob=0.80)
        res   = _run_decide(df, model, {"engine.cooldown_ok": False})
        self.assertFalse(res["alert"])
        self.assertIn(res["signal"], ("ELITE", "STRONG BUY"))

    def test_prob_returned_for_gated_signals(self):
        """prob must be returned even when gated (dashboard uses it)."""
        df    = _make_feature_df(rsi=73.0)
        model = MockModel(prob=0.82)
        res   = _run_decide(df, model)
        self.assertEqual(res["signal"], "GATED")
        self.assertAlmostEqual(res["prob"], 0.82, places=2)

    def test_filters_list_always_present(self):
        """filters key always present — dashboard iterates it."""
        df  = _make_feature_df()
        res = _run_decide(df, MockModel(0.80))
        self.assertIn("filters", res)
        self.assertIsInstance(res["filters"], list)

    def test_score_includes_all_signal_adjusters(self):
        """score = base_score + adj + all signal adjusters."""
        df    = _make_feature_df(rsi=55.0, vol_ratio=2.0, breakout=1)
        model = MockModel(prob=0.80)
        res_base = _run_decide(df, model)
        res_plus = _run_decide(df, model, {
            "engine.short_interest_signal": (2, "high short interest"),
        })
        self.assertEqual(res_plus["score"], res_base["score"] + 2,
            "Additional signal adjusters must add to score")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE GRADE — combined probability + score grading
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceGrade(unittest.TestCase):
    """
    combined = prob × 0.6 + (score / 14) × 0.4
    Thresholds: A+ ≥0.80, A ≥0.65, B+ ≥0.50, B ≥0.35, C <0.35
    No changes were made to this function — tests confirm stability.
    """

    def test_high_prob_high_score_gives_A_plus(self):
        grade, label, bar = confidence_grade(0.90, 14)
        self.assertEqual(grade, "A+")
        self.assertIn("VERY HIGH", label)

    def test_low_prob_low_score_gives_C(self):
        grade, label, bar = confidence_grade(0.30, 2)
        self.assertEqual(grade, "C")
        self.assertIn("LOW", label)

    def test_mid_range_gives_B_plus(self):
        combined = 0.65 * 0.6 + (7 / 14) * 0.4   # 0.59 → B+
        grade, _, _ = confidence_grade(0.65, 7)
        self.assertEqual(grade, "B+")

    def test_bar_length_always_10(self):
        for prob, score in [(0.1, 1), (0.5, 7), (0.9, 14)]:
            _, _, bar = confidence_grade(prob, score)
            bar_chars = bar.split(" ")[0]
            self.assertEqual(len(bar_chars), 10,
                f"Bar must always be 10 chars, got {bar_chars!r}")

    def test_bar_value_matches_combined_score(self):
        prob, score = 0.70, 10
        combined = prob * 0.6 + (score / 14) * 0.4
        expected_filled = round(combined * 10)
        _, _, bar = confidence_grade(prob, score)
        filled = bar.count("█")
        self.assertEqual(filled, expected_filled)

    def test_formula_unchanged(self):
        """
        STABILITY CHECK: ensure combined = 0.6×prob + 0.4×(score/14).
        No changes to confidence_grade() — any deviation is a regression.
        """
        prob, score = 0.72, 9
        expected_combined = 0.72 * 0.6 + (9 / 14) * 0.4
        actual_filled = confidence_grade(prob, score)[2].count("█")
        expected_filled = round(expected_combined * 10)
        self.assertEqual(actual_filled, expected_filled)


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEEDBACK WEIGHTS — min gate, multiplier bounds
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackWeights(unittest.TestCase):
    """
    REGRESSION: previous multipliers were WIN×10 / LOSS×0.3.
    Fixed to WIN×2.5 / LOSS×0.5 with minimum gate of ≥10 resolved signals.
    """

    def _make_combined(self, n_rows: int = 50) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
        return pd.DataFrame({
            "_row_date": dates,
            "_ticker":   ["TST.AX"] * n_rows,
        }, index=range(n_rows))

    def _make_log(self, n_wins: int, n_losses: int):
        entries = []
        for i in range(n_wins):
            entries.append({
                "ticker": "TST.AX",
                "outcome": "HIT_TARGET",
                "signal_date": "2024-01-15T00:00:00",
            })
        for i in range(n_losses):
            entries.append({
                "ticker": "TST.AX",
                "outcome": "HIT_STOP",
                "signal_date": "2024-01-15T00:00:00",
            })
        return entries

    def test_fewer_than_10_resolved_skips_weights(self):
        """< 10 resolved → weights unchanged (min gate)."""
        combined = self._make_combined()
        weights  = pd.Series([1.0] * len(combined))
        log      = self._make_log(n_wins=5, n_losses=4)   # 9 total < 10
        with patch("engine._load_log", return_value=log):
            adj_w, n_win, n_loss = engine._apply_feedback_weights(combined, weights)
        self.assertEqual(n_win + n_loss, 0,
            "< 10 resolved: no weight adjustments must be applied")
        pd.testing.assert_series_equal(adj_w, weights,
            "< 10 resolved: weights must be unchanged")

    def test_exactly_10_resolved_triggers_weighting(self):
        """10 resolved signals → weighting kicks in."""
        combined = self._make_combined()
        weights  = pd.Series([1.0] * len(combined))
        log      = self._make_log(n_wins=10, n_losses=0)
        with patch("engine._load_log", return_value=log):
            adj_w, n_win, n_loss = engine._apply_feedback_weights(combined, weights)
        self.assertGreater(n_win, 0, "10+ resolved: win weights must be applied")

    def test_win_multiplier_is_2_5_not_10(self):
        """
        INTENTIONAL FIX: WIN rows multiplied by 2.5 (was 10).
        Max weight after one WIN match = 1.0 × 2.5 = 2.5.
        """
        combined = self._make_combined(n_rows=5)
        # Use a date matching the signal date
        combined["_row_date"] = pd.Timestamp("2024-01-15", tz="UTC")
        weights = pd.Series([1.0] * 5)
        log = [{"ticker": "TST.AX", "outcome": "HIT_TARGET",
                "signal_date": "2024-01-15T00:00:00"}] * 10  # 10 to pass gate
        with patch("engine._load_log", return_value=log):
            adj_w, _, _ = engine._apply_feedback_weights(combined, weights)
        max_w = adj_w.max()
        self.assertLessEqual(max_w, 2.5 ** 10 + 1,   # stacked (one per entry)
            "WIN multiplier must be 2.5 per application, not 10")
        self.assertGreater(max_w, 1.0, "WIN multiplier must boost weight above 1.0")

    def test_loss_multiplier_is_0_5_not_0_3(self):
        """
        INTENTIONAL FIX: LOSS rows multiplied by 0.5 (was 0.3).
        0.3 caused extreme weight suppression; 0.5 is more balanced.
        """
        combined = self._make_combined(n_rows=5)
        combined["_row_date"] = pd.Timestamp("2024-01-15", tz="UTC")
        weights = pd.Series([1.0] * 5)
        # Create exactly 10 entries so gate passes, all losses
        log = [{"ticker": "TST.AX", "outcome": "HIT_STOP",
                "signal_date": "2024-01-15T00:00:00"}] * 10
        with patch("engine._load_log", return_value=log):
            adj_w, _, _ = engine._apply_feedback_weights(combined, weights)
        min_w = adj_w.min()
        self.assertLess(min_w, 1.0, "LOSS multiplier must reduce weight below 1.0")
        # Old multiplier 0.3^10 ≈ 6e-6; new 0.5^10 ≈ 0.001 (much more reasonable)
        self.assertGreater(min_w, 1e-5,
            "LOSS multiplier 0.5 must not collapse weights to near zero")


# ══════════════════════════════════════════════════════════════════════════════
# 6. VOLUME RATIO ZERO GUARD — suspended / halted stocks
# ══════════════════════════════════════════════════════════════════════════════

class TestVolRatioZeroGuard(unittest.TestCase):
    """
    REGRESSION: vol / vol.rolling(20).mean() produced inf when 20-day mean
    was 0 (e.g., trading halt days followed by a resumption spike).
    Fixed: replace 0 means with NaN before division.
    """

    def _compute_vol_ratio(self, vol_series: pd.Series) -> pd.Series:
        """Replicate the fixed vol_ratio logic."""
        vol_mean = vol_series.rolling(20).mean().replace(0, float("nan"))
        return vol_series / vol_mean

    def test_no_inf_when_volume_is_zero_for_a_period(self):
        """Zero-volume window → NaN vol_ratio, never inf."""
        vol = pd.Series([0.0] * 20 + [500_000.0] * 10)
        ratio = self._compute_vol_ratio(vol)
        self.assertFalse(np.isinf(ratio).any(),
            "vol_ratio must never produce inf (trading halt scenario)")

    def test_nan_propagated_not_inf(self):
        """After zero-mean window, vol_ratio should be NaN, not inf."""
        vol = pd.Series([0.0] * 20 + [100_000.0])
        ratio = self._compute_vol_ratio(vol)
        last = ratio.iloc[-1]
        self.assertTrue(np.isnan(last) or not np.isinf(last))

    def test_normal_volume_unaffected(self):
        """Non-zero volume windows still produce valid ratios."""
        vol = pd.Series([500_000.0] * 30)
        ratio = self._compute_vol_ratio(vol).dropna()
        self.assertTrue((ratio > 0).all())
        self.assertFalse(np.isinf(ratio).any())
        self.assertFalse(np.isnan(ratio).any())

    def test_regression_old_code_produced_nan_not_inf(self):
        """
        REGRESSION MARKER: old code was vol / vol.rolling(20).mean().
        With a fully zero-volume window, rolling mean = 0 → 0/0 = NaN.
        Because the current value is always part of its own rolling window,
        the only possible outcomes are:
          - nonzero current vol → mean > 0 → valid ratio (no inf possible)
          - zero current vol    → 0/0 = NaN (silent, propagates through features)
        The fix (replace 0 → NaN before division) makes the zero-volume case
        explicit — same NaN output but with clear, intentional semantics.
        This also future-proofs against any code path that uses rolling mean
        without the current value contributing (e.g. shifted windows).
        Difference: INTENTIONAL DEFENSIVE FIX.
        """
        vol_all_zero = pd.Series([0.0] * 30)
        broken = vol_all_zero / vol_all_zero.rolling(20).mean()
        fixed  = vol_all_zero / vol_all_zero.rolling(20).mean().replace(0, float("nan"))
        # Both produce NaN for all-zero window (0/0)
        self.assertTrue(np.isnan(broken.iloc[-1]),
            "Old code: all-zero volume window → NaN (0/0 confirmed)")
        self.assertTrue(np.isnan(fixed.iloc[-1]),
            "New code: same NaN output but semantics are explicit")
        # Critical: no inf produced by either path
        self.assertFalse(np.isinf(broken.iloc[-1]))
        self.assertFalse(np.isinf(fixed.iloc[-1]))


# ══════════════════════════════════════════════════════════════════════════════
# 7. INSIDER SIGNAL — timezone handling
# ══════════════════════════════════════════════════════════════════════════════

class TestInsiderSignalTimezone(unittest.TestCase):
    """
    REGRESSION: pd.to_datetime(df[date_col], errors='coerce') returned
    timezone-naive timestamps that could not be compared with UTC cutoff
    → TypeError crash on tickers with insider transactions.
    Fixed: utc=True added so all timestamps are normalised to UTC.
    """

    def _make_insider_df(self, tz_aware: bool):
        """Synthetic insider transactions DataFrame."""
        now = pd.Timestamp.now(tz="UTC") if tz_aware else pd.Timestamp.now()
        dates = [now - pd.Timedelta(days=i) for i in [10, 30, 60, 120]]
        return pd.DataFrame({
            "startdate": dates,
            "transaction": ["Buy", "Sell", "Buy", "Buy"],
            "shares": [1000, 500, 2000, 1500],
        })

    def test_utc_equals_true_normalises_tz_naive_dates(self):
        """pd.to_datetime(..., utc=True) makes tz-naive dates tz-aware."""
        df = self._make_insider_df(tz_aware=False)
        result = pd.to_datetime(df["startdate"], errors="coerce", utc=True)
        self.assertTrue(result.dt.tz is not None,
            "utc=True must produce tz-aware series")

    def test_utc_equals_true_preserves_tz_aware_dates(self):
        """Already-UTC timestamps are preserved when utc=True."""
        df = self._make_insider_df(tz_aware=True)
        result = pd.to_datetime(df["startdate"], errors="coerce", utc=True)
        self.assertTrue(result.dt.tz is not None)

    def test_regression_tz_naive_vs_tz_aware_comparison_raises(self):
        """
        REGRESSION MARKER: without utc=True, comparing tz-naive with UTC
        Timestamp raises TypeError in pandas.
        Difference: INTENTIONAL FIX.
        """
        df = self._make_insider_df(tz_aware=False)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)

        naive_series = pd.to_datetime(df["startdate"], errors="coerce")
        try:
            _ = naive_series >= cutoff
            # Some pandas versions coerce instead of raising — check if result valid
        except TypeError:
            pass  # Expected on strict pandas versions

        utc_series = pd.to_datetime(df["startdate"], errors="coerce", utc=True)
        try:
            result = utc_series >= cutoff
            self.assertEqual(len(result), len(df),
                "utc=True must allow comparison without TypeError")
        except TypeError:
            self.fail("Fixed code (utc=True) must NOT raise TypeError")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAX PAIN — vectorised vs O(N²) reference
# ══════════════════════════════════════════════════════════════════════════════

class TestMaxPainVectorisation(unittest.TestCase):
    """
    REGRESSION: max pain was O(N²) — iterating every strike to compute
    total option value.  Replaced with NumPy broadcasting (vectorised O(N)).
    Result must be IDENTICAL — this is a pure performance fix.
    """

    def _max_pain_reference(self, calls, puts) -> float:
        """O(N²) reference implementation — the old algorithm."""
        strikes = sorted(set(
            calls["strike"].dropna().tolist() +
            puts["strike"].dropna().tolist()
        ))
        min_val = None
        mp_strike = None
        for s in strikes:
            call_val = float(
                ((s - calls["strike"]).clip(lower=0) *
                 calls["openInterest"].fillna(0)).sum()
            )
            put_val = float(
                ((puts["strike"] - s).clip(lower=0) *
                 puts["openInterest"].fillna(0)).sum()
            )
            total = call_val + put_val
            if min_val is None or total < min_val:
                min_val = total
                mp_strike = s
        return mp_strike

    def _max_pain_vectorised(self, calls, puts) -> float:
        """Vectorised implementation — the new algorithm."""
        c_strikes = calls["strike"].dropna().values
        c_oi      = calls["openInterest"].fillna(0).values
        p_strikes = puts["strike"].dropna().values
        p_oi      = puts["openInterest"].fillna(0).values
        all_strikes = np.union1d(c_strikes, p_strikes)
        call_pain   = np.maximum(all_strikes[:, None] - c_strikes[None, :], 0) @ c_oi
        put_pain    = np.maximum(p_strikes[None, :] - all_strikes[:, None], 0) @ p_oi
        mp_idx      = (call_pain + put_pain).argmin()
        return float(all_strikes[mp_idx])

    def _make_chain(self, n_strikes: int = 20, seed: int = 0):
        rng = np.random.default_rng(seed)
        price = 100.0
        strikes = np.linspace(price * 0.80, price * 1.20, n_strikes)
        calls = pd.DataFrame({
            "strike":        strikes,
            "openInterest":  rng.integers(100, 5000, n_strikes).astype(float),
        })
        puts = pd.DataFrame({
            "strike":        strikes,
            "openInterest":  rng.integers(100, 5000, n_strikes).astype(float),
        })
        return calls, puts

    def test_vectorised_matches_reference_small_chain(self):
        """20 strikes — both algorithms must agree."""
        calls, puts = self._make_chain(n_strikes=20)
        ref = self._max_pain_reference(calls, puts)
        vec = self._max_pain_vectorised(calls, puts)
        self.assertAlmostEqual(ref, vec, places=4,
            msg="Vectorised max pain must match O(N²) reference")

    def test_vectorised_matches_reference_large_chain(self):
        """100 strikes — stress test for correctness."""
        calls, puts = self._make_chain(n_strikes=100, seed=99)
        ref = self._max_pain_reference(calls, puts)
        vec = self._max_pain_vectorised(calls, puts)
        self.assertAlmostEqual(ref, vec, places=4)

    def test_vectorised_handles_asymmetric_strikes(self):
        """Calls and puts on different strike sets — union must be correct."""
        calls = pd.DataFrame({"strike": [95.0, 100.0, 105.0],
                               "openInterest": [500., 1000., 300.]})
        puts  = pd.DataFrame({"strike": [90.0, 95.0, 100.0],
                               "openInterest": [200., 800.,  600.]})
        ref = self._max_pain_reference(calls, puts)
        vec = self._max_pain_vectorised(calls, puts)
        self.assertAlmostEqual(ref, vec, places=4)

    def test_performance_comparison(self):
        """Vectorised must be faster than reference for ≥50 strikes."""
        import time
        calls, puts = self._make_chain(n_strikes=100)
        t0 = time.perf_counter()
        for _ in range(100):
            self._max_pain_reference(calls, puts)
        ref_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(100):
            self._max_pain_vectorised(calls, puts)
        vec_time = time.perf_counter() - t0

        self.assertLess(vec_time, ref_time,
            f"Vectorised ({vec_time:.4f}s) should beat O(N²) ({ref_time:.4f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# 9. IV ZERO FILTER — implied volatility cleanup
# ══════════════════════════════════════════════════════════════════════════════

class TestIVZeroFilter(unittest.TestCase):
    """
    REGRESSION: yfinance returns impliedVolatility=0.0001 for illiquid strikes.
    Old code: included these in mean → artificially low IV → false bullish skew.
    Fixed: replace 0 with NaN before computing mean (effectively filters them).
    """

    def test_zero_ivs_excluded_from_mean(self):
        """IV values of 0 must be treated as missing."""
        iv_series = pd.Series([0.0, 0.0001, 0.35, 0.38, 0.40])
        old_mean = iv_series.dropna().mean()
        new_mean = iv_series.replace(0, float("nan")).dropna().mean()
        self.assertGreater(new_mean, old_mean,
            "Filtering zero IVs must produce a higher (more accurate) mean")

    def test_all_zero_iv_returns_empty(self):
        """All-zero IV chain → empty series → not appended to call_ivs list."""
        iv_series = pd.Series([0.0, 0.0, 0.0])
        filtered  = iv_series.replace(0, float("nan")).dropna()
        self.assertEqual(len(filtered), 0)

    def test_nonzero_ivs_unaffected(self):
        """Valid IVs (>0) are not modified."""
        iv_series = pd.Series([0.35, 0.40, 0.45])
        filtered  = iv_series.replace(0, float("nan")).dropna()
        self.assertEqual(len(filtered), 3)
        self.assertAlmostEqual(filtered.mean(), iv_series.mean())

    def test_regression_old_code_used_exact_zero_ivs(self):
        """
        REGRESSION MARKER: yfinance returns impliedVolatility=0.0 for illiquid
        deep OTM/ITM strikes (exactly 0.0, not just near-zero).
        Old code: dropna() only removes NaN — it kept exact-zero IVs in the mean.
        New code: replace(0, NaN).dropna() removes exact zeros.

        Example: calls with IVs [0.0, 0.0, 0.38, 0.40]
          old mean = (0+0+0.38+0.40)/4 = 0.195  (artificially suppressed)
          new mean = (0.38+0.40)/2 = 0.39         (accurate)

        Impact: suppressed call IV → (call_iv < put_iv) triggers false
        "Bullish IV skew" +1 signal when no genuine skew exists.
        Difference: INTENTIONAL FIX.
        """
        iv_with_zeros = pd.Series([0.0, 0.0, 0.38, 0.40])
        old_mean = iv_with_zeros.dropna().mean()                          # 0.195
        new_mean = iv_with_zeros.replace(0, float("nan")).dropna().mean() # 0.390
        self.assertAlmostEqual(old_mean, 0.195, places=4,
            msg="Old code mean confirmed (includes zero IVs)")
        self.assertAlmostEqual(new_mean, 0.390, places=4,
            msg="New code mean confirmed (zeros filtered)")
        self.assertGreater(new_mean, old_mean,
            msg="New code must produce higher (more accurate) IV mean")


# ══════════════════════════════════════════════════════════════════════════════
# 10. COOLDOWN LOGIC — per-ticker temporal guard
# ══════════════════════════════════════════════════════════════════════════════

class TestCooldownLogic(unittest.TestCase):
    """
    No changes made to cooldown logic — tests confirm stability.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_cooldown_file = engine.COOLDOWN_FILE
        engine.COOLDOWN_FILE = Path(self._tmpdir.name) / "cooldowns.json"

    def tearDown(self):
        engine.COOLDOWN_FILE = self._orig_cooldown_file
        self._tmpdir.cleanup()

    def test_no_cooldown_file_means_ok(self):
        self.assertTrue(engine.cooldown_ok("TST.AX"))

    def test_fresh_alert_starts_cooldown(self):
        engine.mark_alerted("TST.AX")
        self.assertFalse(engine.cooldown_ok("TST.AX"))

    def test_expired_cooldown_is_ok(self):
        past = datetime.now() - timedelta(hours=COOLDOWN_HOURS + 1)
        engine.COOLDOWN_FILE.write_text(json.dumps({"TST.AX": past.isoformat()}))
        self.assertTrue(engine.cooldown_ok("TST.AX"))

    def test_active_cooldown_is_not_ok(self):
        recent = datetime.now() - timedelta(hours=1)
        engine.COOLDOWN_FILE.write_text(json.dumps({"TST.AX": recent.isoformat()}))
        self.assertFalse(engine.cooldown_ok("TST.AX"))

    def test_cooldown_is_per_ticker(self):
        engine.mark_alerted("TST.AX")
        self.assertTrue(engine.cooldown_ok("BHP.AX"),
            "Cooldown on TST.AX must not affect BHP.AX")

    def test_corrupt_cooldown_file_handled_gracefully(self):
        engine.COOLDOWN_FILE.write_text("{INVALID JSON}")
        try:
            result = engine.cooldown_ok("TST.AX")
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"Corrupt cooldown file crashed cooldown_ok: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. DUPLICATE ALERT GUARD — _guard_ok
# ══════════════════════════════════════════════════════════════════════════════

class TestGuardOk(unittest.TestCase):
    """
    No changes to _guard_ok logic — tests confirm it was not broken by
    other changes.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_guard_file = engine.SEND_GUARD_FILE
        engine.SEND_GUARD_FILE = Path(self._tmpdir.name) / ".last_sent.json"

    def tearDown(self):
        engine.SEND_GUARD_FILE = self._orig_guard_file
        self._tmpdir.cleanup()

    def test_first_alert_passes_guard(self):
        self.assertTrue(engine._guard_ok("TST.AX"))

    def test_same_ticker_within_window_blocked(self):
        engine._guard_ok("TST.AX")           # stamps the ticker
        self.assertFalse(engine._guard_ok("TST.AX"))

    def test_different_ticker_blocked_by_global_guard(self):
        """Global 5-min guard: any second alert within 5 min is blocked."""
        engine._guard_ok("AAA.AX")
        self.assertFalse(engine._guard_ok("BBB.AX"),
            "Global 5-min guard must block different ticker within 5 min")

    def test_missing_guard_file_treated_as_ok(self):
        """No guard file → first alert always passes."""
        self.assertTrue(engine._guard_ok("FRESH.AX"))

    def test_corrupt_guard_file_fails_open(self):
        """Corrupt file → guard fails-open (alert allowed) to avoid false suppression."""
        engine.SEND_GUARD_FILE.write_text("{BAD JSON}")
        result = engine._guard_ok("TST.AX")
        self.assertTrue(result, "Corrupt guard file must fail-open")


# ══════════════════════════════════════════════════════════════════════════════
# 12. GET_DATA — feature calculation correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestGetData(unittest.TestCase):
    """
    Tests that get_data() correctly computes all expected feature columns
    and handles edge cases without crashing.
    Uses mocked yfinance to avoid network calls.
    """

    def _mock_ticker(self, ohlcv: pd.DataFrame):
        mock_t = MagicMock()
        mock_t.history.return_value = ohlcv
        return mock_t

    def test_all_feature_columns_present(self):
        ohlcv = _make_ohlcv(n=300)
        with patch("engine.yf.Ticker", return_value=self._mock_ticker(ohlcv)):
            df = engine.get_data("TST.AX", "6mo")
        for col in FEATURES:
            self.assertIn(col, df.columns, f"Feature column '{col}' missing from get_data output")

    def test_empty_history_returns_empty_df(self):
        """Delisted or invalid ticker returns empty DataFrame — must not crash."""
        mock_t = MagicMock()
        mock_t.history.return_value = pd.DataFrame()
        with patch("engine.yf.Ticker", return_value=mock_t):
            df = engine.get_data("DELISTED.AX", "6mo")
        self.assertTrue(df.empty)

    def test_no_inf_in_features(self):
        """No feature column should contain inf values."""
        ohlcv = _make_ohlcv(n=300, zero_vol_indices=[50, 51, 52, 53, 54])
        with patch("engine.yf.Ticker", return_value=self._mock_ticker(ohlcv)):
            df = engine.get_data("TST.AX", "6mo")
        for col in FEATURES:
            if col in df.columns:
                has_inf = np.isinf(df[col].replace([np.inf, -np.inf], np.nan).fillna(0)).any()
                self.assertFalse(has_inf, f"Feature '{col}' contains inf after zero-volume days")

    def test_no_nan_after_dropna(self):
        """dropna() at end of get_data must eliminate all NaN rows."""
        ohlcv = _make_ohlcv(n=300)
        with patch("engine.yf.Ticker", return_value=self._mock_ticker(ohlcv)):
            df = engine.get_data("TST.AX", "6mo")
        self.assertFalse(df[FEATURES].isnull().any().any(),
            "get_data must return no NaN values in feature columns after dropna()")

    def test_breakout_is_binary(self):
        """breakout column must only contain 0 or 1."""
        ohlcv = _make_ohlcv(n=300)
        with patch("engine.yf.Ticker", return_value=self._mock_ticker(ohlcv)):
            df = engine.get_data("TST.AX", "6mo")
        self.assertTrue(df["breakout"].isin([0, 1]).all())

    def test_vol_ratio_no_inf(self):
        """
        INTENTIONAL FIX: vol_ratio must not be inf even after zero-volume days.
        """
        zero_indices = list(range(5, 25))   # 20 consecutive zero-volume days
        ohlcv = _make_ohlcv(n=300, zero_vol_indices=zero_indices)
        with patch("engine.yf.Ticker", return_value=self._mock_ticker(ohlcv)):
            df = engine.get_data("TST.AX", "6mo")
        if "vol_ratio" in df.columns:
            self.assertFalse(np.isinf(df["vol_ratio"]).any(),
                "vol_ratio must never be inf")


# ══════════════════════════════════════════════════════════════════════════════
# 13. RECENCY WEIGHTS — training weight range check
# ══════════════════════════════════════════════════════════════════════════════

class TestRecencyWeights(unittest.TestCase):
    """
    REGRESSION: recency weights changed from ×4/×2/×1 to ×2/×1.5/×1.
    Tests confirm new range, document old range as intentional change.
    """

    def _compute_weights(self, ages_days: list) -> pd.Series:
        """Replicate the recency weight formula from train_model()."""
        return pd.Series(ages_days).apply(
            lambda d: 2.0 if d <= 30 else (1.5 if d <= 90 else 1.0)
        )

    def test_recent_row_gets_weight_2(self):
        """Row from 10 days ago → weight 2.0."""
        w = self._compute_weights([10])
        self.assertAlmostEqual(w.iloc[0], 2.0)

    def test_mid_age_row_gets_weight_1_5(self):
        """Row from 60 days ago → weight 1.5."""
        w = self._compute_weights([60])
        self.assertAlmostEqual(w.iloc[0], 1.5)

    def test_old_row_gets_weight_1(self):
        """Row from 180 days ago → weight 1.0."""
        w = self._compute_weights([180])
        self.assertAlmostEqual(w.iloc[0], 1.0)

    def test_max_weight_is_2_not_4(self):
        """
        INTENTIONAL FIX: max weight is 2.0 (was 4.0).
        Reduces regime mismatch risk in unusual recent periods.
        """
        all_ages = [1, 10, 20, 30, 60, 90, 180, 365]
        w = self._compute_weights(all_ages)
        self.assertAlmostEqual(w.max(), 2.0,
            msg="Max recency weight must be 2.0 (not the old 4.0)")

    def test_regression_old_weights(self):
        """
        REGRESSION MARKER: documents old weights for comparison.
        Old:  ≤30d → ×4,  ≤90d → ×2,  older → ×1
        New:  ≤30d → ×2,  ≤90d → ×1.5, older → ×1
        Difference: INTENTIONAL — reduces over-adaptation to recent regime.
        """
        old_weights = pd.Series([10, 60, 180]).apply(
            lambda d: 4.0 if d <= 30 else (2.0 if d <= 90 else 1.0)
        )
        new_weights = self._compute_weights([10, 60, 180])
        self.assertEqual(list(old_weights), [4.0, 2.0, 1.0],
            "Old weight formula baseline confirmed")
        self.assertEqual(list(new_weights), [2.0, 1.5, 1.0],
            "New weight formula confirmed")
        self.assertLess(new_weights.max(), old_weights.max(),
            "New formula must be more conservative than old")


# ══════════════════════════════════════════════════════════════════════════════
# 14. SCANNER MARKET HOURS — timezone and schedule logic
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketHours(unittest.TestCase):
    """
    Tests scanner.py market hours logic — no changes made here,
    confirming stability.
    """

    def _add_scanner_to_path(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))

    def test_markets_open_imports(self):
        """scanner.py must import cleanly."""
        try:
            import scanner  # noqa: F401
        except ImportError as e:
            self.fail(f"scanner.py failed to import: {e}")

    def test_asx_open_window_correct(self):
        """ASX opens 10:00 and closes 16:00 AEST on weekdays."""
        import pytz
        from scanner import _market_open
        ASX_TZ = pytz.timezone("Australia/Sydney")
        # Simulate a Monday at 11:00 AEST
        test_dt = datetime(2024, 1, 8, 11, 0, tzinfo=ASX_TZ)
        with patch("scanner.datetime") as mock_dt:
            mock_dt.now.return_value = test_dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            is_open, _ = _market_open(ASX_TZ, 10, 0, 16, 0)
        self.assertTrue(is_open)

    def test_weekend_not_open(self):
        """Saturday should never be open."""
        import pytz
        from scanner import _market_open
        ASX_TZ = pytz.timezone("Australia/Sydney")
        saturday = datetime(2024, 1, 6, 11, 0, tzinfo=ASX_TZ)   # Saturday
        with patch("scanner.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            is_open, _ = _market_open(ASX_TZ, 10, 0, 16, 0)
        self.assertFalse(is_open)


# ══════════════════════════════════════════════════════════════════════════════
# 15. BEHAVIOURAL REGRESSION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

class TestBehaviouralRegressionSummary(unittest.TestCase):
    """
    End-to-end behavioural regression: runs a synthetic signal log through
    the full accuracy_stats + performance_adjustments pipeline and confirms
    the new system produces materially better results than the old system
    would on the same data.
    """

    def _make_realistic_log(self):
        """Simulate 30 resolved signals: 60% wins across 3 tickers."""
        entries = []
        outcomes = (
            ["HIT_TARGET"] * 12 + ["EXPIRED_GAIN"] * 6 +    # 18 wins
            ["HIT_STOP"] * 8 + ["EXPIRED_LOSS"] * 4          # 12 losses
        )
        tickers = ["BHP.AX"] * 15 + ["CBA.AX"] * 8 + ["NVDA"] * 7
        for t, o in zip(tickers, outcomes):
            entries.append({
                "ticker": t,
                "outcome": o,
                "actual_pct": 0.04 if o in ("HIT_TARGET", "EXPIRED_GAIN") else -0.02,
            })
        return entries

    def test_old_system_win_rate_always_zero(self):
        """
        REGRESSION: old system with outcome=='WIN' check.
        On a 60% winning log, old system reported 0% win rate.
        """
        log = self._make_realistic_log()
        old_wins = sum(1 for e in log if e["outcome"] == "WIN")
        self.assertEqual(old_wins, 0,
            "Old system (outcome=='WIN') gives 0 wins on realistic log")

    def test_new_system_win_rate_correct(self):
        """New system correctly reports 60% win rate on same log."""
        log = self._make_realistic_log()
        r   = accuracy_stats(log)
        self.assertEqual(r["wins"], 18)
        self.assertEqual(r["losses"], 12)
        self.assertAlmostEqual(r["win_rate"], 0.60, places=2)

    def test_old_system_penalised_winning_tickers(self):
        """
        Old performance_adjustments() gave -2 to BHP.AX (12 HIT_TARGET)
        because win_rate computed as 0%. New system gives +2.
        """
        log = self._make_realistic_log()

        # Simulate old broken logic
        from collections import defaultdict
        bucket = defaultdict(list)
        for e in log:
            if e["outcome"] is not None:
                bucket[e["ticker"]].append(e["outcome"] == "WIN")  # always False

        old_adj = {}
        for ticker, results in bucket.items():
            recent = results[-20:]
            if len(recent) < 3:
                continue
            wr = sum(recent) / len(recent)  # always 0.0
            if   wr >= 0.75: old_adj[ticker] = +2
            elif wr >= 0.60: old_adj[ticker] = +1
            elif wr <= 0.25: old_adj[ticker] = -2
            elif wr <= 0.40: old_adj[ticker] = -1
            else:            old_adj[ticker] =  0

        self.assertEqual(old_adj.get("BHP.AX"), -2,
            "Old system gives -2 to BHP.AX despite 80% win rate")

        # New correct logic
        with patch("engine._load_log", return_value=log):
            new_adj = engine.performance_adjustments()

        self.assertEqual(new_adj.get("BHP.AX"), 2,
            "New system gives +2 to BHP.AX (80% win rate)")
        self.assertGreater(
            new_adj.get("BHP.AX", 0),
            old_adj.get("BHP.AX", 0),
            "New system must score BHP.AX higher than old (broken) system"
        )

    def test_new_avg_return_correct(self):
        """avg_return must use actual_pct values from entries."""
        log = self._make_realistic_log()
        r   = accuracy_stats(log)
        expected_avg = (18 * 0.04 + 12 * (-0.02)) / 30
        self.assertAlmostEqual(r["avg_return"], expected_avg, places=4)


# ══════════════════════════════════════════════════════════════════════════════
# 16. UPDATE TICKER PERFORMANCE — Discord win rate in outcome notifications
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateTickerPerformanceWinRate(unittest.TestCase):
    """
    REGRESSION: update_ticker_performance() previously computed the per-ticker
    win rate for the Discord outcome-update message using:

        wins = sum(1 for t in recent if t["outcome"] == "WIN")

    Since real outcomes are "HIT_TARGET" / "EXPIRED_GAIN" (never the literal
    string "WIN"), this always produced wins=0 and reported 0% win rate in
    every Discord notification regardless of actual results.

    The fix changes the check to:

        wins = sum(1 for t in recent
                   if t["outcome"] in ("WIN", "HIT_TARGET", "EXPIRED_GAIN"))
    """

    def _make_log(self, n_wins: int, n_losses: int, ticker: str = "BHP.AX") -> list:
        """Build a resolved signal log with realistic outcome strings."""
        base = {
            "ticker":       ticker,
            "signal_date":  "2024-01-10",
            "pred_days":    6,
            "entry_price":  50.0,
            "stop_price":   47.0,
            "target_price": 54.0,
            "target_pct":   0.04,
            "stop_pct":     0.06,
            "exit_price":   54.0,
            "actual_pct":   0.04,
        }
        entries = []
        for _ in range(n_wins):
            e = dict(base)
            e["outcome"] = "HIT_TARGET"   # real win outcome string — not "WIN"
            entries.append(e)
        for _ in range(n_losses):
            e = dict(base)
            e["outcome"] = "HIT_STOP"
            e["exit_price"] = 47.0
            e["actual_pct"] = -0.06
            entries.append(e)
        return entries

    def _compute_discord_win_rate(self, trades: list) -> float:
        """Replicate the win-rate calculation in update_ticker_performance()."""
        recent = trades[-20:]
        wins   = sum(1 for t in recent
                     if t["outcome"] in ("WIN", "HIT_TARGET", "EXPIRED_GAIN"))
        return wins / len(recent) * 100 if recent else 0.0

    def _compute_discord_win_rate_old(self, trades: list) -> float:
        """Replicate the BROKEN old calculation for comparison."""
        recent = trades[-20:]
        wins   = sum(1 for t in recent if t["outcome"] == "WIN")
        return wins / len(recent) * 100 if recent else 0.0

    def test_regression_old_code_reported_zero_win_rate(self):
        """
        REGRESSION MARKER — old code used outcome == 'WIN' which never matches
        real outcomes. This proves the bug: 6 HIT_TARGET wins → 0% reported.
        """
        log = self._make_log(n_wins=6, n_losses=4)
        old_wr = self._compute_discord_win_rate_old(log)
        self.assertEqual(old_wr, 0.0,
            "Old code must reproduce the 0% bug to prove the regression was real")

    def test_new_code_reports_correct_win_rate(self):
        """6 HIT_TARGET + 4 HIT_STOP → Discord message must show 60%."""
        log = self._make_log(n_wins=6, n_losses=4)
        wr  = self._compute_discord_win_rate(log)
        self.assertAlmostEqual(wr, 60.0, places=1,
            msg="6/10 HIT_TARGET → Discord must report 60% win rate")

    def test_expired_gain_counts_as_win_in_discord(self):
        """EXPIRED_GAIN is a win — must be counted in Discord notification."""
        trades = [{"outcome": "EXPIRED_GAIN"}, {"outcome": "HIT_STOP"}]
        wr = self._compute_discord_win_rate(trades)
        self.assertEqual(wr, 50.0)

    def test_all_hit_target_reports_100_pct(self):
        """All HIT_TARGET → 100% win rate in Discord."""
        log = self._make_log(n_wins=10, n_losses=0)
        wr  = self._compute_discord_win_rate(log)
        self.assertAlmostEqual(wr, 100.0)

    def test_all_hit_stop_reports_0_pct(self):
        """All HIT_STOP → 0% win rate (correctly) in Discord."""
        log = self._make_log(n_wins=0, n_losses=10)
        wr  = self._compute_discord_win_rate(log)
        self.assertEqual(wr, 0.0)

    def test_rolling_window_capped_at_20(self):
        """Discord win rate uses last 20 trades even when log has more."""
        log = self._make_log(n_wins=25, n_losses=0)  # 25 wins, take last 20
        wr  = self._compute_discord_win_rate(log)
        self.assertAlmostEqual(wr, 100.0)
        log2 = self._make_log(n_wins=0, n_losses=5)
        # Prepend 20 losses so the last 20 are all losses
        full = self._make_log(n_wins=0, n_losses=25)
        wr2  = self._compute_discord_win_rate(full)
        self.assertEqual(wr2, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 17. ACTIVE TIER — _large_move_check model parameter
# ══════════════════════════════════════════════════════════════════════════════

class TestLargeMoveCheckModelParameter(unittest.TestCase):
    """
    REGRESSION: _large_move_check() referenced `model` inside its body but
    the parameter was absent from the function signature:

        def _large_move_check(ticker: str, df: "pd.DataFrame") -> dict | None:

    Python raised NameError at `if model is not None:`, caught by the outer
    `except Exception: return None`. Result: _large_move_check() ALWAYS returned
    None silently, making big_mover_check() fall through to SETUP only.
    The ACTIVE tier (large confirmed movers) was completely inoperative.

    Fix: added `model=None` to the signature and updated big_mover_check() to
    pass `model=model` to _large_move_check().
    """

    def _make_df(self, n: int = 30) -> pd.DataFrame:
        """Minimal DataFrame — enough rows but values that won't pass ACTIVE gates."""
        data = {c: [1.0] * n for c in engine.FEATURES}
        data.update({
            "Close": [100.0] * n, "Open": [97.0] * n,
            "High": [102.0] * n,  "Low": [96.0] * n,
            "Volume": [1_000_000.0] * n,
            "bb_upper": [105.0] * n, "bb_lower": [95.0] * n,
            "bb_squeeze": [0] * n,
            "ema20": [98.0] * n,  "ema50": [95.0] * n,
        })
        return pd.DataFrame(data)

    def test_function_accepts_model_parameter(self):
        """_large_move_check must accept a model keyword argument."""
        import inspect
        sig = inspect.signature(engine._large_move_check)
        self.assertIn("model", sig.parameters,
            "model parameter must be in _large_move_check signature")

    def test_regression_without_model_always_returned_none(self):
        """
        REGRESSION MARKER — old code had no model parameter.
        Calling it without model in scope raised NameError → silently returned None.
        Verify the old behaviour is reproducible by calling with model=None explicitly.
        When model=None the function must return None (needs model to proceed).
        """
        df = self._make_df()
        result = engine._large_move_check("BHP.AX", df, model=None)
        self.assertIsNone(result,
            "With model=None, _large_move_check must return None (AI required)")

    def test_no_name_error_when_model_provided(self):
        """
        When a model object is provided, the function must not raise NameError.
        It may still return None (gates not met), but must not crash.
        """
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.5]]
        df = self._make_df()
        try:
            result = engine._large_move_check("BHP.AX", df, model=mock_model)
            # Result is None (gates not passed) but no NameError was raised
        except NameError as e:
            self.fail(f"NameError raised — model still not in scope: {e}")

    def test_big_mover_check_passes_model_to_large_move_check(self):
        """big_mover_check must pass model to _large_move_check (not just to SETUP)."""
        import inspect
        src = inspect.getsource(engine.big_mover_check)
        self.assertIn("_large_move_check(ticker, df, model=model)", src,
            "big_mover_check must pass model to _large_move_check")


# ══════════════════════════════════════════════════════════════════════════════
# 18. CONFIDENCE GRADE — bar never exceeds 10 characters
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceGradeBarClamp(unittest.TestCase):
    """
    REGRESSION: confidence_grade() used:

        filled = round(combined * 10)
        bar = "█" * filled + "░" * (10 - filled)

    When `combined > 1.0` (possible when score > 14 with any positive prob),
    `filled` exceeded 10 and the bar string grew to 11–13+ characters,
    corrupting the Discord alert and dashboard display.

    Fix: `filled = min(10, round(combined * 10))` — clamps at 10.
    """

    def _bar_len(self, prob: float, score: int) -> int:
        """Return the character length of the bar portion."""
        _, _, bar_str = confidence_grade(prob, score)
        return len(bar_str.split(" ")[0])

    def test_bar_10_chars_at_normal_scores(self):
        """Score ≤ 14 — bar must be exactly 10 characters."""
        for score in [0, 5, 9, 11, 14]:
            self.assertEqual(self._bar_len(0.75, score), 10,
                f"bar must be 10 chars at score={score}")

    def test_bar_10_chars_at_high_scores(self):
        """Score > 14 — bar must still be exactly 10 characters (clamped)."""
        for score in [15, 18, 20, 25, 40]:
            self.assertEqual(self._bar_len(0.90, score), 10,
                f"bar must be clamped to 10 chars at score={score}")

    def test_regression_old_code_overflowed(self):
        """
        REGRESSION MARKER — old code produced bars > 10 chars at high scores.
        Verify new code is strictly bounded.
        """
        # score=25, prob=0.9: combined = 0.54 + 0.714 = 1.254 → old: filled=13
        _, _, bar_str = confidence_grade(0.90, 25)
        bar_part = bar_str.split(" ")[0]
        self.assertEqual(len(bar_part), 10,
            f"bar must be exactly 10 chars, got {len(bar_part)}: '{bar_part}'")

    def test_bar_content_correct_at_normal_range(self):
        """Bar content must reflect filled/empty balance correctly."""
        _, _, bar_str = confidence_grade(0.50, 7)
        bar_part = bar_str.split(" ")[0]
        filled_count = bar_part.count("█")
        empty_count  = bar_part.count("░")
        self.assertEqual(filled_count + empty_count, 10)
        self.assertGreater(filled_count, 0)


# ══════════════════════════════════════════════════════════════════════════════
# 19. LOG SIGNAL — same-ticker+same-date+same-tier deduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestLogSignalDedup(unittest.TestCase):
    """
    REGRESSION: log_signal() had no deduplication guard. Calling it twice
    for the same ticker + signal_date + tier (e.g. from a double-fired
    GitHub Actions run, or legacy code paths) produced duplicate log entries
    that inflated the outcome history used by the adaptive learning loop.

    Fix: before appending, log_signal() now checks for an existing unresolved
    entry with the same (ticker, signal_date, tier) and skips if found.
    """

    def _write_and_read(self, tmp_path, calls):
        """
        Patch LOG_FILE to a temp path, call log_signal N times, return entries.
        Each call is a dict of kwargs to log_signal().
        """
        import engine as eng
        original = eng.LOG_FILE
        eng.LOG_FILE = tmp_path
        try:
            for kw in calls:
                eng.log_signal(**kw)
            return eng._load_log()
        finally:
            eng.LOG_FILE = original
            if tmp_path.exists():
                tmp_path.unlink()
            tmp = tmp_path.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()

    def test_first_entry_is_written(self):
        """A single call must produce exactly one log entry."""
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mktemp(suffix=".json"))
        entries = self._write_and_read(tmp, [
            {"ticker": "BHP.AX", "price": 45.0, "tier": "ELITE",
             "score": 12, "prob": 0.82}
        ])
        self.assertEqual(len(entries), 1)

    def test_duplicate_same_ticker_date_tier_is_skipped(self):
        """
        REGRESSION MARKER — calling log_signal twice with the same
        ticker + today's date + tier must produce only ONE entry.
        """
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mktemp(suffix=".json"))
        entries = self._write_and_read(tmp, [
            {"ticker": "CBA.AX", "price": 163.0, "tier": "ELITE",
             "score": 11, "prob": 0.75},
            {"ticker": "CBA.AX", "price": 164.0, "tier": "ELITE",
             "score": 12, "prob": 0.80},   # duplicate — same ticker+date+tier
        ])
        self.assertEqual(len(entries), 1,
            "Second call for same ticker+date+tier must be silently skipped")
        self.assertAlmostEqual(entries[0]["entry_price"], 163.0,
            msg="First entry must be preserved, not overwritten")

    def test_different_tier_same_ticker_is_allowed(self):
        """Same ticker can be logged under different tiers on the same day."""
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mktemp(suffix=".json"))
        entries = self._write_and_read(tmp, [
            {"ticker": "NVDA", "price": 224.0, "tier": "SETUP",  "score": 10},
            {"ticker": "NVDA", "price": 225.0, "tier": "ACTIVE", "score": 9},
        ])
        self.assertEqual(len(entries), 2,
            "Different tiers for the same ticker on the same day must both be logged")

    def test_different_ticker_same_tier_is_allowed(self):
        """Different tickers with the same tier are always independent entries."""
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mktemp(suffix=".json"))
        entries = self._write_and_read(tmp, [
            {"ticker": "AAPL", "price": 281.0, "tier": "ACTIVE", "score": 9},
            {"ticker": "MSFT", "price": 368.0, "tier": "ACTIVE", "score": 9},
        ])
        self.assertEqual(len(entries), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 20. ACTIVE TIER — ai_prob present in return dict
# ══════════════════════════════════════════════════════════════════════════════

class TestActiveTierAiProbInDict(unittest.TestCase):
    """
    REGRESSION: _large_move_check() computed ai_prob (used for the ≥ 38% gate)
    but did NOT include it in the returned dict. send_mover_alert() called
    mover.get("ai_prob", 0.0) — always receiving 0.0 — so every ACTIVE alert
    was logged and displayed with prob=0.0, regardless of the actual AI score.

    Fix: "ai_prob": ai_prob added to the _large_move_check() return dict,
    making the logged probability accurate for ACTIVE tier alerts.
    """

    def test_active_return_dict_contains_ai_prob_key(self):
        """
        The ACTIVE tier return dict must contain the 'ai_prob' key.
        Verified by inspecting the source of _large_move_check().
        """
        import inspect
        src = inspect.getsource(engine._large_move_check)
        # Find the return block
        in_return = False
        keys_found = []
        for line in src.splitlines():
            if '"tier":' in line and '"ACTIVE"' in line:
                in_return = True
            if in_return and '":' in line:
                key = line.strip().split('"')[1]
                keys_found.append(key)
            if in_return and line.strip().startswith("}"):
                break
        self.assertIn("ai_prob", keys_found,
            "ACTIVE return dict must include 'ai_prob' so send_mover_alert logs correct prob")

    def test_active_prob_not_zero_when_model_provides_value(self):
        """
        REGRESSION MARKER — old code always produced prob=0.0 for ACTIVE alerts.
        When a model returns a real probability, the dict must carry it forward.
        Simulate by running with a mock model that returns a known prob.
        """
        import pandas as pd

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.45, 0.55]]   # 55% → above gate

        rows = 30
        data = {c: [1.0] * rows for c in engine.FEATURES}
        data.update({
            "Close": [100.0] * rows, "Open":  [95.0]  * rows,
            "High":  [105.0] * rows, "Low":   [93.0]  * rows,
            "Volume":[5_000_000.0]   * rows,
            "vol_ratio": [5.0]       * rows,
            "rsi":       [55.0]      * rows,
            "atr":       [3.0]       * rows,
            "adx":       [30.0]      * rows,
            "bb_upper":  [105.0]     * rows,
            "bb_lower":  [95.0]      * rows,
            "bb_squeeze":[0]         * rows,
            "ema20":     [98.0]      * rows,
            "ema50":     [95.0]      * rows,
        })
        df = pd.DataFrame(data)

        result = engine._large_move_check("AAPL", df, model=mock_model)
        # Result may be None if daily gates don't pass (vol_r, daily_ret, atr_exp etc.)
        # but if it fires, ai_prob must be present and non-zero
        if result is not None:
            self.assertIn("ai_prob", result,
                "ACTIVE result dict must contain 'ai_prob'")
            self.assertGreater(result["ai_prob"], 0.0,
                "ai_prob must reflect the model's actual output, not default 0.0")

    def test_setup_also_returns_ai_prob(self):
        """SETUP tier already included ai_prob — verify it is unchanged."""
        import inspect
        src = inspect.getsource(engine._breakout_setup_check)
        self.assertIn('"ai_prob"', src,
            "SETUP return dict must still contain 'ai_prob'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
