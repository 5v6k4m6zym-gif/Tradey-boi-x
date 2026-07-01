"""
Tests for opportunity.alerts — Enhanced Discord Alert Formatter (Phase 3)
No actual Discord requests are made — all HTTP calls are mocked.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.alerts as alerts_mod
from opportunity.alerts import (
    format_opportunity_alert,
    format_outcome_alert,
    send_opportunity_alert,
)


def _make_opp(overrides: dict | None = None) -> dict:
    base = {
        "ticker":               "TST.AX",
        "opportunity_score":    74,
        "confidence":           0.71,
        "expected_upside_pct":  0.225,
        "expected_downside_pct": 0.06,
        "est_holding_days":     18,
        "prob_target_hit":      0.64,
        "prob_stop_hit":        0.23,
        "risk_level":           "MEDIUM",
        "rr_ratio":             3.75,
        "entry_zone":           [1.42, 1.48],
        "stop_loss":            1.31,
        "take_profit":          [1.65, 1.85, 2.10],
        "trailing_stop_pct":    0.06,
        "regime":               "BULLISH",
        "reasons_for":          ["Strong volume surge", "EMA uptrend"],
        "reasons_against":      ["RSI approaching overbought"],
        "technical_summary":    "ADX 31, EMA aligned",
        "momentum_summary":     "+4% this week",
        "component_scores":     {
            "expected_return":    80,
            "technical_strength": 75,
            "volume_expansion":   60,
            "momentum":           70,
            "news_catalyst":      50,
            "institutional":      55,
            "risk_reward":        90,
        },
    }
    if overrides:
        base.update(overrides)
    return base


def _make_outcome(overrides: dict | None = None) -> dict:
    base = {
        "ticker":           "TST.AX",
        "outcome":          "WIN",
        "entry_price":      1.45,
        "exit_price":       1.78,
        "actual_pct":       0.228,
        "signal_date":      "2026-06-01",
        "pred_days":        18,
        "opportunity_score": 74,
        "confidence":       0.71,
        "stop_price":       1.31,
        "target_price":     1.85,
    }
    if overrides:
        base.update(overrides)
    return base


class TestFormatOpportunityAlert(unittest.TestCase):

    def test_returns_string(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIsInstance(msg, str)

    def test_contains_ticker(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("TST.AX", msg)

    def test_contains_score(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("74", msg)

    def test_contains_confidence_pct(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("71%", msg)

    def test_contains_expected_return(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("22.5", msg)

    def test_contains_entry_zone(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("1.42", msg)
        self.assertIn("1.48", msg)

    def test_contains_stop_loss(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("1.31", msg)

    def test_contains_all_take_profit_levels(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("1.65", msg)
        self.assertIn("1.85", msg)
        self.assertIn("2.10", msg)

    def test_contains_holding_days(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("18", msg)

    def test_contains_regime(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("BULLISH", msg)

    def test_contains_reasons_for(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("Strong volume surge", msg)

    def test_contains_reasons_against(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("RSI approaching overbought", msg)

    def test_under_2000_chars(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertLessEqual(len(msg), 2000)

    def test_empty_reasons_does_not_crash(self):
        opp = _make_opp({"reasons_for": [], "reasons_against": []})
        msg = format_opportunity_alert(opp)
        self.assertIsInstance(msg, str)

    def test_rr_ratio_in_message(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("3.75", msg)

    def test_risk_level_in_message(self):
        msg = format_opportunity_alert(_make_opp())
        self.assertIn("MEDIUM", msg)

    def test_bearish_regime_reflected(self):
        opp = _make_opp({"regime": "BEARISH"})
        msg = format_opportunity_alert(opp)
        self.assertIn("BEARISH", msg)


class TestFormatOutcomeAlert(unittest.TestCase):

    def test_returns_string(self):
        msg = format_outcome_alert(_make_outcome())
        self.assertIsInstance(msg, str)

    def test_win_outcome_in_message(self):
        msg = format_outcome_alert(_make_outcome({"outcome": "WIN"}))
        self.assertIn("WIN", msg) or self.assertIn("✅", msg)

    def test_loss_outcome_in_message(self):
        msg = format_outcome_alert(_make_outcome({"outcome": "HIT_STOP"}))
        self.assertIn("STOP", msg) or self.assertIn("❌", msg)

    def test_contains_ticker(self):
        msg = format_outcome_alert(_make_outcome())
        self.assertIn("TST.AX", msg)

    def test_contains_actual_pct(self):
        msg = format_outcome_alert(_make_outcome())
        self.assertIn("22.8", msg)

    def test_under_2000_chars(self):
        msg = format_outcome_alert(_make_outcome())
        self.assertLessEqual(len(msg), 2000)

    def test_missing_optional_fields_degrade_gracefully(self):
        sparse = {"ticker": "TST.AX", "outcome": "WIN",
                  "actual_pct": 0.15, "entry_price": 1.00, "exit_price": 1.15}
        msg = format_outcome_alert(sparse)
        self.assertIsInstance(msg, str)
        self.assertIn("TST.AX", msg)


class TestSendOpportunityAlert(unittest.TestCase):

    def test_returns_false_when_flag_off(self):
        with patch.object(alerts_mod, "ENABLE_ENHANCED_ALERTS", False):
            self.assertFalse(send_opportunity_alert(_make_opp()))

    def test_returns_false_when_no_webhook(self):
        with patch.object(alerts_mod, "ENABLE_ENHANCED_ALERTS", True), \
             patch.dict("os.environ", {}, clear=True):
            self.assertFalse(send_opportunity_alert(_make_opp()))

    def test_returns_true_on_successful_post(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__  = MagicMock(return_value=False)
        with patch.object(alerts_mod, "ENABLE_ENHANCED_ALERTS", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/webhook"}), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = send_opportunity_alert(_make_opp())
        self.assertTrue(result)

    def test_returns_false_on_network_error(self):
        with patch.object(alerts_mod, "ENABLE_ENHANCED_ALERTS", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/webhook"}), \
             patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = send_opportunity_alert(_make_opp())
        self.assertFalse(result)

    def test_does_not_raise_on_malformed_opp(self):
        with patch.object(alerts_mod, "ENABLE_ENHANCED_ALERTS", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/webhook"}), \
             patch("urllib.request.urlopen", side_effect=Exception("err")):
            result = send_opportunity_alert({"ticker": "X"})
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
