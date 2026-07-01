"""
Tests for opportunity.performance — Phase 5 Performance Learning & Calibration
All file I/O and Discord calls are mocked.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.performance as perf_mod
from opportunity.performance import (
    calibration_buckets,
    sector_performance,
    performance_summary,
    run_performance_analytics,
    send_weekly_performance_report,
    WIN_OUTCOMES,
    CALIB_BUCKETS,
)


# ─── Synthetic data helpers ───────────────────────────────────────────────────

def _entry(
    outcome:    str   = "WIN",
    actual_pct: float = 0.12,
    prob:       float = 0.70,
    days_ago:   int   = 3,
    ticker:     str   = "BHP.AX",
) -> dict:
    date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "ticker":       ticker,
        "tier":         "B",
        "score":        70,
        "prob":         prob,
        "entry_price":  1.00,
        "stop_price":   0.90,
        "target_price": 1.15,
        "signal_date":  date,
        "target_pct":   0.15,
        "pred_days":    14,
        "outcome":      outcome,
        "exit_price":   round(1.00 * (1 + actual_pct), 4),
        "actual_pct":   actual_pct,
    }


def _mixed_entries(n_wins: int = 6, n_losses: int = 4, prob: float = 0.70) -> list[dict]:
    entries = []
    for i in range(n_wins):
        entries.append(_entry("WIN", 0.12, prob, days_ago=i + 1))
    for i in range(n_losses):
        entries.append(_entry("HIT_STOP", -0.06, prob, days_ago=i + 1))
    return entries


# ─── calibration_buckets ─────────────────────────────────────────────────────

class TestCalibrationBuckets(unittest.TestCase):

    def test_empty_returns_all_no_data(self):
        result = calibration_buckets([])
        self.assertTrue(all(b["calibration_status"] == "NO_DATA" for b in result))

    def test_returns_correct_number_of_buckets(self):
        self.assertEqual(len(calibration_buckets([])), len(CALIB_BUCKETS))

    def test_all_wins_in_bucket_well_calibrated_or_underconfident(self):
        entries = [_entry("WIN", prob=0.65) for _ in range(20)]
        result = calibration_buckets(entries)
        mid_bucket = next(b for b in result if b["label"] == "60–70%")
        self.assertIn(
            mid_bucket["calibration_status"],
            ("WELL_CALIBRATED", "UNDERCONFIDENT"),
        )

    def test_all_losses_in_high_bucket_overconfident(self):
        entries = [_entry("HIT_STOP", prob=0.85) for _ in range(20)]
        result = calibration_buckets(entries)
        high_bucket = next(b for b in result if b["label"] == "80%+")
        self.assertEqual(high_bucket["calibration_status"], "OVERCONFIDENT")

    def test_actual_win_rate_correct(self):
        # 4 wins, 1 loss in 60-70% bucket → actual = 0.80
        entries = [_entry("WIN",     prob=0.65) for _ in range(8)] + \
                  [_entry("HIT_STOP", prob=0.65) for _ in range(2)]
        result = calibration_buckets(entries)
        bucket = next(b for b in result if b["label"] == "60–70%")
        self.assertAlmostEqual(bucket["actual_win_rate"], 0.80, places=2)

    def test_count_correct(self):
        entries = [_entry("WIN", prob=0.72) for _ in range(5)]
        result = calibration_buckets(entries)
        bucket = next(b for b in result if b["label"] == "70–80%")
        self.assertEqual(bucket["count"], 5)

    def test_no_data_bucket_has_none_win_rate(self):
        entries = [_entry("WIN", prob=0.55)]  # only 50-60% bucket filled
        result = calibration_buckets(entries)
        high_bucket = next(b for b in result if b["label"] == "80%+")
        self.assertIsNone(high_bucket["actual_win_rate"])


# ─── sector_performance ───────────────────────────────────────────────────────

class TestSectorPerformance(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(sector_performance([]), [])

    def test_known_ticker_classified(self):
        entries = [_entry("WIN", ticker="BHP.AX")]
        result = sector_performance(entries)
        sectors = [r["sector"] for r in result]
        self.assertIn("Resources", sectors)

    def test_unknown_ticker_goes_to_other(self):
        entries = [_entry("WIN", ticker="XYZ.AX")]
        result = sector_performance(entries)
        sectors = [r["sector"] for r in result]
        self.assertIn("Other", sectors)

    def test_win_rate_correct_per_sector(self):
        entries = (
            [_entry("WIN",     ticker="BHP.AX")] * 3 +
            [_entry("HIT_STOP", ticker="BHP.AX")] * 1
        )
        result = sector_performance(entries)
        r_sector = next(r for r in result if r["sector"] == "Resources")
        self.assertAlmostEqual(r_sector["win_rate"], 0.75, places=2)

    def test_sorted_by_avg_pct_descending(self):
        entries = (
            [_entry("WIN",     0.20, ticker="BHP.AX")] * 5 +   # Resources high
            [_entry("HIT_STOP", -0.10, ticker="CBA.AX")] * 5    # Banks low
        )
        result = sector_performance(entries)
        avg_pcts = [r["avg_pct"] for r in result]
        self.assertEqual(avg_pcts, sorted(avg_pcts, reverse=True))


# ─── performance_summary ──────────────────────────────────────────────────────

class TestPerformanceSummary(unittest.TestCase):

    def test_empty_entries_returns_zero_counts(self):
        s = performance_summary([])
        self.assertEqual(s["resolved_count"], 0)

    def test_win_rate_correct(self):
        entries = _mixed_entries(6, 4)
        s = performance_summary(entries, lookback_days=30)
        self.assertAlmostEqual(s["win_rate"], 0.60, places=2)

    def test_expectancy_positive_for_good_trades(self):
        entries = [_entry("WIN", 0.20)] * 8 + [_entry("HIT_STOP", -0.05)] * 2
        s = performance_summary(entries, lookback_days=30)
        self.assertGreater(s["expectancy_r"], 0)

    def test_calibration_in_summary(self):
        s = performance_summary(_mixed_entries(), lookback_days=30)
        self.assertIn("calibration", s)
        self.assertIsInstance(s["calibration"], list)

    def test_sector_breakdown_in_summary(self):
        s = performance_summary(_mixed_entries(), lookback_days=30)
        self.assertIn("sector_breakdown", s)

    def test_old_entries_excluded_by_lookback(self):
        old_entries = [_entry("WIN", days_ago=60)]
        s = performance_summary(old_entries, lookback_days=7)
        self.assertEqual(s["resolved_count"], 0)

    def test_all_required_keys_present(self):
        s = performance_summary(_mixed_entries(), lookback_days=30)
        for key in ("period_days", "cutoff_date", "resolved_count", "win_count",
                    "loss_count", "win_rate", "expectancy_r", "avg_hold_days",
                    "calibration", "sector_breakdown"):
            self.assertIn(key, s)


# ─── run_performance_analytics ────────────────────────────────────────────────

class TestRunPerformanceAnalytics(unittest.TestCase):

    def test_returns_none_when_flag_off(self):
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", False):
            self.assertIsNone(run_performance_analytics())

    def test_returns_none_on_empty_log(self):
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", True), \
             patch.object(perf_mod, "_resolved_entries", return_value=[]):
            self.assertIsNone(run_performance_analytics())

    def test_returns_dict_with_entries(self):
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", True), \
             patch.object(perf_mod, "_resolved_entries", return_value=_mixed_entries()):
            result = run_performance_analytics(lookback_days=30)
        self.assertIsNotNone(result)
        self.assertIn("win_rate", result)


# ─── send_weekly_performance_report ──────────────────────────────────────────

class TestSendWeeklyPerformanceReport(unittest.TestCase):

    def test_returns_false_when_flag_off(self):
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", False):
            self.assertFalse(send_weekly_performance_report())

    def test_returns_false_with_no_webhook(self):
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", True), \
             patch.dict("os.environ", {}, clear=True), \
             patch.object(perf_mod, "_resolved_entries", return_value=[]):
            self.assertFalse(send_weekly_performance_report())

    def test_sends_ok_with_webhook_and_data(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__  = MagicMock(return_value=False)
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/hook"}), \
             patch.object(perf_mod, "_resolved_entries",
                          return_value=_mixed_entries(6, 4)), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = send_weekly_performance_report()
        self.assertTrue(result)

    def test_handles_no_resolved_trades_gracefully(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__  = MagicMock(return_value=False)
        with patch.object(perf_mod, "ENABLE_PERFORMANCE_ANALYTICS", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/hook"}), \
             patch.object(perf_mod, "_resolved_entries", return_value=[]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = send_weekly_performance_report()
        self.assertTrue(result)   # still sends "no trades" message


if __name__ == "__main__":
    unittest.main(verbosity=2)
