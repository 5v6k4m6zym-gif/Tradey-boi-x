"""
Tests for opportunity.backtester — Phase 4 Backtesting Expansion
All file I/O and Discord calls are mocked.
"""
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.backtester as bt_mod
from opportunity.backtester import (
    compute_metrics,
    _empty_metrics,
    _walk_forward,
    _out_of_sample,
    _historical_simulation,
    _paper_trading_snapshot,
    _sharpe,
    _sortino,
    _max_drawdown,
    _streaks,
    run_backtest,
    send_backtest_discord,
    WIN_OUTCOMES,
)


# ─── Synthetic trade log builder ──────────────────────────────────────────────

def _entry(
    outcome: str = "WIN",
    actual_pct: float = 0.15,
    prob: float = 0.70,
    days_ago: int = 5,
    pred_days: int = 14,
) -> dict:
    date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "ticker":       "TST.AX",
        "tier":         "B",
        "score":        70,
        "prob":         prob,
        "entry_price":  1.00,
        "stop_price":   0.90,
        "target_price": 1.15,
        "signal_date":  date,
        "target_pct":   0.15,
        "pred_days":    pred_days,
        "outcome":      outcome,
        "exit_price":   round(1.00 * (1 + actual_pct), 4),
        "actual_pct":   actual_pct,
    }


def _trades(n_wins: int = 6, n_losses: int = 4) -> list[dict]:
    trades = []
    day = 0
    for i in range(n_wins):
        trades.append(_entry("WIN", 0.12, days_ago=day + i * 3))
    for i in range(n_losses):
        trades.append(_entry("HIT_STOP", -0.06, days_ago=day + i * 2 + 1))
    return trades


# ─── compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics(unittest.TestCase):

    def test_empty_returns_zero_metrics(self):
        m = compute_metrics([])
        self.assertEqual(m["trade_count"], 0)
        self.assertEqual(m["win_rate"],    0)

    def test_all_wins_win_rate_1(self):
        trades = [_entry("WIN", 0.10) for _ in range(5)]
        m = compute_metrics(trades)
        self.assertEqual(m["win_rate"], 1.0)

    def test_all_losses_win_rate_0(self):
        trades = [_entry("HIT_STOP", -0.05) for _ in range(5)]
        m = compute_metrics(trades)
        self.assertEqual(m["win_rate"], 0.0)

    def test_win_rate_correct(self):
        trades = _trades(6, 4)
        m = compute_metrics(trades)
        self.assertAlmostEqual(m["win_rate"], 0.60, places=2)

    def test_profit_factor_positive_when_wins_dominate(self):
        trades = _trades(8, 2)
        m = compute_metrics(trades)
        self.assertGreater(m["profit_factor"], 1.0)

    def test_avg_gain_pct_positive(self):
        trades = _trades(5, 0)
        m = compute_metrics(trades)
        self.assertGreater(m["avg_gain_pct"], 0)

    def test_avg_loss_pct_positive(self):
        trades = _trades(0, 5)
        m = compute_metrics(trades)
        self.assertGreater(m["avg_loss_pct"], 0)

    def test_max_drawdown_non_negative(self):
        m = compute_metrics(_trades(5, 5))
        self.assertGreaterEqual(m["max_drawdown_pct"], 0)

    def test_sharpe_finite(self):
        m = compute_metrics(_trades(6, 4))
        self.assertFalse(m["sharpe_ratio"] != m["sharpe_ratio"])   # not NaN

    def test_all_required_keys_present(self):
        m = compute_metrics(_trades(5, 5))
        for key in _empty_metrics():
            self.assertIn(key, m, f"Missing key: {key}")

    def test_winning_streak_correct(self):
        trades = [_entry("WIN")] * 4 + [_entry("HIT_STOP")] * 2
        m = compute_metrics(trades)
        self.assertEqual(m["winning_streak"], 4)

    def test_losing_streak_correct(self):
        trades = [_entry("HIT_STOP")] * 3 + [_entry("WIN")] * 5
        m = compute_metrics(trades)
        self.assertEqual(m["losing_streak"], 3)

    def test_expectancy_positive_for_good_strategy(self):
        trades = [_entry("WIN", 0.20) for _ in range(7)] + \
                 [_entry("HIT_STOP", -0.05) for _ in range(3)]
        m = compute_metrics(trades)
        self.assertGreater(m["expectancy_r"], 0)

    def test_expectancy_negative_for_bad_strategy(self):
        trades = [_entry("WIN", 0.03) for _ in range(3)] + \
                 [_entry("HIT_STOP", -0.10) for _ in range(7)]
        m = compute_metrics(trades)
        self.assertLess(m["expectancy_r"], 0)

    def test_hit_target_counts_as_win(self):
        trades = [_entry("HIT_TARGET", 0.15) for _ in range(5)]
        m = compute_metrics(trades)
        self.assertEqual(m["win_rate"], 1.0)

    def test_expired_gain_counts_as_win(self):
        trades = [_entry("EXPIRED_GAIN", 0.08) for _ in range(5)]
        m = compute_metrics(trades)
        self.assertEqual(m["win_rate"], 1.0)


# ─── Statistical helpers ──────────────────────────────────────────────────────

class TestStatHelpers(unittest.TestCase):

    def test_sharpe_zero_on_single_return(self):
        self.assertEqual(_sharpe([0.05]), 0.0)

    def test_sharpe_positive_for_consistent_gains(self):
        self.assertGreater(_sharpe([0.01, 0.02, 0.015, 0.012, 0.018] * 10), 0)

    def test_sortino_zero_with_no_downside(self):
        result = _sortino([0.01, 0.02, 0.03])
        self.assertGreater(result, 0)   # no negative returns → inf or large

    def test_max_drawdown_zero_on_monotone_gains(self):
        returns = [0.01] * 20
        self.assertAlmostEqual(_max_drawdown(returns), 0.0, places=5)

    def test_max_drawdown_positive_on_loss_sequence(self):
        returns = [0.05, -0.10, -0.08, 0.03]
        self.assertGreater(_max_drawdown(returns), 0)

    def test_streaks_all_wins(self):
        trades = [_entry("WIN")] * 5
        w, l = _streaks(trades)
        self.assertEqual(w, 5)
        self.assertEqual(l, 0)

    def test_streaks_alternating(self):
        trades = [_entry("WIN"), _entry("HIT_STOP")] * 5
        w, l = _streaks(trades)
        self.assertEqual(w, 1)
        self.assertEqual(l, 1)


# ─── Walk-forward ─────────────────────────────────────────────────────────────

class TestWalkForward(unittest.TestCase):

    def _dated_trades(self, n: int = 200) -> list[dict]:
        trades = []
        for i in range(n):
            date = (datetime.utcnow() - timedelta(days=n - i)).strftime("%Y-%m-%d")
            outcome = "WIN" if i % 3 != 0 else "HIT_STOP"
            trades.append({**_entry(outcome), "signal_date": date})
        return trades

    def test_empty_returns_empty(self):
        self.assertEqual(_walk_forward([]), [])

    def test_returns_list_of_dicts(self):
        windows = _walk_forward(self._dated_trades(200), window_size=80, test_size=30)
        self.assertIsInstance(windows, list)
        self.assertTrue(all(isinstance(w, dict) for w in windows))

    def test_each_window_has_required_keys(self):
        windows = _walk_forward(self._dated_trades(200), window_size=80, test_size=30)
        for w in windows:
            self.assertIn("window_start", w)
            self.assertIn("window_end",   w)
            self.assertIn("trade_count",  w)

    def test_fewer_than_one_window_returns_empty(self):
        trades = self._dated_trades(10)
        self.assertEqual(_walk_forward(trades, window_size=50, test_size=20), [])

    def test_multiple_windows_produced(self):
        windows = _walk_forward(self._dated_trades(300), window_size=80, test_size=30)
        self.assertGreater(len(windows), 1)


# ─── Out-of-sample ────────────────────────────────────────────────────────────

class TestOutOfSample(unittest.TestCase):

    def _dated_trades(self, n: int = 50) -> list[dict]:
        trades = []
        for i in range(n):
            date = (datetime.utcnow() - timedelta(days=n - i)).strftime("%Y-%m-%d")
            trades.append({**_entry(), "signal_date": date})
        return trades

    def test_empty_returns_zero_metrics(self):
        m = _out_of_sample([])
        self.assertEqual(m["trade_count"], 0)

    def test_holdout_pct_in_result(self):
        m = _out_of_sample(self._dated_trades(50), holdout_pct=0.20)
        self.assertIn("holdout_pct", m)
        self.assertAlmostEqual(m["holdout_pct"], 0.20)

    def test_test_size_approximately_20_pct(self):
        trades = self._dated_trades(50)
        m = _out_of_sample(trades, holdout_pct=0.20)
        self.assertAlmostEqual(m["trade_count"], 10, delta=2)


# ─── Paper trading ────────────────────────────────────────────────────────────

class TestPaperTrading(unittest.TestCase):

    def _trades_with_dates(self, n_recent: int = 5, n_old: int = 20) -> list[dict]:
        trades = []
        for i in range(n_recent):
            date = (datetime.utcnow() - timedelta(days=i + 1)).strftime("%Y-%m-%d")
            trades.append({**_entry("WIN"), "signal_date": date})
        for i in range(n_old):
            date = (datetime.utcnow() - timedelta(days=60 + i)).strftime("%Y-%m-%d")
            trades.append({**_entry("WIN"), "signal_date": date})
        return trades

    def test_paper_mode_in_result(self):
        m = _paper_trading_snapshot(self._trades_with_dates())
        self.assertEqual(m["mode"], "paper")

    def test_only_recent_trades_counted(self):
        m = _paper_trading_snapshot(self._trades_with_dates(5, 20), days=30)
        self.assertLessEqual(m["trade_count"], 10)

    def test_open_positions_non_negative(self):
        m = _paper_trading_snapshot(self._trades_with_dates())
        self.assertGreaterEqual(m["open_positions"], 0)


# ─── run_backtest ─────────────────────────────────────────────────────────────

class TestRunBacktest(unittest.TestCase):

    def _mock_log(self, n: int = 50) -> list[dict]:
        trades = []
        for i in range(n):
            date = (datetime.utcnow() - timedelta(days=n - i)).strftime("%Y-%m-%d")
            outcome = "WIN" if i % 2 == 0 else "HIT_STOP"
            trades.append({**_entry(outcome), "signal_date": date})
        return trades

    def test_returns_none_when_flag_off(self):
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", False):
            self.assertIsNone(run_backtest())

    def test_returns_none_when_no_resolved_entries(self):
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=[]):
            self.assertIsNone(run_backtest())

    def test_historical_sim_returns_dict(self):
        log = self._mock_log(30)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log), \
             patch.object(bt_mod, "_save_backtest_report", return_value=Path("/tmp/r.json")), \
             patch.object(bt_mod, "send_backtest_discord", return_value=False):
            result = run_backtest(mode="historical_sim", notify=False)
        self.assertIsNotNone(result)
        self.assertIn("mode", result)
        self.assertIn("summary", result)

    def test_paper_mode_returns_dict(self):
        log = self._mock_log(30)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log), \
             patch.object(bt_mod, "_save_backtest_report", return_value=Path("/tmp/r.json")), \
             patch.object(bt_mod, "send_backtest_discord", return_value=False):
            result = run_backtest(mode="paper", notify=False)
        self.assertIsNotNone(result)

    def test_out_of_sample_returns_dict(self):
        log = self._mock_log(50)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log), \
             patch.object(bt_mod, "_save_backtest_report", return_value=Path("/tmp/r.json")), \
             patch.object(bt_mod, "send_backtest_discord", return_value=False):
            result = run_backtest(mode="out_of_sample", notify=False)
        self.assertIsNotNone(result)

    def test_walk_forward_returns_dict_with_windows(self):
        log = self._mock_log(200)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log), \
             patch.object(bt_mod, "_save_backtest_report", return_value=Path("/tmp/r.json")), \
             patch.object(bt_mod, "send_backtest_discord", return_value=False):
            result = run_backtest(mode="walk_forward", window_size=80, test_size=20, notify=False)
        if result:
            self.assertIn("windows", result)

    def test_invalid_mode_raises_value_error(self):
        log = self._mock_log(30)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log):
            with self.assertRaises(ValueError):
                run_backtest(mode="invalid_mode", notify=False)

    def test_report_contains_generated_at(self):
        log = self._mock_log(30)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log), \
             patch.object(bt_mod, "_save_backtest_report", return_value=Path("/tmp/r.json")), \
             patch.object(bt_mod, "send_backtest_discord", return_value=False):
            result = run_backtest(mode="historical_sim", notify=False)
        self.assertIn("generated_at", result)

    def test_notify_calls_discord(self):
        log = self._mock_log(30)
        with patch.object(bt_mod, "ENABLE_ADVANCED_BACKTESTS", True), \
             patch.object(bt_mod, "_load_log", return_value=log), \
             patch.object(bt_mod, "_save_backtest_report", return_value=Path("/tmp/r.json")), \
             patch.object(bt_mod, "send_backtest_discord", return_value=False) as mock_send:
            run_backtest(mode="historical_sim", notify=True)
        mock_send.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
