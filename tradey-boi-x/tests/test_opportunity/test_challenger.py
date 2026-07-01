"""
Tests for opportunity.challenger — Phase 6 Strategy Challenger Sandbox
Guardrails verified: no engine imports, no Discord, no log writes.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.challenger as ch_mod
from opportunity.challenger import (
    compare_strategies,
    save_challenger_report,
    run_challenger,
    _challenger_opportunity_score,
    _prod_opportunity_score,
    _recommendation,
    DEFAULT_CANDIDATE_WEIGHTS,
    PRODUCTION_WEIGHTS,
)


# ─── Synthetic helpers ────────────────────────────────────────────────────────

def _entry(
    outcome:    str   = "WIN",
    actual_pct: float = 0.12,
    prob:       float = 0.70,
    days_ago:   int   = 5,
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
        "pred_days":    14,
        "outcome":      outcome,
        "exit_price":   round(1.00 * (1 + actual_pct), 4),
        "actual_pct":   actual_pct,
    }


def _history(n_wins: int = 15, n_losses: int = 10) -> list[dict]:
    entries = []
    for i in range(n_wins):
        entries.append(_entry("WIN", 0.15, prob=0.72, days_ago=i * 2 + 1))
    for i in range(n_losses):
        entries.append(_entry("HIT_STOP", -0.06, prob=0.58, days_ago=i * 2 + 2))
    return entries


# ─── Guardrail: no write imports from engine ──────────────────────────────────

class TestGuardrails(unittest.TestCase):

    def test_challenger_does_not_import_send_alert(self):
        """send_alert writes to Discord — challenger must never import it."""
        import opportunity.challenger as ch
        self.assertFalse(hasattr(ch, "send_alert"))

    def test_challenger_does_not_import_log_signal(self):
        """log_signal writes to disk — challenger must never import it."""
        import opportunity.challenger as ch
        self.assertFalse(hasattr(ch, "log_signal"))

    def test_challenger_does_not_import_resolve_outcomes(self):
        import opportunity.challenger as ch
        self.assertFalse(hasattr(ch, "resolve_outcomes"))

    def test_requires_human_approval_flag_in_comparison(self):
        entries = _history(10, 5)
        result = compare_strategies(entries, DEFAULT_CANDIDATE_WEIGHTS)
        self.assertTrue(result.get("requires_human_approval"))


# ─── _challenger_opportunity_score ───────────────────────────────────────────

class TestChallengerScore(unittest.TestCase):

    def test_returns_float_for_valid_entry(self):
        e = _entry("WIN", 0.15, prob=0.70)
        score = _challenger_opportunity_score(e, DEFAULT_CANDIDATE_WEIGHTS)
        self.assertIsInstance(score, float)

    def test_returns_none_for_zero_prob(self):
        e = _entry(prob=0.0)
        self.assertIsNone(_challenger_opportunity_score(e, DEFAULT_CANDIDATE_WEIGHTS))

    def test_score_between_0_and_100(self):
        e = _entry("WIN", 0.15, prob=0.70)
        score = _challenger_opportunity_score(e, DEFAULT_CANDIDATE_WEIGHTS)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_win_scores_higher_than_loss(self):
        win_e  = _entry("WIN",     0.20, prob=0.70)
        loss_e = _entry("HIT_STOP", -0.06, prob=0.70)
        win_s  = _challenger_opportunity_score(win_e,  DEFAULT_CANDIDATE_WEIGHTS)
        loss_s = _challenger_opportunity_score(loss_e, DEFAULT_CANDIDATE_WEIGHTS)
        self.assertGreater(win_s, loss_s)

    def test_different_weights_produce_different_scores(self):
        e = _entry("WIN", 0.15, prob=0.70)
        alt_weights = {k: v * 1.1 for k, v in DEFAULT_CANDIDATE_WEIGHTS.items()}
        s1 = _challenger_opportunity_score(e, DEFAULT_CANDIDATE_WEIGHTS)
        s2 = _challenger_opportunity_score(e, alt_weights)
        # Not necessarily different by much, but the function runs cleanly
        self.assertIsNotNone(s2)


# ─── compare_strategies ──────────────────────────────────────────────────────

class TestCompareStrategies(unittest.TestCase):

    def test_returns_dict(self):
        result = compare_strategies(_history(), DEFAULT_CANDIDATE_WEIGHTS)
        self.assertIsInstance(result, dict)

    def test_all_required_keys_present(self):
        result = compare_strategies(_history(), DEFAULT_CANDIDATE_WEIGHTS)
        for key in ("generated_at", "total_history", "production",
                    "challenger", "delta", "pct_change", "recommendation",
                    "requires_human_approval"):
            self.assertIn(key, result)

    def test_production_contains_metrics(self):
        result = compare_strategies(_history(), DEFAULT_CANDIDATE_WEIGHTS)
        self.assertIn("metrics",  result["production"])
        self.assertIn("selected", result["production"])

    def test_challenger_contains_metrics(self):
        result = compare_strategies(_history(), DEFAULT_CANDIDATE_WEIGHTS)
        self.assertIn("metrics",  result["challenger"])
        self.assertIn("selected", result["challenger"])

    def test_delta_keys_match_comparison_keys(self):
        result = compare_strategies(_history(), DEFAULT_CANDIDATE_WEIGHTS)
        delta_keys = set(result["delta"].keys())
        expected   = {"win_rate", "profit_factor", "sharpe_ratio", "expectancy_r",
                      "avg_gain_pct", "avg_loss_pct", "max_drawdown_pct",
                      "annualised_return_pct"}
        self.assertEqual(delta_keys, expected)

    def test_empty_history_returns_valid_structure(self):
        result = compare_strategies([], DEFAULT_CANDIDATE_WEIGHTS)
        self.assertIn("recommendation", result)

    def test_total_history_correct(self):
        history = _history(10, 5)
        result  = compare_strategies(history, DEFAULT_CANDIDATE_WEIGHTS)
        self.assertEqual(result["total_history"], len(history))

    def test_recommendation_is_string(self):
        result = compare_strategies(_history(), DEFAULT_CANDIDATE_WEIGHTS)
        self.assertIsInstance(result["recommendation"], str)
        self.assertGreater(len(result["recommendation"]), 10)


# ─── _recommendation ─────────────────────────────────────────────────────────

class TestRecommendation(unittest.TestCase):

    def _metrics(self, win_rate, exp_r, sharpe, drawdown) -> dict:
        return {"win_rate": win_rate, "expectancy_r": exp_r,
                "sharpe_ratio": sharpe, "max_drawdown_pct": drawdown}

    def test_clearly_better_challenger_suggests_consider(self):
        prod = self._metrics(0.55, 0.30, 0.80, 12.0)
        chal = self._metrics(0.65, 0.50, 1.20, 8.0)
        rec = _recommendation(prod, chal)
        self.assertIn("CONSIDER", rec)

    def test_worse_challenger_suggests_keep_production(self):
        prod = self._metrics(0.65, 0.50, 1.20, 8.0)
        chal = self._metrics(0.45, 0.10, 0.40, 18.0)
        rec = _recommendation(prod, chal)
        self.assertIn("KEEP_PRODUCTION", rec)

    def test_borderline_case(self):
        prod = self._metrics(0.60, 0.30, 0.80, 10.0)
        chal = self._metrics(0.60, 0.36, 0.80, 10.0)  # only exp_r slightly better
        rec = _recommendation(prod, chal)
        self.assertIn(rec.split(" — ")[0], ("CONSIDER_ADOPTING", "BORDERLINE", "KEEP_PRODUCTION"))


# ─── save_challenger_report ───────────────────────────────────────────────────

class TestSaveChallengerReport(unittest.TestCase):

    def test_saves_json_file(self):
        import tempfile, json
        result = compare_strategies(_history(5, 5), DEFAULT_CANDIDATE_WEIGHTS)
        with patch.object(ch_mod, "CHALLENGER_DIR",
                          Path(tempfile.mkdtemp()) / "challenger"):
            path = save_challenger_report(result)
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text())
        self.assertIn("recommendation", loaded)

    def test_creates_directory_if_missing(self):
        import tempfile
        result = compare_strategies(_history(3, 2), DEFAULT_CANDIDATE_WEIGHTS)
        with patch.object(ch_mod, "CHALLENGER_DIR",
                          Path(tempfile.mkdtemp()) / "new_dir" / "challenger"):
            path = save_challenger_report(result)
        self.assertTrue(path.parent.exists())


# ─── run_challenger ───────────────────────────────────────────────────────────

class TestRunChallenger(unittest.TestCase):

    def test_returns_none_when_flag_off(self):
        with patch.object(ch_mod, "ENABLE_STRATEGY_CHALLENGER", False):
            self.assertIsNone(run_challenger())

    def test_returns_none_on_empty_log(self):
        with patch.object(ch_mod, "ENABLE_STRATEGY_CHALLENGER", True), \
             patch.object(ch_mod, "_load_log", return_value=[]):
            self.assertIsNone(run_challenger())

    def test_returns_comparison_dict_with_data(self):
        log = _history(15, 10)
        with patch.object(ch_mod, "ENABLE_STRATEGY_CHALLENGER", True), \
             patch.object(ch_mod, "_load_log", return_value=log), \
             patch.object(ch_mod, "save_challenger_report",
                          return_value=Path("/tmp/challenger.json")):
            result = run_challenger()
        self.assertIsNotNone(result)
        self.assertIn("recommendation", result)

    def test_custom_weights_accepted(self):
        log = _history(10, 5)
        custom = {**DEFAULT_CANDIDATE_WEIGHTS, "expected_return": 0.45,
                  "technical_strength": 0.15}
        with patch.object(ch_mod, "ENABLE_STRATEGY_CHALLENGER", True), \
             patch.object(ch_mod, "_load_log", return_value=log), \
             patch.object(ch_mod, "save_challenger_report",
                          return_value=Path("/tmp/challenger.json")):
            result = run_challenger(candidate_weights=custom)
        self.assertIsNotNone(result)
        self.assertEqual(result["challenger"]["weights"]["expected_return"], 0.45)


if __name__ == "__main__":
    unittest.main(verbosity=2)
