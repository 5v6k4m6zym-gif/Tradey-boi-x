"""
Tests for opportunity.costs — Realistic Backtesting trading cost model (T003).
Pure computation only; never touches signal generation.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.costs as costs_mod
from opportunity.costs import is_asx_ticker, round_trip_cost_pct, apply_cost


class TestIsAsxTicker(unittest.TestCase):
    def test_asx_suffix_detected(self):
        self.assertTrue(is_asx_ticker("BHP.AX"))

    def test_us_ticker_not_asx(self):
        self.assertFalse(is_asx_ticker("AAPL"))

    def test_empty_string_is_not_asx(self):
        self.assertFalse(is_asx_ticker(""))

    def test_case_insensitive(self):
        self.assertTrue(is_asx_ticker("bhp.ax"))


class TestRoundTripCostPct(unittest.TestCase):
    def test_disabled_returns_zero(self):
        costs_mod.ENABLE_REALISTIC_COSTS = False
        try:
            self.assertEqual(round_trip_cost_pct("BHP.AX"), 0.0)
        finally:
            costs_mod.ENABLE_REALISTIC_COSTS = True

    def test_asx_cost_is_positive_and_larger_than_us(self):
        us_cost = round_trip_cost_pct("AAPL")
        asx_cost = round_trip_cost_pct("BHP.AX")
        self.assertGreater(us_cost, 0)
        self.assertGreater(asx_cost, 0)
        self.assertGreater(asx_cost, us_cost)

    def test_cost_is_small_relative_to_typical_move(self):
        # Sanity check: round-trip cost should be a small fraction (<2%) of
        # trade value, not large enough to dominate any real trading signal.
        self.assertLess(round_trip_cost_pct("BHP.AX"), 0.02)
        self.assertLess(round_trip_cost_pct("AAPL"), 0.02)


class TestApplyCost(unittest.TestCase):
    def test_reduces_positive_return(self):
        net = apply_cost(0.15, "BHP.AX")
        self.assertLess(net, 0.15)

    def test_increases_magnitude_of_negative_return(self):
        net = apply_cost(-0.06, "BHP.AX")
        self.assertLess(net, -0.06)

    def test_none_actual_pct_treated_as_zero(self):
        net = apply_cost(None, "BHP.AX")
        self.assertLessEqual(net, 0.0)

    def test_disabled_leaves_return_unchanged(self):
        costs_mod.ENABLE_REALISTIC_COSTS = False
        try:
            self.assertEqual(apply_cost(0.10, "BHP.AX"), 0.10)
        finally:
            costs_mod.ENABLE_REALISTIC_COSTS = True


if __name__ == "__main__":
    unittest.main()
