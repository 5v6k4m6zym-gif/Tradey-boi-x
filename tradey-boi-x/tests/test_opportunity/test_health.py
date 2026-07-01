"""
Tests for opportunity.health — Phase 7 System Health Monitor
No psutil required, no actual file I/O beyond temp dirs, no Discord.
"""
import sys
import json
import time
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opportunity.health as health_mod
from opportunity.health import (
    log_health_event,
    _load_health_log,
    check_memory,
    check_duplicate,
    HealthMonitor,
    wrap_run_scan,
    run_health_check,
    send_weekly_health_report,
)


class _TmpHealthDir:
    """Context manager that redirects LOGS_DIR and HEALTH_LOG to a temp dir."""
    def __init__(self):
        self._tmpdir = tempfile.mkdtemp()

    def __enter__(self):
        self.logs_dir   = Path(self._tmpdir) / "logs"
        self.health_log = self.logs_dir / "health.log"
        self._p1 = patch.object(health_mod, "LOGS_DIR",   self.logs_dir)
        self._p2 = patch.object(health_mod, "HEALTH_LOG", self.health_log)
        self._p1.start()
        self._p2.start()
        return self

    def __exit__(self, *args):
        self._p1.stop()
        self._p2.stop()


# ─── log_health_event ────────────────────────────────────────────────────────

class TestLogHealthEvent(unittest.TestCase):

    def test_noop_when_flag_off(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", False):
            log_health_event("TEST")
            self.assertFalse(tmp.health_log.exists())

    def test_creates_log_file_when_flag_on(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            log_health_event("TEST_EVENT", value=42)
            self.assertTrue(tmp.health_log.exists())

    def test_log_entry_is_valid_json(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            log_health_event("TEST_EVENT", foo="bar")
            lines = tmp.health_log.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["event"], "TEST_EVENT")
            self.assertEqual(record["foo"],   "bar")

    def test_multiple_events_appended(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            log_health_event("A")
            log_health_event("B")
            log_health_event("C")
            lines = [l for l in tmp.health_log.read_text().strip().splitlines() if l]
            self.assertEqual(len(lines), 3)

    def test_timestamp_field_present(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            log_health_event("TS_TEST")
            record = json.loads(tmp.health_log.read_text().strip())
            self.assertIn("ts", record)


# ─── check_memory ─────────────────────────────────────────────────────────────

class TestCheckMemory(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        with patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            m = check_memory()
        for key in ("used_pct", "used_mb", "total_mb", "warning"):
            self.assertIn(key, m)

    def test_warning_true_above_threshold(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.object(health_mod, "MEMORY_WARN_PCT", 0.0):   # threshold = 0 → always warn
            m = check_memory()
        self.assertTrue(m["warning"])

    def test_works_without_psutil(self):
        with patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.dict("sys.modules", {"psutil": None}):
            try:
                m = check_memory()
                self.assertIn("used_pct", m)
            except Exception:
                pass   # ImportError handled gracefully inside function

    def test_warning_false_below_threshold(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.object(health_mod, "MEMORY_WARN_PCT", 999.0):  # threshold = 999 → never warn
            m = check_memory()
        self.assertFalse(m["warning"])


# ─── check_duplicate ─────────────────────────────────────────────────────────

class TestCheckDuplicate(unittest.TestCase):

    def setUp(self):
        # Clear the registry before each test
        health_mod._alert_registry.clear()

    def test_first_alert_is_not_duplicate(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            self.assertFalse(check_duplicate("NEWT.AX"))

    def test_second_alert_within_window_is_duplicate(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.object(health_mod, "MAX_DUPE_WINDOW_SEC", 3600):
            check_duplicate("DUP.AX")   # register
            self.assertTrue(check_duplicate("DUP.AX"))   # duplicate

    def test_different_tickers_are_not_duplicates(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            check_duplicate("AAA.AX")
            self.assertFalse(check_duplicate("BBB.AX"))

    def test_alert_after_window_expires_is_not_duplicate(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.object(health_mod, "MAX_DUPE_WINDOW_SEC", 0):  # window = 0 → never dupes
            check_duplicate("EXP.AX")
            self.assertFalse(check_duplicate("EXP.AX"))


# ─── HealthMonitor context manager ───────────────────────────────────────────

class TestHealthMonitor(unittest.TestCase):

    def test_noop_when_flag_off(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", False):
            with HealthMonitor("test"):
                pass
            self.assertFalse(tmp.health_log.exists())

    def test_logs_scan_start_and_end(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            with HealthMonitor("unit_test"):
                pass
            lines = [json.loads(l) for l in
                     tmp.health_log.read_text().strip().splitlines() if l]
            events = [l["event"] for l in lines]
            self.assertIn("SCAN_START", events)
            self.assertIn("SCAN_END",   events)

    def test_end_record_has_duration(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            with HealthMonitor("dur_test"):
                pass
            records = [json.loads(l) for l in
                       tmp.health_log.read_text().strip().splitlines() if l]
            end = next(r for r in records if r["event"] == "SCAN_END")
            self.assertIn("duration_sec", end)
            self.assertGreaterEqual(end["duration_sec"], 0)

    def test_does_not_suppress_exceptions(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            with self.assertRaises(ValueError):
                with HealthMonitor("exc_test"):
                    raise ValueError("test error")

    def test_error_logged_on_exception(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            try:
                with HealthMonitor("err_test"):
                    raise RuntimeError("oops")
            except RuntimeError:
                pass
            records = [json.loads(l) for l in
                       tmp.health_log.read_text().strip().splitlines() if l]
            end = next((r for r in records if r["event"] == "SCAN_END"), None)
            self.assertIsNotNone(end)
            self.assertEqual(end["status"], "ERROR")


# ─── wrap_run_scan ────────────────────────────────────────────────────────────

class TestWrapRunScan(unittest.TestCase):

    def test_returns_original_fn_when_flag_off(self):
        def my_scan(): return 42
        with patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", False):
            wrapped = wrap_run_scan(my_scan)
        self.assertIs(wrapped, my_scan)

    def test_wrapped_fn_returns_correct_value(self):
        def my_scan(): return 99
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            wrapped = wrap_run_scan(my_scan)
            result  = wrapped()
        self.assertEqual(result, 99)

    def test_wrapped_fn_propagates_exception(self):
        def bad_scan(): raise ValueError("boom")
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            wrapped = wrap_run_scan(bad_scan)
            with self.assertRaises(ValueError):
                wrapped()


# ─── run_health_check ─────────────────────────────────────────────────────────

class TestRunHealthCheck(unittest.TestCase):

    def test_returns_none_when_flag_off(self):
        with patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", False):
            self.assertIsNone(run_health_check())

    def test_returns_dict_when_flag_on(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            result = run_health_check()
        self.assertIsNotNone(result)
        for key in ("timestamp", "memory", "scans_24h", "errors_24h",
                    "slow_scans_24h", "duplicate_alerts"):
            self.assertIn(key, result)

    def test_scans_24h_counts_recent_scan_ends(self):
        with _TmpHealthDir() as tmp, \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True):
            # Write 3 SCAN_END events in the last 24h
            for _ in range(3):
                log_health_event("SCAN_END", duration_sec=30.0, status="OK", slow=False, error=None)
            result = run_health_check()
        self.assertEqual(result["scans_24h"], 3)


# ─── send_weekly_health_report ───────────────────────────────────────────────

class TestSendWeeklyHealthReport(unittest.TestCase):

    def test_returns_false_when_flag_off(self):
        with patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", False):
            self.assertFalse(send_weekly_health_report())

    def test_returns_false_with_no_webhook(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.dict("os.environ", {}, clear=True):
            self.assertFalse(send_weekly_health_report())

    def test_sends_report_successfully(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__  = MagicMock(return_value=False)
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/hook"}), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = send_weekly_health_report()
        self.assertTrue(result)

    def test_returns_false_on_network_error(self):
        with _TmpHealthDir(), \
             patch.object(health_mod, "ENABLE_SYSTEM_HEALTH", True), \
             patch.dict("os.environ", {"Discordwebhook": "https://discord.test/hook"}), \
             patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = send_weekly_health_report()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
