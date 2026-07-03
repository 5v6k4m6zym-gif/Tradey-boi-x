from diagnostics.morning_evaluation_debugger import (
    MorningEvaluationDebugger,
    _simulate_restart_regression,
)
from diagnostics.event_system_auditor import EventSystemAuditor
from diagnostics import trace_logger, config


def test_regression_simulation_reproduces_old_bug_and_confirms_fix():
    result = _simulate_restart_regression()
    assert result["old_in_memory_mechanism_would_resend_on_restart"] is True
    assert result["new_persisted_mechanism_prevents_resend_on_restart"] is True
    assert result["regression_guard_passes"] is True
    assert result["expected_evaluations_per_calendar_day"] == 1


def test_event_system_auditor_finds_single_scheduler_loop():
    result = EventSystemAuditor().audit()
    assert "error" not in result
    assert result["while_true_scheduler_loops"] == 1
    assert result["send_morning_brief_call_sites"] == 1
    assert result["double_registration_detected"] is False
    assert result["persisted_dedup_state_present"] is True


def test_debugger_produces_high_confidence_root_cause_when_fix_present():
    result = MorningEvaluationDebugger().diagnose()
    assert "error" not in result
    assert result["confidence_score"] >= 80
    assert "root_cause_hypothesis" in result
    assert result["event_system_audit"]["persisted_dedup_state_present"] is True


def test_trace_logger_round_trip(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(config, "TRACE_LOG_PATH", trace_path)
    monkeypatch.setattr(trace_logger.config, "TRACE_LOG_PATH", trace_path)

    ok = trace_logger.log_trace("test_event", foo="bar")
    assert ok is True
    records = trace_logger.read_traces(trace_path)
    assert len(records) == 1
    assert records[0]["event"] == "test_event"
    assert records[0]["foo"] == "bar"


def test_scanner_state_persistence_round_trip(tmp_path, monkeypatch):
    state_path = tmp_path / "scanner_state.json"
    monkeypatch.setattr(trace_logger.config, "SCANNER_STATE_PATH", state_path)

    assert trace_logger.load_scanner_state() == {}
    ok = trace_logger.save_scanner_state({"brief_sent_date": "2026-07-03"})
    assert ok is True
    assert trace_logger.load_scanner_state() == {"brief_sent_date": "2026-07-03"}


def test_trace_logger_never_raises_on_bad_state_path(monkeypatch):
    monkeypatch.setattr(trace_logger.config, "SCANNER_STATE_PATH",
                         "/nonexistent-root-dir/impossible/state.json")
    assert trace_logger.load_scanner_state() == {}
    assert trace_logger.save_scanner_state({"x": 1}) is False
