import json

from diagnostics.alert_behaviour_tester import AlertBehaviourTester
from diagnostics import config


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_pass_through_and_filtered_classification(tmp_path, monkeypatch):
    signal_log = tmp_path / "signal_log.json"
    _write_json(signal_log, [
        {"ticker": "AAA", "signal_date": "2026-07-01"},
        {"ticker": "BBB", "signal_date": "2026-07-01"},
    ])
    eval_log = tmp_path / "trade_evaluations.jsonl"
    _write_jsonl(eval_log, [
        {"symbol": "AAA", "timestamp": "2026-07-01T01:00:00", "passed": True},
        {"symbol": "BBB", "timestamp": "2026-07-01T01:00:00", "passed": False,
         "rejection_reasons": ["low edge"]},
    ])

    monkeypatch.setattr(config, "SIGNAL_LOG_PATH", signal_log)
    import diagnostics.filter_impact_analyzer as fia
    monkeypatch.setattr(fia, "LAYERS", {"trade_evaluator": eval_log})
    monkeypatch.setattr("diagnostics.alert_behaviour_tester.LAYERS", {"trade_evaluator": eval_log})

    result = AlertBehaviourTester().run()
    assert result["pass_through"] == 1
    assert result["filtered_out"] == 1
    assert result["duplicated"] == 0
    assert result["critical_issues_found"] is False


def test_duplicated_signal_detected(tmp_path, monkeypatch):
    signal_log = tmp_path / "signal_log.json"
    _write_json(signal_log, [{"ticker": "AAA", "signal_date": "2026-07-01"}])
    eval_log = tmp_path / "trade_evaluations.jsonl"
    _write_jsonl(eval_log, [
        {"symbol": "AAA", "timestamp": "2026-07-01T01:00:00", "passed": True},
        {"symbol": "AAA", "timestamp": "2026-07-01T02:00:00", "passed": True},
    ])
    monkeypatch.setattr(config, "SIGNAL_LOG_PATH", signal_log)
    monkeypatch.setattr("diagnostics.alert_behaviour_tester.LAYERS", {"trade_evaluator": eval_log})

    result = AlertBehaviourTester().run()
    assert result["duplicated"] == 1
    assert result["critical_issues_found"] is True


def test_missing_signal_log_returns_zero_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIGNAL_LOG_PATH", tmp_path / "missing.json")
    result = AlertBehaviourTester().run()
    assert result["total_original_signals"] == 0
    assert "error" not in result
