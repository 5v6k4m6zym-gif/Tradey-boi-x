import json

from diagnostics.filter_impact_analyzer import FilterImpactAnalyzer, analyze_layer


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_analyze_layer_counts_pass_and_suppress(tmp_path):
    path = tmp_path / "layer.jsonl"
    _write_jsonl(path, [
        {"symbol": "AAA", "timestamp": "2026-07-01T00:00:00", "passed": True},
        {"symbol": "BBB", "timestamp": "2026-07-01T00:00:00", "passed": False,
         "rejection_reasons": ["x"]},
        {"symbol": "CCC", "timestamp": "2026-07-01T00:00:00", "passed": False,
         "rejection_reasons": ["y"]},
    ])
    stats = analyze_layer("custom", path)
    assert stats["activations"] == 3
    assert stats["outputs_passed"] == 1
    assert stats["suppressed"] == 2
    assert stats["duplication_rate"] == 0.0


def test_analyze_layer_detects_duplication(tmp_path):
    path = tmp_path / "layer.jsonl"
    _write_jsonl(path, [
        {"symbol": "AAA", "timestamp": "2026-07-01T00:00:00", "passed": True},
        {"symbol": "AAA", "timestamp": "2026-07-01T05:00:00", "passed": True},
    ])
    stats = analyze_layer("custom", path)
    assert stats["duplicated_signal_keys"] == 1
    assert stats["duplication_rate"] == 1.0


def test_analyze_layer_missing_file_returns_zeroes(tmp_path):
    stats = analyze_layer("custom", tmp_path / "does_not_exist.jsonl")
    assert stats["activations"] == 0
    assert "error" not in stats


def test_analyze_layer_never_raises_on_corrupt_data(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n{also not json\n")
    stats = analyze_layer("custom", path)
    assert stats["activations"] == 0


def test_filter_impact_analyzer_analyze_all_runs_without_error():
    result = FilterImpactAnalyzer().analyze_all()
    assert "layers" in result
    assert "layers_with_duplication" in result
    assert set(result["layers"].keys()) == {
        "trade_evaluator", "adaptive_core", "strategy_optimizer", "audit_engine"
    }
