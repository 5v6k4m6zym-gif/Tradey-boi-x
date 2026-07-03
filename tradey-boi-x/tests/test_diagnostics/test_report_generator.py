from diagnostics.report_generator import DiagnosticReportGenerator


def test_generate_returns_all_required_parts():
    report = DiagnosticReportGenerator().generate()
    assert "error" not in report
    required_keys = [
        "1_morning_evaluation_duplication_root_cause",
        "2_pipeline_stage_responsible",
        "3_severity_level",
        "4_fix_recommendation",
        "5_alert_integrity_score",
        "6_system_execution_trace_summary",
    ]
    for key in required_keys:
        assert key in report
    assert report["3_severity_level"] in {"Low", "Medium", "Critical"}
    assert 0 <= report["5_alert_integrity_score"] <= 100


def test_generate_markdown_produces_nonempty_string():
    md = DiagnosticReportGenerator().generate_markdown()
    assert isinstance(md, str)
    assert "System Behaviour Audit Report" in md
    assert "Severity Level" in md
