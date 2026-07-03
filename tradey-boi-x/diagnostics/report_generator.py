"""Part 8 — Diagnostic Report Generator.

Assembles AlertBehaviourTester, FilterImpactAnalyzer, MorningEvaluationDebugger
(which itself wraps EventSystemAuditor) into the final structured report:

  1. Morning Evaluation Duplication Root Cause
  2. Pipeline Stage Responsible
  3. Severity Level (Low / Medium / Critical)
  4. Fix Recommendation (safe, non-breaking)
  5. Alert Integrity Score (0-100)
  6. System Execution Trace Summary

Read-only. Never raises.
"""
from datetime import datetime, timezone
from typing import Any

from diagnostics import config
from diagnostics.alert_behaviour_tester import AlertBehaviourTester
from diagnostics.filter_impact_analyzer import FilterImpactAnalyzer
from diagnostics.morning_evaluation_debugger import MorningEvaluationDebugger


def _severity(debugger_result: dict, auditor_result: dict, behaviour_result: dict) -> str:
    if auditor_result.get("double_registration_detected"):
        return "Critical"
    if behaviour_result.get("critical_issues_found"):
        return "Critical"
    if not debugger_result.get("event_system_audit", {}).get("persisted_dedup_state_present"):
        return "Medium"
    return "Low"


def _alert_integrity_score(behaviour: dict, filters: dict) -> int:
    try:
        total = behaviour.get("total_original_signals", 0)
        if total == 0:
            return 100
        duplicated = behaviour.get("duplicated", 0)
        # lost_or_layer_disabled is not penalized on its own (can be expected
        # when a layer was disabled at signal time) but duplication always is.
        penalty = (duplicated / total) * 100
        layer_dup_penalty = 5 * len(filters.get("layers_with_duplication", []))
        score = max(0, round(100 - penalty - layer_dup_penalty))
        return score
    except Exception:
        return 0


class DiagnosticReportGenerator:
    """Part 8 entry point."""

    def generate(self) -> dict[str, Any]:
        try:
            behaviour = AlertBehaviourTester().run()
            filters = FilterImpactAnalyzer().analyze_all()
            debugger_result = MorningEvaluationDebugger().diagnose()

            severity = _severity(
                debugger_result, debugger_result.get("event_system_audit", {}), behaviour
            )
            score = _alert_integrity_score(behaviour, filters)

            fix_recommendation = (
                "Persist the morning-brief dedup marker to disk (implemented: "
                "`scanner_state.json`, mirroring the existing `cooldowns.json` "
                "pattern) so a scanner process restart does not defeat the "
                "once-per-calendar-day gate. This is a scheduling-state change "
                "only — it does not modify send_morning_brief(), the prediction "
                "model, signal generation, regime detection, or execution logic."
            )

            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "1_morning_evaluation_duplication_root_cause":
                    debugger_result.get("root_cause_hypothesis"),
                "2_pipeline_stage_responsible":
                    debugger_result.get("pipeline_stage"),
                "3_severity_level": severity,
                "4_fix_recommendation": fix_recommendation,
                "5_alert_integrity_score": score,
                "6_system_execution_trace_summary": {
                    "alert_behaviour": behaviour,
                    "filter_impact": filters,
                    "morning_evaluation_debug": debugger_result,
                },
                "confidence_score": debugger_result.get("confidence_score"),
            }
            return report
        except Exception as e:
            return {"error": str(e)}

    def generate_markdown(self) -> str:
        r = self.generate()
        if "error" in r:
            return f"# System Audit Report — ERROR\n\n{r['error']}\n"

        behaviour = r["6_system_execution_trace_summary"]["alert_behaviour"]
        filters = r["6_system_execution_trace_summary"]["filter_impact"]
        debug = r["6_system_execution_trace_summary"]["morning_evaluation_debug"]

        lines = [
            "# Tradey Boi X — System Behaviour Audit Report",
            f"_Generated: {r['generated_at']}_",
            "",
            "## 1. Morning Evaluation Duplication — Root Cause",
            r["1_morning_evaluation_duplication_root_cause"] or "n/a",
            "",
            "## 2. Pipeline Stage Responsible",
            r["2_pipeline_stage_responsible"] or "n/a",
            "",
            f"## 3. Severity Level: **{r['3_severity_level']}**",
            "",
            "## 4. Fix Recommendation",
            r["4_fix_recommendation"],
            "",
            f"## 5. Alert Integrity Score: **{r['5_alert_integrity_score']}/100**",
            "",
            f"## Diagnosis Confidence: {r.get('confidence_score')}/100",
            "",
            "## 6. System Execution Trace Summary",
            "",
            "### Alert Behaviour (Part 1)",
            f"- Total original signals: {behaviour.get('total_original_signals')}",
            f"- Pass-through: {behaviour.get('pass_through')}",
            f"- Filtered out: {behaviour.get('filtered_out')}",
            f"- Duplicated (CRITICAL if >0): {behaviour.get('duplicated')}",
            f"- Lost or layer-disabled: {behaviour.get('lost_or_layer_disabled')}",
            "",
            "### Filter Impact (Part 3)",
        ]
        for name, stats in filters.get("layers", {}).items():
            if "error" in stats:
                lines.append(f"- **{name}**: error reading log ({stats['error']})")
                continue
            lines.append(
                f"- **{name}**: activations={stats['activations']}, "
                f"passed={stats['outputs_passed']}, suppressed={stats['suppressed']}, "
                f"duplication_rate={stats['duplication_rate']}, "
                f"suppression_rate={stats['suppression_rate']}"
            )
        lines += [
            "",
            "### Morning Evaluation Debug (Parts 2, 5, 6)",
            f"- Regression guard passes: "
            f"{debug.get('regression_simulation', {}).get('regression_guard_passes')}",
            f"- While-True scheduler loops found: "
            f"{debug.get('event_system_audit', {}).get('while_true_scheduler_loops')}",
            f"- send_morning_brief() call sites: "
            f"{debug.get('event_system_audit', {}).get('send_morning_brief_call_sites')}",
            f"- Persisted dedup state present: "
            f"{debug.get('event_system_audit', {}).get('persisted_dedup_state_present')}",
            f"- Trace log records found: "
            f"{debug.get('trace_log_analysis', {}).get('trace_records_found')}",
            f"- Dates with multiple sends (should be empty): "
            f"{debug.get('trace_log_analysis', {}).get('dates_with_multiple_sends')}",
        ]
        return "\n".join(lines)

    def write_report(self) -> str:
        md = self.generate_markdown()
        try:
            config.AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.AUDIT_REPORT_PATH.write_text(md)
        except Exception as e:
            return f"(failed to write report file: {e})\n\n{md}"
        return md
