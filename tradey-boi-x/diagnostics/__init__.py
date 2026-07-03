"""Additive, observe-only system-behaviour audit tooling for Tradey Boi X.

FAIL-SAFE CONTRACT (applies to every module in this package):
  - Never modifies the prediction model, signal generation, or execution logic.
  - Never raises out of a public function — all internal errors are caught and
    reported as part of the diagnostic output, never propagated.
  - Never blocks, delays, or mutates a live trade/alert. These tools only read
    existing logs/state and, where noted, append to their own trace log.

Public surface:
  - AlertBehaviourTester   (Part 1)
  - MorningEvaluationDebugger (Part 2)
  - FilterImpactAnalyzer   (Part 3)
  - TraceLogger            (Part 4)
  - EventSystemAuditor     (Part 6)
  - DiagnosticReportGenerator (Part 8)
"""
