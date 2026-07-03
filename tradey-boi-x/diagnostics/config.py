"""Flags and paths for the observe-only system-behaviour audit tooling.

All flags default True because every component here is read-only /
append-only observation — there is no gating or trading-logic risk in
having them enabled by default (unlike the opportunity/ package's
ENABLE_* flags, which gate trades).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"

ENABLE_TRACE_LOGGER = os.getenv("ENABLE_TRACE_LOGGER", "true").lower() == "true"
ENABLE_ALERT_BEHAVIOUR_TESTER = os.getenv("ENABLE_ALERT_BEHAVIOUR_TESTER", "true").lower() == "true"

TRACE_LOG_PATH = Path(os.getenv("MORNING_TRACE_LOG_PATH", str(LOGS_DIR / "morning_eval_trace.jsonl")))
SCANNER_STATE_PATH = Path(os.getenv("SCANNER_STATE_PATH", str(BASE_DIR / "scanner_state.json")))

SIGNAL_LOG_PATH = BASE_DIR / "signal_log.json"
TRADE_EVAL_LOG_PATH = LOGS_DIR / "trade_evaluations.jsonl"
ADAPTIVE_CORE_LOG_PATH = LOGS_DIR / "adaptive_core_decisions.jsonl"
STRATEGY_OPTIMIZER_LOG_PATH = LOGS_DIR / "strategy_optimizer_decisions.jsonl"
AUDIT_TRADES_LOG_PATH = LOGS_DIR / "audit_trades.jsonl"

AUDIT_REPORT_PATH = LOGS_DIR / "system_audit_report.md"
