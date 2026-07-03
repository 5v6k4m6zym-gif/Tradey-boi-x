---
name: Tradey Boi X opportunity-layer tests must isolate their logger
description: Unit tests that call process_trade_signal()/similar entry points without mocking the layer's log_*/_log_decision function silently write test fixture symbols into production JSONL logs.
---

Every `opportunity/*` layer (trade_evaluator, adaptive_core, strategy_optimizer, audit_engine, ...) appends decisions to a real production JSONL log file by default (e.g. `logs/trade_evaluations.jsonl`, `logs/adaptive_core_decisions.jsonl`). Unit tests that call `process_trade_signal()` (or any function that internally logs) without patching the module's logger function will write real entries using test fixture ticker names (e.g. "BAD.AX", "TEST.AX") straight into production logs, on every single pytest run.

**Why:** This was found while building `diagnostics/filter_impact_analyzer.py` — a "100% duplication rate" reading for `trade_evaluator` and `adaptive_core` turned out to be dozens of accumulated "BAD.AX"/"TEST.AX" entries from `test_trade_evaluator.py::test_live_mode_blocks_only_failing_trades` and `test_adaptive_core.py::test_internal_exception_falls_back_to_passthrough`, which called `process_trade_signal()` without mocking `log_trade_decision`/`_log_decision`. It was a test-hygiene bug, not a real production issue, but it polluted real logs and skewed any log-based analysis.

**How to apply:** Any new/edited opportunity-layer test that exercises `process_trade_signal()` (or similar) must either (a) `patch.object(<module>, "<log_function_name>")` for the duration of the call, or (b) redirect the module's `*_LOG_PATH` to a `tmp_path`. Before trusting any log-based diagnostic (duplication rate, activation counts, etc.), first check for fixture-looking symbols (ALL-CAPS placeholder names like "TEST.AX"/"BAD.AX"/"GOOD.AX") in the underlying JSONL and treat them as test pollution, not real signal.
