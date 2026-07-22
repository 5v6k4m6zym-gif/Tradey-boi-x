"""
Backtest state singleton.

Python caches imported modules in sys.modules, so this module is loaded
ONCE per process lifetime regardless of how many times Streamlit reruns
pro_dashboard.py.  The background thread and the main thread both reference
the exact same dict object — writes from the thread are immediately visible
to the next Streamlit rerun without any locking needed (GIL protects dict
key/value swaps).
"""

STATE: dict = {
    "running":   False,
    "done":      False,
    "progress":  (0, 1, "Starting…"),
    "result":    None,
    "error":     None,
    "traceback": None,
}


def reset() -> None:
    """Reset to a clean pre-run state (call from main thread before starting a run)."""
    STATE.update({
        "running":   True,
        "done":      False,
        "progress":  (0, 1, "Starting download…"),
        "result":    None,
        "error":     None,
        "traceback": None,
    })


def finish_ok(result) -> None:
    """Mark run complete with results (call from worker thread)."""
    STATE["result"]  = result
    STATE["running"] = False
    STATE["done"]    = True


def finish_err(err: str, tb: str) -> None:
    """Mark run failed (call from worker thread)."""
    STATE["error"]     = err
    STATE["traceback"] = tb
    STATE["running"]   = False
    STATE["done"]      = True


def set_progress(done: int, total: int, msg: str) -> None:
    """Update progress (call from worker thread)."""
    STATE["progress"] = (done, total, msg)
