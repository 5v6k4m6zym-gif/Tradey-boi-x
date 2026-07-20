"""
Gate History & Automatic Rollback — v4.0 Adaptive Gate Validation.

Maintains a versioned snapshot directory of every gate configuration
change. If live performance after a change deteriorates beyond
configurable thresholds, the previous snapshot is automatically restored.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

HISTORY_DIR        = Path(__file__).parent.parent / "config" / "gate_history"
MAX_SNAPSHOTS      = 30
ROLLBACK_PF_DROP   = 0.20
ROLLBACK_EXP_DROP  = 0.25
MIN_TRADES_TO_JUDGE = 10


def save_snapshot(cfg: dict, reason: str, metrics: dict) -> Path:
    """Snapshot current gate config before applying a proposed change."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = HISTORY_DIR / f"{ts}.json"
    path.write_text(json.dumps({
        "timestamp": ts,
        "reason":    reason,
        "config":    cfg,
        "metrics":   metrics,
    }, indent=2))
    _prune()
    return path


def load_snapshots(n: int = 5) -> list[dict]:
    """Return up to N most-recent snapshots (newest first)."""
    if not HISTORY_DIR.exists():
        return []
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:n]
    out: list[dict] = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


def restore_previous(adaptive_config_path: Path) -> dict | None:
    """
    Restore the config from the snapshot just before the most recent one.
    Returns the restored config dict, or None if no prior snapshot exists.
    """
    snaps = load_snapshots(2)
    if len(snaps) < 2:
        return None
    prev = snaps[1]
    cfg  = prev.get("config", {})
    if not cfg:
        return None
    adaptive_config_path.write_text(json.dumps({
        **cfg,
        "last_updated":      datetime.now().strftime("%Y-%m-%d"),
        "_rollback_from":    snaps[0]["timestamp"],
        "_rollback_reason":  "auto-rollback: live performance deteriorated after gate change",
    }, indent=2))
    return cfg


def check_needs_rollback(
    post_change_trades: list[dict],
    pre_change_metrics: dict,
) -> tuple[bool, str]:
    """
    Determine whether live performance since the last gate change is bad
    enough to trigger automatic rollback.

    Returns (needs_rollback, reason_string).
    """
    resolved = [t for t in post_change_trades
                if t.get("actual_pct") is not None and t.get("outcome")]
    if len(resolved) < MIN_TRADES_TO_JUDGE:
        return False, f"Too few post-change trades ({len(resolved)}) to judge"

    from .metrics import compute_metrics
    live = compute_metrics(resolved)

    pf_drop  = pre_change_metrics.get("profit_factor",  1.0) - live["profit_factor"]
    exp_drop = pre_change_metrics.get("expectancy",      0.0) - live["expectancy"]

    reasons: list[str] = []
    if pf_drop  > ROLLBACK_PF_DROP:
        reasons.append(
            f"PF dropped {pf_drop:.2f} "
            f"({pre_change_metrics.get('profit_factor', 0):.2f}→{live['profit_factor']:.2f})"
        )
    if exp_drop > ROLLBACK_EXP_DROP:
        reasons.append(
            f"Expectancy dropped {exp_drop:.3f} "
            f"({pre_change_metrics.get('expectancy', 0):.3f}→{live['expectancy']:.3f})"
        )

    if reasons:
        return True, "Rollback triggered — " + "; ".join(reasons)
    return False, "Performance within acceptable bounds"


def _prune() -> None:
    if not HISTORY_DIR.exists():
        return
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    for old in files[MAX_SNAPSHOTS:]:
        old.unlink(missing_ok=True)
