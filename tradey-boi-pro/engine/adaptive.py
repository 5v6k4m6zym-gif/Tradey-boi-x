"""
Tradey Boi Pro — Adaptive Threshold Learning

Same algorithm as Tradey Boi X's adaptive_threshold_update():
  • Reads the last 30 resolved ELITE/STRONG BUY trades from Pro's own DB
  • Computes R-multiple expectancy (same formula as X's _expectancy_stats)
  • Nudges prob_floor / sb_base_score by one step if expectancy is clearly
    too low (tighten) or clearly strong and low-volume (ease)
  • Writes result to tradey-boi-pro/config/adaptive_thresholds.json

Pro learns from its OWN live execution outcomes — independent of X's
Discord-alert outcomes. X's thresholds are used as the cold-start default;
once Pro has ≥10 resolved trades it manages its own gates.

Safety bounds (identical to X):
  prob_floor    : 0.50 – 0.60
  sb_base_score : 5    – 9
  min resolved trades before any change: 10
  max one step change per call
"""
from __future__ import annotations

import json
import logging
import pathlib
from collections import defaultdict
from datetime import datetime, timedelta

import db.database as db

log = logging.getLogger("ProAdaptive")

# ── Per-ticker score adjustments — refreshed every 15 min ─────────────────────
_ticker_adj_cache: dict[str, int] = {}
_ticker_adj_ts: datetime | None   = None
_TICKER_ADJ_TTL_MINUTES           = 15

_PRO_DIR      = pathlib.Path(__file__).parent.parent
_X_DIR        = _PRO_DIR.parent / "tradey-boi-x"
_PRO_CFG_PATH = _PRO_DIR / "config" / "adaptive_thresholds.json"
_X_CFG_PATH   = _X_DIR   / "config" / "adaptive_thresholds.json"

# Outcome strings that count as wins
_WIN_OUTCOMES = ("TARGET_HIT", "WIN", "HIT_TARGET", "EXPIRED_GAIN")


def _load_pro_cfg() -> dict:
    """Load Pro's own adaptive config; fall back to X's if Pro's doesn't exist yet."""
    if _PRO_CFG_PATH.exists():
        try:
            return json.loads(_PRO_CFG_PATH.read_text())
        except Exception:
            pass
    # Cold-start — inherit X's current thresholds
    if _X_CFG_PATH.exists():
        try:
            base = json.loads(_X_CFG_PATH.read_text())
            return {
                "prob_floor":    base.get("prob_floor",    0.53),
                "sb_base_score": base.get("sb_base_score", 7),
                "last_updated":  None,
                "last_checked":  None,
                "last_expectancy": None,
                "last_win_rate": None,
                "recent_n":      0,
                "adjustment_history": [],
                "_note": "Auto-managed by Pro's adaptive learning. "
                         "Safety bounds: prob_floor 0.50–0.60, sb_base_score 5–9.",
            }
        except Exception:
            pass
    return {
        "prob_floor": 0.53, "sb_base_score": 7,
        "last_updated": None, "last_checked": None,
        "last_expectancy": None, "last_win_rate": None,
        "recent_n": 0, "adjustment_history": [],
    }


def _save_pro_cfg(cfg: dict) -> None:
    _PRO_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PRO_CFG_PATH.write_text(json.dumps(cfg, indent=2))


def _expectancy_stats(trades: list[dict]) -> tuple[float, float, float]:
    """
    Compute (win_rate, avg_win_R, avg_loss_R, expectancy_R) from closed trades.
    R-multiple = actual_pnl_pct / risk_pct where risk_pct = |entry - stop| / entry.
    Mirrors X's _expectancy_stats() formula.
    """
    win_Rs, loss_Rs = [], []
    for t in trades:
        try:
            actual   = float(t.get("pnl_pct") or 0.0)
            entry_px = float(t.get("entry_price") or 0.0)
            stop_px  = t.get("stop_price")
            if entry_px > 0 and stop_px and float(stop_px) > 0:
                risk_pct = abs(entry_px - float(stop_px)) / entry_px
            else:
                risk_pct = 0.04   # 4% proxy
            if risk_pct <= 0:
                risk_pct = 0.04
            r = max(-5.0, min(5.0, actual / risk_pct))
            outcome = t.get("outcome") or t.get("exit_reason") or ""
            if outcome in _WIN_OUTCOMES or r > 0:
                win_Rs.append(r)
            else:
                loss_Rs.append(abs(r))
        except Exception:
            continue

    n = len(win_Rs) + len(loss_Rs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    win_rate   = len(win_Rs) / n
    avg_win_R  = sum(win_Rs)  / len(win_Rs)  if win_Rs  else 0.0
    avg_loss_R = sum(loss_Rs) / len(loss_Rs) if loss_Rs else 0.0
    # Same formula as X — 1.3× loss weight for slippage/commission drag
    expectancy = avg_win_R * win_rate - avg_loss_R * 1.3 * (1 - win_rate)
    return win_rate, avg_win_R, avg_loss_R, expectancy


def adaptive_threshold_update() -> dict:
    """
    Run Pro's own adaptive gate tuning. Call this after every batch of resolved
    trades (bot_runner calls it weekly or after every 10 closed positions).

    Returns a summary dict describing what was decided and why.
    """
    cfg        = _load_pro_cfg()
    prob_floor = float(cfg.get("prob_floor",    0.53))
    sb_base    = int(  cfg.get("sb_base_score", 7))

    today = datetime.utcnow().strftime("%Y-%m-%d")
    cfg["last_checked"] = today

    # ── Pull Pro's own closed trades from DB ──────────────────────────────────
    all_trades = db.all_trades(limit=200)
    resolved   = [
        t for t in all_trades
        if t.get("outcome") or t.get("exit_reason")
    ]
    # Focus on the last 30 ELITE/STRONG BUY exits
    elite_resolved = [
        t for t in resolved
        if t.get("tier") in ("ELITE", "STRONG BUY")
    ][-30:]
    recent = elite_resolved if elite_resolved else resolved[-30:]

    cfg["recent_n"] = len(recent)
    result = {
        "prob_floor":    prob_floor,
        "sb_base_score": sb_base,
        "recent_n":      len(recent),
        "adjustment":    None,
        "reason":        None,
    }

    if len(recent) < 10:
        result["reason"] = (
            f"Skipped — only {len(recent)} resolved trades (need ≥10). "
            f"Using {'Pro' if _PRO_CFG_PATH.exists() else 'X'} thresholds as-is."
        )
        _save_pro_cfg(cfg)
        log.info(f"Adaptive update: {result['reason']}")
        return result

    win_rate, avg_win_R, avg_loss_R, expectancy = _expectancy_stats(recent)
    cfg["last_expectancy"] = round(expectancy, 3)
    cfg["last_win_rate"]   = round(win_rate,   3)
    result["expectancy"]   = round(expectancy, 3)
    result["win_rate"]     = round(win_rate,   3)

    # ── How many ELITE/SB trades fired in the last 14 days? ──────────────────
    cutoff = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
    recent_alerts = sum(
        1 for t in all_trades
        if t.get("tier") in ("ELITE", "STRONG BUY")
        and (t.get("entry_date") or t.get("signal_date") or "") >= cutoff
    )

    # ── Adjustment rules (identical to X) ────────────────────────────────────
    new_prob = prob_floor
    new_sb   = sb_base
    reason   = "No change"

    if expectancy < -0.30:
        # Performance is poor — tighten gates
        new_prob = min(round(prob_floor + 0.01, 4), 0.60)
        new_sb   = min(sb_base + 1, 9)
        reason   = (
            f"Tightened: expectancy {expectancy:.3f}R < -0.30R threshold. "
            f"prob {prob_floor:.2f}→{new_prob:.2f}, score {sb_base}→{new_sb}"
        )
    elif expectancy > 0.80 and recent_alerts < 3:
        # Strong performance AND signals are rare — ease up slightly
        new_prob = max(round(prob_floor - 0.01, 4), 0.50)
        new_sb   = max(sb_base - 1, 5)
        reason   = (
            f"Eased: expectancy {expectancy:.3f}R > 0.80R and only "
            f"{recent_alerts} signals in 14 days. "
            f"prob {prob_floor:.2f}→{new_prob:.2f}, score {sb_base}→{new_sb}"
        )
    else:
        reason = (
            f"No change: expectancy {expectancy:.3f}R in acceptable range "
            f"[-0.30, +0.80] or signal volume adequate ({recent_alerts} in 14d)."
        )

    cfg["prob_floor"]    = new_prob
    cfg["sb_base_score"] = new_sb
    if new_prob != prob_floor or new_sb != sb_base:
        cfg["last_updated"] = today
        adj_entry = {
            "date":          today,
            "prob_floor":    f"{prob_floor:.2f}→{new_prob:.2f}",
            "sb_base_score": f"{sb_base}→{new_sb}",
            "expectancy":    round(expectancy, 3),
            "win_rate":      round(win_rate,   3),
            "recent_n":      len(recent),
            "reason":        reason,
        }
        history = cfg.get("adjustment_history", [])
        history.append(adj_entry)
        cfg["adjustment_history"] = history[-20:]   # keep last 20
        result["adjustment"] = adj_entry

    result["reason"] = reason
    _save_pro_cfg(cfg)
    log.info(f"Adaptive update complete: {reason}")
    return result


def get_per_ticker_adjustments() -> dict[str, int]:
    """
    Per-ticker score adjustments learned from Pro's own closed trade outcomes.
    Identical algorithm to X's performance_adjustments().

    Rolling window: last 20 resolved trades per ticker, min 3 before applying.

    weighted_expectancy = avg_win_R × win_rate − avg_loss_R × 1.3 × (1 − win_rate)

    Expectancy  →  adj
    ≥ +1.0R     →  +2  (strong positive expectancy)
    ≥ +0.2R     →  +1  (positive)
    −0.2..+0.2  →   0  (noise zone)
    ≤ −0.2R     →  -1  (slight negative)
    ≤ −1.0R     →  -2  (deeply negative)

    Cached for 15 minutes — safe to call on every ticker eval without DB spam.
    """
    global _ticker_adj_cache, _ticker_adj_ts
    now = datetime.utcnow()
    if (
        _ticker_adj_ts is not None
        and (now - _ticker_adj_ts).total_seconds() < _TICKER_ADJ_TTL_MINUTES * 60
    ):
        return _ticker_adj_cache

    try:
        all_t    = db.all_trades(limit=2000)
        resolved = [t for t in all_t if t.get("outcome") or t.get("exit_reason")]
        bucket: dict[str, list] = defaultdict(list)
        for t in resolved:
            ticker = t.get("ticker", "")
            if ticker:
                bucket[ticker].append(t)

        adj: dict[str, int] = {}
        for ticker, trades in bucket.items():
            recent = trades[-20:]
            if len(recent) < 3:
                continue
            win_rate, avg_win_R, avg_loss_R, expectancy = _expectancy_stats(recent)
            if   expectancy >= 1.0:  adj[ticker] = +2
            elif expectancy >= 0.2:  adj[ticker] = +1
            elif expectancy <= -1.0: adj[ticker] = -2
            elif expectancy <= -0.2: adj[ticker] = -1
            else:                    adj[ticker] =  0

        _ticker_adj_cache = adj
        _ticker_adj_ts    = now
        if adj:
            log.debug(f"Per-ticker adjustments refreshed: {len(adj)} tickers")
    except Exception as e:
        log.warning(f"Per-ticker adjustment fetch failed (non-fatal): {e}")

    return _ticker_adj_cache


def get_pro_thresholds() -> dict:
    """
    Return the current effective thresholds for Pro.
    Used by the dashboard to display current gate values.
    """
    cfg = _load_pro_cfg()
    return {
        "prob_floor":    float(cfg.get("prob_floor",    0.53)),
        "sb_base_score": int(  cfg.get("sb_base_score", 7)),
        "last_updated":  cfg.get("last_updated"),
        "last_checked":  cfg.get("last_checked"),
        "last_expectancy": cfg.get("last_expectancy"),
        "last_win_rate": cfg.get("last_win_rate"),
        "recent_n":      cfg.get("recent_n", 0),
        "source":        "pro" if _PRO_CFG_PATH.exists() else "x_inherited",
        "adjustment_history": cfg.get("adjustment_history", []),
    }
