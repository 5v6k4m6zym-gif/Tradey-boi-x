"""
Continuous Learning Database — v4.0 Adaptive Gate Validation.

SQLite store for every resolved trade with full context as required
by spec §7: regime, features, gate values, AI confidence, prediction,
outcome, P&L, MAE, MFE, holding period, reason for success/failure.

All public functions are fire-and-forget safe: any DB error is caught
and printed without propagating to the live scanner.
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "logs" / "trade_learning.db"

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at             TEXT NOT NULL,
    ticker                TEXT NOT NULL,
    signal_date           TEXT,
    entry_price           REAL,
    exit_price            REAL,
    outcome               TEXT,
    actual_pct            REAL,
    pnl_est               REAL,
    hold_days             INTEGER,
    tier                  TEXT,
    score                 REAL,
    prob                  REAL,
    atr_pct               REAL,
    macro_regime          TEXT,
    micro_regime          TEXT,
    weighted_score        REAL,
    prob_floor_at_signal  REAL,
    sb_score_at_signal    INTEGER,
    mae                   REAL,
    mfe                   REAL,
    features_json         TEXT,
    gate_values_json      TEXT,
    reason_success        TEXT,
    reason_failure        TEXT
);
CREATE INDEX IF NOT EXISTS idx_tdb_signal_date  ON trades (signal_date);
CREATE INDEX IF NOT EXISTS idx_tdb_outcome      ON trades (outcome);
CREATE INDEX IF NOT EXISTS idx_tdb_regime       ON trades (macro_regime);
CREATE INDEX IF NOT EXISTS idx_tdb_ticker       ON trades (ticker);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(_DDL)
    return con


def log_trade(
    *,
    ticker:               str,
    signal_date:          str | None   = None,
    entry_price:          float | None = None,
    exit_price:           float | None = None,
    outcome:              str | None   = None,
    actual_pct:           float | None = None,
    pnl_est:              float | None = None,
    hold_days:            int | None   = None,
    tier:                 str | None   = None,
    score:                float | None = None,
    prob:                 float | None = None,
    atr_pct:              float | None = None,
    macro_regime:         str | None   = None,
    micro_regime:         str | None   = None,
    weighted_score:       float | None = None,
    prob_floor_at_signal: float | None = None,
    sb_score_at_signal:   int | None   = None,
    mae:                  float | None = None,
    mfe:                  float | None = None,
    features:             dict | None  = None,
    gate_values:          dict | None  = None,
    reason_success:       str | None   = None,
    reason_failure:       str | None   = None,
) -> bool:
    try:
        con = _conn()
        con.execute(
            """
            INSERT INTO trades (
                logged_at, ticker, signal_date, entry_price, exit_price,
                outcome, actual_pct, pnl_est, hold_days, tier, score, prob,
                atr_pct, macro_regime, micro_regime, weighted_score,
                prob_floor_at_signal, sb_score_at_signal, mae, mfe,
                features_json, gate_values_json, reason_success, reason_failure
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().isoformat(), ticker, signal_date, entry_price, exit_price,
                outcome, actual_pct, pnl_est, hold_days, tier, score, prob,
                atr_pct, macro_regime, micro_regime, weighted_score,
                prob_floor_at_signal, sb_score_at_signal, mae, mfe,
                json.dumps(features)    if features    else None,
                json.dumps(gate_values) if gate_values else None,
                reason_success, reason_failure,
            ),
        )
        con.commit()
        con.close()
        return True
    except Exception as exc:
        print(f"[trade_db] log_trade error: {exc}")
        return False


def log_from_signal_entry(
    entry: dict,
    *,
    prob_floor: float = 0.53,
    sb_base:    int   = 7,
    macro_regime: str | None = None,
) -> bool:
    """Log a resolved signal_log.json entry into the learning database."""
    if entry.get("actual_pct") is None or entry.get("outcome") is None:
        return False

    WIN_OUTCOMES = {"HIT_TARGET", "EXPIRED_WIN", "WIN", "TARGET", "PARTIAL_WIN"}
    out = entry.get("outcome", "")
    is_win = out.upper() in WIN_OUTCOMES

    pct = float(entry["actual_pct"])
    ep  = float(entry.get("entry_price") or 0)
    sp  = entry.get("stop_price")
    if ep > 0 and sp:
        risk_pct = abs(ep - float(sp)) / ep
        pnl_est  = round(pct / risk_pct * 0.02 * 10_000, 2) if risk_pct > 0 else None
    else:
        pnl_est = None

    return log_trade(
        ticker               = entry.get("ticker", ""),
        signal_date          = entry.get("signal_date") or entry.get("date"),
        entry_price          = entry.get("entry_price"),
        exit_price           = entry.get("exit_price"),
        outcome              = out,
        actual_pct           = pct,
        pnl_est              = pnl_est,
        tier                 = entry.get("tier"),
        score                = entry.get("score"),
        prob                 = entry.get("prob"),
        atr_pct              = entry.get("atr_pct"),
        macro_regime         = macro_regime,
        weighted_score       = entry.get("weighted_score"),
        prob_floor_at_signal = prob_floor,
        sb_score_at_signal   = sb_base,
        features             = {k: entry[k] for k in ("target_pct", "pred_days", "atr_pct") if k in entry},
        gate_values          = {"prob_floor": prob_floor, "sb_base_score": sb_base},
        reason_success       = out if is_win else None,
        reason_failure       = out if not is_win else None,
    )


def get_recent_trades(n: int = 100, *, regime: str | None = None) -> list[dict]:
    try:
        con  = _conn()
        args: list[Any] = []
        q    = "SELECT * FROM trades WHERE outcome IS NOT NULL"
        if regime:
            q += " AND macro_regime = ?"
            args.append(regime)
        q += " ORDER BY signal_date DESC, id DESC LIMIT ?"
        args.append(n)
        rows = con.execute(q, args).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[trade_db] get_recent_trades error: {exc}")
        return []


def get_regime_summary() -> dict[str, dict]:
    try:
        con  = _conn()
        rows = con.execute(
            """
            SELECT macro_regime,
                   COUNT(*) as n,
                   SUM(CASE WHEN actual_pct >= 0 THEN 1 ELSE 0 END) as wins,
                   AVG(actual_pct) as avg_pct
            FROM trades
            WHERE outcome IS NOT NULL AND macro_regime IS NOT NULL
            GROUP BY macro_regime
            """
        ).fetchall()
        con.close()
        return {
            r["macro_regime"]: {
                "n":        r["n"],
                "wins":     r["wins"],
                "win_rate": r["wins"] / r["n"] if r["n"] else 0.0,
                "avg_pct":  r["avg_pct"] or 0.0,
            }
            for r in rows
        }
    except Exception as exc:
        print(f"[trade_db] get_regime_summary error: {exc}")
        return {}


def total_count() -> int:
    try:
        con = _conn()
        n   = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        con.close()
        return n
    except Exception:
        return 0
