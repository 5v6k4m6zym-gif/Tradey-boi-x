"""
SQLite database layer for Tradey Boi Pro.
Tables: settings, positions, trades, performance_log, error_log
"""
import sqlite3, json, threading
from pathlib import Path
from datetime import datetime

DB_PATH = Path.home() / "TradeyBoiPro" / "pro.db"
_lock   = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            exchange        TEXT NOT NULL DEFAULT 'ASX',
            entry_price     REAL NOT NULL,
            stop_price      REAL NOT NULL,
            target_price    REAL NOT NULL,
            quantity        REAL NOT NULL,
            entry_date      TEXT NOT NULL,
            max_hold_days   INTEGER NOT NULL DEFAULT 15,
            ibkr_order_id   INTEGER,
            stop_order_id   INTEGER,
            status          TEXT NOT NULL DEFAULT 'OPEN',
            signal_score    REAL,
            signal_prob     REAL,
            atr_pct         REAL,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            exchange        TEXT NOT NULL DEFAULT 'ASX',
            entry_price     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            quantity        REAL NOT NULL,
            entry_date      TEXT NOT NULL,
            exit_date       TEXT NOT NULL,
            outcome         TEXT NOT NULL,
            pnl             REAL NOT NULL,
            pnl_pct         REAL NOT NULL,
            hold_days       INTEGER,
            signal_score    REAL,
            signal_prob     REAL,
            exit_reason     TEXT,
            mode            TEXT NOT NULL DEFAULT 'PAPER'
        );

        CREATE TABLE IF NOT EXISTS performance_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at       TEXT NOT NULL,
            account_value   REAL,
            cash            REAL,
            open_positions  INTEGER,
            daily_pnl       REAL,
            total_pnl       REAL,
            win_rate        REAL,
            profit_factor   REAL,
            mode            TEXT NOT NULL DEFAULT 'PAPER'
        );

        CREATE TABLE IF NOT EXISTS error_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT NOT NULL,
            source      TEXT,
            level       TEXT DEFAULT 'ERROR',
            message     TEXT NOT NULL
        );
        """)


def get_setting(key: str, default=None):
    with _lock, _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]


def set_setting(key: str, value):
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)",
                  (key, json.dumps(value)))


def get_all_settings() -> dict:
    with _lock, _conn() as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]
        return out


def open_positions() -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY entry_date"
        ).fetchall()
        return [dict(r) for r in rows]


def upsert_position(data: dict) -> int:
    cols   = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))
    vals   = list(data.values())
    with _lock, _conn() as c:
        cur = c.execute(
            f"INSERT OR REPLACE INTO positions({cols}) VALUES({placeholders})", vals
        )
        return cur.lastrowid


def close_position(position_id: int, exit_price: float, exit_reason: str):
    with _lock, _conn() as c:
        pos = c.execute("SELECT * FROM positions WHERE id=?",
                        (position_id,)).fetchone()
        if not pos:
            return
        pos = dict(pos)
        pnl     = (exit_price - pos["entry_price"]) * pos["quantity"]
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        entry_dt = datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d")
        hold_days = (datetime.utcnow() - entry_dt).days

        mode = get_setting("mode", "PAPER")

        c.execute("""
            INSERT INTO trades
            (ticker,exchange,entry_price,exit_price,quantity,entry_date,
             exit_date,outcome,pnl,pnl_pct,hold_days,signal_score,
             signal_prob,exit_reason,mode)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pos["ticker"], pos["exchange"], pos["entry_price"], exit_price,
            pos["quantity"], pos["entry_date"],
            datetime.utcnow().strftime("%Y-%m-%d"),
            "WIN" if pnl >= 0 else "LOSS",
            round(pnl, 4), round(pnl_pct, 6), hold_days,
            pos.get("signal_score"), pos.get("signal_prob"),
            exit_reason, mode
        ))
        c.execute("UPDATE positions SET status='CLOSED' WHERE id=?",
                  (position_id,))


def all_trades(limit: int = 500) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM trades ORDER BY exit_date DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def log_performance(data: dict):
    data["logged_at"] = datetime.utcnow().isoformat()
    data.setdefault("mode", get_setting("mode", "PAPER"))
    cols   = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))
    with _lock, _conn() as c:
        c.execute(f"INSERT INTO performance_log({cols}) VALUES({placeholders})",
                  list(data.values()))


def log_error(source: str, message: str, level: str = "ERROR"):
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO error_log(logged_at,source,level,message) VALUES(?,?,?,?)",
            (datetime.utcnow().isoformat(), source, level, message)
        )


def recent_errors(limit: int = 20) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM error_log ORDER BY logged_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
