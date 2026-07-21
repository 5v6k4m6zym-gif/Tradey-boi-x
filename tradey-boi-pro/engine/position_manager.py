"""
Position manager — monitors open positions every N minutes and triggers exits.
Runs as a background thread. Checks stops, targets, max hold time.
"""
from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import yfinance as yf
import db.database as db
import config.settings as cfg

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient

log = logging.getLogger("PositionManager")


class PositionManager:
    def __init__(self, broker: "IBKRClient"):
        self._broker  = broker
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()
        self.last_run: datetime | None = None
        self.last_actions: list[str]   = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="PositionManager"
        )
        self._thread.start()
        log.info("PositionManager started")

    def stop(self):
        self._stop.set()
        log.info("PositionManager stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Main loop ────────────────────────────────────────────────────────────

    def _loop(self):
        interval = max(int((cfg.get("scan_interval_mins") or 60) * 60), 60)
        while not self._stop.is_set():
            try:
                self.check_all_positions()
            except Exception as e:
                log.error(f"Position check error: {e}")
                db.log_error("PositionManager", str(e))
            self._stop.wait(interval)

    def check_all_positions(self):
        positions = db.open_positions()
        if not positions:
            self.last_run = datetime.utcnow()
            return

        actions = []
        for pos in positions:
            result = self._check_position(pos)
            if result:
                actions.append(result)

        self.last_run    = datetime.utcnow()
        self.last_actions = actions[-10:]   # keep last 10

    # ── Per-position logic ───────────────────────────────────────────────────

    def _check_position(self, pos: dict) -> str | None:
        ticker    = pos["ticker"]
        exchange  = pos["exchange"]
        pos_id    = pos["id"]

        # ── Max hold check ───────────────────────────────────────────────────
        max_hold  = pos.get("max_hold_days") or cfg.get("hold_days") or 15
        entry_dt  = datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d")
        days_held = (datetime.utcnow() - entry_dt).days
        if days_held >= max_hold:
            price = self._get_price(ticker, exchange)
            self._exit(pos_id, price or pos["entry_price"], "MAX_HOLD_EXPIRED")
            return f"CLOSED {ticker} — max hold {max_hold}d reached"

        # ── Price check ──────────────────────────────────────────────────────
        price = self._get_price(ticker, exchange)
        if price is None:
            log.warning(f"Could not fetch price for {ticker}")
            return None

        stop   = pos["stop_price"]
        target = pos["target_price"]

        if price <= stop:
            self._exit(pos_id, stop, "STOP_HIT")
            return f"STOPPED {ticker} @ ${stop:.3f} (price ${price:.3f})"

        if price >= target:
            self._exit(pos_id, target, "TARGET_HIT")
            return f"TARGET {ticker} @ ${target:.3f} (price ${price:.3f})"

        return None

    def _exit(self, position_id: int, exit_price: float, reason: str):
        pos = next(
            (p for p in db.open_positions() if p["id"] == position_id), None
        )
        if not pos:
            return
        try:
            self._broker.close_position(
                pos["ticker"], pos["exchange"], pos["quantity"],
                "AUD" if pos["exchange"] == "ASX" else "USD"
            )
        except Exception as e:
            log.error(f"Exit order failed for {pos['ticker']}: {e}")
            db.log_error("PositionManager", f"Exit order failed: {e}")

        db.close_position(position_id, exit_price, reason)
        log.info(f"Position closed: {pos['ticker']} @ ${exit_price:.3f} ({reason})")

    # ── Price fetching ───────────────────────────────────────────────────────

    def _get_price(self, ticker: str, exchange: str) -> float | None:
        # Try IBKR first (fastest, real-time)
        price = self._broker.get_current_price(
            ticker, exchange,
            "AUD" if exchange == "ASX" else "USD"
        )
        if price:
            return price

        # Fallback to yfinance (15-min delayed but free)
        try:
            yt = ticker if not ticker.endswith(".AX") else ticker
            df = yf.download(yt, period="1d", interval="1m",
                             progress=False, auto_adjust=True)
            if df.empty:
                return None
            df.columns = [c.title() if isinstance(c, str) else c for c in df.columns]
            close = df["Close"].iloc[-1]
            if hasattr(close, "item"):
                return float(close.item())
            return float(close)
        except Exception:
            return None
