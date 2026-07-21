"""
Bot runner — the main scan-and-trade loop.
Runs as a background thread when the bot is enabled from the dashboard.
"""
from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import config.settings as cfg
import db.database as db
from engine.signal_bridge import get_pending_signals
from engine.executor import execute_signal
from engine.position_manager import PositionManager

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient

log = logging.getLogger("BotRunner")


class BotRunner:
    def __init__(self, broker: "IBKRClient"):
        self._broker   = broker
        self._pm       = PositionManager(broker)
        self._thread:  threading.Thread | None = None
        self._stop     = threading.Event()
        self.status    = "STOPPED"
        self.last_scan: datetime | None = None
        self.scan_log:  list[str]       = []

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="BotRunner"
        )
        self._thread.start()
        self._pm.start()
        self.status = "RUNNING"
        log.info("BotRunner started")

    def stop(self):
        self._stop.set()
        self._pm.stop()
        self.status = "STOPPED"
        log.info("BotRunner stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        interval_secs = max(int((cfg.get("scan_interval_mins") or 60) * 60), 300)
        while not self._stop.is_set():
            try:
                self._scan_cycle()
            except Exception as e:
                log.error(f"Scan cycle error: {e}")
                db.log_error("BotRunner", str(e))
            self._stop.wait(interval_secs)

    def _scan_cycle(self):
        self.last_scan = datetime.utcnow()
        mode = cfg.get("mode") or "PAPER"

        if not self._broker.connected:
            self._log("Broker not connected — skipping scan")
            return

        signals = get_pending_signals(lookback_hours=48)
        if not signals:
            self._log("No pending signals")
            return

        self._log(f"Found {len(signals)} pending signal(s)")

        account_value = self._broker.get_account_value()
        for sig in signals[:3]:    # max 3 new trades per cycle
            result = execute_signal(sig, self._broker)
            if result["ok"]:
                self._log(
                    f"[{mode}] TRADED {sig['ticker']} "
                    f"qty={result['qty']:.0f} "
                    f"stop=${result['stop']:.3f} "
                    f"target=${result['target']:.3f}"
                )
            else:
                self._log(f"SKIPPED {sig['ticker']}: {result['reason']}")

    def _log(self, msg: str):
        ts  = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        log.info(msg)
        self.scan_log.insert(0, entry)
        self.scan_log = self.scan_log[:50]    # keep last 50 entries
