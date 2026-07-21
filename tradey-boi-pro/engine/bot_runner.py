"""
Bot runner — orchestrates ContinuousScanner → SignalBridge → Executor.
Runs as a background thread. Scanner runs independently on its own cadence.
"""
from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import config.settings as cfg
import db.database as db
from scanner.market_scanner import ContinuousScanner
from engine.signal_bridge import get_pending_signals
from engine.executor import execute_signal
from engine.position_manager import PositionManager

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient

log = logging.getLogger("BotRunner")


class BotRunner:
    def __init__(self, broker: "IBKRClient"):
        self._broker   = broker
        self._scanner  = ContinuousScanner()
        self._pm       = PositionManager(broker)
        self._thread:  threading.Thread | None = None
        self._stop     = threading.Event()
        self.status    = "STOPPED"
        self.last_trade_cycle: datetime | None = None
        self.scan_log:  list[str] = []

    # ── Public lifecycle ──────────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="BotRunner"
        )
        self._thread.start()
        self._scanner.start()
        self._pm.start()
        self.status = "RUNNING"
        log.info("BotRunner started")

    def stop(self):
        self._stop.set()
        self._scanner.stop()
        self._pm.stop()
        self.status = "STOPPED"
        log.info("BotRunner stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def force_scan(self):
        """Trigger an immediate scanner run (non-blocking)."""
        self._scanner.force_scan()

    # ── Properties exposing scanner state for the dashboard ───────────────────

    @property
    def scanner(self) -> ContinuousScanner:
        return self._scanner

    @property
    def position_manager(self) -> PositionManager:
        return self._pm

    # ── Trade execution loop (runs independently of scanner) ─────────────────

    def _loop(self):
        """
        Check scanner results every 5 minutes and attempt to trade any new
        qualifying signals. The scanner itself runs on its own cadence.
        """
        while not self._stop.is_set():
            try:
                self._trade_cycle()
            except Exception as e:
                log.error(f"Trade cycle error: {e}")
                db.log_error("BotRunner", str(e))
            self._stop.wait(300)   # check every 5 minutes

    def _trade_cycle(self):
        if not self._broker.connected:
            return

        scanner_signals = self._scanner.signals   # thread-safe snapshot
        pending = get_pending_signals(scanner_signals=scanner_signals)
        if not pending:
            return

        self.last_trade_cycle = datetime.utcnow()
        mode = cfg.get("mode") or "PAPER"
        self._log(f"Trade check — {len(pending)} qualifying signal(s)")

        for sig in pending[:3]:    # max 3 new positions per cycle
            result = execute_signal(sig, self._broker)
            src    = sig.get("source", "pro_scanner")
            if result["ok"]:
                self._log(
                    f"[{mode}] TRADED {sig['ticker']} "
                    f"qty={result['qty']:.0f} "
                    f"stop=${result['stop']:.3f} "
                    f"target=${result['target']:.3f} "
                    f"[{src}]"
                )
            else:
                self._log(f"SKIPPED {sig['ticker']}: {result['reason']} [{src}]")

    def _log(self, msg: str):
        ts    = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        log.info(msg)
        self.scan_log.insert(0, entry)
        self.scan_log = self.scan_log[:100]
