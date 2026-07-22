"""
Bot runner — orchestrates TieredMonitor → SignalBridge → Executor.
Runs as a background thread. The TieredMonitor handles its own 3-tier cadence
independently. This loop just checks for actionable signals and executes them.

Only ELITE and STRONG BUY signals from the ranked output are ever executed.
BUY and WATCH tiers are displayed in the dashboard but not traded.
"""
from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import config.settings as cfg
import db.database as db
from scanner.monitor import TieredMonitor
from engine.signal_bridge import get_pending_signals
from engine.executor import execute_signal
from engine.position_manager import PositionManager
from engine.adaptive import adaptive_threshold_update
from engine.ibkr_sync import sync_ibkr_positions

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient

log = logging.getLogger("BotRunner")


class BotRunner:
    def __init__(self, broker: "IBKRClient"):
        self._broker  = broker
        self._monitor = TieredMonitor()
        self._pm      = PositionManager(broker)
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()
        self.status   = "STOPPED"
        self.last_trade_cycle: datetime | None   = None
        self.scan_log: list[str]                 = []
        self._last_adaptive_run: datetime | None = None

    # ── Public lifecycle ──────────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="BotRunner"
        )
        self._thread.start()
        self._monitor.start()
        self._pm.start()
        self.status = "RUNNING"
        log.info("BotRunner started — TieredMonitor active (Tier1=60m, Tier2=15m, Tier3=5m)")

    def stop(self):
        self._stop.set()
        self._monitor.stop()
        self._pm.stop()
        self.status = "STOPPED"
        log.info("BotRunner stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def force_scan(self):
        self._monitor.force_scan()

    # ── Properties exposing monitor state to the dashboard ───────────────────

    @property
    def scanner(self) -> TieredMonitor:
        """Return monitor as 'scanner' for dashboard compatibility."""
        return self._monitor

    @property
    def monitor(self) -> TieredMonitor:
        return self._monitor

    @property
    def position_manager(self) -> PositionManager:
        return self._pm

    # ── Trade execution loop (checks every 5 min for new actionable signals) ──

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._trade_cycle()
                self._maybe_run_adaptive_update()
            except Exception as e:
                log.error(f"Trade cycle error: {e}")
                db.log_error("BotRunner", str(e))
            self._stop.wait(300)

    def _maybe_run_adaptive_update(self):
        """Run adaptive threshold learning once per week (or on first start)."""
        now = datetime.utcnow()
        if (
            self._last_adaptive_run is None
            or (now - self._last_adaptive_run).days >= 7
        ):
            try:
                result = adaptive_threshold_update()
                self._last_adaptive_run = now
                reason = result.get("reason", "")
                self._log(f"[ADAPTIVE LEARNING] {reason}")
            except Exception as e:
                log.warning(f"Adaptive update failed (non-fatal): {e}")

    def _trade_cycle(self):
        if not self._broker.connected:
            return

        # Sync any IBKR positions that aren't in the database yet
        try:
            imported = sync_ibkr_positions(self._broker)
            for ticker in imported:
                self._log(f"[SYNC] Auto-imported IBKR position: {ticker}")
        except Exception as e:
            log.warning(f"IBKR position sync error (non-fatal): {e}")

        # Only pass ELITE + STRONG BUY signals to the bridge
        actionable = self._monitor.actionable_signals

        pending = get_pending_signals(scanner_signals=actionable)
        if not pending:
            return

        self.last_trade_cycle = datetime.utcnow()
        mode = cfg.get("mode") or "PAPER"
        max_new = int(cfg.get("max_positions") or 5)

        # Apply regime size multiplier (reduces sizing in neutral environments)
        regimes    = self._monitor.regimes
        asx_regime = regimes.get("ASX")
        us_regime  = regimes.get("US")

        self._log(
            f"Trade check — {len(pending)} qualifying signal(s)  "
            f"ASX={asx_regime.regime.value if asx_regime else '?'}  "
            f"US={us_regime.regime.value if us_regime else '?'}"
        )

        for sig in pending[:3]:
            # Apply regime size multiplier to each signal
            ticker = sig.get("ticker", "")
            market = "ASX" if ticker.endswith(".AX") else "US"
            regime = regimes.get(market)
            if regime:
                sig = {**sig, "_regime_size_mult": regime.size_multiplier}

            result = execute_signal(sig, self._broker)
            src    = sig.get("source", "pro_scanner")
            tier   = sig.get("tier", "?")
            csc    = sig.get("composite_score", sig.get("score", "?"))

            if result["ok"]:
                self._log(
                    f"[{mode}] TRADED {sig['ticker']}  "
                    f"tier={tier}  score={csc}  "
                    f"qty={result['qty']:.0f}  "
                    f"stop=${result['stop']:.3f}  "
                    f"target=${result['target']:.3f}  "
                    f"[{src}]"
                )
            else:
                self._log(
                    f"SKIPPED {sig['ticker']}: {result['reason']}  "
                    f"[{tier}  {src}]"
                )

    def _log(self, msg: str):
        ts    = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        log.info(msg)
        self.scan_log.insert(0, entry)
        self.scan_log = self.scan_log[:100]
