"""
IBKR broker client for Tradey Boi Pro.
Uses ib_insync's synchronous API with ib.sleep() to maintain the heartbeat
that keeps the connection alive — prevents the drops caused by asyncio misuse.
"""
from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger("IBKRClient")

try:
    from ib_insync import IB, Stock, MarketOrder, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    log.warning("ib_insync not installed — broker will run in SIMULATION mode")


class IBKRClient:
    """
    Thread-safe wrapper around ib_insync (synchronous API).
    ib.sleep() is used instead of asyncio.sleep() — it properly processes
    TWS heartbeats and keeps the connection alive indefinitely.
    """

    def __init__(self):
        self._ib:       Optional[object] = None
        self._thread:   Optional[threading.Thread] = None
        self._lock      = threading.Lock()
        self._connected = False
        self._error_msg = ""
        self._last_ping: Optional[datetime] = None
        self._host      = "127.0.0.1"
        self._port      = 4002
        self._client_id = 1

        self.account_summary: dict = {}
        self.positions:       list = []
        self.open_orders:     list = []

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self, host: str, port: int, client_id: int = 1) -> bool:
        if not IB_AVAILABLE:
            self._connected = True
            self._error_msg = ""
            self._start_sim_thread()
            log.info("SIMULATION mode — no ib_insync")
            return True

        if self._connected:
            return True

        # Store params for auto-reconnect
        self._host, self._port, self._client_id = host, port, client_id

        # Kill any existing thread cleanly
        if self._thread and self._thread.is_alive():
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._thread.join(timeout=5)

        self._error_msg = ""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="IBKRThread"
        )
        self._thread.start()

        # Wait up to 15 s for connection
        for _ in range(30):
            time.sleep(0.5)
            if self._connected or self._error_msg:
                break
        return self._connected

    def disconnect(self):
        self._connected = False
        try:
            if self._ib:
                self._ib.disconnect()
        except Exception:
            pass

    def _run_loop(self):
        """
        Runs in a background daemon thread.
        Uses ib.sleep() to process TWS messages and send heartbeats —
        this is the correct way to maintain a stable ib_insync connection.
        Auto-reconnects on drop.
        """
        import asyncio
        # ib_insync needs an event loop in this thread even in sync mode
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        host, port, client_id = self._host, self._port, self._client_id
        slow_tick = 0

        while True:
            try:
                self._ib = IB()
                self._ib.connect(host, port, clientId=client_id, timeout=10)
                self._connected = True
                self._error_msg = ""
                log.info(f"Connected to IBKR {host}:{port} (client {client_id})")

                # Subscribe so TWS pushes account data automatically
                self._ib.reqAccountUpdates(True)
                self._ib.sleep(2)              # let TWS push initial data
                self._refresh_sync()           # immediate first refresh
                slow_tick = 0

                # ib.sleep() keeps the event loop alive and sends TWS heartbeats
                while self._ib.isConnected():
                    slow_tick += 1
                    if slow_tick >= 5:          # refresh account every 25s
                        self._refresh_sync()
                        slow_tick = 0
                    self._ib.sleep(5)           # heartbeat-safe sleep

                log.warning("IB connection dropped — reconnecting in 5s…")
                self._connected = False
                self._error_msg = "Reconnecting…"

            except Exception as exc:
                self._connected = False
                self._error_msg = str(exc)
                log.error(f"IBKR error: {exc} — retrying in 5s")

            try:
                self._ib.disconnect()
            except Exception:
                pass
            time.sleep(5)

    def _refresh_sync(self):
        """Refresh account summary, positions, and open orders (synchronous)."""
        try:
            vals = self._ib.accountValues()
            summary = {}
            for v in vals:
                if v.currency in ("AUD", "USD", "BASE") and v.tag in (
                    "NetLiquidation", "TotalCashValue", "BuyingPower",
                    "UnrealizedPnL", "RealizedPnL", "GrossPositionValue"
                ):
                    try:
                        summary[v.tag] = float(v.value)
                    except ValueError:
                        pass
            with self._lock:
                self.account_summary = summary
        except Exception as e:
            log.error(f"Account refresh error: {e}")

        try:
            positions = []
            for p in self._ib.positions():
                try:
                    positions.append({
                        "ticker":   p.contract.symbol,
                        "exchange": p.contract.exchange or "ASX",
                        "quantity": p.position,
                        "avg_cost": p.avgCost,
                    })
                except Exception:
                    pass
            with self._lock:
                self.positions = positions
        except Exception as e:
            log.error(f"Position refresh error: {e}")

        try:
            orders = []
            for t in self._ib.openTrades():
                orders.append({
                    "order_id": t.order.orderId,
                    "ticker":   t.contract.symbol,
                    "action":   t.order.action,
                    "qty":      t.order.totalQuantity,
                    "type":     t.order.orderType,
                    "status":   t.orderStatus.status,
                })
            with self._lock:
                self.open_orders = orders
        except Exception as e:
            log.error(f"Order refresh error: {e}")

        self._last_ping = datetime.utcnow()

    # ── Simulation mode ──────────────────────────────────────────────────────

    def _start_sim_thread(self):
        def _sim():
            while True:
                self.account_summary = {
                    "NetLiquidation": 10_000.0,
                    "TotalCashValue": 10_000.0,
                    "BuyingPower":    10_000.0,
                    "UnrealizedPnL":  0.0,
                }
                self._last_ping = datetime.utcnow()
                time.sleep(30)
        threading.Thread(target=_sim, daemon=True, name="SimThread").start()

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> str:
        return self._error_msg

    def get_account_value(self) -> float:
        return self.account_summary.get("NetLiquidation", 0.0)

    def get_cash(self) -> float:
        return self.account_summary.get("TotalCashValue", 0.0)

    def get_buying_power(self) -> float:
        return self.account_summary.get("BuyingPower", 0.0)

    def place_bracket_order(
        self,
        ticker:   str,
        exchange: str,
        quantity: float,
        entry:    float,
        stop:     float,
        target:   float,
        currency: str = "AUD",
    ) -> dict:
        if not IB_AVAILABLE or not self._ib or not self._connected:
            fake_id = int(time.time())
            log.info(f"[SIM] Bracket: BUY {quantity:.0f} {ticker} "
                     f"@ {entry:.3f}  stop={stop:.3f}  target={target:.3f}")
            return {"entry_id": fake_id, "stop_id": fake_id + 1,
                    "target_id": fake_id + 2, "simulated": True}
        try:
            contract = Stock(ticker, exchange, currency)
            self._ib.qualifyContracts(contract)
            bracket = self._ib.bracketOrder(
                action="BUY",
                quantity=round(quantity),
                limitPrice=round(entry, 3),
                takeProfitPrice=round(target, 3),
                stopLossPrice=round(stop, 3),
            )
            ids = {}
            for order in bracket:
                trade = self._ib.placeOrder(contract, order)
                ids[order.orderType.lower().replace(" ", "_") + "_id"] = trade.order.orderId
            self._ib.sleep(1)
            return ids
        except Exception as exc:
            log.error(f"place_bracket_order failed: {exc}")
            return {"error": str(exc), "simulated": False}

    def close_position(
        self,
        ticker:   str,
        exchange: str,
        quantity: float,
        currency: str = "AUD",
    ) -> dict:
        if not IB_AVAILABLE or not self._ib or not self._connected:
            log.info(f"[SIM] Close: SELL {quantity:.0f} {ticker}")
            return {"order_id": int(time.time()), "simulated": True}
        try:
            contract = Stock(ticker, exchange, currency)
            self._ib.qualifyContracts(contract)
            order = MarketOrder("SELL", round(quantity))
            trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(1)
            return {"order_id": trade.order.orderId, "simulated": False}
        except Exception as exc:
            log.error(f"close_position failed: {exc}")
            return {"error": str(exc), "simulated": False}

    def get_current_price(
        self,
        ticker:   str,
        exchange: str,
        currency: str = "AUD",
    ) -> Optional[float]:
        if not IB_AVAILABLE or not self._ib or not self._connected:
            return None
        try:
            contract = Stock(ticker, exchange, currency)
            self._ib.qualifyContracts(contract)
            ticker_obj = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(2)
            price = ticker_obj.last or ticker_obj.close
            self._ib.cancelMktData(contract)
            return float(price) if price and price == price else None
        except Exception:
            return None
