"""
IBKR broker client for Tradey Boi Pro.

Architecture:
  - One background thread runs its own asyncio event loop forever.
  - ib_insync's async API runs on that loop — heartbeats fire automatically
    as long as the loop is running (no manual ib.sleep() juggling needed).
  - All calls from other threads use asyncio.run_coroutine_threadsafe(),
    which schedules work onto the running loop safely without any race.
"""
from __future__ import annotations

import asyncio
import threading
import time
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger("IBKRClient")

try:
    from ib_insync import IB, Stock, MarketOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    log.warning("ib_insync not installed — broker will run in SIMULATION mode")


class IBKRClient:

    def __init__(self):
        self._ib:       Optional[object] = None
        self._loop:     Optional[asyncio.AbstractEventLoop] = None
        self._thread:   Optional[threading.Thread] = None
        self._lock      = threading.Lock()
        self._connected = False
        self._error_msg = ""
        self._last_ping: Optional[datetime] = None
        self._host      = "127.0.0.1"
        self._port      = 4002
        self._client_id = 1
        self._stop_flag = False

        self.account_summary: dict = {}
        self.positions:       list = []
        self.open_orders:     list = []

    # ── Public lifecycle ─────────────────────────────────────────────────────

    def connect(self, host: str, port: int, client_id: int = 1) -> bool:
        if not IB_AVAILABLE:
            self._connected = True
            self._error_msg = ""
            self._start_sim_thread()
            return True

        if self._connected:
            return True

        self._host, self._port, self._client_id = host, port, client_id
        self._stop_flag = False

        # Kill any existing thread
        if self._thread and self._thread.is_alive():
            self._stop_flag = True
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=8)
            self._stop_flag = False

        self._thread = threading.Thread(
            target=self._thread_main, daemon=True, name="IBKRThread"
        )
        self._thread.start()

        # Wait up to 15 s for connection
        for _ in range(30):
            time.sleep(0.5)
            if self._connected or self._error_msg.startswith("Fatal"):
                break
        return self._connected

    def disconnect(self):
        self._stop_flag = True
        self._connected = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Background thread ────────────────────────────────────────────────────

    def _thread_main(self):
        """Creates a dedicated event loop and runs it forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as e:
            log.error(f"IBKRThread fatal: {e}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_main(self):
        """
        Async reconnect loop. The event loop is running the whole time,
        so ib_insync heartbeats fire automatically — no ib.sleep() needed.
        """
        host, port, client_id = self._host, self._port, self._client_id

        while not self._stop_flag:
            self._ib = IB()
            try:
                await self._ib.connectAsync(host, port, clientId=client_id, timeout=10)
                self._connected = True
                self._error_msg = ""
                log.info(f"Connected to IBKR {host}:{port} (client {client_id})")

                # Immediate account data pull
                await self._async_refresh()

                # Stay alive — event loop running keeps heartbeats going
                tick = 0
                while self._ib.isConnected() and not self._stop_flag:
                    await asyncio.sleep(5)
                    tick += 1
                    if tick >= 6:       # refresh every 30 s
                        await self._async_refresh()
                        tick = 0

                if not self._stop_flag:
                    log.warning("IBKR disconnected — reconnecting in 5s…")

            except Exception as exc:
                self._error_msg = str(exc)
                log.error(f"IBKR connect error: {exc}")
            finally:
                self._connected = False
                try:
                    self._ib.disconnect()
                except Exception:
                    pass

            if not self._stop_flag:
                await asyncio.sleep(5)

    # ── Account refresh (runs on the event loop) ─────────────────────────────

    async def _async_refresh(self):
        try:
            TAGS = {
                "NetLiquidation", "TotalCashValue", "BuyingPower",
                "UnrealizedPnL", "RealizedPnL", "GrossPositionValue"
            }

            def _parse(vals):
                out = {}
                for v in vals:
                    if v.tag in TAGS:
                        try:
                            out[v.tag] = float(v.value)
                        except (ValueError, TypeError):
                            pass
                return out

            # First try: cached account values (already pushed by TWS)
            summary = _parse(self._ib.accountValues())

            # Second try: explicit account summary request
            if not summary:
                acct_vals = await self._ib.reqAccountSummaryAsync()
                await asyncio.sleep(1)
                summary = _parse(acct_vals) or _parse(self._ib.accountValues())

            # Third try: managed accounts + per-account update
            if not summary:
                accounts = self._ib.managedAccounts()
                if accounts:
                    self._ib.reqAccountUpdates(True, accounts[0])
                    await asyncio.sleep(2)
                    summary = _parse(self._ib.accountValues())

            with self._lock:
                self.account_summary = summary
            if summary:
                log.info(f"Account: ${summary.get('NetLiquidation', 0):,.0f}")
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

    # ── Thread-safe dispatch to the event loop ────────────────────────────────

    def _run_on_loop(self, coro, timeout: float = 15.0):
        """Schedule a coroutine on the IBKRThread event loop and wait for result."""
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("Event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── Simulation mode ───────────────────────────────────────────────────────

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

    # ── Public API (safe to call from any thread) ─────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> str:
        return self._error_msg

    def refresh_account_summary(self) -> None:
        """Re-fetch account values from IBKR (NetLiq, Cash, BuyingPower, P&L).
        Call before displaying metrics so values reflect current positions."""
        if not IB_AVAILABLE or not self._connected:
            return
        try:
            self._run_on_loop(self._async_refresh(), timeout=10)
        except Exception as exc:
            log.warning(f"refresh_account_summary: {exc}")

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
        if not IB_AVAILABLE or not self._connected:
            fake_id = int(time.time())
            log.info(f"[SIM] Bracket: BUY {quantity:.0f} {ticker} "
                     f"@ {entry:.3f}  stop={stop:.3f}  target={target:.3f}")
            return {"entry_id": fake_id, "stop_id": fake_id + 1,
                    "target_id": fake_id + 2, "simulated": True}

        async def _place():
            contract = Stock(ticker, exchange, currency)
            await self._ib.qualifyContractsAsync(contract)
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
            await asyncio.sleep(1)
            return ids

        try:
            return self._run_on_loop(_place(), timeout=20)
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
        if not IB_AVAILABLE or not self._connected:
            log.info(f"[SIM] Close: SELL {quantity:.0f} {ticker}")
            return {"order_id": int(time.time()), "simulated": True}

        async def _close():
            contract = Stock(ticker, exchange, currency)
            await self._ib.qualifyContractsAsync(contract)
            order = MarketOrder("SELL", round(quantity))
            trade = self._ib.placeOrder(contract, order)
            await asyncio.sleep(1)
            return {"order_id": trade.order.orderId, "simulated": False}

        try:
            return self._run_on_loop(_close(), timeout=20)
        except Exception as exc:
            log.error(f"close_position failed: {exc}")
            return {"error": str(exc), "simulated": False}

    def get_current_price(
        self,
        ticker:   str,
        exchange: str,
        currency: str = "AUD",
    ) -> Optional[float]:
        if not IB_AVAILABLE or not self._connected:
            return None

        async def _price():
            import math as _math
            contract = Stock(ticker, exchange, currency)
            await self._ib.qualifyContractsAsync(contract)
            ticker_obj = self._ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(2)
            self._ib.cancelMktData(contract)
            # NaN is truthy in Python — must check explicitly, not use `or`
            def _valid(v):
                try:
                    return v is not None and not _math.isnan(float(v)) and float(v) > 0
                except (TypeError, ValueError):
                    return False
            for candidate in (ticker_obj.last, ticker_obj.close, ticker_obj.bid, ticker_obj.ask):
                if _valid(candidate):
                    return float(candidate)
            return None

        try:
            return self._run_on_loop(_price(), timeout=10)
        except Exception:
            return None
