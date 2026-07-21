"""
IBKR broker client for Tradey Boi Pro.
Runs ib_insync in a background thread with its own asyncio event loop.
All public methods are thread-safe and can be called from Streamlit.
"""
from __future__ import annotations

import threading
import asyncio
import time
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger("IBKRClient")

try:
    from ib_insync import IB, Stock, Order, MarketOrder, LimitOrder, StopOrder, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    log.warning("ib_insync not installed — broker will run in SIMULATION mode")


class IBKRClient:
    """
    Thread-safe wrapper around ib_insync.
    Call connect() to start; the background thread handles the IB event loop.
    """

    def __init__(self):
        self._ib:          Optional[object] = None
        self._thread:      Optional[threading.Thread] = None
        self._loop:        Optional[asyncio.AbstractEventLoop] = None
        self._lock         = threading.Lock()
        self._connected    = False
        self._error_msg    = ""
        self._last_ping    = None

        # Shared state (written by background thread, read by dashboard)
        self.account_summary: dict     = {}
        self.positions:       list     = []
        self.open_orders:     list     = []

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self, host: str, port: int, client_id: int = 1) -> bool:
        if not IB_AVAILABLE:
            self._connected = True
            self._error_msg = ""
            log.info("SIMULATION mode — no ib_insync")
            self._start_sim_thread()
            return True

        if self._connected:
            return True

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(host, port, client_id),
            daemon=True,
            name="IBKRThread"
        )
        self._thread.start()
        # Wait up to 10 s for connection
        for _ in range(20):
            time.sleep(0.5)
            if self._connected or self._error_msg:
                break
        return self._connected

    def disconnect(self):
        if self._ib and IB_AVAILABLE:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_disconnect(), self._loop
                ).result(timeout=5)
            except Exception:
                pass
        self._connected = False

    def _run_loop(self, host: str, port: int, client_id: int):
        # asyncio.run() wraps the coroutine in a proper Task automatically,
        # which satisfies asyncio.timeout() on Python 3.11/3.12+.
        # We grab the running loop reference from inside so disconnect() can use it.
        async def _main():
            self._loop = asyncio.get_running_loop()
            await self._async_main(host, port, client_id)
        asyncio.run(_main())

    async def _async_main(self, host, port, client_id):
        self._ib = IB()
        try:
            await self._ib.connectAsync(host, port, clientId=client_id, timeout=10)
            self._connected = True
            self._error_msg = ""
            log.info(f"Connected to IBKR {host}:{port}")
            await self._poll_loop()
        except Exception as e:
            self._error_msg = str(e)
            self._connected = False
            log.error(f"IBKR connection failed: {e}")

    async def _async_disconnect(self):
        if self._ib:
            self._ib.disconnect()

    async def _poll_loop(self):
        while self._connected:
            try:
                await self._refresh_account()
                await self._refresh_positions()
                await self._refresh_orders()
                self._last_ping = datetime.utcnow()
            except Exception as e:
                log.error(f"Poll error: {e}")
            await asyncio.sleep(15)

    # ── Simulation mode (no IBKR installed) ─────────────────────────────────

    def _start_sim_thread(self):
        def _sim():
            while True:
                self.account_summary = {
                    "NetLiquidation":  10_000.0,
                    "TotalCashValue":  10_000.0,
                    "BuyingPower":     10_000.0,
                    "UnrealizedPnL":   0.0,
                }
                self._last_ping = datetime.utcnow()
                time.sleep(30)
        t = threading.Thread(target=_sim, daemon=True, name="SimThread")
        t.start()

    # ── Account & Positions ─────────────────────────────────────────────────

    async def _refresh_account(self):
        if not self._ib:
            return
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

    async def _refresh_positions(self):
        if not self._ib:
            return
        positions = []
        for p in self._ib.positions():
            try:
                ticker = p.contract.symbol
                positions.append({
                    "ticker":    ticker,
                    "exchange":  p.contract.exchange or "ASX",
                    "quantity":  p.position,
                    "avg_cost":  p.avgCost,
                })
            except Exception:
                pass
        with self._lock:
            self.positions = positions

    async def _refresh_orders(self):
        if not self._ib:
            return
        orders = []
        for t in self._ib.openTrades():
            orders.append({
                "order_id":  t.order.orderId,
                "ticker":    t.contract.symbol,
                "action":    t.order.action,
                "qty":       t.order.totalQuantity,
                "type":      t.order.orderType,
                "status":    t.orderStatus.status,
            })
        with self._lock:
            self.open_orders = orders

    # ── Public API ──────────────────────────────────────────────────────────

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
        ticker:     str,
        exchange:   str,
        quantity:   float,
        entry:      float,
        stop:       float,
        target:     float,
        currency:   str = "AUD",
    ) -> dict:
        """
        Place a limit buy + attached stop-loss + take-profit.
        Returns dict with order_ids.
        In simulation mode, returns fake IDs.
        """
        if not IB_AVAILABLE or not self._ib:
            fake_id = int(time.time())
            log.info(f"[SIM] Bracket order: BUY {quantity:.0f} {ticker} "
                     f"@ {entry:.3f}  stop={stop:.3f}  target={target:.3f}")
            return {"entry_id": fake_id, "stop_id": fake_id + 1,
                    "target_id": fake_id + 2, "simulated": True}

        fut = asyncio.run_coroutine_threadsafe(
            self._async_bracket(ticker, exchange, quantity, entry,
                                stop, target, currency),
            self._loop
        )
        return fut.result(timeout=15)

    async def _async_bracket(self, ticker, exchange, qty, entry,
                              stop_px, target_px, currency):
        contract = Stock(ticker, exchange, currency)
        await self._ib.qualifyContractsAsync(contract)

        bracket = self._ib.bracketOrder(
            action="BUY",
            quantity=round(qty),
            limitPrice=round(entry, 3),
            takeProfitPrice=round(target_px, 3),
            stopLossPrice=round(stop_px, 3),
        )
        ids = {}
        for order in bracket:
            trade = self._ib.placeOrder(contract, order)
            ids[order.orderType.lower().replace(" ", "_") + "_id"] = trade.order.orderId
        await asyncio.sleep(1)
        return ids

    def close_position(self, ticker: str, exchange: str,
                       quantity: float, currency: str = "AUD") -> dict:
        if not IB_AVAILABLE or not self._ib:
            log.info(f"[SIM] Close position: SELL {quantity:.0f} {ticker}")
            return {"order_id": int(time.time()), "simulated": True}

        fut = asyncio.run_coroutine_threadsafe(
            self._async_close(ticker, exchange, quantity, currency),
            self._loop
        )
        return fut.result(timeout=15)

    async def _async_close(self, ticker, exchange, qty, currency):
        contract = Stock(ticker, exchange, currency)
        await self._ib.qualifyContractsAsync(contract)
        order = MarketOrder("SELL", round(qty))
        trade = self._ib.placeOrder(contract, order)
        await asyncio.sleep(1)
        return {"order_id": trade.order.orderId, "simulated": False}

    def get_current_price(self, ticker: str, exchange: str,
                          currency: str = "AUD") -> Optional[float]:
        if not IB_AVAILABLE or not self._ib:
            return None
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._async_price(ticker, exchange, currency),
                self._loop
            )
            return fut.result(timeout=10)
        except Exception:
            return None

    async def _async_price(self, ticker, exchange, currency):
        contract = Stock(ticker, exchange, currency)
        await self._ib.qualifyContractsAsync(contract)
        ticker_obj = self._ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)
        price = ticker_obj.last or ticker_obj.close
        self._ib.cancelMktData(contract)
        return float(price) if price and price == price else None
