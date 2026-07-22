"""
Position manager — monitors open positions every N minutes.

Implements the same exit mechanics as backtest/engine.py so live results
match what the backtest predicts:

  1. Min hold days  — stop cannot trigger during the first N calendar days
                      (entry-day spread / gap noise protection)
  2. Break-even stop — once intraday price hits entry+be_trigger_r×1R,
                       slides stop to entry (converts round-trips to scratches)
  3. Trailing stop  — once peak ≥ entry+trail_trigger_r×1R,
                       trails trail_dist_r×1R below rolling peak (locks profit)
  4. Max hold exit  — force-close at market after hold_days
  5. Stop / target  — close via market order when price crosses either level

All BE/trailing stop changes are written to the DB and, if connected,
sent to IBKR via modify_stop_order() so the bracket order reflects reality.
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

        # In-memory caches — keyed by position DB id
        self._peak_cache: dict[int, float] = {}   # rolling peak price

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
        interval = max(int((cfg.get("scan_interval_mins") or 15) * 60), 60)
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

        self.last_run     = datetime.utcnow()
        self.last_actions = actions[-10:]

    # ── Per-position logic ───────────────────────────────────────────────────

    def _check_position(self, pos: dict) -> str | None:
        ticker    = pos["ticker"]
        exchange  = pos["exchange"]
        pos_id    = pos["id"]
        entry     = pos["entry_price"]

        # ── Compute 1R from stored atr_pct (consistent across restarts) ──────
        # Even if the stop has been slid (BE/trail), 1R is always computed from
        # the original ATR so it stays stable throughout the trade.
        atr_pct = float(pos.get("atr_pct") or 2.0)
        atr     = entry * atr_pct / 100
        if atr_pct >= 3.0:
            sl_mult = float(cfg.get("sl_mult_hi")  or 1.2)
        elif atr_pct >= 1.5:
            sl_mult = float(cfg.get("sl_mult_mid") or 1.0)
        else:
            sl_mult = float(cfg.get("sl_mult_lo")  or 0.8)
        one_r = max(sl_mult * atr, entry * 0.005)   # at least 0.5% of entry

        # ── Min hold check — no stop exits in first N days ────────────────────
        min_hold = int(cfg.get("min_hold_days") or 2)
        entry_dt  = datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d")
        days_held = (datetime.utcnow() - entry_dt).days
        past_min_hold = days_held >= min_hold

        # ── Max hold — always applies regardless of min_hold ──────────────────
        max_hold = pos.get("max_hold_days") or int(cfg.get("hold_days") or 15)
        if days_held >= max_hold:
            price = self._get_price(ticker, exchange)
            self._exit(pos_id, price or entry, "MAX_HOLD_EXPIRED")
            self._peak_cache.pop(pos_id, None)
            return f"CLOSED {ticker} — max hold {max_hold}d reached"

        # ── Current price ─────────────────────────────────────────────────────
        price = self._get_price(ticker, exchange)
        if price is None:
            log.warning(f"Could not fetch price for {ticker}")
            return None

        # ── Rolling peak ──────────────────────────────────────────────────────
        peak = self._peak_cache.get(pos_id, entry)
        if price > peak:
            peak = price
            self._peak_cache[pos_id] = peak

        stop   = pos["stop_price"]
        target = pos["target_price"]

        # ── Break-even stop ───────────────────────────────────────────────────
        # When price reaches entry + be_trigger_r × 1R, slide stop to entry.
        # Converts potential losses on round-trips into flat scratches.
        be_r = float(cfg.get("be_trigger_r") or 1.0)
        if price >= entry + be_r * one_r and stop < entry and past_min_hold:
            new_stop = round(entry, 4)
            self._update_stop(pos_id, pos, new_stop)
            stop = new_stop
            log.info(f"{ticker}: BE stop — moved to entry {entry:.4f}")

        # ── Trailing stop ─────────────────────────────────────────────────────
        # Once peak ≥ entry + trail_trigger_r × 1R, trail trail_dist_r × 1R
        # below rolling peak. Locks in profit on extended runners.
        trail_r = float(cfg.get("trail_trigger_r") or 1.5)
        dist_r  = float(cfg.get("trail_dist_r")    or 0.7)
        if peak >= entry + trail_r * one_r:
            trail_stop = round(peak - dist_r * one_r, 4)
            if trail_stop > stop and past_min_hold:
                self._update_stop(pos_id, pos, trail_stop)
                stop = trail_stop
                log.info(f"{ticker}: Trail stop → {trail_stop:.4f} (peak={peak:.4f})")

        # ── Stop hit (with min_hold guard) ────────────────────────────────────
        if price <= stop and past_min_hold:
            self._exit(pos_id, stop, "STOP_HIT")
            self._peak_cache.pop(pos_id, None)
            return f"STOPPED {ticker} @ ${stop:.3f}  (price ${price:.3f})"

        # ── Target hit ────────────────────────────────────────────────────────
        if price >= target:
            self._exit(pos_id, target, "TARGET_HIT")
            self._peak_cache.pop(pos_id, None)
            return f"TARGET {ticker} @ ${target:.3f}  (price ${price:.3f})"

        return None

    # ── Stop modification ────────────────────────────────────────────────────

    def _update_stop(self, pos_id: int, pos: dict, new_stop: float):
        """Write new stop to DB and (if connected) modify the IBKR bracket order."""
        db.update_position_stop(pos_id, new_stop)

        stop_order_id = pos.get("stop_order_id")
        if stop_order_id:
            try:
                self._broker.modify_stop_order(int(stop_order_id), new_stop)
            except Exception as e:
                log.warning(f"Could not modify IBKR stop order {stop_order_id}: {e}")

    # ── Exit ─────────────────────────────────────────────────────────────────

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
        price = self._broker.get_current_price(
            ticker, exchange,
            "AUD" if exchange == "ASX" else "USD"
        )
        if price:
            return price

        try:
            df = yf.download(ticker, period="1d", interval="1m",
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
