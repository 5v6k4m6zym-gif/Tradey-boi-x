"""
Execution engine — converts a validated signal into a live/paper bracket order.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import db.database as db
import config.settings as cfg
from engine.risk import position_size, sl_and_target, can_open_new_position

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient

log = logging.getLogger("Executor")


def execute_signal(signal: dict, broker: "IBKRClient") -> dict:
    """
    Evaluate a signal and, if all checks pass, place a bracket order.
    Returns a result dict: {"ok": bool, "reason": str, "position_id": int|None}
    """
    ticker     = signal["ticker"]
    entry      = signal["entry_price"]
    atr_pct    = signal["atr_pct"]
    exchange   = signal.get("exchange", "ASX")
    currency   = signal.get("currency", "AUD")

    # ── Account check ────────────────────────────────────────────────────────
    account_value = broker.get_account_value()
    if account_value <= 0:
        return _fail("Account value is 0 — check IBKR connection")

    ok, reason = can_open_new_position(account_value)
    if not ok:
        return _fail(reason)

    # ── Position sizing ──────────────────────────────────────────────────────
    stop, target = sl_and_target(entry, atr_pct)
    qty = position_size(account_value, entry, stop)
    if qty < 1:
        return _fail(f"Position size < 1 share (account too small for this risk/stop)")

    trade_value = qty * entry
    if trade_value > broker.get_cash() * 0.95:
        return _fail(f"Insufficient cash (need ${trade_value:.0f}, have ${broker.get_cash():.0f})")

    # ── Place order ──────────────────────────────────────────────────────────
    mode = cfg.get("mode") or "PAPER"
    log.info(
        f"[{mode}] Placing bracket: BUY {qty:.0f} {ticker} "
        f"@ ${entry:.3f}  stop=${stop:.3f}  target=${target:.3f}"
    )

    try:
        order_result = broker.place_bracket_order(
            ticker=ticker,
            exchange=exchange,
            quantity=qty,
            entry=entry,
            stop=stop,
            target=target,
            currency=currency,
        )
    except Exception as e:
        db.log_error("Executor", f"Order placement failed for {ticker}: {e}")
        return _fail(f"Order error: {e}")

    # ── Record position ──────────────────────────────────────────────────────
    pos_id = db.upsert_position({
        "ticker":        ticker,
        "exchange":      exchange,
        "entry_price":   entry,
        "stop_price":    stop,
        "target_price":  target,
        "quantity":      qty,
        "entry_date":    datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "max_hold_days": cfg.get("hold_days") or 15,
        "ibkr_order_id": order_result.get("entry_id"),
        "stop_order_id": order_result.get("stop_id"),
        "status":        "OPEN",
        "signal_score":  signal.get("score"),
        "signal_prob":   signal.get("prob"),
        "atr_pct":       atr_pct,
        "notes":         f"mode={mode} simulated={order_result.get('simulated',False)}",
    })

    log.info(f"Position recorded: id={pos_id} {ticker} qty={qty:.0f}")
    return {"ok": True, "reason": "Order placed", "position_id": pos_id,
            "stop": stop, "target": target, "qty": qty}


def manual_close(position_id: int, broker: "IBKRClient") -> dict:
    """Force-close a position at market price."""
    positions = db.open_positions()
    pos = next((p for p in positions if p["id"] == position_id), None)
    if not pos:
        return _fail("Position not found")

    current_price = broker.get_current_price(
        pos["ticker"], pos["exchange"],
        "AUD" if pos["exchange"] == "ASX" else "USD"
    ) or pos["entry_price"]

    try:
        broker.close_position(pos["ticker"], pos["exchange"], pos["quantity"])
    except Exception as e:
        db.log_error("Executor", f"Manual close failed for {pos['ticker']}: {e}")
        return _fail(str(e))

    db.close_position(position_id, current_price, "MANUAL_CLOSE")
    log.info(f"Manual close: {pos['ticker']} @ ${current_price:.3f}")
    return {"ok": True, "reason": "Position closed", "exit_price": current_price}


def _fail(reason: str) -> dict:
    log.warning(f"Signal rejected: {reason}")
    return {"ok": False, "reason": reason, "position_id": None}
