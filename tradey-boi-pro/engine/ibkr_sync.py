"""
IBKR portfolio sync — imports any open IBKR positions that aren't in the database.

Called on every trade cycle so a fresh-database install automatically recovers
any positions that already exist in the broker account.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import db.database as db
import config.settings as cfg
from engine.risk import sl_and_target, position_size

if TYPE_CHECKING:
    from broker.ibkr_client import IBKRClient

log = logging.getLogger("IBKRSync")

_DEFAULT_ATR_PCT = 2.0   # fallback if we can't fetch real ATR


def _fetch_atr_pct(yf_symbol: str) -> float:
    """
    Download 30 days of daily data and return the 14-day ATR as % of price.
    Returns _DEFAULT_ATR_PCT if the fetch fails.
    """
    try:
        import yfinance as yf
        import contextlib, io, warnings as _w
        end   = datetime.utcnow()
        start = end - timedelta(days=40)
        sink  = io.StringIO()
        with contextlib.redirect_stderr(sink), _w.catch_warnings():
            _w.simplefilter("ignore")
            raw = yf.download(
                yf_symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        raw.columns = [str(c).lower() for c in raw.columns]
        if raw.empty or len(raw) < 5:
            return _DEFAULT_ATR_PCT

        high  = raw["high"]
        low   = raw["low"]
        close = raw["close"].shift(1)
        tr    = (high - low).combine(
            (high - close).abs(), max
        ).combine(
            (low  - close).abs(), max
        )
        atr14 = tr.rolling(14).mean().iloc[-1]
        price = float(raw["close"].iloc[-1])
        if price > 0 and atr14 == atr14:   # nan-check
            return round(float(atr14) / price * 100, 4)
    except Exception as e:
        log.debug(f"ATR fetch failed for {yf_symbol}: {e}")
    return _DEFAULT_ATR_PCT


def sync_ibkr_positions(broker: "IBKRClient") -> list[str]:
    """
    Compare broker.positions (live IBKR account) against db.open_positions().
    For any IBKR position that isn't tracked in the database, import it.

    Returns a list of ticker symbols that were newly imported.
    """
    ibkr_positions = getattr(broker, "positions", [])
    if not ibkr_positions:
        return []

    # Build set of tickers already tracked (normalised to uppercase, no suffix)
    tracked = set()
    for p in db.open_positions():
        tracked.add(p["ticker"].upper())

    mode     = cfg.get("mode") or "PAPER"
    hold_days = int(cfg.get("hold_days") or 10)
    imported  = []

    for ibkr_pos in ibkr_positions:
        raw_ticker = str(ibkr_pos.get("ticker", "")).upper()
        exchange   = str(ibkr_pos.get("exchange", "")).upper()
        quantity   = float(ibkr_pos.get("quantity", 0))
        avg_cost   = float(ibkr_pos.get("avg_cost", 0))

        if not raw_ticker or quantity <= 0 or avg_cost <= 0:
            continue

        if raw_ticker in tracked:
            continue

        # Normalise exchange — IBKR sometimes returns "SMART" or blank
        if not exchange or exchange in ("SMART", ""):
            exchange = "ASX" if raw_ticker.endswith(".AX") else "NASDAQ"

        # Build the yfinance symbol (ASX tickers need ".AX" suffix)
        if exchange == "ASX" and not raw_ticker.endswith(".AX"):
            yf_symbol = raw_ticker + ".AX"
        else:
            yf_symbol = raw_ticker

        # Strip ".AX" from the DB ticker (the DB stores bare symbol + exchange separately)
        db_ticker = raw_ticker.replace(".AX", "") if exchange == "ASX" else raw_ticker

        log.info(
            f"[SYNC] Found untracked IBKR position: {db_ticker} "
            f"({exchange})  qty={quantity:.0f}  avg_cost={avg_cost:.3f}"
        )

        # Compute stop/target from ATR
        atr_pct = _fetch_atr_pct(yf_symbol)
        stop, target = sl_and_target(avg_cost, atr_pct)

        db.upsert_position({
            "ticker":        db_ticker,
            "exchange":      exchange,
            "entry_price":   avg_cost,
            "stop_price":    stop,
            "target_price":  target,
            "quantity":      quantity,
            "entry_date":    datetime.utcnow().strftime("%Y-%m-%d"),
            "max_hold_days": hold_days,
            "status":        "OPEN",
            "atr_pct":       atr_pct,
            "notes":         f"auto-imported from IBKR  mode={mode}",
        })

        tracked.add(raw_ticker)
        imported.append(db_ticker)
        log.info(
            f"[SYNC] Imported {db_ticker}: stop={stop:.3f}  target={target:.3f}  "
            f"atr={atr_pct:.2f}%"
        )

    return imported
