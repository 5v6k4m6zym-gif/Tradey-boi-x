"""
Persistent ticker-history cache populator — read-only data fetch, NOT part of
signal/execution logic.

Downloads 2y OHLCV history for every ticker in engine.WATCHLIST plus the
KNOWN_WINNERS list (see full_watchlist_backtest.py) and writes each to
tradey-boi-x/.cache/ticker_history/<TICKER>.pkl.

WHY THIS EXISTS: full_watchlist_backtest.py / full_pipeline_live_gating_
validation.py only READ from this cache, they never populate it — the cache
those scripts depend on previously lived under /tmp and was lost whenever the
environment restarted, forcing a full 412-ticker re-download every time
(unreliable/slow in this sandbox). This script populates a cache under the
project directory (tradey-boi-x/.cache/, gitignored, never committed) so it
survives environment restarts.

Resumable: skips any ticker that already has a cached file with >= 150 rows.
Safe to interrupt (foreground timeout, Ctrl-C) and re-run — it will pick up
where it left off. Writes progress to stdout every N tickers.

Run with: python3 tests/populate_ticker_cache.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

import engine
from full_watchlist_backtest import ALL_TICKERS, CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def already_cached(ticker: str) -> bool:
    f = CACHE_DIR / f"{ticker.replace('.', '_')}.pkl"
    if not f.exists() or f.stat().st_size == 0:
        return False
    try:
        import pandas as pd
        df = pd.read_pickle(f)
        return len(df) >= 150
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Stop after fetching this many NEW tickers this run "
                              "(useful for chunking within a foreground timeout).")
    args = parser.parse_args()

    todo = [t for t in ALL_TICKERS if not already_cached(t)]
    print(f"{len(ALL_TICKERS) - len(todo)} / {len(ALL_TICKERS)} tickers already cached.")
    print(f"{len(todo)} remaining to fetch.")

    if args.limit:
        todo = todo[: args.limit]
        print(f"(this run capped at {args.limit} new tickers)")

    fetched, failed = 0, 0
    for i, ticker in enumerate(todo, 1):
        out_file = CACHE_DIR / f"{ticker.replace('.', '_')}.pkl"
        try:
            df = engine.get_data(ticker, period="2y")
            if df is None or df.empty or len(df) < 150:
                out_file.write_bytes(b"")
                failed += 1
            else:
                df.to_pickle(out_file)
                fetched += 1
        except Exception as e:
            print(f"  \u26a0\ufe0f  {ticker}: fetch failed ({e})")
            out_file.write_bytes(b"")
            failed += 1

        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] fetched={fetched} failed={failed} (last: {ticker})", flush=True)

        time.sleep(0.15)  # be gentle on the data provider

    print(f"\nDone this run: {fetched} fetched, {failed} failed/empty.")
    remaining = [t for t in ALL_TICKERS if not already_cached(t)]
    print(f"Total remaining across all tickers: {len(remaining)} / {len(ALL_TICKERS)}")


if __name__ == "__main__":
    main()
