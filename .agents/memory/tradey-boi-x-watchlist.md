---
name: ASX watchlist delisted ticker cleanup
description: How delisted tickers were identified and removed from engine.py WATCHLIST
---

The scanner logs a `"$TICKER.AX: possibly delisted"` warning (from yfinance failing
to return data) for tickers that no longer trade. Before removing tickers from
`WATCHLIST` in `engine.py`, always cross-check the extracted delisted set against
actual `WATCHLIST` membership — a naive `grep -o` without `-h` will pick up file path
prefixes and corrupt the ticker list, producing false positives.

**Why:** An early pass without `-h` polluted the "delisted" set with log file paths,
which made every ticker appear to have zero overlap with the watchlist. Re-running
with `-h` (or explicit `--no-filename`) gave the correct 86-ticker overlap.

**How to apply:** When parsing multi-file grep/log output programmatically, always
verify field extraction on a small sample before trusting bulk results, and
double-check set differences both ways (delisted-not-in-list AND list-not-in-delisted).
