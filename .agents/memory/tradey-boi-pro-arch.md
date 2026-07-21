---
name: Tradey Boi Pro architecture
description: Standalone autonomous trading platform built on top of Tradey Boi X signals. Key design decisions and constraints.
---

## Structure
`tradey-boi-pro/` in the same repo as Tradey Boi X (never touches X files)

```
broker/ibkr_client.py      — ib_insync background thread, thread-safe, sim mode if no ib_insync
config/settings.py         — all config from SQLite, DEFAULTS dict defines everything
db/database.py             — SQLite: positions/trades/settings/performance_log/error_log
engine/risk.py             — position_size(), circuit_breaker_active(), can_open_new_position()
engine/signal_bridge.py    — reads ../tradey-boi-x/signal_log.json, filters by prob/score/recency
engine/executor.py         — signal → bracket order → DB position record
engine/position_manager.py — background thread, yfinance fallback price, checks stop/target/maxhold
engine/bot_runner.py       — main scan-and-trade loop, owns PositionManager
pro_dashboard.py           — Streamlit, 5 tabs: Dashboard/Positions/Performance/Health/Settings
start_pro.py               — launcher (installs deps, starts streamlit on port 8502)
SETUP.md                   — end-user setup guide
```

## Key design decisions

**Why ib_insync in background thread?**  
ib_insync runs its own asyncio event loop. Streamlit runs in the main thread. Background thread + `nest_asyncio` is the correct isolation pattern; all shared state protected by `threading.Lock`.

**Why SQLite, not file/memory?**  
Survives restarts. Position state must persist across crashes — in-memory state was the root cause of dedup bugs in Tradey Boi X scheduler.

**Why read signal_log.json, not import X directly?**  
Zero coupling. X runs on GitHub Actions; Pro runs locally. Reading the JSON file is the correct loose-coupling interface. No risk of importing X code that has side effects.

**Optimal X parameters (baked into defaults):**
- prob_floor=0.53, min_score=7, hold_days=15
- sl_mult: 1.2/1.0/0.8 × ATR (hi/mid/lo vol)
- targets: 12/8/5% (hi/mid/lo vol)
- circuit breaker: 3 consecutive losses → 7-day pause

**IBKR ports:** paper=7497, live=7496

**Simulation mode:** if ib_insync not installed, broker auto-switches to simulation (logs orders, no real execution). All DB recording still works.

**Why:** Tradey Boi Pro must run on user's home PC/VPS, not on Replit, because it needs a local IBKR Gateway connection. The files live in the repo for versioning/download.
