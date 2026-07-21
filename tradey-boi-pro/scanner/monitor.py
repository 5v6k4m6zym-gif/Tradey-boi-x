"""
Tradey Boi Pro — Tiered Continuous Monitor

Three-tier scanning cadence (all independent of Tradey Boi X GH Actions schedule):

  Tier 1: Full universe scan   — every 60 min (all 400-900 tickers)
  Tier 2: Top-50 re-scan       — every 15 min (shortlist from Tier 1)
  Tier 3: Top-20 deep watch    — every  5 min (highest-ranked only, intraday data)

Tier 1 handles broad discovery.
Tier 2 refreshes the shortlist as prices move during the session.
Tier 3 provides near-real-time confidence updates on the best setups.

Regime is refreshed before each Tier 1 scan and applied to all rankings.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

import db.database as db
import config.settings as cfg
from scanner.market_scanner import (
    scan_all, scan_batch, market_is_open, next_open_seconds,
)
from scanner.market_regime  import get_all_regimes, RegimeData, regime_summary
from scanner.ranker          import rank_signals
from scanner.universe        import build_universe

log = logging.getLogger("TieredMonitor")

# Tier intervals (seconds)
TIER1_INTERVAL = 60 * 60   # 60 min
TIER2_INTERVAL = 15 * 60   # 15 min
TIER3_INTERVAL =  5 * 60   #  5 min

TIER2_SIZE = 50
TIER3_SIZE = 20


class TieredMonitor:
    """
    Runs three background scan tiers, each on its own cadence.
    All public properties are thread-safe.
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self._stop      = threading.Event()

        # ── Signal stores ─────────────────────────────────────────────────────
        self._all_signals:   list[dict] = []    # latest full ranked list
        self._tier2_signals: list[dict] = []    # last tier-2 refresh
        self._tier3_signals: list[dict] = []    # last tier-3 deep watch

        # ── State ─────────────────────────────────────────────────────────────
        self._regimes:       dict[str, RegimeData] = {}
        self._universe:      list[str] = []
        self._df_cache:      dict[str, pd.DataFrame] = {}  # ticker → OHLCV

        self._tier1_last:    Optional[datetime] = None
        self._tier2_last:    Optional[datetime] = None
        self._tier3_last:    Optional[datetime] = None

        self._tier1_scanning = False
        self._tier2_scanning = False
        self._tier3_scanning = False

        self._scan_count     = 0
        self._status         = "IDLE"
        self._progress       = (0, 0)

        # ── Threads ───────────────────────────────────────────────────────────
        self._t1: Optional[threading.Thread] = None
        self._t2: Optional[threading.Thread] = None
        self._t3: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if self._t1 and self._t1.is_alive():
            return
        self._stop.clear()
        self._t1 = threading.Thread(target=self._tier1_loop, daemon=True, name="ScanTier1")
        self._t2 = threading.Thread(target=self._tier2_loop, daemon=True, name="ScanTier2")
        self._t3 = threading.Thread(target=self._tier3_loop, daemon=True, name="ScanTier3")
        self._t1.start()
        self._t2.start()
        self._t3.start()
        log.info("TieredMonitor started — Tier1=60m, Tier2=15m, Tier3=5m")

    def stop(self):
        self._stop.set()
        self._status = "STOPPED"
        log.info("TieredMonitor stopped")

    def is_running(self) -> bool:
        return self._t1 is not None and self._t1.is_alive()

    def force_scan(self):
        threading.Thread(target=self._run_tier1, daemon=True, name="ForceScan").start()

    # ── Public properties (thread-safe reads) ──────────────────────────────────

    @property
    def signals(self) -> list[dict]:
        with self._lock:
            return list(self._all_signals)

    @property
    def elite_signals(self) -> list[dict]:
        with self._lock:
            return [s for s in self._all_signals if s.get("tier") == "ELITE"]

    @property
    def strong_buy_signals(self) -> list[dict]:
        with self._lock:
            return [s for s in self._all_signals if s.get("tier") == "STRONG BUY"]

    @property
    def actionable_signals(self) -> list[dict]:
        """ELITE + STRONG BUY — the only ones Pro will execute."""
        with self._lock:
            return [s for s in self._all_signals
                    if s.get("tier") in ("ELITE", "STRONG BUY")]

    @property
    def regimes(self) -> dict[str, RegimeData]:
        with self._lock:
            return dict(self._regimes)

    @property
    def universe_size(self) -> int:
        return len(self._universe)

    @property
    def status(self) -> str:
        return self._status

    @property
    def progress(self) -> tuple[int, int]:
        return self._progress

    @property
    def is_scanning(self) -> bool:
        return self._tier1_scanning or self._tier2_scanning or self._tier3_scanning

    @property
    def last_scans(self) -> dict[str, Optional[datetime]]:
        return {"tier1": self._tier1_last, "tier2": self._tier2_last, "tier3": self._tier3_last}

    @property
    def scan_count(self) -> int:
        return self._scan_count

    # ── Tier 1 — Full universe, 60 min ────────────────────────────────────────

    def _tier1_loop(self):
        self._run_tier1()   # immediate on start
        while not self._stop.is_set():
            self._stop.wait(TIER1_INTERVAL)
            if not self._stop.is_set():
                if market_is_open():
                    self._run_tier1()
                else:
                    wait = min(next_open_seconds(), 1800)
                    self._status = f"Market closed — next open in {wait//60}m"
                    self._stop.wait(wait)

    def _run_tier1(self):
        self._tier1_scanning = True
        self._status         = "Tier 1: refreshing regime + universe…"
        log.info("Tier 1 scan starting")

        # 1. Refresh market regime
        try:
            regimes = get_all_regimes()
            with self._lock:
                self._regimes = regimes
        except Exception as e:
            log.error(f"Regime refresh failed: {e}")
            regimes = self._regimes or {}

        # 2. Build universe
        enabled = cfg.get("enabled_markets") or ["ASX", "US"]
        self._status = "Tier 1: building universe…"
        try:
            universe = build_universe(markets=enabled, apply_liquidity=False)
            self._universe = universe
        except Exception as e:
            log.error(f"Universe build failed: {e}")
            universe = self._universe

        # 3. Scan + rank
        self._status   = f"Tier 1: scanning {len(universe)} tickers…"
        self._progress = (0, len(universe))

        def _prog(done, total, msg=""):
            self._progress = (done, total)
            self._status   = f"Tier 1: {msg}" if msg else f"Tier 1: {done}/{total}"

        try:
            raw_signals, df_cache = scan_all(
                universe, batch_size=50, progress_cb=_prog, return_cache=True
            )
            with self._lock:
                self._df_cache = df_cache

            ranked = rank_signals(raw_signals, df_cache, regimes)

            with self._lock:
                self._all_signals   = ranked
                # Seed tier-2 shortlist
                self._tier2_signals = ranked[:TIER2_SIZE]

            self._scan_count  += 1
            self._tier1_last   = datetime.utcnow()
            elite = len([s for s in ranked if s.get("tier") == "ELITE"])
            sbuy  = len([s for s in ranked if s.get("tier") == "STRONG BUY"])
            self._status = (
                f"Tier 1 done — {len(ranked)} signals  "
                f"ELITE={elite}  STRONG BUY={sbuy}"
            )
            log.info(self._status)
        except Exception as e:
            log.error(f"Tier 1 scan error: {e}")
            db.log_error("TieredMonitor.T1", str(e))
            self._status = f"Tier 1 error: {e}"
        finally:
            self._tier1_scanning = False

    # ── Tier 2 — Top-50 refresh, 15 min ───────────────────────────────────────

    def _tier2_loop(self):
        time.sleep(TIER2_INTERVAL)   # let Tier 1 run first
        while not self._stop.is_set():
            if not self._stop.is_set():
                self._run_tier2()
            self._stop.wait(TIER2_INTERVAL)

    def _run_tier2(self):
        with self._lock:
            t2_tickers = [s["ticker"] for s in self._tier2_signals]
            regimes    = dict(self._regimes)

        if not t2_tickers:
            return

        self._tier2_scanning = True
        log.info(f"Tier 2 re-scan: {len(t2_tickers)} tickers")
        try:
            raw, df_cache = scan_batch(t2_tickers, return_cache=True)
            with self._lock:
                self._df_cache.update(df_cache)
            ranked = rank_signals(raw, {**self._df_cache, **df_cache}, regimes)

            with self._lock:
                # Merge: update scores for tier-2 tickers in the full list
                t2_map = {s["ticker"]: s for s in ranked}
                updated = []
                for s in self._all_signals:
                    updated.append(t2_map.get(s["ticker"], s))
                # Re-sort by composite_score
                updated.sort(key=lambda s: s.get("composite_score", 0), reverse=True)
                for i, s in enumerate(updated):
                    s["rank"] = i + 1
                self._all_signals   = updated
                self._tier3_signals = ranked[:TIER3_SIZE]

            self._tier2_last = datetime.utcnow()
            log.info(f"Tier 2 done — {len(ranked)} refreshed")
        except Exception as e:
            log.error(f"Tier 2 error: {e}")
        finally:
            self._tier2_scanning = False

    # ── Tier 3 — Top-20 deep watch, 5 min ────────────────────────────────────

    def _tier3_loop(self):
        time.sleep(TIER3_INTERVAL)
        while not self._stop.is_set():
            if not self._stop.is_set():
                self._run_tier3()
            self._stop.wait(TIER3_INTERVAL)

    def _run_tier3(self):
        with self._lock:
            t3_tickers = [s["ticker"] for s in self._tier3_signals]
            regimes    = dict(self._regimes)

        if not t3_tickers:
            return

        self._tier3_scanning = True
        log.info(f"Tier 3 deep watch: {len(t3_tickers)} tickers")
        try:
            # Use shorter period for speed; also try 1h intraday for freshest close
            raw, df_cache = scan_batch(t3_tickers, period="30d", return_cache=True)
            with self._lock:
                self._df_cache.update(df_cache)
            ranked = rank_signals(raw, {**self._df_cache, **df_cache}, regimes)

            with self._lock:
                t3_map = {s["ticker"]: s for s in ranked}
                updated = []
                for s in self._all_signals:
                    if s["ticker"] in t3_map:
                        updated.append(t3_map[s["ticker"]])
                    else:
                        updated.append(s)
                updated.sort(key=lambda s: s.get("composite_score", 0), reverse=True)
                for i, s in enumerate(updated):
                    s["rank"] = i + 1
                self._all_signals = updated

            self._tier3_last = datetime.utcnow()
            log.info(f"Tier 3 done — {len(ranked)} deep-watched")
        except Exception as e:
            log.error(f"Tier 3 error: {e}")
        finally:
            self._tier3_scanning = False
