"""
Market Regime Detection for Tradey Boi Pro.

Detects whether each market is in BULL / BEAR / NEUTRAL regime.
Regime affects: position sizing, min score threshold, whether to trade at all.
Caches result for 4 hours to avoid excessive downloads.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("MarketRegime")


class Regime(str, Enum):
    BULL    = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR    = "BEAR"


@dataclass
class RegimeData:
    market:     str           # "ASX" | "US" | "GLOBAL"
    regime:     Regime
    confidence: float         # 0–1
    vix:        Optional[float] = None
    index_pct_50ema: Optional[float] = None   # price as % above/below 50 EMA
    index_pct_200ema: Optional[float] = None
    breadth:    Optional[float] = None        # % stocks above 200d MA (if available)
    fetched_at: Optional[datetime] = None

    @property
    def is_tradeable(self) -> bool:
        return self.regime != Regime.BEAR

    @property
    def size_multiplier(self) -> float:
        """Scalar to apply to position size based on regime."""
        if self.regime == Regime.BULL:
            return 1.0 + min(self.confidence * 0.2, 0.2)  # up to 1.2×
        if self.regime == Regime.NEUTRAL:
            return 0.75
        return 0.0   # no trades in BEAR

    @property
    def min_score_delta(self) -> int:
        """Extra score points required in non-bull environments."""
        if self.regime == Regime.BULL:
            return 0
        if self.regime == Regime.NEUTRAL:
            return 1    # require score ≥ min+1
        return 99       # effectively blocks all trades in BEAR


# ── Cache ──────────────────────────────────────────────────────────────────────

_cache: dict[str, RegimeData] = {}
_CACHE_TTL = timedelta(hours=4)


def _is_stale(rd: RegimeData) -> bool:
    if rd.fetched_at is None:
        return True
    return (datetime.utcnow() - rd.fetched_at) > _CACHE_TTL


# ── Regime detection ───────────────────────────────────────────────────────────

def get_regime(market: str = "US") -> RegimeData:
    """
    Return current regime for the given market.
    market: "US", "ASX", or "GLOBAL"
    Results cached 4 hours.
    """
    market = market.upper()
    cached = _cache.get(market)
    if cached and not _is_stale(cached):
        return cached

    try:
        result = _fetch_regime(market)
    except Exception as e:
        log.error(f"Regime detection failed for {market}: {e}")
        result = RegimeData(
            market=market, regime=Regime.NEUTRAL, confidence=0.5,
            fetched_at=datetime.utcnow()
        )
    _cache[market] = result
    return result


def get_all_regimes() -> dict[str, RegimeData]:
    return {
        "US":     get_regime("US"),
        "ASX":    get_regime("ASX"),
    }


def _fetch_regime(market: str) -> RegimeData:
    import yfinance as yf

    if market == "US":
        tickers = {"index": "SPY", "vix": "^VIX", "breadth": "^NYA"}
    elif market == "ASX":
        tickers = {"index": "STW.AX", "vix": "^VIX", "breadth": "^AXJO"}
    else:
        return RegimeData(market=market, regime=Regime.NEUTRAL, confidence=0.5,
                          fetched_at=datetime.utcnow())

    # Download ~1 year of daily data for indicators
    raw = yf.download(
        list(tickers.values()), period="1y", interval="1d",
        auto_adjust=True, progress=False, group_by="ticker", threads=True
    )

    def _get_series(ticker: str) -> pd.Series | None:
        try:
            if len(tickers) == 1:
                df_s = raw.copy()
            else:
                df_s = raw[ticker].copy()
            # Flatten MultiIndex and normalise to Title Case (handles new yfinance)
            if hasattr(df_s.columns, "levels") and df_s.columns.nlevels > 1:
                df_s.columns = df_s.columns.get_level_values(0)
            df_s.columns = [c.title() if isinstance(c, str) else c for c in df_s.columns]
            return df_s["Close"].squeeze().dropna()
        except Exception:
            return None

    index_close   = _get_series(tickers["index"])
    vix_close     = _get_series(tickers["vix"])
    breadth_close = _get_series(tickers["breadth"])

    if index_close is None or len(index_close) < 60:
        return RegimeData(market=market, regime=Regime.NEUTRAL, confidence=0.5,
                          fetched_at=datetime.utcnow())

    curr        = float(index_close.iloc[-1])
    ema50       = float(index_close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200      = float(index_close.ewm(span=200, adjust=False).mean().iloc[-1])
    pct_50ema   = (curr - ema50)  / ema50  * 100
    pct_200ema  = (curr - ema200) / ema200 * 100

    vix_val = None
    if vix_close is not None and len(vix_close) > 0:
        vix_val = float(vix_close.iloc[-1])

    # ── Regime classification (12 possible points — prevents artificial 100%) ──
    bull_points = 0
    bear_points = 0

    # 1. Short-term trend: price vs 50 EMA (max 2 bull / 2 bear)
    if pct_50ema > 2:    bull_points += 2
    elif pct_50ema > 0:  bull_points += 1
    elif pct_50ema < -2: bear_points += 2
    else:                bear_points += 1

    # 2. Long-term trend: price vs 200 EMA (max 2 bull / 2 bear)
    if pct_200ema > 0:    bull_points += 2
    elif pct_200ema < -5: bear_points += 2
    else:                 bear_points += 1

    # 3. EMA alignment: 50 above/below 200 (max 1 bull / 1 bear)
    if ema50 > ema200:   bull_points += 1
    else:                bear_points += 1

    # 4. Volatility: VIX level (max 2 bull / 2 bear)
    if vix_val is not None:
        if vix_val < 18:   bull_points += 2
        elif vix_val < 25: bull_points += 1
        elif vix_val > 35: bear_points += 2
        elif vix_val > 28: bear_points += 1

    # 5. Short-term momentum: 10-day rate of change (max 1 bull / 1 bear)
    if len(index_close) >= 10:
        roc10 = (curr - float(index_close.iloc[-10])) / float(index_close.iloc[-10]) * 100
        if roc10 > 2:    bull_points += 1
        elif roc10 < -3: bear_points += 1

    # 6. Medium-term momentum: 50-day rate of change (max 1 bull / 1 bear)
    if len(index_close) >= 50:
        roc50 = (curr - float(index_close.iloc[-50])) / float(index_close.iloc[-50]) * 100
        if roc50 > 5:     bull_points += 1
        elif roc50 < -10: bear_points += 1

    # 7. Market breadth: broad index vs its 200 EMA (max 1 bull / 1 bear)
    if breadth_close is not None and len(breadth_close) >= 200:
        b_curr   = float(breadth_close.iloc[-1])
        b_ema200 = float(breadth_close.ewm(span=200, adjust=False).mean().iloc[-1])
        # Also check if breadth is declining (divergence warning)
        b_ema50  = float(breadth_close.ewm(span=50, adjust=False).mean().iloc[-1])
        if b_curr > b_ema200 and b_ema50 > b_ema200:
            bull_points += 1   # broad market healthy and trending
        elif b_curr < b_ema200:
            bear_points += 1   # broad market below long-term average

    # ── Regime classification ─────────────────────────────────────────────────
    total = bull_points + bear_points
    confidence = max(bull_points, bear_points) / total if total > 0 else 0.5

    if bull_points >= bear_points * 1.5:
        regime = Regime.BULL
    elif bear_points >= bull_points * 1.5:
        regime = Regime.BEAR
    else:
        regime = Regime.NEUTRAL

    log.info(
        f"Regime [{market}]: {regime.value}  conf={confidence:.2f}  "
        f"bull={bull_points}  bear={bear_points}  total={total}  "
        f"50EMA={pct_50ema:+.1f}%  200EMA={pct_200ema:+.1f}%  VIX={vix_val}"
    )

    return RegimeData(
        market           = market,
        regime           = regime,
        confidence       = round(confidence, 2),
        vix              = round(vix_val, 1) if vix_val else None,
        index_pct_50ema  = round(pct_50ema, 2),
        index_pct_200ema = round(pct_200ema, 2),
        fetched_at       = datetime.utcnow(),
    )


def regime_summary(rd: RegimeData) -> str:
    icon  = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}[rd.regime.value]
    vix_s = f"  VIX={rd.vix}" if rd.vix else ""
    ema_s = f"  50EMA={rd.index_pct_50ema:+.1f}%" if rd.index_pct_50ema is not None else ""
    return f"{icon} {rd.regime.value}  confidence={rd.confidence:.0%}{ema_s}{vix_s}"
