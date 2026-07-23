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
from dataclasses import field

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
    index_pct_50ema:  Optional[float] = None   # price as % above/below 50 EMA
    index_pct_200ema: Optional[float] = None
    breadth:    Optional[float] = None        # % stocks above 200d MA (if available)
    rsi:        Optional[float] = None
    roc10:      Optional[float] = None        # 10-day rate of change %
    roc50:      Optional[float] = None        # 50-day rate of change %
    bull_points: Optional[int]  = None
    bear_points: Optional[int]  = None
    fetched_at: Optional[datetime] = None
    # Per-factor breakdown: list of {"factor", "bull", "bear", "note"}
    factors:    list = field(default_factory=list)

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


def clear_cache():
    """Force the next get_regime() call to re-fetch from Yahoo Finance."""
    _cache.clear()


def _fetch_regime(market: str) -> RegimeData:
    import yfinance as yf
    import contextlib, io, warnings as _w

    if market == "US":
        # SPY = S&P 500 ETF (index), ^VIX = US volatility, ^NYA = NYSE breadth
        tickers = {"index": "SPY", "vix": "^VIX", "breadth": "^NYA"}
    elif market == "ASX":
        # ^AXJO = ASX 200 index, ^AXVI = ASX 200 VIX, STW.AX = ASX 200 ETF for breadth
        tickers = {"index": "^AXJO", "vix": "^AXVI", "breadth": "STW.AX"}
    else:
        return RegimeData(market=market, regime=Regime.NEUTRAL, confidence=0.5,
                          fetched_at=datetime.utcnow())

    # Download ~1 year of daily data for indicators
    _sink = io.StringIO()
    with contextlib.redirect_stderr(_sink), _w.catch_warnings():
        _w.simplefilter("ignore")
        raw = yf.download(
            list(tickers.values()), period="1y", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker", threads=False,
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
    factors: list[dict] = []   # per-factor breakdown for dashboard display

    def _add(name: str, b: int, br: int, note: str):
        nonlocal bull_points, bear_points
        bull_points += b
        bear_points += br
        factors.append({"factor": name, "bull": b, "bear": br, "note": note})

    # 1. Short-term trend: price vs 50 EMA (max 2 bull / 2 bear)
    if pct_50ema > 2:
        _add("50 EMA trend", 2, 0, f"Price {pct_50ema:+.1f}% above 50 EMA ✅")
    elif pct_50ema > 0:
        _add("50 EMA trend", 1, 0, f"Price {pct_50ema:+.1f}% above 50 EMA (marginal)")
    elif pct_50ema < -2:
        _add("50 EMA trend", 0, 2, f"Price {pct_50ema:+.1f}% below 50 EMA ⚠️")
    else:
        _add("50 EMA trend", 0, 1, f"Price {pct_50ema:+.1f}% below 50 EMA (marginal)")

    # 2. Long-term trend: price vs 200 EMA (max 2 bull / 2 bear)
    if pct_200ema > 0:
        _add("200 EMA trend", 2, 0, f"Price {pct_200ema:+.1f}% above 200 EMA ✅")
    elif pct_200ema < -5:
        _add("200 EMA trend", 0, 2, f"Price {pct_200ema:+.1f}% below 200 EMA ⚠️")
    else:
        _add("200 EMA trend", 0, 1, f"Price {pct_200ema:+.1f}% below 200 EMA (marginal)")

    # 3. EMA alignment: 50 above/below 200 (max 1 bull / 1 bear)
    if ema50 > ema200:
        _add("EMA alignment", 1, 0, f"50 EMA above 200 EMA (golden cross) ✅")
    else:
        _add("EMA alignment", 0, 1, f"50 EMA below 200 EMA (death cross) ⚠️")

    # 4. Volatility: VIX level (max 2 bull / 2 bear)
    if vix_val is not None:
        if vix_val < 18:
            _add("Volatility (VIX)", 2, 0, f"VIX {vix_val:.1f} — low fear ✅")
        elif vix_val < 25:
            _add("Volatility (VIX)", 1, 0, f"VIX {vix_val:.1f} — moderate, acceptable")
        elif vix_val > 35:
            _add("Volatility (VIX)", 0, 2, f"VIX {vix_val:.1f} — high fear ⚠️")
        elif vix_val > 28:
            _add("Volatility (VIX)", 0, 1, f"VIX {vix_val:.1f} — elevated ⚠️")
        else:
            _add("Volatility (VIX)", 0, 0, f"VIX {vix_val:.1f} — neutral")

    # 5. Short-term momentum: 10-day rate of change (max 1 bull / 1 bear)
    if len(index_close) >= 10:
        roc10 = (curr - float(index_close.iloc[-10])) / float(index_close.iloc[-10]) * 100
        if roc10 > 2:
            _add("10d momentum", 1, 0, f"10d ROC {roc10:+.1f}% ✅")
        elif roc10 < -3:
            _add("10d momentum", 0, 1, f"10d ROC {roc10:+.1f}% ⚠️")
        else:
            _add("10d momentum", 0, 0, f"10d ROC {roc10:+.1f}% — neutral")

    # 6. Medium-term momentum: 50-day rate of change (max 1 bull / 1 bear)
    if len(index_close) >= 50:
        roc50 = (curr - float(index_close.iloc[-50])) / float(index_close.iloc[-50]) * 100
        if roc50 > 5:
            _add("50d momentum", 1, 0, f"50d ROC {roc50:+.1f}% ✅")
        elif roc50 < -10:
            _add("50d momentum", 0, 1, f"50d ROC {roc50:+.1f}% ⚠️")
        else:
            _add("50d momentum", 0, 0, f"50d ROC {roc50:+.1f}% — neutral")

    # 7. Market breadth: broad index vs its 200 EMA (max 1 bull / 1 bear)
    if breadth_close is not None and len(breadth_close) >= 200:
        b_curr   = float(breadth_close.iloc[-1])
        b_ema200 = float(breadth_close.ewm(span=200, adjust=False).mean().iloc[-1])
        b_ema50  = float(breadth_close.ewm(span=50, adjust=False).mean().iloc[-1])
        if b_curr > b_ema200 and b_ema50 > b_ema200:
            _add("Market breadth", 1, 0, "Broad index above 200 EMA and trending ✅")
        elif b_curr < b_ema200:
            _add("Market breadth", 0, 1, "Broad index below 200 EMA ⚠️")
        else:
            _add("Market breadth", 0, 0, "Broad index near 200 EMA — neutral")

    # 8. RSI(14): healthy uptrend momentum vs overbought/weak (max 1 bull / 1 bear)
    if len(index_close) >= 15:
        _delta = index_close.diff()
        _gain  = _delta.clip(lower=0).ewm(com=13, adjust=False).mean().iloc[-1]
        _loss  = (-_delta.clip(upper=0)).ewm(com=13, adjust=False).mean().iloc[-1]
        _rsi   = 100.0 if _loss == 0 else 100 - (100 / (1 + _gain / _loss))
        if _rsi > 75:
            _add("RSI momentum", 0, 1, f"RSI {_rsi:.0f} — overbought, pullback risk ⚠️")
        elif _rsi < 45:
            _add("RSI momentum", 0, 1, f"RSI {_rsi:.0f} — weak/deteriorating ⚠️")
        elif 52 <= _rsi <= 72:
            _add("RSI momentum", 1, 0, f"RSI {_rsi:.0f} — healthy bullish range ✅")
        else:
            _add("RSI momentum", 0, 0, f"RSI {_rsi:.0f} — neutral zone")

    # ── Confidence uses a fixed denominator (max possible bull points = 12) ──
    # This prevents the artificial 100% that occurs when every metric fires the
    # same direction (old formula: max/total → 100% whenever bear_points == 0).
    MAX_BULL = 12   # theoretical maximum: 2+2+1+2+1+1+1+1 = 11, rounded to 12 for headroom

    if bull_points >= bear_points * 1.5:
        regime     = Regime.BULL
        confidence = round(min(bull_points / MAX_BULL, 1.0), 2)
    elif bear_points >= bull_points * 1.5:
        regime     = Regime.BEAR
        confidence = round(min(bear_points / MAX_BULL, 1.0), 2)
    else:
        regime     = Regime.NEUTRAL
        confidence = 0.5   # genuinely uncertain — don't inflate

    log.info(
        f"Regime [{market}]: {regime.value}  conf={confidence:.2f}  "
        f"bull={bull_points}  bear={bear_points}  "
        f"50EMA={pct_50ema:+.1f}%  200EMA={pct_200ema:+.1f}%  VIX={vix_val}"
    )

    # ── Compute RSI and ROC for storage (already calculated above for scoring) ──
    _roc10 = _roc50 = _rsi_val = None
    if len(index_close) >= 10:
        _roc10 = round((curr - float(index_close.iloc[-10])) / float(index_close.iloc[-10]) * 100, 2)
    if len(index_close) >= 50:
        _roc50 = round((curr - float(index_close.iloc[-50])) / float(index_close.iloc[-50]) * 100, 2)
    if len(index_close) >= 15:
        _delta = index_close.diff()
        _g = _delta.clip(lower=0).ewm(com=13, adjust=False).mean().iloc[-1]
        _l = (-_delta.clip(upper=0)).ewm(com=13, adjust=False).mean().iloc[-1]
        _rsi_val = round(100.0 if _l == 0 else 100 - (100 / (1 + _g / _l)), 1)

    return RegimeData(
        market           = market,
        regime           = regime,
        confidence       = round(confidence, 2),
        vix              = round(vix_val, 1) if vix_val else None,
        index_pct_50ema  = round(pct_50ema, 2),
        index_pct_200ema = round(pct_200ema, 2),
        rsi              = _rsi_val,
        roc10            = _roc10,
        roc50            = _roc50,
        bull_points      = bull_points,
        bear_points      = bear_points,
        fetched_at       = datetime.utcnow(),
        factors          = factors,
    )


# ── ETF movers ─────────────────────────────────────────────────────────────────

_ETF_MOVERS_CACHE: dict[str, tuple[datetime, list]] = {}
_ETF_MOVERS_TTL = timedelta(minutes=30)

_ASX_ETFS = [
    ("STW.AX", "ASX 200"),
    ("VAS.AX", "All Ords"),
    ("QFN.AX", "Financials"),
    ("OZR.AX", "Resources"),
    ("VAP.AX", "Property"),
    ("VHY.AX", "High Yield"),
    ("NDQ.AX", "NASDAQ (AU)"),
    ("GOLD.AX", "Gold"),
    ("BEAR.AX", "Bear hedge"),
]

_US_ETFS = [
    ("SPY",  "S&P 500"),
    ("QQQ",  "NASDAQ 100"),
    ("IWM",  "Russell 2000"),
    ("XLK",  "Tech"),
    ("XLE",  "Energy"),
    ("XLF",  "Financials"),
    ("XLV",  "Health"),
    ("XLI",  "Industrials"),
    ("GLD",  "Gold"),
    ("TLT",  "Bonds (20yr)"),
]


def get_etf_movers(market: str, n: int = 3) -> list[dict]:
    """
    Return top N ETFs by absolute 1-day % change for the given market.
    Each dict: {ticker, name, change_pct, price, direction ("up"/"down")}
    Cached 30 min.
    """
    market = market.upper()
    cached = _ETF_MOVERS_CACHE.get(market)
    if cached and (datetime.utcnow() - cached[0]) < _ETF_MOVERS_TTL:
        return cached[1][:n]

    etf_list = _ASX_ETFS if market == "ASX" else _US_ETFS
    tickers  = [t for t, _ in etf_list]
    name_map = {t: nm for t, nm in etf_list}

    try:
        import yfinance as yf
        import contextlib, io, warnings as _w
        _sink = io.StringIO()
        with contextlib.redirect_stderr(_sink), _w.catch_warnings():
            _w.simplefilter("ignore")
            raw = yf.download(
                tickers, period="5d", interval="1d",
                auto_adjust=True, progress=False,
                group_by="ticker", threads=False,
            )

        results = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()
                if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.title() if isinstance(c, str) else c for c in df.columns]
                closes = df["Close"].squeeze().dropna()
                if len(closes) < 2:
                    continue
                prev  = float(closes.iloc[-2])
                curr  = float(closes.iloc[-1])
                if prev <= 0:
                    continue
                chg   = (curr - prev) / prev * 100
                results.append({
                    "ticker":     ticker,
                    "name":       name_map.get(ticker, ticker),
                    "change_pct": round(chg, 2),
                    "price":      round(curr, 3),
                    "direction":  "up" if chg >= 0 else "down",
                })
            except Exception:
                continue

        results.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        _ETF_MOVERS_CACHE[market] = (datetime.utcnow(), results)
        return results[:n]

    except Exception as e:
        log.warning(f"get_etf_movers({market}): {e}")
        return []


def regime_summary(rd: RegimeData) -> str:
    icon  = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}[rd.regime.value]
    vix_s = f"  VIX={rd.vix}" if rd.vix else ""
    ema_s = f"  50EMA={rd.index_pct_50ema:+.1f}%" if rd.index_pct_50ema is not None else ""
    return f"{icon} {rd.regime.value}  confidence={rd.confidence:.0%}{ema_s}{vix_s}"
