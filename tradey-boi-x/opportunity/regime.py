"""
Market Regime Detector — Phase 1
Classifies the ASX market as BULLISH / BEARISH / SIDEWAYS / HIGH_VOL / LOW_VOL.
Controlled by ENABLE_MARKET_REGIME feature flag — returns None when flag is off.
No side effects on engine.py or scanner.py.
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from opportunity.config import ENABLE_MARKET_REGIME

_SYMBOL = "^AXJO"
_PERIOD = "6mo"

REGIMES = ("BULLISH", "BEARISH", "SIDEWAYS", "HIGH_VOL", "LOW_VOL")

_REGIME_EMOJI: dict[str, str] = {
    "BULLISH":  "🟢",
    "BEARISH":  "🔴",
    "SIDEWAYS": "🟡",
    "HIGH_VOL": "⚡",
    "LOW_VOL":  "😴",
}


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index from OHLC data."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0.0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0.0)

    atr_s    = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def detect_regime() -> dict | None:
    """
    Fetch ASX 200 data and classify the current market regime.

    Returns a dict or None if the flag is off or data is unavailable.

    Example return value::

        {
            "regime":         "BULLISH",
            "confidence":     0.78,
            "asx200_ret_20d": 0.034,
            "adx":            28.4,
            "atr_pct":        0.009,
            "atr_pct_rank":   0.42,
            "above_ema50":    True,
            "price":          7840.0,
            "ema50":          7720.0,
        }
    """
    if not ENABLE_MARKET_REGIME:
        return None

    try:
        df = yf.Ticker(_SYMBOL).history(period=_PERIOD)
        if df is None or len(df) < 60:
            return None

        close = df["Close"]

        ema50     = close.ewm(span=50, adjust=False).mean()
        adx_series = _compute_adx(df)

        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"]  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        atr_pct      = float(atr.iloc[-1] / close.iloc[-1])
        atr_hist_pct = (atr / close).dropna()
        atr_pct_rank = float((atr_hist_pct < atr_pct).mean())

        current    = float(close.iloc[-1])
        ema50_now  = float(ema50.iloc[-1])
        above_ema  = current > ema50_now
        adx_val    = float(adx_series.iloc[-1])
        ret_20d    = float(
            (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
        ) if len(close) > 21 else 0.0

        # ── Classification ───────────────────────────────────────────────────
        if atr_pct_rank >= 0.80:
            regime     = "HIGH_VOL"
            confidence = min(0.95, 0.50 + (atr_pct_rank - 0.80) * 2.5)
        elif atr_pct_rank <= 0.20:
            regime     = "LOW_VOL"
            confidence = min(0.95, 0.50 + (0.20 - atr_pct_rank) * 2.5)
        elif above_ema and adx_val > 20 and ret_20d > 0:
            regime     = "BULLISH"
            confidence = min(0.95, 0.50 + adx_val / 100 + ret_20d)
        elif not above_ema and adx_val > 20 and ret_20d < 0:
            regime     = "BEARISH"
            confidence = min(0.95, 0.50 + adx_val / 100 + abs(ret_20d))
        else:
            regime     = "SIDEWAYS"
            confidence = max(0.40, 0.70 - adx_val / 100)

        return {
            "regime":         regime,
            "confidence":     round(confidence, 3),
            "asx200_ret_20d": round(ret_20d, 4),
            "adx":            round(adx_val, 1),
            "atr_pct":        round(atr_pct, 4),
            "atr_pct_rank":   round(atr_pct_rank, 3),
            "above_ema50":    above_ema,
            "price":          round(current, 1),
            "ema50":          round(ema50_now, 1),
        }

    except Exception:
        return None


def regime_label(regime_dict: dict | None) -> str:
    """One-line human-readable label for Discord messages."""
    if not regime_dict:
        return ""
    em   = _REGIME_EMOJI.get(regime_dict["regime"], "")
    conf = round(regime_dict["confidence"] * 100)
    return f"{em} {regime_dict['regime']} ({conf}% confidence)"
