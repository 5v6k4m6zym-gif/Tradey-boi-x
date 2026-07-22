"""
Tradey Boi Pro — Core Scanner

Uses Tradey Boi X's REAL trained EnsembleModel (XGBoost + RandomForest, 60/40 blend)
and the identical 15-feature pipeline, hard filters, scoring rules, and adaptive
thresholds from X's engine.py.  The only difference from X is execution instead of
Discord alerts.

Feature pipeline (matches X's get_data() exactly):
  rsi, macd_diff, bb_width, atr, ret_5, ret_10, ret_20, ret_63,
  vol_ratio, breakout, obv_ratio, adx, mfi, bb_squeeze, gap_up

Hard filters (X's decide() gates, unchanged):
  EMA20 > EMA50 today and prior day, MACD diff > 0 today and prior day,
  RSI 25–72, vol_ratio ≥ 0.5, AI prob ≥ 0.40

Scoring (X's rules):
  +3/2/1 AI prob tiers, +3 52-week breakout, +2 vol surge >1.5×,
  +2/1 RSI sweet-spot, +1 EMA uptrend

Tier gates (X's adaptive thresholds):
  ELITE: score ≥ elite_min AND prob ≥ prob_floor AND expected_R > 0
  STRONG BUY: score ≥ sb_min AND prob ≥ prob_floor AND expected_R > 0
"""
from __future__ import annotations

import json
import logging
import pathlib
import pickle
import sys
import threading
import time
from datetime import datetime, timedelta, date as _date
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import ta

log = logging.getLogger("MarketScanner")

# deferred import — avoids circular import during scanner module load
def _get_ticker_adj(ticker: str) -> int:
    try:
        from engine.adaptive import get_per_ticker_adjustments
        return get_per_ticker_adjustments().get(ticker, 0)
    except Exception:
        return 0

_ASX_TZ = pytz.timezone("Australia/Sydney")
_US_TZ  = pytz.timezone("America/New_York")

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCANNER_DIR   = pathlib.Path(__file__).parent
_PRO_DIR       = _SCANNER_DIR.parent

# Model search order: bundled inside Pro's data/ → dev env X cache
_BUNDLED_MODEL = _PRO_DIR / "data" / "model.pkl"
_X_DIR         = _PRO_DIR.parent / "tradey-boi-x"
_X_MODEL_PATH  = _X_DIR / ".cache" / "backtest_checkpoint" / "model.pkl"
_X_ADAPTIVE    = _X_DIR / "config" / "adaptive_thresholds.json"

# X's 15-feature list — must match engine.py exactly
FEATURES = [
    "rsi", "macd_diff", "bb_width", "atr",
    "ret_5", "ret_10", "ret_20", "ret_63",
    "vol_ratio", "breakout", "obv_ratio",
    "adx", "mfi", "bb_squeeze", "gap_up",
]

# ── Earnings date cache ────────────────────────────────────────────────────────
# Fetches upcoming earnings dates per ticker via yfinance and caches 4 hours.
# Used to block entries within N days of earnings — prevents stop-bypass gaps.

_earnings_cache: dict[str, tuple[datetime, list]] = {}
_EARNINGS_TTL = timedelta(hours=4)


def _earnings_within_days(ticker: str, days: int = 5) -> bool:
    """
    Return True if the ticker has a known earnings date within the next `days`
    calendar days from today (UTC).  Caches per-ticker for 4 hours.
    Fails OPEN — returns False if data is unavailable so the trade is not blocked.
    Never called in backtest mode (yfinance only returns future earnings dates,
    not the historical ones needed to correctly simulate past trade entry dates).
    """
    if days <= 0:
        return False

    now   = datetime.utcnow()
    today = now.date()
    cutoff = today + timedelta(days=days)

    cached = _earnings_cache.get(ticker)
    if cached:
        fetched_at, dates = cached
        if (now - fetched_at) < _EARNINGS_TTL:
            return any(today <= d <= cutoff for d in dates)

    try:
        t   = yf.Ticker(ticker)
        cal = t.get_earnings_dates(limit=6)
        dates: list[_date] = []
        if cal is not None and not cal.empty:
            for idx in cal.index:
                try:
                    d = idx.date() if hasattr(idx, "date") else _date.fromisoformat(str(idx)[:10])
                    dates.append(d)
                except Exception:
                    pass
        _earnings_cache[ticker] = (now, dates)
        log.debug(f"Earnings dates for {ticker}: {dates[:4]}")
        return any(today <= d <= cutoff for d in dates)
    except Exception as _e:
        log.debug(f"Earnings fetch failed for {ticker} (non-fatal): {_e}")
        _earnings_cache[ticker] = (now, [])   # cache the miss to avoid hammering yfinance
        return False   # fail open — don't block the trade if we can't get data


# ── Market hours ───────────────────────────────────────────────────────────────

def _is_asx_open() -> bool:
    now = datetime.now(_ASX_TZ)
    return now.weekday() < 5 and 10 <= now.hour < 16


def _is_us_open() -> bool:
    now = datetime.now(_US_TZ)
    return now.weekday() < 5 and (
        (now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16)
    )


def market_is_open() -> bool:
    return _is_asx_open() or _is_us_open()


# ── Higher-timeframe confirmation cache (30-min TTL) ──────────────────────────
_htf_cache: dict[str, tuple[float, tuple]] = {}   # ticker → (timestamp, result)
_HTF_TTL   = 1800   # seconds


def _weekly_trend_ok(df: pd.DataFrame) -> bool:
    """
    Weekly EMA20 > EMA50 — computed by resampling the existing daily df.
    No extra download. Fails open (returns True) if data is insufficient.
    """
    try:
        idx = pd.to_datetime(df.index)
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)   # strip tz — resample requires tz-naive
        close = pd.Series(df["Close"].squeeze().values, index=idx)
        weekly = close.resample("W").last().dropna()
        if len(weekly) < 50:
            return True
        ema20 = float(weekly.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(weekly.ewm(span=50, adjust=False).mean().iloc[-1])
        return ema20 > ema50
    except Exception:
        return True


def _vwap_mtf_check(ticker: str) -> tuple[int, bool]:
    """
    Downloads 5-day 1h data (cached 30 min) and returns:
      vwap_score : +2 cross-above on volume surge, +1 above VWAP, -1 below VWAP, 0 unavailable
      mtf_ok     : True when 1h EMA20 > EMA50 AND 1h MACD > 0 (or data unavailable)
    Fails open on both dimensions when data is missing.
    """
    now = time.time()
    if ticker in _htf_cache:
        ts, cached = _htf_cache[ticker]
        if now - ts < _HTF_TTL:
            return cached

    try:
        df1h = yf.download(ticker, period="5d", interval="1h",
                           progress=False, auto_adjust=True)
        if df1h.empty or len(df1h) < 26:
            result = (0, True)
            _htf_cache[ticker] = (now, result)
            return result

        df1h.columns = [c.title() if isinstance(c, str) else c for c in df1h.columns]
        close = df1h["Close"].squeeze().astype(float)
        vol   = df1h["Volume"].squeeze().astype(float)
        high  = df1h["High"].squeeze().astype(float)
        low   = df1h["Low"].squeeze().astype(float)

        # VWAP (intraday, today only — reset at day boundary)
        typical = (high + low + close) / 3
        vwap = (typical * vol).cumsum() / vol.cumsum()
        last_close = float(close.iloc[-1])
        last_vwap  = float(vwap.iloc[-1])
        prev_close = float(close.iloc[-2])
        prev_vwap  = float(vwap.iloc[-2])
        last_vol   = float(vol.iloc[-1])
        avg_vol    = float(vol.mean())

        crossed_above = prev_close < prev_vwap and last_close > last_vwap
        above         = last_close > last_vwap
        vol_surge     = last_vol > avg_vol * 1.2

        if crossed_above and vol_surge:
            vwap_score = 2
        elif above:
            vwap_score = 1
        else:
            vwap_score = -1

        # Multi-timeframe (1h EMA + MACD)
        ema20_1h  = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50_1h  = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        macd_1h   = float(
            (close.ewm(span=12, adjust=False).mean()
             - close.ewm(span=26, adjust=False).mean()).iloc[-1]
        )
        mtf_ok = bool(ema20_1h > ema50_1h and macd_1h > 0)

        result = (vwap_score, mtf_ok)
    except Exception:
        result = (0, True)   # fail open

    _htf_cache[ticker] = (now, result)
    return result


def _apply_live_filters(signals: list[dict], min_score: int) -> list[dict]:
    """
    Post-filter applied only during live scanning (NOT in the backtest).
    Applies VWAP, multi-timeframe checks, and extended-universe quality gate.

    Quality gate: stocks outside the quality universe (large/liquid mid-cap)
    are only traded when their signal reaches ELITE tier.  This prevents noisy
    signals from speculative small caps from dragging down overall win rate while
    still capturing rare genuine breakouts from the extended universe.
    """
    from scanner.universe import is_quality_ticker

    kept = []
    for sig in signals:
        ticker      = sig["ticker"]
        score       = sig["score"]
        tier        = sig.get("tier", "")

        # Extended universe gate: non-quality stocks only on ELITE signals
        if not is_quality_ticker(ticker) and tier != "ELITE":
            log.debug(f"Extended-universe gate {ticker}: tier={tier} (need ELITE)")
            continue

        vwap_score, mtf_ok = _vwap_mtf_check(ticker)

        # Below VWAP AND 1h misaligned — veto
        if vwap_score < 0 and not mtf_ok:
            log.debug(f"HTF veto {ticker}: below VWAP + 1h misaligned")
            continue

        # Below VWAP alone — require one extra score point
        if vwap_score < 0 and score < min_score + 1:
            log.debug(f"HTF filter {ticker}: below VWAP, score {score} < {min_score+1}")
            continue

        # 1h misaligned alone — require one extra score point
        if not mtf_ok and score < min_score + 1:
            log.debug(f"HTF filter {ticker}: 1h misaligned, score {score} < {min_score+1}")
            continue

        sig = dict(sig)
        sig["vwap_score"] = vwap_score
        sig["mtf_ok"]     = mtf_ok
        kept.append(sig)

    removed = len(signals) - len(kept)
    if removed:
        log.info(f"_apply_live_filters: removed {removed}/{len(signals)} signals via quality gate/VWAP/MTF")
    return kept


def next_open_seconds() -> int:
    now_asx = datetime.now(_ASX_TZ)
    now_us  = datetime.now(_US_TZ)
    secs    = []
    for tz, now, oh, om in [(_ASX_TZ, now_asx, 10, 0), (_US_TZ, now_us, 9, 30)]:
        if now.weekday() < 5:
            op = now.replace(hour=oh, minute=om, second=0, microsecond=0)
            if now < op:
                secs.append(int((op - now).total_seconds()))
    return min(secs) if secs else 3600


# ── X's trained EnsembleModel — loaded once, shared across all scans ──────────

_X_MODEL      = None
_X_MODEL_LOCK = threading.Lock()


def _load_x_model():
    """
    Load the trained XGBoost+RF EnsembleModel (60/40 blend).
    Search order:
      1. Bundled model at data/model.pkl  (standalone PC install)
      2. Dev-env X cache at ../tradey-boi-x/.cache/...  (Replit / dev)
    Thread-safe singleton — loads exactly once per process.
    Falls back to None if unavailable (heuristic prob used instead).
    """
    global _X_MODEL
    if _X_MODEL is not None:
        return _X_MODEL
    with _X_MODEL_LOCK:
        if _X_MODEL is not None:
            return _X_MODEL

        # Register the EnsembleModel class so pickle can deserialise it.
        # Try the bundled stub first; fall back to X's full engine.py.
        _stub_dir = str(_PRO_DIR / "data")
        if _stub_dir not in sys.path:
            sys.path.insert(0, _stub_dir)
        try:
            import engine_stub as engine  # noqa: F401
            import sys as _sys
            _sys.modules.setdefault("engine", engine)
        except Exception:
            pass
        if str(_X_DIR) not in sys.path:
            sys.path.insert(0, str(_X_DIR))
        try:
            import engine as _x_engine  # noqa: F401
        except Exception:
            pass

        # Try bundled model first (standalone install), then dev-env path
        for model_path in (_BUNDLED_MODEL, _X_MODEL_PATH):
            if not model_path.exists():
                continue
            try:
                with open(model_path, "rb") as fh:
                    _X_MODEL = pickle.load(fh)
                log.info(f"EnsembleModel loaded from {model_path.name} — real AI probability active")
                break
            except Exception as exc:
                log.warning(f"Model load failed ({model_path.name}): {exc}")

        if _X_MODEL is None:
            log.warning("No model found — heuristic probability fallback active")
    return _X_MODEL


# ── X's adaptive thresholds (auto-updated by X's weekly learning cycle) ───────

_PRO_ADAPTIVE = _PRO_DIR / "config" / "adaptive_thresholds.json"


def _load_x_adaptive_cfg() -> dict:
    """
    Load adaptive thresholds. Priority:
      1. Pro's own config (written by engine/adaptive.py after live trades)
      2. X's config (inherited until Pro has ≥10 resolved trades of its own)
      3. Hard-coded defaults
    """
    for path in (_PRO_ADAPTIVE, _X_ADAPTIVE):
        try:
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            continue
    return {"prob_floor": 0.53, "sb_base_score": 7}


def _regime_score_thresholds(sb_base: int) -> tuple[int, int]:
    """
    Return (elite_min, sb_min) using X's default neutral-regime offsets.
    Pro doesn't run the full regime classifier, so we use the neutral baseline
    (same as X for weak_bull / sideways market).
    """
    return sb_base + 2, sb_base


# ── X's expected-value R filter (mirrors engine.py:expected_value_r) ──────────

def _expected_value_r(price: float, atr: float, prob: float, breakout: bool) -> float:
    """
    Reject setups where the ATR-implied reward:risk is too thin even at the
    given win probability.  Targets match config/settings.py DEFAULTS exactly
    (15%/10%/7%) so the EV calculation reflects the actual exit parameters.
    """
    if price <= 0 or atr <= 0:
        return -1.0
    atr_pct = atr / price * 100
    if atr_pct >= 3.0:
        base_target_pct, sl_mult = 15.0, 1.2   # was 8.0 — now matches settings
    elif atr_pct >= 1.5:
        base_target_pct, sl_mult = 10.0, 1.0   # was 5.0
    else:
        base_target_pct, sl_mult =  7.0, 0.8   # was 3.0
    if breakout:
        base_target_pct *= 1.10   # breakout bonus (conservative — was 1.25, harder to hit 15%)
    denom = sl_mult * atr_pct
    if denom <= 0:
        return -1.0
    reward_r = base_target_pct / denom
    return prob * reward_r - (1 - prob) * 1.0


# ── Feature computation (mirrors X's get_data() exactly) ──────────────────────

_OHLCV_NAMES = {"open", "high", "low", "close", "volume", "adj close",
                "Open", "High", "Low", "Close", "Volume", "Adj Close"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly normalise yfinance DataFrame columns to Title Case OHLCV names.

    yfinance column structure has changed across versions:
      - Old:      flat Index, Title Case  ('Close', 'Open', ...)
      - Mid:      MultiIndex (Price, Ticker) — level 0 has OHLCV names
      - New 0.2.58+: MultiIndex (Ticker, Price) — level 1 has OHLCV names,
                     OR flat Index with lowercase ('close', 'open', ...)
    """
    df = df.copy()
    if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
        # Pick whichever level contains more OHLCV-style names
        lvl0 = [str(c) for c in df.columns.get_level_values(0)]
        lvl1 = [str(c) for c in df.columns.get_level_values(1)]
        score0 = sum(1 for c in lvl0 if c in _OHLCV_NAMES)
        score1 = sum(1 for c in lvl1 if c in _OHLCV_NAMES)
        df.columns = df.columns.get_level_values(1 if score1 > score0 else 0)
    # Normalise every column label to Title Case string
    df.columns = [str(c).title() for c in df.columns]
    return df


def _compute_x_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute all 15 FEATURES from a raw OHLCV DataFrame.
    Uses min_periods=1 on long rolling windows so shorter histories still
    produce a result for the last row (the model was trained with full history
    but degrades gracefully with partial data).
    Returns None if any required column is missing or computation fails.
    """
    try:
        df = _normalize_columns(df)
        close = df["Close"].squeeze().astype(float)
        high  = df["High"].squeeze().astype(float)
        low   = df["Low"].squeeze().astype(float)
        vol   = df["Volume"].squeeze().astype(float)

        if len(close) < 30:
            return None

        # ── Trend indicators ──────────────────────────────────────────────────
        macd_ind          = ta.trend.MACD(close)
        df["macd"]        = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_diff"]   = macd_ind.macd_diff()
        df["ema20"]       = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df["ema50"]       = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        df["adx"]         = ta.trend.ADXIndicator(high, low, close, window=14).adx()

        # ── Momentum ──────────────────────────────────────────────────────────
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        df["mfi"] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()

        # ── Volatility ────────────────────────────────────────────────────────
        bb_ind         = ta.volatility.BollingerBands(close)
        df["bb_upper"] = bb_ind.bollinger_hband()
        df["bb_lower"] = bb_ind.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / close
        df["atr"]      = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        df["bb_squeeze"] = (
            df["bb_width"] < df["bb_width"].rolling(126, min_periods=20).quantile(0.20)
        ).astype(int)

        # ── Volume features ───────────────────────────────────────────────────
        vol_mean        = vol.rolling(20, min_periods=5).mean().replace(0, float("nan"))
        df["vol_ratio"] = vol / vol_mean
        df["dollar_vol"] = (close * vol).rolling(20, min_periods=5).mean()
        obv             = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        obv_chg         = obv.diff(5)
        df["obv_ratio"] = obv_chg / (obv.rolling(20, min_periods=5).std() + 1e-10)

        # ── Returns ───────────────────────────────────────────────────────────
        df["ret_5"]  = close.pct_change(5)
        df["ret_10"] = close.pct_change(10)
        df["ret_20"] = close.pct_change(20)
        df["ret_63"] = close.pct_change(63)

        # ── Breakout (52-week high) ───────────────────────────────────────────
        df["breakout"] = (close >= close.rolling(252, min_periods=60).max() * 0.98).astype(int)

        # ── Gap up ────────────────────────────────────────────────────────────
        if "Open" in df.columns:
            df["gap_up"] = ((df["Open"].squeeze().astype(float) / close.shift(1) - 1) > 0.02).astype(int)
        else:
            df["gap_up"] = 0

        return df

    except Exception as exc:
        log.debug(f"Feature computation failed: {exc}")
        return None


# ── Core signal detection — identical strategy logic to X's decide() ──────────

def _score_signal(df: pd.DataFrame, ticker: str, params: dict) -> Optional[dict]:
    """
    Evaluate one ticker using X's real EnsembleModel + identical hard filters,
    scoring rules, adaptive thresholds, and expected-value gate.
    Returns a signal dict or None if the ticker doesn't qualify.
    """
    # _reasons is a shared mutable dict passed from the backtest engine.
    # Each return None increments the matching counter so the caller can show
    # a breakdown of why 0 trades occurred.
    _reasons: Optional[dict] = params.get("_reasons")

    def _reject(reason: str):
        if _reasons is not None:
            _reasons[reason] = _reasons.get(reason, 0) + 1
        return None

    if df is None or len(df) < 60:
        return _reject("insufficient_history (<60 rows)")

    try:
        feat_df = _compute_x_features(df)
        if feat_df is None or len(feat_df) < 2:
            return _reject("feature_computation_failed")

        row  = feat_df.iloc[-1]
        prev = feat_df.iloc[-2]

        # ── Guard against NaN in critical indicators ───────────────────────────
        for col in ("ema20", "ema50", "rsi", "macd_diff", "vol_ratio", "atr"):
            if pd.isna(row.get(col)):
                return _reject(f"nan_in_{col}")

        # ── AI probability — X's real EnsembleModel ───────────────────────────
        model = _load_x_model()
        if model is not None:
            try:
                feat_row = pd.DataFrame([{f: row.get(f, 0) for f in FEATURES}])
                prob = float(model.predict_proba(feat_row)[0][1])
            except Exception as exc:
                log.debug(f"Model predict failed for {ticker}: {exc}")
                prob = None
        else:
            prob = None

        # Heuristic fallback (only used when model is unavailable)
        # Stocks that reach this point have already passed all hard filters,
        # so the base probability should reflect that pre-qualification.
        if prob is None:
            rsi_raw = float(row["rsi"])
            vr_raw  = float(row["vol_ratio"])
            rsi_component = max(0.0, (rsi_raw - 40) / 120)
            vr_component  = min((max(vr_raw - 0.5, 0)) / 20, 0.15)
            prob = min(0.52 + rsi_component + vr_component, 0.82)
            prob = max(prob, 0.40)

        # ── Hard filters ──────────────────────────────────────────────────────
        if float(row["ema20"])  <= float(row["ema50"]):
            return _reject("ema_downtrend_today")
        if float(prev["ema20"]) <= float(prev["ema50"]):
            return _reject("ema_downtrend_prev_day")
        if not pd.isna(row.get("macd_diff")) and float(row["macd_diff"])  <= 0:
            return _reject("macd_bearish_today")
        if not pd.isna(prev.get("macd_diff")) and float(prev["macd_diff"]) <= 0:
            return _reject("macd_bearish_prev_day")
        rsi = float(row["rsi"])
        if rsi >= 72 or rsi <= 38:
            return _reject(f"rsi_out_of_range ({rsi:.0f})")
        vr = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 0
        if vr < 1.5:
            return _reject("low_volume_ratio (<1.5×)")
        if prob < 0.40:
            return _reject("prob_below_floor")

        # ── Momentum confirmation ──────────────────────────────────────────────
        # Price must be rising today — no signals on down days
        if float(row["Close"]) <= float(prev["Close"]):
            return _reject("price_falling_today")
        # EMA20 itself must be accelerating upward — not just above EMA50
        if float(row["ema20"]) <= float(prev["ema20"]):
            return _reject("ema20_not_rising")
        # 20-day trend: stock must be net positive over the past month.
        # Prevents dead-cat bounce entries where EMA20>EMA50 but the stock
        # is still broadly falling off a peak.
        if len(feat_df) >= 21:
            close_20d = float(feat_df.iloc[-21].get("Close", float("nan")))
            if not pd.isna(close_20d) and close_20d > 0:
                ret_20d = (float(row["Close"]) - close_20d) / close_20d
                if ret_20d < 0:
                    return _reject("20d_downtrend")

        # ── Scoring (X's rules) ───────────────────────────────────────────────
        is_breakout = bool(int(row.get("breakout", 0)))
        score = 0
        if   prob >= 0.80: score += 3
        elif prob >= 0.70: score += 2
        elif prob >= 0.60: score += 1
        if is_breakout:    score += 3
        if vr > 1.5:       score += 2
        if 35 <= rsi <= 65:  score += 2
        elif rsi < 70:       score += 1
        if float(row["ema20"]) > float(row["ema50"]): score += 1   # always True here but mirrors X

        # ── Per-ticker learning adjustment ────────────────────────────────────
        # Skip in backtest mode — applying today's learned adjustments to
        # historical data would be lookahead bias.
        if not params.get("backtest_mode"):
            ticker_adj = _get_ticker_adj(ticker)
            score      = max(0, score + ticker_adj)

        # ── Thresholds: backtest uses slider value; live uses adaptive file ───
        # In live mode the adaptive file is authoritative, BUT it can only raise
        # the bar above the settings floor — it cannot ease below manual settings.
        # This prevents X's cold-start defaults (0.53/7) from silently overriding
        # the tighter gates set in config/settings.py (0.58/8).
        if params.get("backtest_mode"):
            prob_floor = float(params.get("min_prob",  0.50))
            sb_base    = int(  params.get("min_score", 6))
        else:
            acfg            = _load_x_adaptive_cfg()
            settings_prob   = float(params.get("min_prob",  0.58))
            settings_score  = int(  params.get("min_score", 8))
            # Take whichever is tighter — adaptive OR manual settings floor
            prob_floor = max(float(acfg.get("prob_floor",    0.58)), settings_prob)
            sb_base    = max(int(  acfg.get("sb_base_score", 8)),    settings_score)
        elite_min, sb_min = _regime_score_thresholds(sb_base)

        # ── ATR / stop / target ───────────────────────────────────────────────
        curr_price  = float(row["Close"])
        atr         = float(row["atr"]) if not pd.isna(row["atr"]) else curr_price * 0.015
        atr_pct     = atr / curr_price * 100 if curr_price > 0 else 0

        if atr_pct >= 3.0:
            sl_mult = params.get("sl_mult_hi",  1.2);  tp_pct = params.get("target_hi",  12.0)
        elif atr_pct >= 1.5:
            sl_mult = params.get("sl_mult_mid", 1.0);  tp_pct = params.get("target_mid",  8.0)
        else:
            sl_mult = params.get("sl_mult_lo",  0.8);  tp_pct = params.get("target_lo",   5.0)

        stop_price   = max(curr_price - sl_mult * atr, curr_price * 0.88)
        target_price = curr_price * (1 + tp_pct / 100)

        # ── Expected-value gate (X's formula) ─────────────────────────────────
        expected_r = _expected_value_r(curr_price, atr, prob, is_breakout)

        # ── Weekly trend gate (X's hard gate — uses existing daily data) ────────
        if not _weekly_trend_ok(df):
            return _reject("weekly_ema_downtrend")

        # ── Earnings gate (live only — block entries within N days of earnings) ─
        # Earnings announcements can gap a stock ±15-30% overnight, completely
        # bypassing the stop loss.  Skipped in backtest mode because yfinance only
        # returns current/future earnings dates, not accurate historical ones.
        if not params.get("backtest_mode"):
            guard_days = int(params.get("earnings_guard_days", 5))
            if guard_days > 0 and _earnings_within_days(ticker, guard_days):
                return _reject(f"earnings_within_{guard_days}d")

        # ── Tier classification (X's thresholds) ──────────────────────────────
        if   score >= elite_min and prob >= prob_floor and expected_r > 0:
            tier = "ELITE"
        elif score >= sb_min    and prob >= prob_floor and expected_r > 0:
            tier = "STRONG BUY"
        elif score >= 5:
            tier = "BUY"
        else:
            return _reject(f"score_too_low ({score})")

        # BUY tier is display-only — only STRONG BUY and ELITE qualify for execution.
        # In backtest mode the engine's own min_score gate handles filtering,
        # so BUY signals are passed through there for visibility.
        if not params.get("backtest_mode") and tier == "BUY":
            return _reject("live_scan_buy_tier_suppressed")

        exchange = "ASX" if ticker.endswith(".AX") else "SMART"

        return {
            "ticker":         ticker,
            "entry_price":    round(curr_price,   4),
            "stop_price":     round(stop_price,   4),
            "target_price":   round(target_price, 4),
            "atr_pct":        round(atr_pct,      2),
            "score":          score,
            "prob":           round(prob,          3),
            "ai_confidence":  round(prob,          3),
            "rsi":            round(rsi,           1),
            "vol_ratio":      round(vr,            1),
            "breakout":       is_breakout,
            "expected_r":     round(expected_r,    3),
            "signal_date":    datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "exchange":       exchange,
            "currency":       "AUD" if exchange == "ASX" else "USD",
            "tier":           tier,
            "source":         "pro_scanner",
        }

    except Exception as exc:
        log.debug(f"Score error {ticker}: {exc}")
        return None


# ── Batch download ─────────────────────────────────────────────────────────────

def _download_batch(
    tickers: list[str],
    period:  str = "15mo",      # 15 months ≈ 325 trading days → enough for 252-day breakout
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    try:
        raw = yf.download(
            " ".join(tickers), period=period, interval="1d",
            auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        _norm = _normalize_columns

        if len(tickers) == 1:
            df = _norm(raw.dropna(how="all"))
            return {tickers[0]: df} if not df.empty else {}
        result = {}
        for t in tickers:
            try:
                df = _norm(raw[t].dropna(how="all"))
                if not df.empty and len(df) >= 60:
                    result[t] = df
            except (KeyError, TypeError):
                pass
        return result
    except Exception as exc:
        log.error(f"Batch download error: {exc}")
        return {}


# ── Public scan functions ──────────────────────────────────────────────────────

def scan_all(
    tickers:      list[str],
    batch_size:   int = 50,
    progress_cb:  Callable[[int, int, str], None] | None = None,
    return_cache: bool = False,
    params:       dict | None = None,
) -> tuple[list[dict], dict] | list[dict]:
    """
    Full scan of a ticker list. Returns raw signals (pre-ranking).
    If return_cache=True, also returns {ticker: DataFrame} for ranker reuse.
    """
    if params is None:
        params = _default_params()

    _load_x_model()   # warm up model before scan loop starts

    signals:  list[dict]              = []
    df_cache: dict[str, pd.DataFrame] = {}
    batches   = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    n_total   = len(tickers)

    for b_idx, batch in enumerate(batches):
        done = b_idx * batch_size
        if progress_cb:
            progress_cb(done, n_total, f"Batch {b_idx+1}/{len(batches)}")

        data = _download_batch(batch)
        df_cache.update(data)

        for ticker, df in data.items():
            sig = _score_signal(df, ticker, params)
            if sig:
                signals.append(sig)
                log.info(
                    f"  RAW SIGNAL: {ticker}  tier={sig['tier']}  "
                    f"score={sig['score']}  prob={sig['prob']:.2f}  "
                    f"expectedR={sig['expected_r']:+.2f}"
                )

    signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
    log.info(f"scan_all: {len(signals)} raw signals from {len(tickers)} tickers")

    # VWAP + multi-timeframe post-filter (live only — not called from backtest)
    min_score = params.get("min_score", 7)
    signals = _apply_live_filters(signals, min_score)

    log.info(f"scan_all: {len(signals)} signals after HTF filter")
    return (signals, df_cache) if return_cache else signals


def scan_batch(
    tickers:      list[str],
    period:       str  = "15mo",
    return_cache: bool = False,
    params:       dict | None = None,
) -> tuple[list[dict], dict] | list[dict]:
    """
    Fast re-scan of a small ticker subset (Tier 2/3 refresh).
    """
    if params is None:
        params = _default_params()

    data    = _download_batch(tickers, period=period)
    signals = []
    for ticker, df in data.items():
        sig = _score_signal(df, ticker, params)
        if sig:
            signals.append(sig)

    signals.sort(key=lambda s: (s["score"], s["prob"]), reverse=True)
    return (signals, data) if return_cache else signals


def _default_params() -> dict:
    try:
        import config.settings as cfg
        return {
            "min_score":    int(cfg.get("min_score")    or 8),
            "min_prob":     float(cfg.get("min_prob")   or 0.58),
            "sl_mult_hi":   float(cfg.get("sl_mult_hi") or 1.2),
            "sl_mult_mid":  float(cfg.get("sl_mult_mid")or 1.0),
            "sl_mult_lo":   float(cfg.get("sl_mult_lo") or 0.8),
            "target_hi":    float(cfg.get("target_hi")  or 15.0),
            "target_mid":   float(cfg.get("target_mid") or 10.0),
            "target_lo":    float(cfg.get("target_lo")  or 7.0),
        }
    except Exception:
        return {
            "min_score": 8, "min_prob": 0.58,
            "sl_mult_hi": 1.2, "sl_mult_mid": 1.0, "sl_mult_lo": 0.8,
            "target_hi": 15.0, "target_mid": 10.0, "target_lo": 7.0,
        }
