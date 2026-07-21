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
from datetime import datetime
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
    given win probability.  Identical formula to X's expected_value_r().
    """
    if price <= 0 or atr <= 0:
        return -1.0
    atr_pct = atr / price * 100
    if atr_pct >= 3.0:
        base_target_pct, sl_mult = 8.0, 1.2
    elif atr_pct >= 1.5:
        base_target_pct, sl_mult = 5.0, 1.0
    else:
        base_target_pct, sl_mult = 3.0, 0.8
    if breakout:
        base_target_pct *= 1.25
    denom = sl_mult * atr_pct
    if denom <= 0:
        return -1.0
    reward_r = base_target_pct / denom
    return prob * reward_r - (1 - prob) * 1.0


# ── Feature computation (mirrors X's get_data() exactly) ──────────────────────

def _compute_x_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute all 15 FEATURES from a raw OHLCV DataFrame.
    Uses min_periods=1 on long rolling windows so shorter histories still
    produce a result for the last row (the model was trained with full history
    but degrades gracefully with partial data).
    Returns None if any required column is missing or computation fails.
    """
    try:
        df = df.copy()
        df.columns = (
            df.columns.get_level_values(0)
            if hasattr(df.columns, "levels") and df.columns.nlevels > 1
            else df.columns
        )
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
    if df is None or len(df) < 60:
        return None

    try:
        feat_df = _compute_x_features(df)
        if feat_df is None or len(feat_df) < 2:
            return None

        row  = feat_df.iloc[-1]
        prev = feat_df.iloc[-2]

        # ── Guard against NaN in critical indicators ───────────────────────────
        for col in ("ema20", "ema50", "rsi", "macd_diff", "vol_ratio", "atr"):
            if pd.isna(row.get(col)):
                return None

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
        if prob is None:
            rsi_raw = float(row["rsi"])
            vr_raw  = float(row["vol_ratio"])
            prob = min(0.40 + (rsi_raw - 50) / 200 + (vr_raw - 1) * 0.05, 0.82)
            prob = max(prob, 0.35)

        # ── Hard filters (X's decide() gates, unchanged) ──────────────────────
        if float(row["ema20"])     <= float(row["ema50"]):          return None  # uptrend today
        if float(prev["ema20"])    <= float(prev["ema50"]):         return None  # uptrend prior day
        if not pd.isna(row.get("macd_diff")) and float(row["macd_diff"])  <= 0: return None  # MACD bull today
        if not pd.isna(prev.get("macd_diff")) and float(prev["macd_diff"]) <= 0: return None  # MACD bull prior
        rsi = float(row["rsi"])
        if rsi >= 72 or rsi <= 25:                                  return None  # RSI safe zone
        vr = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 0
        if vr < 0.5:                                                return None  # liquidity
        if prob < 0.40:                                             return None  # AI floor

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

        # ── Per-ticker learning adjustment (identical to X's performance_adjustments) ──
        # Pro learns from its OWN live trade outcomes — +2 to -2 based on rolling
        # expectancy of the last 20 resolved trades for this specific ticker.
        # Cache is refreshed every 15 min; safe to call on every ticker eval.
        ticker_adj = _get_ticker_adj(ticker)
        score      = max(0, score + ticker_adj)

        # ── Adaptive thresholds from Pro's own learning (falls back to X's) ───
        acfg        = _load_x_adaptive_cfg()
        prob_floor  = float(acfg.get("prob_floor",    params.get("min_prob",  0.53)))
        sb_base     = int(  acfg.get("sb_base_score", params.get("min_score", 7)))
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

        # ── Tier classification (X's thresholds) ──────────────────────────────
        if   score >= elite_min and prob >= prob_floor and expected_r > 0:
            tier = "ELITE"
        elif score >= sb_min    and prob >= prob_floor and expected_r > 0:
            tier = "STRONG BUY"
        elif score >= 5:
            tier = "BUY"
        else:
            return None   # IGNORE — don't surface to ranker

        # Respect per-params overrides (backtest / analysis mode with lowered gates)
        min_score_override = int(params.get("min_score", 0))
        if min_score_override == 0 and tier == "BUY":
            return None   # live scan: only ELITE / STRONG BUY pass through

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
        if len(tickers) == 1:
            df = raw.dropna(how="all")
            return {tickers[0]: df} if not df.empty else {}
        result = {}
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
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

        # Throttle between batches so the PC stays responsive
        if b_idx < len(batches) - 1:
            time.sleep(3)

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
    log.info(f"scan_all: {len(signals)} signals from {len(tickers)} tickers")

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
            "min_score":    int(cfg.get("min_score")    or 7),
            "min_prob":     float(cfg.get("min_prob")   or 0.53),
            "sl_mult_hi":   float(cfg.get("sl_mult_hi") or 1.2),
            "sl_mult_mid":  float(cfg.get("sl_mult_mid")or 1.0),
            "sl_mult_lo":   float(cfg.get("sl_mult_lo") or 0.8),
            "target_hi":    float(cfg.get("target_hi")  or 12.0),
            "target_mid":   float(cfg.get("target_mid") or 8.0),
            "target_lo":    float(cfg.get("target_lo")  or 5.0),
        }
    except Exception:
        return {
            "min_score": 7, "min_prob": 0.53,
            "sl_mult_hi": 1.2, "sl_mult_mid": 1.0, "sl_mult_lo": 0.8,
            "target_hi": 12.0, "target_mid": 8.0, "target_lo": 5.0,
        }
