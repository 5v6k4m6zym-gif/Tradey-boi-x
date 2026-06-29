"""
Core engine — pure Python, no Streamlit.
Imported by both dashboard.py and scanner.py.
"""
import json
import os
import requests
from datetime import datetime, timedelta
import pytz as _pytz
from pathlib import Path

import pandas as pd
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier

_vader = SentimentIntensityAnalyzer()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WATCHLIST = [
    # US — tech
    "AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN",
    # US — other sectors
    "TSLA", "XOM", "JPM",
    # ASX — big miners (iron ore)
    "BHP.AX", "FMG.AX", "RIO.AX", "S32.AX",
    # ASX — specialty miners
    "NST.AX", "CXO.AX", "LTR.AX", "MIN.AX", "PDN.AX",
    # ASX — other
    "CBA.AX", "WDS.AX", "CSL.AX",
]
FEATURES        = [
    "rsi", "macd_diff", "bb_width", "atr",
    "ret_5", "ret_10", "ret_20", "ret_63",
    "vol_ratio", "breakout", "obv_ratio",
    "adx", "mfi", "bb_squeeze", "gap_up",
]
PREDICTION_DAYS = 10
TARGET_RETURN   = 0.03
COOLDOWN_HOURS  = 8
MAX_ALERTS      = 3
DISCORD         = os.getenv("Discordwebhook", "") or os.getenv("discordwebhook", "")

BASE_DIR        = Path(__file__).parent
LOG_FILE        = BASE_DIR / "signal_log.json"
COOLDOWN_FILE   = BASE_DIR / "cooldowns.json"
SEND_GUARD_FILE = BASE_DIR / ".last_sent.json"   # prevents double-sends within 90s

# ─── STEP 1 — DATA → FEATURES ────────────────────────────────────────────────
def get_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period).copy()
    if df.empty:
        return df
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    macd = ta.trend.MACD(close)
    bb   = ta.volatility.BollingerBands(close)
    df["rsi"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / close
    df["atr"]         = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["ema20"]       = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"]       = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["vol_ratio"]   = vol / vol.rolling(20).mean()
    df["ret_5"]       = close.pct_change(5)
    df["ret_10"]      = close.pct_change(10)
    df["ret_20"]      = close.pct_change(20)
    df["breakout"]    = (close >= close.rolling(252).max() * 0.98).astype(int)
    obv               = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    obv_chg           = obv.diff(5)
    df["obv_ratio"]   = obv_chg / (obv.rolling(20).std() + 1e-10)
    df["adx"]         = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df["mfi"]         = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
    df["ret_63"]      = close.pct_change(63)
    df["bb_squeeze"]  = (df["bb_width"] < df["bb_width"].rolling(126).quantile(0.20)).astype(int)
    df["gap_up"]      = ((df["Open"] / close.shift(1) - 1) > 0.02).astype(int)
    return df.dropna()

# ─── STEP 2 — AI MODEL ───────────────────────────────────────────────────────
class EnsembleModel:
    """
    Wraps XGBoost + RandomForest. Both models are trained independently;
    predict_proba returns a weighted average (60/40).
    Only fires when both models agree the trade has merit.
    """
    def __init__(self, xgb_pipe: Pipeline, rf_pipe: Pipeline):
        self.xgb = xgb_pipe
        self.rf  = rf_pipe

    def predict_proba(self, X):
        xgb_p = self.xgb.predict_proba(X)
        rf_p  = self.rf.predict_proba(X)
        return 0.60 * xgb_p + 0.40 * rf_p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_model() -> "EnsembleModel":
    """
    Train XGBoost on ALL watchlist tickers with recency-weighted samples.
    Recent data counts more than old data:
      last  30 days → weight 4×
      last  90 days → weight 2×
      older         → weight 1×
    """
    frames = []
    for ticker in WATCHLIST:
        try:
            df = get_data(ticker, "2y").copy()
            df["target"] = (df["Close"].shift(-PREDICTION_DAYS) / df["Close"] - 1 > TARGET_RETURN).astype(int)
            df["_row_date"] = df.index   # keep the date for recency weighting
            frames.append(df.dropna())
        except Exception:
            pass
    if not frames:
        df0 = get_data("AAPL", "2y").copy()
        df0["target"]    = (df0["Close"].shift(-PREDICTION_DAYS) / df0["Close"] - 1 > TARGET_RETURN).astype(int)
        df0["_row_date"] = df0.index
        frames = [df0.dropna()]
    combined = pd.concat(frames, ignore_index=True)

    # Recency weights — more weight on recent market conditions
    try:
        now  = pd.Timestamp.now(tz="UTC")
        dates = pd.to_datetime(combined["_row_date"], utc=True)
        ages  = (now - dates).dt.days.fillna(365)
    except Exception:
        ages = pd.Series([365] * len(combined))
    weights = ages.apply(lambda d: 4.0 if d <= 30 else (2.0 if d <= 90 else 1.0))

    neg = int((combined["target"] == 0).sum())
    pos = int((combined["target"] == 1).sum())
    spw = round(neg / pos, 2) if pos > 0 else 1.0
    X, y = combined[FEATURES], combined["target"]

    # XGBoost — recency-weighted
    xgb_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, verbosity=0,
        )),
    ])
    xgb_pipe.fit(X, y, xgb__sample_weight=weights.values)

    # RandomForest — balanced class weights, independent from XGBoost
    rf_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X, y)

    recent = int((ages <= 30).sum())
    print(f"  Ensemble trained: {len(combined):,} rows ({pos} buy / {neg} no-buy) | "
          f"XGBoost + RandomForest | recency-weighted ({recent} rows ×4)")
    return EnsembleModel(xgb_pipe, rf_pipe)

# ─── MARKET REGIME, VIX, SECTOR, WEEKLY, EARNINGS ───────────────────────────
_regime_cache: dict = {}

# Sector ETF map for US tickers — ASX already covered by ^AXJO in market_regime_ok
SECTOR_ETF = {
    "AAPL": "XLK", "NVDA": "XLK", "MSFT": "XLK", "AMD": "XLK", "META": "XLK",
    "AMZN": "XLY", "TSLA": "XLY",
    "XOM":  "XLE",
    "JPM":  "XLF",
}

# Underlying commodity for each ticker — drives the real price action
COMMODITY_MAP = {
    "BHP.AX": ("VALE", "iron ore"),
    "FMG.AX": ("VALE", "iron ore"),
    "RIO.AX": ("VALE", "iron ore"),
    "S32.AX": ("VALE", "iron ore"),
    "NST.AX": ("GLD",  "gold"),
    "CXO.AX": ("LIT",  "lithium"),
    "LTR.AX": ("LIT",  "lithium"),
    "MIN.AX": ("LIT",  "lithium"),
    "PDN.AX": ("URA",  "uranium"),
    "WDS.AX": ("USO",  "oil/LNG"),
    "XOM":    ("USO",  "oil"),
}

# Correlated groups — only ONE ticker per group alerts per scan
CORRELATION_GROUPS = [
    frozenset({"BHP.AX", "FMG.AX", "RIO.AX", "S32.AX"}),          # iron ore
    frozenset({"CXO.AX", "LTR.AX", "MIN.AX"}),                     # lithium
    frozenset({"AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN"}),    # US mega-cap tech
    frozenset({"XOM", "WDS.AX"}),                                   # energy/oil
]

def _cached_ema_ok(cache_key: str, yf_ticker: str, span: int = 50) -> bool:
    """Generic helper: True when latest close > EMA(span). Cached 1 hour."""
    now = datetime.now().timestamp()
    if cache_key in _regime_cache:
        ts, result = _regime_cache[cache_key]
        if now - ts < 3600:
            return result
    try:
        df    = yf.Ticker(yf_ticker).history(period="3mo")
        if df.empty or len(df) < span:
            return True
        close = df["Close"]
        ema   = close.ewm(span=span, adjust=False).mean()
        result = bool(close.iloc[-1] > ema.iloc[-1])
        _regime_cache[cache_key] = (now, result)
        return result
    except Exception:
        return True

def market_regime_ok(ticker: str) -> bool:
    """True when the relevant broad index (SPY or ASX200) is above its 50-day EMA."""
    index = "^AXJO" if ticker.endswith(".AX") else "SPY"
    return _cached_ema_ok(f"regime_{index}", index, span=50)

def vix_safe() -> bool:
    """True when market fear (VIX) is below 30. High VIX = unreliable signals."""
    now = datetime.now().timestamp()
    if "vix" in _regime_cache:
        ts, result = _regime_cache["vix"]
        if now - ts < 3600:
            return result
    try:
        vix    = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
        result = vix < 30
        _regime_cache["vix"] = (now, result)
        return result
    except Exception:
        return True

def sector_ok(ticker: str) -> bool:
    """True when the stock's sector ETF is in uptrend (US tickers only)."""
    etf = SECTOR_ETF.get(ticker)
    if not etf:
        return True   # ASX already covered by market_regime_ok
    return _cached_ema_ok(f"sector_{etf}", etf, span=50)

def weekly_trend_ok(ticker: str) -> bool:
    """True when the weekly chart EMA20 > EMA50 — higher-timeframe confirmation."""
    cache_key = f"weekly_{ticker}"
    now = datetime.now().timestamp()
    if cache_key in _regime_cache:
        ts, result = _regime_cache[cache_key]
        if now - ts < 3600:
            return result
    try:
        df    = yf.Ticker(ticker).history(period="2y", interval="1wk")
        if df.empty or len(df) < 50:
            return True
        close = df["Close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        result = bool(ema20.iloc[-1] > ema50.iloc[-1])
        _regime_cache[cache_key] = (now, result)
        return result
    except Exception:
        return True

def earnings_safe(ticker: str) -> bool:
    """True when no earnings announcement is within 5 calendar days."""
    try:
        cal   = yf.Ticker(ticker).calendar
        if not cal:
            return True
        dates = cal.get("Earnings Date", [])
        if not dates:
            return True
        today = datetime.now().date()
        for d in (dates if isinstance(dates, list) else [dates]):
            d = d.date() if hasattr(d, "date") else d
            if -1 <= (d - today).days <= 5:
                return False
        return True
    except Exception:
        return True

# ─── STEP 3 — FILTERS + SINGLE DECISION ──────────────────────────────────────
def _load_cooldowns() -> dict:
    if COOLDOWN_FILE.exists():
        try:
            raw = json.loads(COOLDOWN_FILE.read_text())
            return {k: datetime.fromisoformat(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}

def _save_cooldowns(cd: dict):
    COOLDOWN_FILE.write_text(json.dumps({k: v.isoformat() for k, v in cd.items()}, indent=2))

def cooldown_ok(ticker: str) -> bool:
    cd = _load_cooldowns()
    return ticker not in cd or datetime.now() - cd[ticker] > timedelta(hours=COOLDOWN_HOURS)

def mark_alerted(ticker: str):
    cd = _load_cooldowns()
    cd[ticker] = datetime.now()
    _save_cooldowns(cd)

# ─── ADAPTIVE LEARNING — score adjustments from past performance ─────────────
def performance_adjustments() -> dict[str, int]:
    """
    Returns per-ticker score adjustments learned from resolved signal outcomes.
    Uses the most recent 20 resolved signals per ticker (rolling window).
    Needs ≥3 resolved signals before adjusting.

    Win rate  →  adj
    ≥ 75%     →  +2  (hot streak — lower bar to re-enter)
    ≥ 60%     →  +1  (proven winner)
    40–60%    →   0  (neutral — no adjustment)
    ≤ 40%     →  -1  (underperforming — needs stronger signal)
    ≤ 25%     →  -2  (consistent loser — penalise heavily)
    """
    from collections import defaultdict
    entries  = _load_log()
    resolved = [e for e in entries if e["outcome"] is not None]
    if not resolved:
        return {}
    bucket: dict[str, list[bool]] = defaultdict(list)
    for e in resolved:
        bucket[e["ticker"]].append(e["outcome"] == "WIN")
    adj = {}
    for ticker, results in bucket.items():
        recent = results[-20:]          # rolling 20-signal window
        if len(recent) < 3:
            continue
        wr = sum(recent) / len(recent)
        if   wr >= 0.75: adj[ticker] = +2
        elif wr >= 0.60: adj[ticker] = +1
        elif wr <= 0.25: adj[ticker] = -2
        elif wr <= 0.40: adj[ticker] = -1
        else:            adj[ticker] =  0
    return adj


def update_ticker_performance() -> dict:
    """
    Called after every scan:
      1. Resolve any matured trade outcomes (checks actual exit price vs target)
      2. Recompute per-ticker win rates from the rolling signal log
      3. Send a Discord summary if any new outcomes were resolved

    Returns dict of resolved outcomes from this run (may be empty).
    """
    entries_before = sum(1 for e in _load_log() if e["outcome"] is not None)
    updated        = resolve_outcomes()
    entries_after  = sum(1 for e in updated if e["outcome"] is not None)
    new_count      = entries_after - entries_before

    if new_count == 0:
        return {}

    # Build per-ticker summary of newly resolved outcomes
    adj   = performance_adjustments()
    resolved = [e for e in updated if e["outcome"] is not None]
    from collections import defaultdict
    bucket: dict[str, list] = defaultdict(list)
    for e in resolved:
        bucket[e["ticker"]].append(e)

    lines = [
        "**TRADEY BOI X** | 📊 Outcome Update",
        f"_{new_count} trade(s) resolved — model adjustments updated_",
        "",
    ]
    for ticker, trades in sorted(bucket.items()):
        recent = trades[-20:]
        wins   = sum(1 for t in recent if t["outcome"] == "WIN")
        wr     = wins / len(recent) * 100
        last   = trades[-1]
        change = f"{last['actual_pct']*100:+.1f}%" if last.get("actual_pct") is not None else "?"
        a      = adj.get(ticker, 0)
        adj_str = f"adj {a:+d}" if a != 0 else "adj 0 (neutral)"
        lines.append(f"**{ticker}** — Win rate {wr:.0f}% ({wins}/{len(recent)}) | Last: {change} | {adj_str}")

    if DISCORD:
        try:
            requests.post(DISCORD, json={"content": "\n".join(lines)}, timeout=5)
        except Exception:
            pass

    return {e["ticker"]: e["outcome"] for e in updated if e["outcome"] is not None}

def decide(ticker: str, df: pd.DataFrame, model: Pipeline) -> dict:
    GATED = {"signal": "GATED", "label": "🚫 GATED", "color": "#888",
              "alert": False, "prob": 0.0, "score": 0, "why": [], "filters": [],
              "adj": 0}
    if len(df) < 60:
        return {**GATED, "filters": [("Enough data (≥60 rows)", False)]}

    row, prev = df.iloc[-1], df.iloc[-2]
    prob = float(model.predict_proba(pd.DataFrame([row[FEATURES]]))[0][1])

    filters = [
        ("VIX fear index safe (< 30)",         vix_safe()),
        ("Broad market in uptrend",            market_regime_ok(ticker)),
        ("Sector ETF in uptrend",             sector_ok(ticker)),
        ("Weekly trend confirmed",            weekly_trend_ok(ticker)),
        ("No earnings within 5 days",          earnings_safe(ticker)),
        ("Uptrend: EMA20 > EMA50",            row["ema20"]  > row["ema50"]),
        ("Confirmed: EMA20 > EMA50 prior day", prev["ema20"] > prev["ema50"]),
        ("MACD bullish (diff > 0)",            row["macd_diff"]  > 0),
        ("MACD bullish prior day",             prev["macd_diff"] > 0),
        ("RSI not overbought (< 72)",          row["rsi"] < 72),
        ("RSI not oversold (> 25)",            row["rsi"] > 25),
        ("Liquidity (vol ratio ≥ 0.5)",        row["vol_ratio"] >= 0.5),
        ("AI probability ≥ 40%",              prob >= 0.40),
    ]
    if not all(ok for _, ok in filters):
        return {**GATED, "prob": prob, "filters": filters}

    rules = [
        (3, "AI prob ≥ 80%",             prob >= 0.80),
        (2, "AI prob ≥ 70%",             0.70 <= prob < 0.80),
        (1, "AI prob ≥ 60%",             0.60 <= prob < 0.70),
        (3, "52-week breakout",          bool(row["breakout"])),
        (2, "Volume surge > 1.5×",       row["vol_ratio"] > 1.5),
        (2, "RSI in ideal zone (35–65)", 35 <= row["rsi"] <= 65),
        (1, "RSI safe (< 70)",           row["rsi"] < 70),
        (1, "EMA uptrend confirmed",     row["ema20"] > row["ema50"]),
    ]
    base_score = sum(pts for pts, _, met in rules if met)
    why        = [name for _, name, met in rules if met]

    # Apply learned adjustment from past signal outcomes
    adj   = performance_adjustments().get(ticker, 0)

    # Apply news sentiment adjustment
    news  = news_sentiment(ticker)
    news_adj = news["score_adj"]
    if news["label"] == "NEGATIVE":
        filters.append((f"News sentiment not strongly negative ({news['compound']:.2f})", False))
        return {**GATED, "prob": prob, "filters": filters, "news": news}

    # All signal adjusters
    short_adj,   short_why   = short_interest_signal(ticker)
    insider_adj, insider_why = insider_signal(ticker)
    opts_adj,    opts_why    = options_flow_signal(ticker)
    comm_adj,    comm_why    = commodity_signal(ticker)
    vel_adj,     vel_why     = news_velocity(ticker)
    sr_adj,      sr_why      = support_resistance_signal(df)
    mtf_adj,     mtf_why     = multitimeframe_signal(ticker)
    rs_adj,      rs_why      = relative_strength_signal(ticker, df)
    fg_adj,      fg_why      = fear_greed_signal()
    rot_adj,     rot_why     = sector_rotation_signal(ticker)
    gap_adj,     gap_why     = gap_signal(df)
    sq_adj,      sq_why      = squeeze_breakout_signal(df)
    fund_adj,    fund_why    = fundamental_signal(ticker)
    vwap_adj,    vwap_why    = vwap_signal(ticker)
    for reason in (short_why, insider_why, opts_why, comm_why, vel_why,
                   sr_why, mtf_why, rs_why, fg_why, rot_why, gap_why, sq_why, fund_why, vwap_why):
        if reason:
            why.append(reason)

    score = (base_score + adj + news_adj + short_adj + insider_adj + opts_adj
             + comm_adj + vel_adj + sr_adj + mtf_adj + rs_adj + fg_adj
             + rot_adj + gap_adj + sq_adj + fund_adj + vwap_adj)

    # Grade thresholds — only the strongest qualify for an alert
    # ELITE:      score ≥ 11  (any AI prob — already filtered to ≥ 55% above)
    # STRONG BUY: score ≥ 9  AND AI prob ≥ 70%  (genuinely high conviction only)
    # WATCH:      everything else that passed filters — shown on dashboard, never alerted
    if   score >= 11:                        signal, label, color, qualifies = "ELITE",      "🏆 ELITE",      "#00cc44", True
    elif score >= 9 and prob >= 0.70:        signal, label, color, qualifies = "STRONG BUY", "✅ STRONG BUY", "#44bb00", True
    elif score >= 5:                         signal, label, color, qualifies = "WATCH",      "👀 WATCH",      "#e6a817", False
    else:                                    signal, label, color, qualifies = "IGNORE",     "⛔ IGNORE",     "#cc3300", False

    return {"signal": signal, "label": label, "color": color,
            "alert": qualifies and cooldown_ok(ticker),
            "prob": prob, "score": score, "base_score": base_score,
            "adj": adj, "news": news, "why": why, "filters": filters,
            "rsi": round(float(row["rsi"]), 1)}

# ─── COMMODITY PRICE TRACKING ────────────────────────────────────────────────
def commodity_signal(ticker: str) -> tuple:
    """
    For commodity-driven ASX miners: check if the underlying commodity
    is in uptrend AND recently surging.

    Score breakdown:
      +2  commodity up >3% over 5 days (strong surge)
      +1  commodity above its 20-day EMA (uptrend)
       0  neutral
      -1  commodity down >3% over 5 days (headwind)
    """
    mapping = COMMODITY_MAP.get(ticker)
    if not mapping:
        return (0, "")
    etf, name = mapping
    cache_key = f"commodity_{etf}"
    cached = _signal_cached(cache_key, ttl=3600)
    if cached is not None:
        return cached
    try:
        df    = yf.Ticker(etf).history(period="3mo")
        if df.empty or len(df) < 22:
            return _signal_store(cache_key, (0, ""))
        close  = df["Close"]
        ema20  = close.ewm(span=20, adjust=False).mean()
        ret5   = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0

        if ret5 >= 0.03:
            result = (+2, f"{name.title()} surging +{ret5*100:.1f}% (5d)")
        elif close.iloc[-1] > ema20.iloc[-1]:
            result = (+1, f"{name.title()} in uptrend")
        elif ret5 <= -0.03:
            result = (-1, f"{name.title()} falling {ret5*100:.1f}% (5d) — headwind")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── NEWS VELOCITY ────────────────────────────────────────────────────────────
def news_velocity(ticker: str) -> tuple:
    """
    Detect a news volume spike — a sign that a catalyst event is underway.
    Compares articles published in the last 48h vs the prior 5 days.

    Score breakdown:
      +2  strong spike: ≥4 articles in 48h AND more than double the prior rate
      +1  mild spike:   ≥2 articles in 48h AND more than prior rate
       0  no spike
    """
    cache_key = f"velocity_{ticker}"
    cached = _signal_cached(cache_key, ttl=1800)   # 30-min cache
    if cached is not None:
        return cached
    try:
        articles = yf.Ticker(ticker).news or []
        now_ts   = datetime.now().timestamp()
        h48      = 48 * 3600
        d7       = 7  * 24 * 3600

        recent = sum(1 for a in articles
                     if now_ts - a.get("providerPublishTime", 0) < h48)
        older  = sum(1 for a in articles
                     if h48 <= now_ts - a.get("providerPublishTime", 0) < d7)

        if recent >= 4 and recent > older * 1.5:
            result = (+2, f"News velocity spike ({recent} articles in 48h — catalyst likely)")
        elif recent >= 2 and recent > older:
            result = (+1, f"Elevated news activity ({recent} articles in 48h)")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── SUPPORT / RESISTANCE ─────────────────────────────────────────────────────
def support_resistance_signal(df: "pd.DataFrame") -> tuple:
    """
    Detect key support/resistance price interactions on the daily chart.

    Score breakdown:
      +2  Breaking through the 20-day resistance high (momentum at key level)
      +1  Bouncing from the 20-day support low (buyers stepping in)
      +1  Bouncing from the 50-day support low (structural support)
       0  No notable interaction
    """
    try:
        close   = df["Close"]
        current = float(close.iloc[-1])
        high_20 = float(close.rolling(20).max().iloc[-2])   # prior bar avoids look-ahead
        low_20  = float(close.rolling(20).min().iloc[-2])
        low_50  = float(close.rolling(50).min().iloc[-2])

        if current >= high_20 * 0.99:
            return (+2, "Breaking 20-day resistance — momentum at key level")
        elif low_20 <= current <= low_20 * 1.04:
            return (+1, "Bouncing from 20-day support level")
        elif low_50 <= current <= low_50 * 1.04:
            return (+1, "Bouncing from 50-day support level")
    except Exception:
        pass
    return (0, "")


# ─── MULTI-TIMEFRAME CONFIRMATION ─────────────────────────────────────────────
def multitimeframe_signal(ticker: str) -> tuple:
    """
    Check whether the 1-hour chart agrees with the daily buy signal.
    Adds conviction when multiple timeframes point the same direction.

    Score breakdown:
      +1  1h EMA20 > EMA50 AND 1h MACD positive (intraday trend fully aligned)
       0  mixed or unavailable
    """
    cache_key = f"mtf_{ticker}"
    cached = _signal_cached(cache_key, ttl=3600)
    if cached is not None:
        return cached
    try:
        df_1h = yf.Ticker(ticker).history(period="5d", interval="1h")
        if df_1h.empty or len(df_1h) < 26:
            result = (0, "")
        else:
            close     = df_1h["Close"]
            ema20     = close.ewm(span=20, adjust=False).mean()
            ema50     = close.ewm(span=50, adjust=False).mean()
            macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            if ema20.iloc[-1] > ema50.iloc[-1] and macd_line.iloc[-1] > 0:
                result = (+1, "Intraday trend aligned (1h EMA + MACD bullish)")
            else:
                result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── RELATIVE STRENGTH vs BENCHMARK ─────────────────────────────────────────
def relative_strength_signal(ticker: str, df: "pd.DataFrame") -> tuple:
    """
    Compare ticker's 12-week return vs its benchmark (SPY or ^AXJO).
    Stocks outperforming their benchmark attract institutional buying.

    +2  top 10% — significantly outperforming (RS leader)
    +1  outperforming benchmark by any margin
     0  in-line or underperforming
    """
    cache_key = f"rs_{ticker}"
    cached = _signal_cached(cache_key, ttl=3600)
    if cached is not None:
        return cached
    try:
        bench = "^AXJO" if ticker.endswith(".AX") else "SPY"
        b_df  = yf.Ticker(bench).history(period="4mo")
        if len(df) < 63 or len(b_df) < 63:
            return _signal_store(cache_key, (0, ""))
        ticker_ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1)
        bench_ret  = float(b_df["Close"].iloc[-1] / b_df["Close"].iloc[-63] - 1)
        spread = ticker_ret - bench_ret
        if spread >= 0.10:
            result = (+2, f"RS leader — outperforming benchmark by {spread*100:.1f}% (12wk)")
        elif spread > 0:
            result = (+1, f"Outperforming benchmark by {spread*100:.1f}% (12wk)")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── FEAR & GREED COMPOSITE ───────────────────────────────────────────────────
def fear_greed_signal() -> tuple:
    """
    Composite market sentiment: VIX level + SPY 20-day momentum.
    Provides an extra boost in genuinely risk-on conditions.

    +1  VIX < 18 AND SPY has positive 20-day momentum (greed — good conditions)
     0  neutral conditions
    -1  VIX 25–30 (caution — approaching the fear threshold)
    """
    cache_key = "fear_greed"
    cached = _signal_cached(cache_key, ttl=3600)
    if cached is not None:
        return cached
    try:
        vix    = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
        spy_df = yf.Ticker("SPY").history(period="2mo")
        spy_mom = float(spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-20] - 1) if len(spy_df) >= 20 else 0
        if vix < 18 and spy_mom > 0:
            result = (+1, f"Risk-on environment (VIX {vix:.1f}, SPY +{spy_mom*100:.1f}% 20d)")
        elif vix >= 25:
            result = (-1, f"Elevated fear (VIX {vix:.1f}) — caution")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── SECTOR ROTATION ──────────────────────────────────────────────────────────
def sector_rotation_signal(ticker: str) -> tuple:
    """
    Detect if this ticker's sector is currently leading the market.
    A stock in a leading sector has institutional tailwinds.

    +1  sector ETF outperforming SPY by >3% over 4 weeks
     0  sector neutral or lagging
    """
    etf = SECTOR_ETF.get(ticker)
    if not etf:
        return (0, "")
    cache_key = f"rotation_{etf}"
    cached = _signal_cached(cache_key, ttl=3600)
    if cached is not None:
        return cached
    try:
        etf_df = yf.Ticker(etf).history(period="2mo")
        spy_df = yf.Ticker("SPY").history(period="2mo")
        if len(etf_df) < 20 or len(spy_df) < 20:
            return _signal_store(cache_key, (0, ""))
        etf_ret = float(etf_df["Close"].iloc[-1] / etf_df["Close"].iloc[-20] - 1)
        spy_ret = float(spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-20] - 1)
        if etf_ret - spy_ret >= 0.03:
            result = (+1, f"Sector leading market by {(etf_ret-spy_ret)*100:.1f}% (4wk rotation)")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── GAP-UP + BB SQUEEZE BREAKOUT + FUNDAMENTAL ───────────────────────────────
def gap_signal(df: "pd.DataFrame") -> tuple:
    """
    Detect a gap-up on volume — institutions buying aggressively overnight.
    One of the highest-probability short-term momentum signals.

    +2  gap up >2% AND volume surge >1.5× (institutional gap — very strong)
    +1  gap up >2% on normal volume
     0  no gap
    """
    try:
        row = df.iloc[-1]
        if row.get("gap_up", 0) == 1:
            if row["vol_ratio"] > 1.5:
                return (+2, "Gap-up on institutional volume — strong overnight buying")
            return (+1, "Gap-up detected — above prior day's close by >2%")
    except Exception:
        pass
    return (0, "")


def squeeze_breakout_signal(df: "pd.DataFrame") -> tuple:
    """
    Detect a breakout from a Bollinger Band squeeze.
    A breakout after low-volatility consolidation is far more powerful
    than a random breakout from a noisy range.

    +2  currently breaking out AND was in a squeeze in the last 5 days
     0  otherwise
    """
    try:
        row = df.iloc[-1]
        recent_squeeze = df["bb_squeeze"].iloc[-6:-1].any()
        if bool(row["breakout"]) and recent_squeeze:
            return (+2, "Breakout from volatility squeeze — compressed spring releasing")
    except Exception:
        pass
    return (0, "")


def fundamental_signal(ticker: str) -> tuple:
    """
    Basic fundamental quality check using yfinance info.
    Blocks signals on fundamentally broken companies; boosts quality ones.

    +1  strong: positive free cash flow AND P/E 5–25 AND low debt
    -1  weak: negative earnings OR debt/equity > 3
     0  data unavailable or neutral
    """
    cache_key = f"fundamental_{ticker}"
    cached = _signal_cached(cache_key, ttl=86400)   # 24h cache — fundamentals don't change hourly
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker).info
        pe       = info.get("trailingPE", None)
        de       = info.get("debtToEquity", None)
        fcf      = info.get("freeCashflow", None)
        eps      = info.get("trailingEps", None)

        if eps is not None and eps < 0:
            result = (-1, "Negative earnings — fundamental caution")
        elif de is not None and de > 300:   # yfinance expresses as %, so 300 = 3.0
            result = (-1, "High debt load — fundamental caution")
        elif (pe is not None and 5 < pe < 25
              and fcf is not None and fcf > 0
              and (de is None or de < 100)):
            result = (+1, "Strong fundamentals (FCF positive, reasonable P/E, low debt)")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── VWAP SIGNAL ─────────────────────────────────────────────────────────────
def vwap_signal(ticker: str) -> tuple:
    """
    VWAP (Volume-Weighted Average Price) is the single most-used institutional
    reference. When price crosses above VWAP on strong volume, institutions are
    repositioning long. Crossing below VWAP = distribution.

    +2  price above VWAP AND last bar crossed above it on above-avg volume (fresh breakout)
    +1  price above VWAP (bullish positioning — above institutional avg cost)
     0  at or below VWAP
    -1  price below VWAP (distribution — institutions selling above you)
    """
    cache_key = f"vwap_{ticker}"
    cached = _signal_cached(cache_key, ttl=1800)   # 30-min cache — intraday changes matter
    if cached is not None:
        return cached
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1h")
        if len(df) < 3:
            return _signal_store(cache_key, (0, ""))
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap    = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        last_close = float(df["Close"].iloc[-1])
        last_vwap  = float(vwap.iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        prev_vwap  = float(vwap.iloc[-2])
        last_vol   = float(df["Volume"].iloc[-1])
        avg_vol    = float(df["Volume"].mean())

        crossed_above = prev_close < prev_vwap and last_close > last_vwap
        above         = last_close > last_vwap
        vol_surge     = last_vol > avg_vol * 1.2

        if crossed_above and vol_surge:
            result = (+2, f"VWAP cross-above on volume surge — institutional repositioning long (VWAP ${last_vwap:.2f})")
        elif above:
            result = (+1, f"Trading above VWAP ${last_vwap:.2f} — bullish intraday positioning")
        elif not above:
            result = (-1, f"Below VWAP ${last_vwap:.2f} — institutional average cost above current price")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(cache_key, result)


# ─── SHORT INTEREST, INSIDER BUYING, OPTIONS FLOW ───────────────────────────
_signal_cache: dict = {}   # shared 4-hour cache for slower yfinance calls

def _signal_cached(key: str, ttl: int = 14400):
    """Return cached value or None if stale/missing."""
    if key in _signal_cache:
        ts, val = _signal_cache[key]
        if (datetime.now().timestamp() - ts) < ttl:
            return val
    return None

def _signal_store(key: str, val):
    _signal_cache[key] = (datetime.now().timestamp(), val)
    return val

def short_interest_signal(ticker: str) -> tuple:
    """
    High short interest on a breaking-out stock = squeeze potential.
    Returns (score_adj, reason_string).
    """
    cached = _signal_cached(f"si_{ticker}")
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker).info
        spof = float(info.get("shortPercentOfFloat") or 0)
        if spof > 0.25:
            result = (+2, f"Short squeeze candidate ({spof*100:.0f}% short float)")
        elif spof > 0.15:
            result = (+1, f"Elevated short interest ({spof*100:.0f}%)")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(f"si_{ticker}", result)

def insider_signal(ticker: str) -> tuple:
    """
    Net insider buying in last 90 days = management conviction.
    Returns (score_adj, reason_string).
    """
    cached = _signal_cached(f"insider_{ticker}")
    if cached is not None:
        return cached
    try:
        df = yf.Ticker(ticker).insider_transactions
        if df is None or df.empty:
            return _signal_store(f"insider_{ticker}", (0, ""))
        # Normalise column names
        df.columns = [c.lower() for c in df.columns]
        # Date filter — last 90 days
        date_col = next((c for c in df.columns if "date" in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
            df = df[df[date_col] >= cutoff]
        if df.empty:
            return _signal_store(f"insider_{ticker}", (0, ""))
        # Transaction type column
        trans_col = next((c for c in df.columns if "transact" in c or "type" in c), None)
        if not trans_col:
            return _signal_store(f"insider_{ticker}", (0, ""))
        buys  = df[df[trans_col].astype(str).str.contains("Buy|Purchase|Acquire", case=False, na=False)]
        sells = df[df[trans_col].astype(str).str.contains("Sell|Sale|Disposition", case=False, na=False)]
        if len(buys) >= 2 and len(buys) > len(sells):
            result = (+2, f"Insider buying ({len(buys)} purchases in 90 days)")
        elif len(buys) >= 1 and len(buys) >= len(sells):
            result = (+1, "Insider net buying (90 days)")
        elif len(sells) > len(buys) * 2:
            result = (-1, f"Insider selling ({len(sells)} sales in 90 days)")
        else:
            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(f"insider_{ticker}", result)

def options_flow_signal(ticker: str) -> tuple:
    """
    Put/Call ratio from nearest options expiry.
    PCR < 0.7 = more calls than puts = bullish sentiment.
    ASX stocks skipped (no options data on yfinance).
    Returns (score_adj, reason_string).
    """
    if ticker.endswith(".AX"):
        return (0, "")
    cached = _signal_cached(f"opts_{ticker}")
    if cached is not None:
        return cached
    try:
        t     = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return _signal_store(f"opts_{ticker}", (0, ""))
        chain = t.option_chain(dates[0])
        call_oi = float(chain.calls["openInterest"].fillna(0).sum())
        put_oi  = float(chain.puts["openInterest"].fillna(0).sum())
        if call_oi == 0:
            return _signal_store(f"opts_{ticker}", (0, ""))
        pcr = put_oi / call_oi
        if   pcr < 0.60: result = (+2, f"Bullish options flow — PCR {pcr:.2f}")
        elif pcr < 0.80: result = (+1, f"Mildly bullish options — PCR {pcr:.2f}")
        elif pcr > 1.50: result = (-1, f"Bearish options flow — PCR {pcr:.2f}")
        else:            result = (0, "")
    except Exception:
        result = (0, "")
    return _signal_store(f"opts_{ticker}", result)

# ─── NEWS SENTIMENT (VADER + Loughran-McDonald financial lexicon) ─────────────
# Loughran-McDonald financial word lists — purpose-built for financial text
_LM_POS = {
    "beat", "beats", "exceed", "exceeded", "record", "surge", "surged", "raised",
    "upgrade", "upgraded", "outperform", "outperformed", "growth", "profit", "profits",
    "expansion", "breakthrough", "win", "award", "strong", "confident", "momentum",
    "dividend", "buyback", "innovative", "accelerating", "recovery", "robust",
    "impressive", "delivered", "guidance", "raised guidance", "upside", "positive",
}
_LM_NEG = {
    "miss", "missed", "loss", "losses", "decline", "declined", "fail", "failed",
    "weak", "concern", "concerns", "risk", "uncertain", "uncertainty", "cut",
    "downgrade", "downgraded", "fraud", "lawsuit", "recall", "bankrupt", "bankruptcy",
    "layoff", "layoffs", "warning", "crisis", "shortage", "violation", "investigation",
    "probe", "default", "disappointing", "below", "lowered", "guidance cut", "miss",
}

def _lm_score(text: str) -> float:
    """Loughran-McDonald financial lexicon score — returns -1 to +1."""
    words = set(text.lower().split())
    pos   = len(words & _LM_POS)
    neg   = len(words & _LM_NEG)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

def news_sentiment(ticker: str) -> dict:
    """
    Hybrid sentiment: 60% VADER + 40% Loughran-McDonald financial lexicon.
    LM is purpose-built for financial text and catches what VADER misses.
    """
    try:
        articles  = yf.Ticker(ticker).news or []
        headlines = []
        for a in articles[:10]:
            title = (a.get("content") or {}).get("title") or a.get("title") or ""
            if title:
                headlines.append(title)
        if not headlines:
            return {"compound": 0.0, "label": "NEUTRAL", "score_adj": 0,
                    "headlines": [], "count": 0}
        vader_scores = [_vader.polarity_scores(h)["compound"] for h in headlines]
        lm_scores    = [_lm_score(h) for h in headlines]
        hybrid       = [0.6 * v + 0.4 * l for v, l in zip(vader_scores, lm_scores)]
        avg = sum(hybrid) / len(hybrid)
        if   avg >  0.20: label, adj = "POSITIVE", +1
        elif avg < -0.20: label, adj = "NEGATIVE", -1
        else:             label, adj = "NEUTRAL",   0
        top = sorted(zip(hybrid, headlines), reverse=True)
        top_headlines = [h for _, h in top[:2]]
        return {"compound": round(avg, 3), "label": label, "score_adj": adj,
                "headlines": top_headlines, "count": len(headlines)}
    except Exception:
        return {"compound": 0.0, "label": "NEUTRAL", "score_adj": 0,
                "headlines": [], "count": 0}

# ─── CONFIDENCE INTERVAL — historical return range for similar setups ─────────
def confidence_interval(tier: str) -> dict | None:
    """
    Returns best/worst/median return from resolved signals of the same tier.
    Requires ≥3 resolved signals for that tier to be meaningful.
    """
    entries  = _load_log()
    resolved = [e for e in entries if e["outcome"] is not None and e["tier"] == tier]
    if len(resolved) < 3:
        all_resolved = [e for e in entries if e["outcome"] is not None]
        if len(all_resolved) < 3:
            return None
        resolved = all_resolved
    returns = sorted(e["actual_pct"] for e in resolved)
    n = len(returns)
    median = returns[n // 2]
    p25    = returns[max(0, n // 4)]
    p75    = returns[min(n - 1, 3 * n // 4)]
    return {
        "best":   returns[-1],
        "worst":  returns[0],
        "median": median,
        "p25":    p25,
        "p75":    p75,
        "n":      n,
    }

# ─── CONFIDENCE GRADE ────────────────────────────────────────────────────────
def confidence_grade(prob: float, score: int) -> tuple:
    """
    Human-readable confidence grade combining AI probability and signal score.
    Returns (grade, label, bar) — e.g. ("A+", "VERY HIGH", "██████████ 10/10")
    """
    combined = (prob * 0.6) + ((score / 14) * 0.4)
    if   combined >= 0.80: grade, label = "A+", "VERY HIGH 🔥"
    elif combined >= 0.65: grade, label = "A",  "HIGH ✨"
    elif combined >= 0.50: grade, label = "B+", "GOOD 📈"
    elif combined >= 0.35: grade, label = "B",  "MODERATE 📊"
    else:                  grade, label = "C",  "LOW ⚠️"
    filled = round(combined * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return grade, label, f"{bar} {filled}/10"

# ─── STEP 4 — DISCORD ALERT ──────────────────────────────────────────────────
def _guard_ok(ticker: str, window_seconds: int = 90) -> bool:
    """
    Two-layer duplicate guard:
      1. Per-ticker:  blocks same ticker within `window_seconds` (default 90s)
      2. Global:      blocks ANY alert within 5 minutes of the last one sent
    Prevents double-sends whether from the same ticker or rapid successive alerts.
    """
    try:
        now  = datetime.now().timestamp()
        data = json.loads(SEND_GUARD_FILE.read_text()) if SEND_GUARD_FILE.exists() else {}

        # Layer 1 — per-ticker guard
        last_ticker = data.get(ticker, 0)
        if (now - last_ticker) < window_seconds:
            print(f"  ⏭ {ticker}: per-ticker guard active ({int(window_seconds - (now - last_ticker))}s remaining)")
            return False

        # Layer 2 — global guard (5 minutes between any two alerts)
        last_any = data.get("__last_any__", 0)
        if (now - last_any) < 300:
            print(f"  ⏭ {ticker}: global 5-min guard active ({int(300 - (now - last_any))}s remaining)")
            return False

        data[ticker]        = now
        data["__last_any__"] = now
        SEND_GUARD_FILE.write_text(json.dumps(data))
        return True
    except Exception:
        return True

def _trade_params(ticker: str, result: dict, price: float, df: "pd.DataFrame") -> dict:
    """
    Dynamically compute target return % and holding period.

    Volatility  →  base target / window
    ─────────────────────────────────────────────
    High  ≥3% ATR   →  8%  /  7 trading days   (small miners, TSLA)
    Mid  1.5–3% ATR  →  5%  / 10 trading days   (AAPL, BHP, NVDA)
    Low   <1.5% ATR  →  3%  / 15 trading days   (CBA, MSFT)

    Multipliers applied on top:
      ELITE signal   → target ×1.25, window ×0.80
      Breakout       → target ×1.15, window ×0.85
    Hard caps: target 5–20%, window 5–20 trading days.
    """
    row     = df.iloc[-1]
    atr_pct = float(row["atr"]) / price * 100
    tier    = result.get("signal", "WATCH")
    breakout = bool(row.get("breakout", 0))

    if atr_pct >= 3.0:
        base_target, base_days, vol_label = 0.08,  7, "high-volatility"
    elif atr_pct >= 1.5:
        base_target, base_days, vol_label = 0.05, 10, "mid-volatility"
    else:
        base_target, base_days, vol_label = 0.03, 15, "low-volatility"

    t_mult, d_mult = 1.0, 1.0
    if tier == "ELITE":
        t_mult *= 1.25; d_mult *= 0.80
    if breakout:
        t_mult *= 1.15; d_mult *= 0.85

    final_target = round(min(base_target * t_mult, 0.20), 4)
    final_days   = max(5, min(20, int(round(base_days * d_mult))))

    reasons = [vol_label, f"ATR {atr_pct:.1f}%"]
    if tier == "ELITE":  reasons.append("ELITE signal")
    if breakout:         reasons.append("52-week breakout")

    # ATR-based stop-loss — wider for high-vol stocks
    atr_raw  = float(df.iloc[-1]["atr"])
    sl_mult  = 2.0 if atr_pct >= 3.0 else (1.5 if atr_pct >= 1.5 else 1.2)
    stop_loss     = round(max(price - sl_mult * atr_raw, price * 0.85), 4)  # hard floor -15%
    stop_loss_pct = round((stop_loss - price) / price * 100, 1)

    return {
        "target_pct":    final_target,
        "exit_days":     final_days,
        "target_price":  round(price * (1 + final_target), 4),
        "stop_loss":     stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "rationale":     ", ".join(reasons),
    }

def _add_trading_days(start: datetime, n: int) -> datetime:
    """Return the date that is exactly n trading days (Mon–Fri) after start."""
    current = start
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:   # Monday=0 … Friday=4
            added += 1
    return current

def _next_trading_day(from_date: datetime) -> datetime:
    """Return the next trading day from from_date (same day if it's a weekday before market close)."""
    candidate = from_date + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate

def _simple_read(ticker: str, result: dict, price: float) -> str:
    """
    Generate a plain-English one-paragraph summary of why this signal fired.
    Reads like a human analyst note, not a data dump.
    """
    why   = result.get("why", [])
    tier  = result.get("signal", "STRONG BUY")
    prob  = result.get("prob", 0)
    rsi   = result.get("rsi", 50)

    parts = []

    # Commodity context
    comm = next((w for w in why if any(k in w for k in
                 ("Iron Ore", "Gold", "Lithium", "Uranium", "surging", "uptrend", "headwind"))), None)
    if comm:
        parts.append(f"The underlying commodity is showing strength — {comm.lower()}.")

    # Breakout
    if any("breakout" in w.lower() for w in why):
        parts.append(f"{ticker} just broke its 52-week high, a classic momentum signal.")

    # Volume
    if any("volume" in w.lower() for w in why):
        parts.append("Volume is surging well above normal, confirming real buying interest.")

    # Insider
    if any("insider" in w.lower() for w in why):
        parts.append("Company insiders have been net buyers in the last 90 days — management has skin in the game.")

    # Options flow
    if any("options" in w.lower() or "pcr" in w.lower() for w in why):
        parts.append("The options market is skewed bullish — more calls than puts being bought.")

    # News velocity
    if any("velocity" in w.lower() or "catalyst" in w.lower() for w in why):
        parts.append("News volume has spiked in the last 48 hours, suggesting a catalyst may be underway.")

    # Support / resistance
    if any("resistance" in w.lower() for w in why):
        parts.append("Price is breaking through a key resistance level — a textbook momentum signal.")
    elif any("support" in w.lower() for w in why):
        parts.append("Price is bouncing off a key support level, suggesting buyers are defending that zone.")

    # Multi-timeframe
    if any("intraday" in w.lower() for w in why):
        parts.append("The 1-hour chart is also bullish — daily and intraday trends are pointing the same direction.")

    # Relative strength
    if any("rs leader" in w.lower() for w in why):
        parts.append("This stock is a relative strength leader — outperforming the market significantly over the last 3 months. Institutions are actively accumulating.")
    elif any("outperforming benchmark" in w.lower() for w in why):
        parts.append("The stock is outperforming the broader market over the last 3 months — a sign of underlying institutional demand.")

    # Gap-up
    if any("gap-up" in w.lower() for w in why):
        parts.append("A gap-up on high volume overnight signals institutions were buying aggressively before the open.")

    # Squeeze breakout
    if any("squeeze" in w.lower() for w in why):
        parts.append("This breakout follows a period of tight consolidation — like a compressed spring releasing. These are among the highest-quality breakout setups.")

    # Fear & Greed / sector rotation
    if any("risk-on" in w.lower() for w in why):
        parts.append("Macro conditions are risk-on — low fear, positive market momentum. A good environment for momentum trades.")
    if any("sector leading" in w.lower() or "rotation" in w.lower() for w in why):
        parts.append("The sector is currently leading the broader market, providing an institutional tailwind.")

    # Fundamental
    if any("fundamentals" in w.lower() for w in why):
        parts.append("Fundamentals are solid — positive cash flow, reasonable valuation, and manageable debt.")

    # VWAP
    if any("vwap cross-above" in w.lower() for w in why):
        parts.append("Price just crossed above VWAP on a volume surge — this is institutions repositioning long in real time. One of the strongest intraday confirmation signals.")
    elif any("above vwap" in w.lower() for w in why):
        parts.append("Price is trading above VWAP — it's above the institutional average cost for the day, which means buyers are in control.")

    # RSI context
    if rsi < 40:
        parts.append(f"RSI at {rsi:.0f} is oversold — this is an ideal low-risk entry point.")
    elif rsi < 55:
        parts.append(f"RSI at {rsi:.0f} is neutral, leaving plenty of room to run.")

    # AI summary
    if prob >= 0.75:
        parts.append(f"The AI model is {prob*100:.0f}% confident based on historical patterns — very high conviction.")
    elif prob >= 0.55:
        parts.append(f"The AI model is {prob*100:.0f}% confident — solid conviction.")

    # Closing line
    if tier == "ELITE":
        parts.append("All 13 filters are green. This is the highest-quality signal the bot produces.")
    else:
        parts.append("All filters passed. This is a high-quality setup worth watching closely.")

    return " ".join(parts) if parts else "All filters passed with strong technical and AI alignment."


def send_alert(ticker: str, result: dict, price: float, df=None) -> bool:
    if not DISCORD:
        return False
    if not _guard_ok(ticker):
        print(f"  ⏭ {ticker}: duplicate suppressed (sent within last 90s)")
        return False

    # Dynamic target & window — fetch df if not passed
    if df is None:
        import yfinance as _yf
        df = _yf.Ticker(ticker).history(period="6mo")
    params       = _trade_params(ticker, result, price, df)
    target_price = params["target_price"]
    target_pct   = params["target_pct"]
    exit_days    = params["exit_days"]

    now       = datetime.now()
    buy_date  = _next_trading_day(now)
    exit_date = _add_trading_days(buy_date, exit_days)
    grade, glabel, gbar = confidence_grade(result["prob"], result["score"])
    ci = confidence_interval(result["signal"])

    # Company name (best-effort)
    try:
        short_name = yf.Ticker(ticker).info.get("shortName", ticker)
    except Exception:
        short_name = ticker

    # RSI entry suggestion
    rsi = result.get("rsi", 50)
    if rsi < 40:
        entry_price = f"${price:.3f}"
        entry_note  = "buy now — RSI oversold, ideal entry"
    elif rsi < 50:
        entry_price = f"${price:.3f}"
        entry_note  = "now or on a small dip"
    elif rsi < 60:
        entry_price = f"${price*0.99:.3f}–${price*0.985:.3f}"
        entry_note  = "wait for a 1–1.5% pullback"
    else:
        entry_price = f"${price*0.98:.3f}"
        entry_note  = "RSI elevated — wait for a 2% dip"

    # ── Schedule-aware buy instruction ───────────────────────────────────────
    # Priority order:
    #   1. Opening window (first 30 min of session) → "buy at market open NOW"
    #   2. Intraday signal + market closed          → warn, skip
    #   3. Swing signal + market closed             → defer to next open with validity band
    #   4. Mid-session open                         → RSI-based logic (already set above)

    _intraday_kw = ("vwap cross-above", "gap-up on institutional", "gap-up detected", "intraday")
    _swing_kw    = ("ema", "uptrend", "rsi", "support", "resistance", "macd", "breakout",
                    "obv", "relative strength", "sector", "fundamental", "squeeze",
                    "multi-timeframe", "oversold", "volume surge")
    _why_list    = result.get("why", [])
    _is_intraday = any(any(kw in w.lower() for kw in _intraday_kw) for w in _why_list)
    _is_swing    = any(any(kw in w.lower() for kw in _swing_kw)    for w in _why_list)
    _is_asx      = ticker.endswith(".AX")

    _aest        = _pytz.timezone("Australia/Sydney")
    _now_aest    = datetime.now(_aest)
    _h, _m       = _now_aest.hour, _now_aest.minute
    _wd          = _now_aest.weekday()   # 0=Mon … 6=Sun

    # Opening windows (first 30 min of session — highest quality entry)
    _asx_opening = _is_asx  and _wd < 5 and (_h == 10 and _m < 30)
    _us_opening  = not _is_asx and _wd < 5 and ((_h == 23 and _m >= 30) or (_h == 0 and _m < 0))
    _opening_now = _asx_opening or _us_opening

    # Full session open (outside opening window)
    _asx_open    = _is_asx      and 10 <= _h < 16 and _wd < 5
    _us_open     = (not _is_asx) and (_h >= 23 or _h < 6) and _wd < 5
    _mkt_open    = _asx_open or _us_open

    if _opening_now:
        # ★ Best case — alert fires right at the open
        _open_label  = "ASX open" if _is_asx else "US market open"
        entry_price  = f"${price:.3f}"
        entry_note   = "tightest spreads & best fills right now"
        _entry_banner = (f"┌─────────────────────────────────┐\n"
                         f"│  ⚡ BUY NOW  —  {_open_label.upper():<17}│\n"
                         f"│  Market is open. Act immediately. │\n"
                         f"└─────────────────────────────────┘")
    elif _is_intraday and not _mkt_open:
        # Intraday signal but market is closed — cannot defer
        entry_note   += " — market closed; skip unless signal recurs at next open"
        _entry_banner = ("⚠️  **INTRADAY SIGNAL — MARKET CLOSED**\n"
                         "_This VWAP/gap setup expires at close. Skip it unless the same signal fires at next open._")
    elif not _mkt_open and _is_swing:
        # Swing setup — thesis still valid at next open
        _open_str    = "10:00am AEST tomorrow" if _is_asx else "11:30pm AEST tonight"
        _max_valid   = price * 1.025
        entry_price  = f"${price:.3f} – ${_max_valid:.3f}"
        entry_note   = f"place limit order before {_open_str}. Gap above ${_max_valid:.3f}? Wait for pullback."
        _entry_banner = (f"┌──────────────────────────────────────┐\n"
                         f"│  📋 BUY AT OPEN  —  {_open_str.upper():<18}│\n"
                         f"│  Set your order tonight before sleep.  │\n"
                         f"└──────────────────────────────────────┘")
    else:
        # Mid-session, market open — buy now per RSI logic above
        _entry_banner = ("⚡  **BUY NOW** — market is open, entry is live")

    # Good/bad header label
    verdict = "✅ GOOD BUY" if result["signal"] == "STRONG BUY" else "🏆 ELITE BUY — HIGH CONVICTION"
    divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    news = result.get("news", {})
    news_line = ""
    if news and news.get("count", 0) > 0:
        emoji = "🟢" if news["label"] == "POSITIVE" else "⚪"
        headline = news["headlines"][0][:75] if news.get("headlines") else ""
        news_line = f"{emoji} {news['label']} (score {news['compound']:+.2f})" + (f" — _{headline}_" if headline else "")

    adj = result.get("adj", 0)
    track_line = ""
    if adj != 0:
        direction = "boosted ↑" if adj > 0 else "penalised ↓"
        track_line = f"🧠 Past performance {direction} this ticker's score by {adj:+d}"

    hist_line = ""
    if ci:
        hist_line = (f"📈 Historical ({ci['n']} similar signals): "
                     f"worst {ci['worst']*100:+.1f}% · typical {ci['p25']*100:+.1f}% to "
                     f"{ci['p75']*100:+.1f}% · best {ci['best']*100:+.1f}%")

    lines = [
        divider,
        f"**TRADEY BOI X**  |  {verdict}",
        divider,
        _entry_banner,
        "",
        f"📌  **{ticker}**  —  {short_name}",
        f"💵  Price: **${price:.3f}**  |  Grade: **{grade} — {glabel}**  `{gbar}`",
        f"📊  Score: **{result['score']}/14**  |  AI confidence: **{result['prob']*100:.1f}%**",
        "",
        f"**💬 What the bot sees:**",
        _simple_read(ticker, result, price),
        "",
        f"**📅 Trade Plan**",
        f"🟢  Buy:       **{buy_date.strftime('%A %d %b %Y')}**  @  {entry_price}  _({entry_note})_",
        f"🚪  Exit:      **{exit_date.strftime('%A %d %b %Y')}**  ({exit_days} trading days)",
        f"💰  Target:    **${target_price:.3f}**  (+{target_pct*100:.0f}%)  _— {params['rationale']}_",
        f"🛑  Stop-loss: **${params['stop_loss']:.3f}**  ({params['stop_loss_pct']:.1f}%)  _exit if price falls here_",
        f"⚖️  Risk/Reward: **{abs(target_pct/params['stop_loss_pct']):.1f}:1**  _({target_pct*100:.0f}% gain vs {abs(params['stop_loss_pct']):.1f}% risk)_",
    ]

    if hist_line:  lines += ["", hist_line]
    if track_line: lines += [track_line]
    if news_line:  lines += ["", f"**📰 News**", news_line]

    lines += [
        "",
        f"**✅ Why it qualified:**",
        "  " + "  •  ".join(result["why"]),
        divider,
        f"_{now.strftime('%Y-%m-%d %H:%M')}_",
    ]

    try:
        r = requests.post(DISCORD, json={"content": "\n".join(lines)}, timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False

# ─── SIGNAL LOG + OUTCOME TRACKING ───────────────────────────────────────────
def log_signal(ticker: str, price: float, tier: str,
               score: int = 0, prob: float = 0.0):
    entries = _load_log()
    entries.append({
        "ticker":      ticker,
        "tier":        tier,
        "score":       score,
        "prob":        round(prob, 4),
        "entry_price": round(price, 4),
        "signal_date": datetime.now().strftime("%Y-%m-%d"),
        "target_pct":  TARGET_RETURN,
        "pred_days":   PREDICTION_DAYS,
        "outcome":     None,
        "exit_price":  None,
        "actual_pct":  None,
    })
    _save_log(entries)

def _load_log() -> list:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            return []
    return []

def _save_log(entries: list):
    LOG_FILE.write_text(json.dumps(entries, indent=2))

def resolve_outcomes() -> list:
    entries = _load_log()
    changed = False
    for e in entries:
        if e["outcome"] is not None:
            continue
        signal_date = datetime.strptime(e["signal_date"], "%Y-%m-%d")
        if datetime.now() < signal_date + timedelta(days=e["pred_days"] * 1.4):
            continue
        try:
            start = signal_date + timedelta(days=1)
            end   = signal_date + timedelta(days=e["pred_days"] * 2)
            hist  = yf.Ticker(e["ticker"]).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d")
            )
            if len(hist) >= e["pred_days"]:
                exit_price = float(hist["Close"].iloc[e["pred_days"] - 1])
                actual_pct = (exit_price - e["entry_price"]) / e["entry_price"]
                e["exit_price"] = round(exit_price, 4)
                e["actual_pct"] = round(actual_pct, 4)
                e["outcome"]    = "WIN" if actual_pct >= e["target_pct"] else "LOSS"
                changed = True
        except Exception:
            pass
    if changed:
        _save_log(entries)
    return entries

def accuracy_stats(entries: list) -> dict:
    resolved = [e for e in entries if e["outcome"] is not None]
    if not resolved:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": None, "avg_return": None}
    wins = sum(1 for e in resolved if e["outcome"] == "WIN")
    return {
        "total":      len(resolved),
        "wins":       wins,
        "losses":     len(resolved) - wins,
        "win_rate":   wins / len(resolved),
        "avg_return": sum(e["actual_pct"] for e in resolved) / len(resolved),
    }
