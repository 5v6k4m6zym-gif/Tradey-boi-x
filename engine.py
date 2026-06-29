"""
Core engine — pure Python, no Streamlit.
Imported by both dashboard.py and scanner.py.
"""
import json
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import ta
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier

_vader = SentimentIntensityAnalyzer()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT",
    "BHP.AX", "CBA.AX", "FMG.AX", "RIO.AX",
    "NST.AX", "CXO.AX", "LTR.AX", "PDN.AX",
]
FEATURES        = ["rsi", "macd_diff", "bb_width", "atr", "ret_5", "ret_10", "ret_20", "vol_ratio", "breakout", "obv_ratio"]
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
    return df.dropna()

# ─── STEP 2 — AI MODEL ───────────────────────────────────────────────────────
def train_model() -> Pipeline:
    """Train XGBoost on ALL watchlist tickers (12× more data than single-ticker)."""
    frames = []
    for ticker in WATCHLIST:
        try:
            df = get_data(ticker, "2y").copy()
            df["target"] = (df["Close"].shift(-PREDICTION_DAYS) / df["Close"] - 1 > TARGET_RETURN).astype(int)
            frames.append(df.dropna())
        except Exception:
            pass
    if not frames:
        frames = [get_data("AAPL", "2y")]
    combined = pd.concat(frames, ignore_index=True)
    neg = int((combined["target"] == 0).sum())
    pos = int((combined["target"] == 1).sum())
    spw = round(neg / pos, 2) if pos > 0 else 1.0   # balance minority class
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, verbosity=0,
        )),
    ])
    pipe.fit(combined[FEATURES], combined["target"])
    print(f"  Model trained on {len(combined):,} rows ({pos} buy / {neg} no-buy) | scale_pos_weight={spw}")
    return pipe

# ─── MARKET REGIME, VIX, SECTOR, WEEKLY, EARNINGS ───────────────────────────
_regime_cache: dict = {}

# Sector ETF map for US tickers — ASX already covered by ^AXJO in market_regime_ok
SECTOR_ETF = {"AAPL": "XLK", "NVDA": "XLK", "MSFT": "XLK", "TSLA": "XLY"}

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
    Returns per-ticker score adjustments based on resolved signal outcomes.
    Needs ≥3 resolved signals per ticker before adjusting.
      win rate ≥ 65% → +1 (proven winner, lower bar)
      win rate ≤ 35% → -2 (consistent loser, need stronger signal)
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
        if len(results) < 3:
            continue
        wr = sum(results) / len(results)
        if   wr >= 0.65: adj[ticker] = +1
        elif wr <= 0.35: adj[ticker] = -2
    return adj

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

    # Short interest, insider buying, options flow
    short_adj,   short_why   = short_interest_signal(ticker)
    insider_adj, insider_why = insider_signal(ticker)
    opts_adj,    opts_why    = options_flow_signal(ticker)
    for reason in (short_why, insider_why, opts_why):
        if reason:
            why.append(reason)

    score = base_score + adj + news_adj + short_adj + insider_adj + opts_adj

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
    """Returns False if we already sent an alert for this ticker within the window."""
    try:
        data = json.loads(SEND_GUARD_FILE.read_text()) if SEND_GUARD_FILE.exists() else {}
        last = data.get(ticker, 0)
        if (datetime.now().timestamp() - last) < window_seconds:
            return False
        data[ticker] = datetime.now().timestamp()
        SEND_GUARD_FILE.write_text(json.dumps(data))
        return True
    except Exception:
        return True

def send_alert(ticker: str, result: dict, price: float) -> bool:
    if not DISCORD:
        return False
    if not _guard_ok(ticker):
        print(f"  ⏭ {ticker}: duplicate suppressed (sent within last 90s)")
        return False
    target_price = price * (1 + TARGET_RETURN)
    target_date  = (datetime.now() + timedelta(days=PREDICTION_DAYS * 1.4)).strftime("%d %b %Y")

    # Suggested entry based on RSI — how aggressively to chase the entry
    rsi = result.get("rsi", 50)
    if rsi < 40:
        entry_note = f"${price:.2f} (buy now — RSI oversold, ideal entry)"
    elif rsi < 50:
        entry_note = f"${price:.2f}–${price * 0.99:.2f} (now or on a small dip)"
    elif rsi < 60:
        entry_note = f"${price * 0.99:.2f}–${price * 0.985:.2f} (wait for 1–1.5% pullback)"
    else:
        entry_note = f"${price * 0.98:.2f} (RSI elevated — wait for 2% dip)"

    ci = confidence_interval(result["signal"])
    grade, glabel, gbar = confidence_grade(result["prob"], result["score"])

    lines = [
        f"**TRADEY BOI X** | {result['label']}",
        f"**{ticker}** @ ${price:.2f}",
        f"🎯 Confidence: **{grade} — {glabel}**  `{gbar}`",
        f"Score {result['score']}/14 | AI {result['prob']*100:.1f}%",
        f"🟢 Entry: {entry_note}",
        f"💰 Target price: ${target_price:.2f} (+{TARGET_RETURN*100:.0f}%) by {target_date}",
        f"⏱ Timeframe: {PREDICTION_DAYS} trading days from today",
    ]

    if ci:
        lines.append(
            f"📈 Historical range ({ci['n']} similar): "
            f"worst {ci['worst']*100:+.1f}% / typical {ci['p25']*100:+.1f}% to {ci['p75']*100:+.1f}% / best {ci['best']*100:+.1f}%"
        )
    if result.get("adj", 0) != 0:
        direction = "boosted" if result["adj"] > 0 else "penalised"
        lines.append(f"🧠 AI-{direction} ticker (base score {result.get('base_score', result['score'])}, adj {result['adj']:+d})")

    news = result.get("news", {})
    if news and news.get("count", 0) > 0:
        emoji = "📰🟢" if news["label"] == "POSITIVE" else "📰⚪"
        lines.append(f"{emoji} News ({news['count']} headlines, sentiment {news['compound']:+.2f}): {news['headlines'][0][:80]}" if news["headlines"] else f"{emoji} News sentiment: {news['label']}")

    lines += [
        "Why: " + ", ".join(result["why"]),
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
    ]

    try:
        r = requests.post(DISCORD, json={"content": "\n".join(lines)}, timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False

# ─── SIGNAL LOG + OUTCOME TRACKING ───────────────────────────────────────────
def log_signal(ticker: str, price: float, tier: str):
    entries = _load_log()
    entries.append({
        "ticker":      ticker,
        "tier":        tier,
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
