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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT",
    "BHP.AX", "CBA.AX", "FMG.AX", "RIO.AX",
    "NST.AX", "CXO.AX", "LTR.AX", "PDN.AX",
]
FEATURES        = ["rsi", "macd_diff", "bb_width", "atr", "ret_5", "ret_10", "ret_20", "vol_ratio", "breakout"]
PREDICTION_DAYS = 10
TARGET_RETURN   = 0.05
COOLDOWN_HOURS  = 8
MAX_ALERTS      = 3
DISCORD         = os.getenv("Discordwebhook", "") or os.getenv("discordwebhook", "")

BASE_DIR        = Path(__file__).parent
LOG_FILE        = BASE_DIR / "signal_log.json"
COOLDOWN_FILE   = BASE_DIR / "cooldowns.json"

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
    return df.dropna()

# ─── STEP 2 — AI MODEL ───────────────────────────────────────────────────────
def train_model() -> Pipeline:
    df = get_data("AAPL", "2y").copy()
    df["target"] = (df["Close"].shift(-PREDICTION_DAYS) / df["Close"] - 1 > TARGET_RETURN).astype(int)
    df = df.dropna()
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        )),
    ])
    pipe.fit(df[FEATURES], df["target"])
    return pipe

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
        ("Uptrend: EMA20 > EMA50",            row["ema20"]  > row["ema50"]),
        ("Confirmed: EMA20 > EMA50 prior day", prev["ema20"] > prev["ema50"]),
        ("MACD bullish (diff > 0)",            row["macd_diff"]  > 0),
        ("MACD bullish prior day",             prev["macd_diff"] > 0),
        ("RSI not overbought (< 72)",          row["rsi"] < 72),
        ("RSI not oversold (> 25)",            row["rsi"] > 25),
        ("Liquidity (vol ratio ≥ 0.5)",        row["vol_ratio"] >= 0.5),
        ("AI probability ≥ 55%",              prob >= 0.55),
    ]
    if not all(ok for _, ok in filters):
        return {**GATED, "filters": filters}

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
        return {**GATED, "filters": filters, "news": news}

    score = base_score + adj + news_adj

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
            "adj": adj, "news": news, "why": why, "filters": filters}

# ─── NEWS SENTIMENT ──────────────────────────────────────────────────────────
def news_sentiment(ticker: str) -> dict:
    """
    Fetches recent headlines via yfinance and scores them with VADER.
    Returns:
      compound   – avg VADER compound score (-1 to +1)
      label      – 'POSITIVE' / 'NEGATIVE' / 'NEUTRAL'
      score_adj  – +1 (strong positive), -1 (strong negative), 0 (neutral)
      headlines  – list of scored headline strings (for alert display)
      count      – number of headlines analysed
    Falls back gracefully if no news is available.
    """
    try:
        articles = yf.Ticker(ticker).news or []
        headlines = []
        for a in articles[:10]:
            title = (a.get("content") or {}).get("title") or a.get("title") or ""
            if title:
                headlines.append(title)
        if not headlines:
            return {"compound": 0.0, "label": "NEUTRAL", "score_adj": 0,
                    "headlines": [], "count": 0}
        scores = [_vader.polarity_scores(h)["compound"] for h in headlines]
        avg = sum(scores) / len(scores)
        if   avg >  0.20: label, adj = "POSITIVE", +1
        elif avg < -0.20: label, adj = "NEGATIVE", -1
        else:             label, adj = "NEUTRAL",   0
        top = sorted(zip(scores, headlines), reverse=True)
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
def send_alert(ticker: str, result: dict, price: float) -> bool:
    if not DISCORD:
        return False
    target_date = (datetime.now() + timedelta(days=PREDICTION_DAYS * 1.4)).strftime("%d %b %Y")

    ci = confidence_interval(result["signal"])
    grade, glabel, gbar = confidence_grade(result["prob"], result["score"])

    lines = [
        f"**TRADEY BOI X** | {result['label']}",
        f"**{ticker}** @ ${price:.2f}",
        f"🎯 Confidence: **{grade} — {glabel}**  `{gbar}`",
        f"Score {result['score']}/14 | AI {result['prob']*100:.1f}%",
        f"⏱ Timeframe: {PREDICTION_DAYS} trading days (by ~{target_date})",
        f"🎯 Target: +{TARGET_RETURN*100:.0f}% gain",
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
