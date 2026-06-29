import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta
import requests
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Tradey Boi X", page_icon="📈", layout="wide")
st.title("📈 Tradey Boi X")

WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT",
    "BHP.AX", "CBA.AX", "FMG.AX", "RIO.AX",
    "NST.AX", "CXO.AX", "LTR.AX", "PDN.AX"
]

# ─────────────────────────────────────────────
# DISCORD
# ─────────────────────────────────────────────
DISCORD_WEBHOOK = os.getenv("discordwebhook", "")

def send_discord(message: str) -> bool:
    if not DISCORD_WEBHOOK:
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    df["rsi"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd              = ta.trend.MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    bb                = ta.volatility.BollingerBands(close)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / close

    df["atr"]         = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["ema20"]       = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"]       = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    df["vol_ma20"]    = vol.rolling(20).mean()
    df["vol_ratio"]   = vol / df["vol_ma20"]

    df["ret_5"]       = close.pct_change(5)
    df["ret_10"]      = close.pct_change(10)
    df["ret_20"]      = close.pct_change(20)

    high_52w          = close.rolling(252).max()
    df["breakout"]    = (close >= high_52w * 0.98).astype(int)

    return df.dropna()

# ─────────────────────────────────────────────
# AI MODEL
# ─────────────────────────────────────────────
FEATURES = [
    "rsi", "macd_diff", "bb_width", "atr",
    "ret_5", "ret_10", "ret_20",
    "vol_ratio", "breakout"
]

@st.cache_resource(show_spinner="Training AI model...")
def train_model(train_ticker="AAPL"):
    df = yf.Ticker(train_ticker).history(period="2y")
    df = add_indicators(df)
    df["future_ret"] = df["Close"].shift(-10) / df["Close"] - 1
    df["target"]     = (df["future_ret"] > 0.05).astype(int)
    df = df.dropna()
    X, y = df[FEATURES], df["target"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight="balanced"
        ))
    ])
    pipe.fit(X, y)
    return pipe

# ─────────────────────────────────────────────
# DECISION HIERARCHY (scored, tiered)
# ─────────────────────────────────────────────
# Each criterion contributes points. The total score determines the tier.
# Only ELITE and STRONG BUY are sent as Discord alerts.
#
# Criteria                      Points
# ─────────────────────────────────────
# AI probability >= 0.80          +3
# AI probability >= 0.70          +2
# AI probability >= 0.60          +1
# Breakout (near 52w high)        +3
# MACD diff > 0 (bullish)         +2
# RSI 35–65 (optimal zone)        +2
# RSI < 70 (not overbought)       +1
# Volume ratio > 1.5 (surge)      +2
# EMA20 > EMA50 (uptrend)         +1
# ─────────────────────────────────────
# Score >= 9  →  🏆 ELITE          alert
# Score >= 7  →  ✅ STRONG BUY     alert
# Score >= 5  →  👀 WATCH          no alert
# Score  < 5  →  ⛔ IGNORE         no alert

TIER_ELITE      = ("🏆 ELITE",      "#00cc44", 9, True)
TIER_STRONG_BUY = ("✅ STRONG BUY", "#44bb00", 7, True)
TIER_WATCH      = ("👀 WATCH",      "#e6a817", 5, False)
TIER_IGNORE     = ("⛔ IGNORE",     "#cc3300", 0, False)

def score_stock(prob, rsi, macd_diff, breakout, vol_ratio, ema20, ema50):
    s = 0
    if prob >= 0.80: s += 3
    elif prob >= 0.70: s += 2
    elif prob >= 0.60: s += 1

    if breakout:       s += 3
    if macd_diff > 0:  s += 2
    if 35 <= rsi <= 65: s += 2
    elif rsi < 70:     s += 1
    if vol_ratio > 1.5: s += 2
    if ema20 > ema50:  s += 1
    return s

def get_tier(score):
    if score >= TIER_ELITE[2]:      return TIER_ELITE
    if score >= TIER_STRONG_BUY[2]: return TIER_STRONG_BUY
    if score >= TIER_WATCH[2]:      return TIER_WATCH
    return TIER_IGNORE

def build_alert_msg(ticker, price, prob, rsi, macd_diff, breakout, vol_ratio, score, tier_name):
    reasons = []
    if prob >= 0.80:    reasons.append(f"AI confidence {prob*100:.0f}%")
    elif prob >= 0.70:  reasons.append(f"AI confidence {prob*100:.0f}%")
    if breakout:        reasons.append("52-week breakout")
    if macd_diff > 0:   reasons.append("MACD bullish")
    if 35 <= rsi <= 65: reasons.append(f"RSI in ideal zone ({rsi:.0f})")
    if vol_ratio > 1.5: reasons.append(f"volume surge {vol_ratio:.1f}x")

    lines = [
        f"**TRADEY BOI X** | {tier_name}",
        f"**{ticker}** @ ${price:.2f}",
        f"Score: {score}/14 | AI: {prob*100:.1f}%",
        "Why: " + ", ".join(reasons) if reasons else "",
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_"
    ]
    return "\n".join(l for l in lines if l)

# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def plot_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price", "RSI", "MACD")
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA 20",
        line=dict(color="orange", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA 50",
        line=dict(color="blue", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
        line=dict(color="gray", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
        line=dict(color="gray", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
        line=dict(color="purple", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=35, line_dash="dot", line_color="lightgreen", row=2, col=1)
    fig.add_hline(y=65, line_dash="dot", line_color="lightsalmon", row=2, col=1)

    colors = ["green" if v >= 0 else "red" for v in df["macd_diff"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="MACD Hist",
        marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
        line=dict(color="blue", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
        line=dict(color="orange", width=1)), row=3, col=1)

    fig.update_layout(height=620, showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0))
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    selected  = st.selectbox("Select Stock", WATCHLIST)
    period    = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    st.divider()
    run_scan  = st.button("🔍 Scan Watchlist", use_container_width=True)
    st.divider()
    st.subheader("Alert Tiers")
    st.markdown("""
| Score | Tier | Alert? |
|-------|------|--------|
| 9–14 | 🏆 ELITE | ✅ Yes |
| 7–8 | ✅ STRONG BUY | ✅ Yes |
| 5–6 | 👀 WATCH | ❌ No |
| 0–4 | ⛔ IGNORE | ❌ No |
""")
    webhook_set = bool(DISCORD_WEBHOOK)
    st.caption("Discord: " + ("✅ Connected" if webhook_set else "❌ Set `discordwebhook` env var"))

# ─────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────
model = train_model()

@st.cache_data(ttl=300, show_spinner="Fetching data...")
def get_data(ticker, period):
    df = yf.Ticker(ticker).history(period=period)
    return add_indicators(df)

df = get_data(selected, period)

# ─────────────────────────────────────────────
# SINGLE STOCK VIEW
# ─────────────────────────────────────────────
if df.empty:
    st.error(f"No data found for {selected}.")
else:
    l    = df.iloc[-1]
    prob = model.predict_proba([l[FEATURES].values])[0][1]
    sc   = score_stock(prob, l["rsi"], l["macd_diff"], l["breakout"],
                       l["vol_ratio"], l["ema20"], l["ema50"])
    tier_name, tier_color, _, alertable = get_tier(sc)

    prev_close = df["Close"].iloc[-2]
    pct_change = (l["Close"] - prev_close) / prev_close * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Price",      f"${l['Close']:.2f}", f"{pct_change:+.2f}%")
    col2.metric("AI Prob",    f"{prob*100:.1f}%")
    col3.metric("RSI",        f"{l['rsi']:.1f}")
    col4.metric("MACD",       f"{l['macd_diff']:.3f}")
    col5.metric("Vol Ratio",  f"{l['vol_ratio']:.2f}x")
    col6.metric("Score",      f"{sc}/14")

    st.markdown(
        f"### Signal: <span style='color:{tier_color}; font-size:1.3em'>"
        f"{tier_name}</span>"
        + (" &nbsp;🔔 Alert-worthy" if alertable else ""),
        unsafe_allow_html=True
    )

    if alertable:
        msg = build_alert_msg(selected, l["Close"], prob, l["rsi"],
                              l["macd_diff"], l["breakout"], l["vol_ratio"], sc, tier_name)
        if st.button(f"📣 Send Discord Alert for {selected}"):
            if send_discord(msg):
                st.success("Alert sent to Discord!")
            else:
                st.warning("Discord webhook not set or failed. Add `discordwebhook` to your Secrets.")

    st.plotly_chart(plot_chart(df, selected), use_container_width=True)

    with st.expander("📊 Score Breakdown"):
        breakdown = {
            "AI prob ≥ 0.80 (+3)":       prob >= 0.80,
            "AI prob ≥ 0.70 (+2)":       0.70 <= prob < 0.80,
            "AI prob ≥ 0.60 (+1)":       0.60 <= prob < 0.70,
            "Breakout near 52w high (+3)": bool(l["breakout"]),
            "MACD bullish (+2)":          l["macd_diff"] > 0,
            "RSI in ideal zone 35–65 (+2)": 35 <= l["rsi"] <= 65,
            "RSI < 70 not overbought (+1)": l["rsi"] < 70,
            "Volume surge > 1.5x (+2)":   l["vol_ratio"] > 1.5,
            "EMA20 > EMA50 uptrend (+1)": l["ema20"] > l["ema50"],
        }
        for crit, met in breakdown.items():
            st.write(("✅ " if met else "❌ ") + crit)

# ─────────────────────────────────────────────
# WATCHLIST SCANNER
# ─────────────────────────────────────────────
if run_scan:
    st.divider()
    st.subheader("🔍 Watchlist Scan Results")

    rows         = []
    auto_alerted = []
    prog         = st.progress(0)

    for i, ticker in enumerate(WATCHLIST):
        try:
            d = get_data(ticker, "6mo")
            if len(d) < 50:
                continue
            ll   = d.iloc[-1]
            p    = model.predict_proba([ll[FEATURES].values])[0][1]
            sc_t = score_stock(p, ll["rsi"], ll["macd_diff"], ll["breakout"],
                               ll["vol_ratio"], ll["ema20"], ll["ema50"])
            t_name, t_color, _, alertable = get_tier(sc_t)

            rows.append({
                "Ticker":   ticker,
                "Price":    round(ll["Close"], 2),
                "AI %":     f"{p*100:.1f}%",
                "RSI":      round(ll["rsi"], 1),
                "Vol Ratio":round(ll["vol_ratio"], 2),
                "Breakout": "✅" if ll["breakout"] else "—",
                "Score":    sc_t,
                "Signal":   t_name,
                "Alert?":   "📣" if alertable else "—"
            })

            if alertable:
                msg = build_alert_msg(ticker, ll["Close"], p, ll["rsi"],
                                      ll["macd_diff"], ll["breakout"],
                                      ll["vol_ratio"], sc_t, t_name)
                sent = send_discord(msg)
                auto_alerted.append((ticker, t_name, sent))

        except Exception:
            pass
        prog.progress((i + 1) / len(WATCHLIST))

    prog.empty()

    if rows:
        results_df = (pd.DataFrame(rows)
                      .sort_values("Score", ascending=False)
                      .reset_index(drop=True))
        st.dataframe(results_df, use_container_width=True)

        if auto_alerted:
            st.subheader("📣 Auto-Alerts Sent")
            for ticker, t_name, sent in auto_alerted:
                status = "✅ sent" if sent else "⚠️ webhook not set"
                st.write(f"**{ticker}** — {t_name} — {status}")
        else:
            st.info("No alert-worthy setups found in this scan (score < 7).")
    else:
        st.warning("No results returned.")
