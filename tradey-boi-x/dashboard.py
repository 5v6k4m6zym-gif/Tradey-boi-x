import os
import requests
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT",
    "BHP.AX", "CBA.AX", "FMG.AX", "RIO.AX",
    "NST.AX", "CXO.AX", "LTR.AX", "PDN.AX",
]
FEATURES       = ["rsi", "macd_diff", "bb_width", "atr", "ret_5", "ret_10", "ret_20", "vol_ratio", "breakout"]
COOLDOWN_HOURS = 8
MAX_ALERTS     = 3          # per scan session
ALERT_MIN_PROB = 0.65       # AI gate
ALERT_MIN_RSI  = (30, 70)   # RSI gate  (lo, hi)
DISCORD        = os.getenv("discordwebhook", "")

# ─── STEP 1 — DATA → FEATURES ────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Fetching…")
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
@st.cache_resource(show_spinner="Training model…")
def train_model() -> Pipeline:
    df = get_data("AAPL", "2y")
    df = df.copy()
    df["target"] = (df["Close"].shift(-10) / df["Close"] - 1 > 0.05).astype(int)
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
# All filtering and scoring live here — nowhere else.
#
# Returns a plain dict:
#   signal   str    "ELITE" | "STRONG BUY" | "WATCH" | "IGNORE" | "GATED"
#   label    str    emoji label for UI
#   color    str    hex colour
#   alert    bool   True only when signal qualifies AND cooldown clear
#   prob     float  raw AI probability
#   score    int    0-10 points (0 if gated)
#   why      list   human-readable reasons for the score
#   filters  list   (name, passed) for each hard filter

def _cooldowns() -> dict:
    if "cd" not in st.session_state:
        st.session_state["cd"] = {}
    return st.session_state["cd"]

def decide(ticker: str, df: pd.DataFrame, model: Pipeline) -> dict:
    GATED = {"signal": "GATED", "label": "🚫 GATED", "color": "#888",
              "alert": False, "prob": 0.0, "score": 0, "why": [], "filters": []}

    if len(df) < 60:
        return {**GATED, "filters": [("Enough data (≥60 rows)", False)]}

    row  = df.iloc[-1]
    prev = df.iloc[-2]
    prob = float(model.predict_proba([row[FEATURES].values])[0][1])

    # Hard filters — all must pass or signal = GATED
    filters = [
        ("Uptrend: EMA20 > EMA50",               row["ema20"] > row["ema50"]),
        ("Confirmed: EMA20 > EMA50 prior day",    prev["ema20"] > prev["ema50"]),
        ("MACD bullish (diff > 0)",               row["macd_diff"] > 0),
        ("MACD bullish prior day",                prev["macd_diff"] > 0),
        ("RSI not overbought (< 72)",             row["rsi"] < 72),
        ("RSI not oversold (> 25)",               row["rsi"] > 25),
        ("Liquidity (vol ratio ≥ 0.5)",           row["vol_ratio"] >= 0.5),
        ("AI probability ≥ 55%",                  prob >= 0.55),
    ]
    if not all(ok for _, ok in filters):
        return {**GATED, "filters": filters}

    # Scoring — weighted points, one place, no duplicates
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
    score = sum(pts for pts, _, met in rules if met)
    why   = [name for _, name, met in rules if met]

    # Single tier lookup
    if   score >= 11: signal, label, color, qualifies = "ELITE",      "🏆 ELITE",       "#00cc44", True
    elif score >= 8:  signal, label, color, qualifies = "STRONG BUY", "✅ STRONG BUY",  "#44bb00", True
    elif score >= 5:  signal, label, color, qualifies = "WATCH",      "👀 WATCH",       "#e6a817", False
    else:             signal, label, color, qualifies = "IGNORE",     "⛔ IGNORE",      "#cc3300", False

    cd    = _cooldowns()
    cd_ok = ticker not in cd or datetime.now() - cd[ticker] > timedelta(hours=COOLDOWN_HOURS)

    return {
        "signal":  signal,
        "label":   label,
        "color":   color,
        "alert":   qualifies and cd_ok,
        "prob":    prob,
        "score":   score,
        "why":     why,
        "filters": filters,
    }

# ─── STEP 4 — DISCORD ALERT ──────────────────────────────────────────────────
def send_alert(ticker: str, result: dict, price: float) -> bool:
    if not DISCORD:
        return False
    msg = "\n".join([
        f"**TRADEY BOI X** | {result['label']}",
        f"**{ticker}** @ ${price:.2f}",
        f"Score {result['score']}/14 | AI {result['prob']*100:.1f}%",
        "Why: " + ", ".join(result["why"]),
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
    ])
    try:
        r = requests.post(DISCORD, json={"content": msg}, timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False

def mark_alerted(ticker: str):
    _cooldowns()[ticker] = datetime.now()

# ─── UI HELPERS ──────────────────────────────────────────────────────────────
def chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
                        subplot_titles=(f"{ticker} Price", "RSI (14)", "MACD"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"]), row=1, col=1)
    for y, name, color in [(df["ema20"], "EMA20", "orange"), (df["ema50"], "EMA50", "royalblue")]:
        fig.add_trace(go.Scatter(x=df.index, y=y, name=name, line=dict(color=color, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(color="gray", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(color="gray", dash="dot", width=1),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color="purple", width=1.5)), row=2, col=1)
    for lvl, dash in [(70, "dash"), (30, "dash"), (65, "dot"), (35, "dot")]:
        fig.add_hline(y=lvl, line_dash=dash, line_color="red" if lvl >= 65 else "green", row=2, col=1)

    colors = ["green" if v >= 0 else "red" for v in df["macd_diff"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color="royalblue", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], line=dict(color="orange", width=1)), row=3, col=1)

    fig.update_layout(height=600, showlegend=False, xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tradey Boi X", page_icon="📈", layout="wide")
st.title("📈 Tradey Boi X")

with st.sidebar:
    st.header("Settings")
    selected = st.selectbox("Stock", WATCHLIST)
    period   = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    st.divider()
    run_scan = st.button("🔍 Scan Watchlist", use_container_width=True)
    st.divider()
    st.caption("**Alert tiers:** ELITE (≥11) · STRONG BUY (≥8)")
    st.caption(f"Cooldown: {COOLDOWN_HOURS}h · Max alerts/scan: {MAX_ALERTS}")
    st.caption("Discord: " + ("✅ Connected" if DISCORD else "❌ Set `discordwebhook` secret"))

model = train_model()

# ── Single stock view ─────────────────────────────────────────────────────────
df = get_data(selected, period)

if df.empty:
    st.error(f"No data for {selected}.")
else:
    res   = decide(selected, df, model)
    row   = df.iloc[-1]
    chg   = (row["Close"] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",     f"${row['Close']:.2f}",      f"{chg:+.2f}%")
    c2.metric("AI Prob",   f"{res['prob']*100:.1f}%")
    c3.metric("RSI",       f"{row['rsi']:.1f}")
    c4.metric("MACD",      f"{row['macd_diff']:.4f}")
    c5.metric("Vol Ratio", f"{row['vol_ratio']:.2f}×")
    c6.metric("Score",     f"{res['score']}/14")

    st.markdown(f"### <span style='color:{res['color']}'>{res['label']}</span>",
                unsafe_allow_html=True)

    if res["alert"]:
        if st.button(f"📣 Send Discord Alert — {selected}"):
            if send_alert(selected, res, row["Close"]):
                mark_alerted(selected)
                st.success("Alert sent!")
            else:
                st.warning("Webhook missing or failed.")
    elif res["signal"] == "GATED":
        st.info("Failed hard filters — see breakdown below.")
    else:
        cd = _cooldowns()
        if selected in cd:
            eta = cd[selected] + timedelta(hours=COOLDOWN_HOURS)
            st.info(f"Cooldown active until {eta.strftime('%H:%M')}.")

    st.plotly_chart(chart(df, selected), use_container_width=True)

    with st.expander("Filter & Score Breakdown"):
        st.write("**Hard Filters**")
        for name, passed in res["filters"]:
            st.write(("✅ " if passed else "❌ ") + name)
        if res["signal"] != "GATED":
            st.write("**Score**")
            for pts, name, met in [(pts, n, m) for pts, n, m in [
                (3,"AI prob ≥ 80%",       res["prob"]>=0.80),
                (2,"AI prob ≥ 70%",       0.70<=res["prob"]<0.80),
                (1,"AI prob ≥ 60%",       0.60<=res["prob"]<0.70),
                (3,"52-week breakout",    bool(row["breakout"])),
                (2,"Volume surge >1.5×",  row["vol_ratio"]>1.5),
                (2,"RSI ideal 35–65",     35<=row["rsi"]<=65),
                (1,"RSI safe < 70",       row["rsi"]<70),
                (1,"EMA uptrend",         row["ema20"]>row["ema50"]),
            ]]:
                st.write(f"{'✅' if met else '—'} `{'+'if met else ' '}{pts if met else 0}` {name}")

# ── Watchlist scan ────────────────────────────────────────────────────────────
if run_scan:
    st.divider()
    st.subheader("🔍 Scan Results")

    rows, fired, prog = [], 0, st.progress(0)

    for i, ticker in enumerate(WATCHLIST):
        try:
            d = get_data(ticker, "6mo")
            if d.empty:
                continue
            r   = decide(ticker, d, model)
            ll  = d.iloc[-1]
            alerted = False

            if r["alert"] and fired < MAX_ALERTS:
                alerted = send_alert(ticker, r, ll["Close"])
                if alerted:
                    mark_alerted(ticker)
                fired += 1

            rows.append({
                "Ticker":    ticker,
                "Price":     round(ll["Close"], 2),
                "AI %":      f"{r['prob']*100:.1f}%",
                "RSI":       round(ll["rsi"], 1),
                "Vol ×":     round(ll["vol_ratio"], 2),
                "Breakout":  "✅" if ll["breakout"] else "—",
                "Score":     r["score"] if r["signal"] != "GATED" else "—",
                "Signal":    r["label"],
                "Alert":     "📣" if alerted else ("⏳" if not r["alert"] and r["signal"] not in ("GATED","IGNORE","WATCH") else "—"),
            })
        except Exception:
            pass
        prog.progress((i + 1) / len(WATCHLIST))

    prog.empty()

    if rows:
        out = pd.DataFrame(rows)
        out["_s"] = out["Score"].apply(lambda v: v if isinstance(v, int) else -1)
        st.dataframe(out.sort_values("_s", ascending=False).drop(columns="_s").reset_index(drop=True),
                     use_container_width=True)
        if fired >= MAX_ALERTS:
            st.caption(f"Alert cap ({MAX_ALERTS}) reached — lower-ranked signals suppressed.")
