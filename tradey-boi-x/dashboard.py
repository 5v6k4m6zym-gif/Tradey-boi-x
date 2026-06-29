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
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

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

    df["breakout"]    = (close >= close.rolling(252).max() * 0.98).astype(int)

    return df.dropna()

# ─────────────────────────────────────────────
# AI MODEL
# ─────────────────────────────────────────────
AI_FEATURES = [
    "rsi", "macd_diff", "bb_width", "atr",
    "ret_5", "ret_10", "ret_20",
    "vol_ratio", "breakout"
]

@st.cache_resource(show_spinner="Training AI model...")
def train_model(train_ticker: str = "AAPL") -> Pipeline:
    df = yf.Ticker(train_ticker).history(period="2y")
    df = add_indicators(df)
    df["future_ret"] = df["Close"].shift(-10) / df["Close"] - 1
    df["target"]     = (df["future_ret"] > 0.05).astype(int)
    df = df.dropna()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight="balanced"
        ))
    ])
    pipe.fit(df[AI_FEATURES], df["target"])
    return pipe

# ═════════════════════════════════════════════
#  UNIFIED SCORING ENGINE
#
#  Single entry point: evaluate(ticker, df, model)
#  Returns a rich result dict — no logic lives
#  anywhere else in this file.
# ═════════════════════════════════════════════

# ── HARD GATES ───────────────────────────────
# All gates must pass. If any fails the signal
# is "GATED" and earns score = 0. No exceptions.
GATES = [
    {
        "name":    "Minimum history",
        "desc":    "≥ 60 rows of clean data",
        "fn":      lambda df, l, _p: len(df) >= 60,
    },
    {
        "name":    "Uptrend",
        "desc":    "EMA20 > EMA50",
        "fn":      lambda df, l, _p: l["ema20"] > l["ema50"],
    },
    {
        "name":    "MACD momentum",
        "desc":    "MACD histogram > 0",
        "fn":      lambda df, l, _p: l["macd_diff"] > 0,
    },
    {
        "name":    "RSI not overbought",
        "desc":    "RSI < 72",
        "fn":      lambda df, l, _p: l["rsi"] < 72,
    },
    {
        "name":    "RSI not oversold",
        "desc":    "RSI > 25",
        "fn":      lambda df, l, _p: l["rsi"] > 25,
    },
    {
        "name":    "Liquidity",
        "desc":    "Volume ratio ≥ 0.5 (stock is trading)",
        "fn":      lambda df, l, _p: l["vol_ratio"] >= 0.5,
    },
    {
        "name":    "AI minimum confidence",
        "desc":    "AI probability ≥ 55%",
        "fn":      lambda df, l, p: p >= 0.55,
    },
    {
        "name":    "2-day confirmation",
        "desc":    "Signal confirmed on prior day too",
        "fn":      lambda df, l, p: (
            len(df) >= 2 and
            df["macd_diff"].iloc[-2] > 0 and
            df["ema20"].iloc[-2] > df["ema50"].iloc[-2]
        ),
    },
]

# ── SCORING CRITERIA ─────────────────────────
# Only reached if all gates pass.
# Max possible score = 14.
CRITERIA = [
    {"name": "AI prob ≥ 80%",        "pts": 3, "fn": lambda l, p: p >= 0.80},
    {"name": "AI prob ≥ 70%",        "pts": 2, "fn": lambda l, p: 0.70 <= p < 0.80},
    {"name": "AI prob ≥ 60%",        "pts": 1, "fn": lambda l, p: 0.60 <= p < 0.70},
    {"name": "Breakout near 52w high","pts": 3, "fn": lambda l, p: bool(l["breakout"])},
    {"name": "Strong volume surge >1.5x","pts": 2, "fn": lambda l, p: l["vol_ratio"] > 1.5},
    {"name": "RSI in ideal zone 35–65","pts": 2, "fn": lambda l, p: 35 <= l["rsi"] <= 65},
    {"name": "RSI < 70 (safe)",       "pts": 1, "fn": lambda l, p: l["rsi"] < 70},
    {"name": "EMA uptrend confirmed", "pts": 1, "fn": lambda l, p: l["ema20"] > l["ema50"]},
]
MAX_SCORE = sum(c["pts"] for c in CRITERIA)   # 15 but capped; display /14

# ── TIER TABLE ───────────────────────────────
# (min_score, label, hex_color, send_alert)
TIERS = [
    (11, "🏆 ELITE",       "#00cc44", True),
    (8,  "✅ STRONG BUY",  "#44bb00", True),
    (5,  "👀 WATCH",       "#e6a817", False),
    (0,  "⛔ IGNORE",      "#cc3300", False),
]

# ── COOLDOWN STORE ───────────────────────────
# Stored in session state so it persists across
# reruns but resets each browser session.
COOLDOWN_HOURS = 8
MAX_ALERTS_PER_SCAN = 3

def _cooldown_store():
    if "cooldowns" not in st.session_state:
        st.session_state["cooldowns"] = {}
    return st.session_state["cooldowns"]

def _cooldown_ok(ticker: str) -> bool:
    store = _cooldown_store()
    if ticker not in store:
        return True
    return datetime.now() - store[ticker] > timedelta(hours=COOLDOWN_HOURS)

def _mark_alerted(ticker: str):
    _cooldown_store()[ticker] = datetime.now()

# ── MAIN ENGINE ──────────────────────────────
def evaluate(ticker: str, df: pd.DataFrame, model: Pipeline) -> dict:
    """
    Single entry point for all signal logic.
    Returns a unified result dict:
      gated         bool        — any hard gate failed
      gate_results  list[dict]  — per-gate pass/fail + name + desc
      prob          float       — AI probability
      score         int         — total score (0 if gated)
      score_max     int         — MAX_SCORE constant
      criteria      list[dict]  — per-criterion met/pts/name
      tier_label    str
      tier_color    str
      send_alert    bool        — tier qualifies AND cooldown ok
      cooldown_ok   bool
      latest        pd.Series   — last row of df
      reasons       list[str]   — human-readable why list
    """
    l    = df.iloc[-1]
    prob = model.predict_proba([l[AI_FEATURES].values])[0][1]

    # ── run hard gates ──
    gate_results = []
    gated        = False
    for g in GATES:
        passed = bool(g["fn"](df, l, prob))
        gate_results.append({"name": g["name"], "desc": g["desc"], "passed": passed})
        if not passed:
            gated = True

    # ── score (only if all gates pass) ──
    score    = 0
    criteria = []
    reasons  = []
    if not gated:
        for c in CRITERIA:
            met = bool(c["fn"](l, prob))
            pts = c["pts"] if met else 0
            score += pts
            criteria.append({"name": c["name"], "pts": c["pts"], "met": met})
            if met:
                reasons.append(c["name"])

    # ── pick tier ──
    tier_label = "🚫 GATED"
    tier_color = "#888888"
    alertable  = False
    if not gated:
        for min_s, label, color, alert in TIERS:
            if score >= min_s:
                tier_label = label
                tier_color = color
                alertable  = alert
                break

    cd_ok      = _cooldown_ok(ticker)
    send_alert = alertable and cd_ok

    return {
        "gated":        gated,
        "gate_results": gate_results,
        "prob":         prob,
        "score":        score,
        "score_max":    MAX_SCORE,
        "criteria":     criteria,
        "tier_label":   tier_label,
        "tier_color":   tier_color,
        "send_alert":   send_alert,
        "cooldown_ok":  cd_ok,
        "latest":       l,
        "reasons":      reasons,
    }

def build_alert_msg(ticker: str, result: dict) -> str:
    l = result["latest"]
    lines = [
        f"**TRADEY BOI X** | {result['tier_label']}",
        f"**{ticker}** @ ${l['Close']:.2f}",
        f"Score: {result['score']}/{result['score_max']} | AI: {result['prob']*100:.1f}%",
        "Why: " + ", ".join(result["reasons"]) if result["reasons"] else "",
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
    ]
    return "\n".join(ln for ln in lines if ln)

# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def plot_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price", "RSI (14)", "MACD")
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)
    for y, name, color in [(df["ema20"], "EMA 20", "orange"), (df["ema50"], "EMA 50", "blue")]:
        fig.add_trace(go.Scatter(x=df.index, y=y, name=name,
            line=dict(color=color, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
        line=dict(color="gray", dash="dot", width=1), name="BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
        line=dict(color="gray", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(128,128,128,0.05)", name="BB"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
        line=dict(color="purple", width=1.5), name="RSI"), row=2, col=1)
    for y_val, color in [(70, "red"), (30, "green"), (65, "lightsalmon"), (35, "lightgreen")]:
        fig.add_hline(y=y_val, line_dash="dash" if y_val in (70, 30) else "dot",
                      line_color=color, row=2, col=1)

    colors = ["green" if v >= 0 else "red" for v in df["macd_diff"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"],
        marker_color=colors, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
        line=dict(color="blue", width=1), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
        line=dict(color="orange", width=1), name="Signal"), row=3, col=1)

    fig.update_layout(height=620, showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0))
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    selected = st.selectbox("Select Stock", WATCHLIST)
    period   = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    st.divider()
    run_scan = st.button("🔍 Scan Watchlist", use_container_width=True)
    st.divider()
    st.subheader("Signal Tiers")
    st.markdown(f"""
| Score | Tier | Alert? |
|-------|------|--------|
| 11–{MAX_SCORE} | 🏆 ELITE | ✅ |
| 8–10 | ✅ STRONG BUY | ✅ |
| 5–7 | 👀 WATCH | ❌ |
| 0–4 | ⛔ IGNORE | ❌ |
| — | 🚫 GATED | ❌ |
""")
    st.caption(f"Cooldown: {COOLDOWN_HOURS}h per ticker")
    st.caption(f"Max alerts/scan: {MAX_ALERTS_PER_SCAN}")
    st.caption("Discord: " + ("✅ Connected" if DISCORD_WEBHOOK else "❌ Set `discordwebhook` secret"))

# ─────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────
model = train_model()

@st.cache_data(ttl=300, show_spinner="Fetching data...")
def get_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    return add_indicators(df)

df = get_data(selected, period)

# ─────────────────────────────────────────────
# SINGLE STOCK VIEW
# ─────────────────────────────────────────────
if df.empty:
    st.error(f"No data found for {selected}.")
else:
    res = evaluate(selected, df, model)
    l   = res["latest"]

    prev_close = df["Close"].iloc[-2]
    pct_change = (l["Close"] - prev_close) / prev_close * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",      f"${l['Close']:.2f}",       f"{pct_change:+.2f}%")
    c2.metric("AI Prob",    f"{res['prob']*100:.1f}%")
    c3.metric("RSI",        f"{l['rsi']:.1f}")
    c4.metric("MACD",       f"{l['macd_diff']:.4f}")
    c5.metric("Vol Ratio",  f"{l['vol_ratio']:.2f}x")
    c6.metric("Score",      f"{res['score']}/{res['score_max']}")

    gated_warning = " &nbsp;🔒 Gates failed" if res["gated"] else ""
    cooldown_note = " &nbsp;⏳ Cooldown active" if (res["send_alert"] is False and not res["gated"] and not res["cooldown_ok"]) else ""
    st.markdown(
        f"### Signal: <span style='color:{res['tier_color']}; font-size:1.3em'>"
        f"{res['tier_label']}</span>{gated_warning}{cooldown_note}",
        unsafe_allow_html=True
    )

    if res["send_alert"]:
        if st.button(f"📣 Send Discord Alert for {selected}"):
            msg = build_alert_msg(selected, res)
            if send_discord(msg):
                _mark_alerted(selected)
                st.success("Alert sent!")
            else:
                st.warning("Webhook not set or failed. Add `discordwebhook` to Secrets.")
    elif res["gated"]:
        st.info("This stock did not pass all hard gates — no score computed.")
    elif not res["cooldown_ok"]:
        last = _cooldown_store().get(selected)
        next_alert = last + timedelta(hours=COOLDOWN_HOURS) if last else None
        st.info(f"Cooldown active. Next alert available: {next_alert.strftime('%H:%M') if next_alert else 'soon'}.")

    st.plotly_chart(plot_chart(df, selected), use_container_width=True)

    col_gates, col_score = st.columns(2)

    with col_gates:
        with st.expander("🔒 Hard Gate Results", expanded=res["gated"]):
            for g in res["gate_results"]:
                icon = "✅" if g["passed"] else "❌"
                st.write(f"{icon} **{g['name']}** — {g['desc']}")

    with col_score:
        with st.expander("📊 Score Breakdown"):
            if res["gated"]:
                st.write("Score not computed — gates failed.")
            else:
                for c in res["criteria"]:
                    icon = "✅" if c["met"] else "—"
                    pts  = f"+{c['pts']}" if c["met"] else "  0"
                    st.write(f"{icon} `{pts}` {c['name']}")

# ─────────────────────────────────────────────
# WATCHLIST SCANNER
# ─────────────────────────────────────────────
if run_scan:
    st.divider()
    st.subheader("🔍 Watchlist Scan")

    rows          = []
    alerts_sent   = 0
    alert_log     = []
    prog          = st.progress(0)

    for i, ticker in enumerate(WATCHLIST):
        try:
            d = get_data(ticker, "6mo")
            if d.empty:
                continue
            res_t = evaluate(ticker, d, model)
            ll    = res_t["latest"]

            # Frequency control: cap alerts per scan
            fire = res_t["send_alert"] and alerts_sent < MAX_ALERTS_PER_SCAN

            rows.append({
                "Ticker":     ticker,
                "Price":      round(ll["Close"], 2),
                "AI %":       f"{res_t['prob']*100:.1f}%",
                "RSI":        round(ll["rsi"], 1),
                "Vol Ratio":  round(ll["vol_ratio"], 2),
                "Breakout":   "✅" if ll["breakout"] else "—",
                "Score":      res_t["score"] if not res_t["gated"] else "—",
                "Signal":     res_t["tier_label"],
                "Alert":      "📣" if fire else ("⏳" if not res_t["cooldown_ok"] else "—"),
            })

            if fire:
                msg  = build_alert_msg(ticker, res_t)
                sent = send_discord(msg)
                if sent:
                    _mark_alerted(ticker)
                alert_log.append((ticker, res_t["tier_label"], res_t["score"], sent))
                alerts_sent += 1

        except Exception:
            pass
        prog.progress((i + 1) / len(WATCHLIST))

    prog.empty()

    if rows:
        def _sort_key(r):
            s = r["Score"]
            return s if isinstance(s, int) else -1

        results_df = (pd.DataFrame(rows)
                      .assign(_sort=lambda df: df["Score"].apply(lambda s: s if isinstance(s, int) else -1))
                      .sort_values("_sort", ascending=False)
                      .drop(columns="_sort")
                      .reset_index(drop=True))
        st.dataframe(results_df, use_container_width=True)

        skipped = WATCHLIST.__len__() - len(rows)
        if skipped:
            st.caption(f"{skipped} ticker(s) skipped (no data).")

    if alert_log:
        st.subheader("📣 Alerts Fired This Scan")
        for ticker, label, score, sent in alert_log:
            status = "✅ sent" if sent else "⚠️ webhook missing"
            st.write(f"**{ticker}** — {label} (score {score}) — {status}")
        if alerts_sent >= MAX_ALERTS_PER_SCAN:
            st.caption(f"Alert cap of {MAX_ALERTS_PER_SCAN} reached. Lower-ranked signals suppressed.")
    else:
        st.info(f"No alert-worthy signals this scan (need score ≥ 8, all gates passing, cooldown clear).")
