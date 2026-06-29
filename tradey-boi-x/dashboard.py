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

st.set_page_config(page_title="Tradey Boi X", page_icon="📈", layout="wide")
st.title("📈 Tradey Boi X")

WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT",
    "BHP.AX", "CBA.AX", "FMG.AX", "RIO.AX",
    "NST.AX", "CXO.AX", "LTR.AX", "PDN.AX"
]

# ─────────────────────────────────────────────
# INDICATORS (RSI + MACD + Breakout + more)
# ─────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    df["rsi"]        = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd             = ta.trend.MACD(close)
    df["macd"]       = macd.macd()
    df["macd_signal"]= macd.macd_signal()
    df["macd_diff"]  = macd.macd_diff()

    bb               = ta.volatility.BollingerBands(close)
    df["bb_upper"]   = bb.bollinger_hband()
    df["bb_lower"]   = bb.bollinger_lband()
    df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / close

    df["atr"]        = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["ema20"]      = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema50"]      = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    df["vol_ma20"]   = vol.rolling(20).mean()
    df["vol_ratio"]  = vol / df["vol_ma20"]

    df["ret_5"]      = close.pct_change(5)
    df["ret_10"]     = close.pct_change(10)
    df["ret_20"]     = close.pct_change(20)

    high_52w         = close.rolling(252).max()
    df["breakout"]   = (close >= high_52w * 0.98).astype(int)

    return df.dropna()

# ─────────────────────────────────────────────
# IMPROVED AI MODEL
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

    X = df[FEATURES]
    y = df["target"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    pipe.fit(X, y)
    return pipe

# ─────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────
def classify(prob, rsi, macd_diff, breakout):
    bullish = (
        prob >= 0.60 and
        rsi < 70 and
        macd_diff > 0
    )
    if breakout and prob >= 0.65:
        return "🚀 BREAKOUT BUY", "green"
    elif bullish and prob >= 0.72:
        return "✅ BUY", "green"
    elif bullish and prob >= 0.58:
        return "👀 WATCH", "orange"
    else:
        return "⛔ IGNORE", "red"

# ─────────────────────────────────────────────
# PRICE + INDICATOR CHART
# ─────────────────────────────────────────────
def plot_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
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
        line=dict(color="gray", dash="dot", width=1), fill="tonexty",
        fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
        line=dict(color="purple", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    colors = ["green" if v >= 0 else "red" for v in df["macd_diff"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="MACD Hist",
        marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
        line=dict(color="blue", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
        line=dict(color="orange", width=1)), row=3, col=1)

    fig.update_layout(
        height=620,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
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

# ─────────────────────────────────────────────
# MAIN — SINGLE STOCK VIEW
# ─────────────────────────────────────────────
model = train_model()

@st.cache_data(ttl=300, show_spinner="Fetching data...")
def get_data(ticker, period):
    df = yf.Ticker(ticker).history(period=period)
    return add_indicators(df)

df = get_data(selected, period)

if df.empty:
    st.error(f"No data found for {selected}.")
else:
    latest = df.iloc[-1]
    prob   = model.predict_proba([latest[FEATURES].values])[0][1]
    signal, sig_color = classify(
        prob, latest["rsi"], latest["macd_diff"], latest["breakout"]
    )

    prev_close = df["Close"].iloc[-2]
    change     = latest["Close"] - prev_close
    pct_change = change / prev_close * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price",    f"${latest['Close']:.2f}", f"{pct_change:+.2f}%")
    col2.metric("RSI",      f"{latest['rsi']:.1f}")
    col3.metric("MACD",     f"{latest['macd_diff']:.3f}")
    col4.metric("Breakout", "Yes 🚀" if latest["breakout"] else "No")
    col5.metric("AI Prob",  f"{prob*100:.1f}%")

    st.markdown(f"### Signal: <span style='color:{sig_color}; font-size:1.3em'>{signal}</span>",
                unsafe_allow_html=True)

    st.plotly_chart(plot_chart(df, selected), use_container_width=True)

# ─────────────────────────────────────────────
# WATCHLIST SCANNER
# ─────────────────────────────────────────────
if run_scan:
    st.divider()
    st.subheader("🔍 Watchlist Scan Results")

    rows = []
    prog = st.progress(0)
    for i, ticker in enumerate(WATCHLIST):
        try:
            d = get_data(ticker, "6mo")
            if len(d) < 50:
                continue
            l = d.iloc[-1]
            p = model.predict_proba([l[FEATURES].values])[0][1]
            s, _ = classify(p, l["rsi"], l["macd_diff"], l["breakout"])
            rows.append({
                "Ticker":   ticker,
                "Price":    round(l["Close"], 2),
                "RSI":      round(l["rsi"], 1),
                "MACD":     round(l["macd_diff"], 4),
                "Breakout": "Yes" if l["breakout"] else "No",
                "AI %":     f"{p*100:.1f}%",
                "Signal":   s
            })
        except Exception:
            pass
        prog.progress((i + 1) / len(WATCHLIST))

    prog.empty()

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("AI %", ascending=False).reset_index(drop=True)
    st.dataframe(results_df, use_container_width=True)
