import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta
from plotly.subplots import make_subplots

from engine import (
    WATCHLIST, FEATURES, PREDICTION_DAYS, TARGET_RETURN,
    COOLDOWN_HOURS, MAX_ALERTS, DISCORD,
    get_data as _get_data, train_model as _train_model,
    decide, send_alert, mark_alerted, log_signal,
    resolve_outcomes, accuracy_stats,
    _load_cooldowns,
)
from opportunity.backtester import compute_metrics, _resolved, run_backtest
from opportunity.config import ENABLE_ADVANCED_BACKTESTS

# ─── CACHED WRAPPERS (Streamlit caching on top of engine functions) ───────────
@st.cache_data(ttl=300, show_spinner="Fetching…")
def get_data(ticker: str, period: str) -> pd.DataFrame:
    return _get_data(ticker, period)

@st.cache_resource(show_spinner="Training model…")
def train_model():
    return _train_model()

# ─── CHART ────────────────────────────────────────────────────────────────────
def chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
                        subplot_titles=(f"{ticker} Price", "RSI (14)", "MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"]), row=1, col=1)
    for y, name, color in [(df["ema20"], "EMA20", "orange"), (df["ema50"], "EMA50", "royalblue")]:
        fig.add_trace(go.Scatter(x=df.index, y=y, name=name,
                                  line=dict(color=color, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
                              line=dict(color="gray", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
                              line=dict(color="gray", dash="dot", width=1),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
                              line=dict(color="purple", width=1.5)), row=2, col=1)
    for lvl, dash in [(70, "dash"), (30, "dash"), (65, "dot"), (35, "dot")]:
        fig.add_hline(y=lvl, line_dash=dash,
                      line_color="red" if lvl >= 65 else "green", row=2, col=1)
    colors = ["green" if v >= 0 else "red" for v in df["macd_diff"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
                              line=dict(color="royalblue", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
                              line=dict(color="orange", width=1)), row=3, col=1)
    fig.update_layout(height=600, showlegend=False, xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tradey Boi X", page_icon="📈", layout="wide")
st.title("📈 Tradey Boi X")

all_signals = resolve_outcomes()
stats       = accuracy_stats(all_signals)

with st.sidebar:
    st.header("Settings")
    selected = st.selectbox("Stock", WATCHLIST)
    period   = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    st.divider()
    run_scan = st.button("🔍 Scan Now", use_container_width=True)
    st.divider()
    st.caption(f"**Prediction window:** {PREDICTION_DAYS} trading days")
    st.caption(f"**Target:** +{TARGET_RETURN*100:.0f}% return")
    st.caption("**Alert tiers:** ELITE (≥11) · STRONG BUY (≥8)")
    st.caption(f"Cooldown: {COOLDOWN_HOURS}h · Max alerts/scan: {MAX_ALERTS}")
    st.caption("Discord: " + ("✅ Connected" if DISCORD else "❌ Set `Discordwebhook` secret"))
    st.divider()
    st.subheader("📊 Signal Accuracy")
    if stats["total"] == 0:
        st.caption("No resolved signals yet — check back after 10 trading days.")
    else:
        st.metric("Win Rate",   f"{stats['win_rate']*100:.1f}%",
                  delta=f"{stats['wins']}W / {stats['losses']}L")
        st.metric("Avg Return", f"{stats['avg_return']*100:+.2f}%")
        st.caption(f"Based on {stats['total']} resolved signal(s)")

model = train_model()

# ── Single stock view ─────────────────────────────────────────────────────────
df = get_data(selected, period)

if df.empty:
    st.error(f"No data for {selected}.")
else:
    res = decide(selected, df, model)
    row = df.iloc[-1]
    chg = (row["Close"] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Price",     f"${row['Close']:.2f}", f"{chg:+.2f}%")
    c2.metric("AI Prob",   f"{res['prob']*100:.1f}%")
    c3.metric("RSI",       f"{row['rsi']:.1f}")
    c4.metric("MACD",      f"{row['macd_diff']:.4f}")
    c5.metric("Vol Ratio", f"{row['vol_ratio']:.2f}×")
    c6.metric("Score",     f"{res['score']}/14")
    c7.metric("Regime",    res["regime"], help="Informational only — does not gate alerts")

    st.markdown(
        f"### <span style='color:{res['color']}'>{res['label']}</span>"
        + (f" &nbsp;<span style='color:gray;font-size:0.8em'>"
           f"⏱ {PREDICTION_DAYS}-day window · 🎯 +{TARGET_RETURN*100:.0f}% target</span>"
           if res["signal"] not in ("GATED", "IGNORE") else ""),
        unsafe_allow_html=True,
    )

    if res["alert"]:
        if st.button(f"📣 Send Discord Alert — {selected}"):
            if send_alert(selected, res, row["Close"]):
                mark_alerted(selected)
                log_signal(selected, row["Close"], res["signal"])
                st.success("Alert sent and logged!")
            else:
                st.warning("Webhook missing or failed.")
    elif res["signal"] == "GATED":
        st.info("Failed hard filters — see breakdown below.")
    else:
        cd = _load_cooldowns()
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
            for pts, name, met in [
                (3, "AI prob ≥ 80%",        res["prob"] >= 0.80),
                (2, "AI prob ≥ 70%",        0.70 <= res["prob"] < 0.80),
                (1, "AI prob ≥ 60%",        0.60 <= res["prob"] < 0.70),
                (3, "52-week breakout",     bool(row["breakout"])),
                (2, "Volume surge >1.5×",   row["vol_ratio"] > 1.5),
                (2, "RSI ideal 35–65",      35 <= row["rsi"] <= 65),
                (1, "RSI safe < 70",        row["rsi"] < 70),
                (1, "EMA uptrend",          row["ema20"] > row["ema50"]),
            ]:
                st.write(f"{'✅' if met else '—'} `{'+'if met else ' '}{pts if met else 0}` {name}")

# ── Signal history ────────────────────────────────────────────────────────────
if all_signals:
    with st.expander(f"📋 Signal History ({len(all_signals)} logged)"):
        rows_h = [{
            "Date":    e["signal_date"],
            "Ticker":  e["ticker"],
            "Tier":    e["tier"],
            "Entry $": e["entry_price"],
            "Exit $":  e["exit_price"] if e["exit_price"] else "pending",
            "Return":  f"{e['actual_pct']*100:+.1f}%" if e["actual_pct"] is not None else "pending",
            "Outcome": ("✅ WIN" if e["outcome"] in ("WIN", "HIT_TARGET", "EXPIRED_GAIN") else "❌ LOSS")
                       if e["outcome"] else f"⏳ ~{e['pred_days']}d window",
        } for e in reversed(all_signals)]
        st.dataframe(pd.DataFrame(rows_h), use_container_width=True)

# ── Institutional metrics (cost-adjusted P&L, regime, walk-forward/Monte Carlo) ─
with st.expander("🏛️ Institutional Metrics"):
    resolved = _resolved(all_signals)
    m = compute_metrics(resolved)
    if m["trade_count"] == 0:
        st.caption("No resolved trades yet — institutional metrics need at "
                    "least one closed signal.")
    else:
        st.caption("Cost-adjusted P&L (commissions + slippage + spread applied; "
                    "see opportunity/costs.py)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Profit Factor", f"{m['profit_factor']:.3f}"
                   if m["profit_factor"] not in (float("inf"),) else "∞")
        m2.metric("Expectancy (R)", f"{m['expectancy_r']:+.3f}")
        m3.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
        m4.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Sortino", f"{m['sortino_ratio']:.2f}"
                   if m["sortino_ratio"] not in (float("inf"),) else "∞")
        m6.metric("Trade Count", m["trade_count"])
        m7.metric("Win Streak (max)", m["winning_streak"])
        m8.metric("Loss Streak (max)", m["losing_streak"])

    st.divider()
    st.caption("Walk-forward validation & Monte Carlo risk-of-ruin "
               "(opportunity/backtester.py)")
    if not ENABLE_ADVANCED_BACKTESTS:
        st.info("Disabled — set `ENABLE_ADVANCED_BACKTESTS=true` to compute "
                "walk-forward and Monte Carlo reports.")
    else:
        wf = run_backtest(mode="walk_forward", notify=False)
        mc = run_backtest(mode="monte_carlo", notify=False)
        if wf and wf.get("windows"):
            st.write(f"**Walk-forward** — {len(wf['windows'])} rolling window(s)")
            st.dataframe(pd.DataFrame(wf["windows"]), use_container_width=True)
        else:
            st.caption("Not enough resolved trades yet for a walk-forward window.")
        if mc and mc.get("summary", {}).get("n_simulations"):
            s = mc["summary"]
            st.write(f"**Monte Carlo** — {s['n_simulations']} resampled sequences "
                      f"of {s['sample_size']} trades")
            c1, c2, c3 = st.columns(3)
            c1.metric("Profit Factor (median)", f"{s['profit_factor_median']:.3f}",
                       help=f"p5 {s['profit_factor_p5']:.3f} · p95 {s['profit_factor_p95']:.3f}")
            c2.metric("Expectancy R (median)", f"{s['expectancy_r_median']:+.3f}",
                       help=f"p5 {s['expectancy_r_p5']:+.3f} · p95 {s['expectancy_r_p95']:+.3f}")
            c3.metric("Risk of Ruin", f"{s['risk_of_ruin_pct']:.1f}%")
        else:
            st.caption("Not enough resolved trades yet for Monte Carlo resampling.")

# ── Watchlist scan ────────────────────────────────────────────────────────────
if run_scan:
    st.divider()
    st.subheader("🔍 Scan Results")
    rows, fired, prog = [], 0, st.progress(0)

    for i, ticker in enumerate(WATCHLIST):
        try:
            d  = get_data(ticker, "6mo")
            if d.empty:
                continue
            r  = decide(ticker, d, model)
            ll = d.iloc[-1]
            alerted = False

            if r["alert"] and fired < MAX_ALERTS:
                alerted = send_alert(ticker, r, ll["Close"])
                if alerted:
                    mark_alerted(ticker)
                    log_signal(ticker, ll["Close"], r["signal"])
                fired += 1

            rows.append({
                "Ticker":   ticker,
                "Price":    round(ll["Close"], 2),
                "AI %":     f"{r['prob']*100:.1f}%",
                "RSI":      round(ll["rsi"], 1),
                "Vol ×":    round(ll["vol_ratio"], 2),
                "Breakout": "✅" if ll["breakout"] else "—",
                "Score":    r["score"] if r["signal"] != "GATED" else "—",
                "Signal":   r["label"],
                "Alert":    "📣" if alerted else (
                            "⏳" if r["signal"] in ("ELITE", "STRONG BUY") and not r["alert"] else "—"),
            })
        except Exception:
            pass
        prog.progress((i + 1) / len(WATCHLIST))

    prog.empty()
    if rows:
        out = pd.DataFrame(rows)
        out["_s"] = out["Score"].apply(lambda v: v if isinstance(v, int) else -1)
        st.dataframe(out.sort_values("_s", ascending=False)
                        .drop(columns="_s").reset_index(drop=True),
                     use_container_width=True)
        if fired >= MAX_ALERTS:
            st.caption(f"Alert cap ({MAX_ALERTS}) reached — lower-ranked signals suppressed.")
