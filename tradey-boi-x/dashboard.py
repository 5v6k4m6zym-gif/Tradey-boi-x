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
    market_regime, vix_safe,
    quality_score_100, quality_score_label,
    rank_opportunities,
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

@st.cache_data(ttl=1800, show_spinner="Checking market regime…")
def _get_market_health():
    """Fetch regime + VIX once per 30 min for the dashboard header."""
    try:
        asx_regime = market_regime("CBA.AX")
        us_regime  = market_regime("SPY")
        vix_ok     = vix_safe()
        return {"asx": asx_regime, "us": us_regime, "vix_ok": vix_ok}
    except Exception:
        return {"asx": "sideways", "us": "sideways", "vix_ok": True}

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

# ─── REGIME DISPLAY HELPERS ───────────────────────────────────────────────────
_REGIME_EMOJI = {
    "strong_bull": "📈 Strong Bull",
    "weak_bull":   "📊 Weak Bull",
    "low_vol":     "😴 Low Volatility (coiled)",
    "sideways":    "↔️  Sideways",
    "weak_bear":   "📉 Weak Bear",
    "strong_bear": "🐻 Strong Bear",
    "high_vol":    "⚡ High Volatility",
}
_REGIME_COLOR = {
    "strong_bull": "green",
    "weak_bull":   "lightgreen",
    "low_vol":     "yellow",
    "sideways":    "orange",
    "weak_bear":   "salmon",
    "strong_bear": "red",
    "high_vol":    "red",
}

# ─── APP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tradey Boi X", page_icon="📈", layout="wide")
st.title("📈 Tradey Boi X")

all_signals = resolve_outcomes()
stats       = accuracy_stats(all_signals)

# ── Executive Market Health Banner ────────────────────────────────────────────
mh = _get_market_health()
asx_r  = mh["asx"]
us_r   = mh["us"]
vix_ok = mh["vix_ok"]

st.markdown("### 🌐 Market Health")
hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("ASX Regime",   _REGIME_EMOJI.get(asx_r, asx_r),
           help="Broad market regime for ASX200 (^AXJO) — drives score thresholds")
hc2.metric("US Regime",    _REGIME_EMOJI.get(us_r, us_r),
           help="Broad market regime for S&P500 (SPY) — drives score thresholds")
hc3.metric("VIX Safety",   "✅ Safe (< 30)" if vix_ok else "⚡ Elevated (≥ 30)",
           help="VIX ≥ 30 gates ALL signals — unreliable environment")

# Regime score threshold hint
_REGIME_SCORE_ADJ = {"strong_bull": -1, "low_vol": -1, "weak_bull": 0,
                      "sideways": 0, "weak_bear": +1, "high_vol": +2, "strong_bear": +3}
us_elite = 8 + _REGIME_SCORE_ADJ.get(us_r, 0)
asx_elite = 8 + _REGIME_SCORE_ADJ.get(asx_r, 0)
hc4.metric("ELITE Threshold", f"US: {us_elite}  ·  ASX: {asx_elite}",
           help="Score needed for ELITE today — regime-adaptive (v3)")

st.divider()

with st.sidebar:
    st.header("Settings")
    selected = st.selectbox("Stock", WATCHLIST)
    period   = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
    st.divider()
    run_scan = st.button("🔍 Scan Now (Ranked)", use_container_width=True)
    st.divider()
    st.caption(f"**Prediction window:** {PREDICTION_DAYS} trading days")
    st.caption(f"**Target:** +{TARGET_RETURN*100:.0f}% return")
    st.caption("**Alert tiers:** ELITE · STRONG BUY (regime-adaptive v3)")
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

    q      = res.get("quality_score", quality_score_100(res))
    qlabel = quality_score_label(q)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Price",         f"${row['Close']:.2f}", f"{chg:+.2f}%")
    c2.metric("AI Confidence", f"{res['prob']*100:.1f}%")
    c3.metric("RSI",           f"{row['rsi']:.1f}")
    c4.metric("Vol Ratio",     f"{row['vol_ratio']:.2f}×")
    c5.metric("Quality Score", f"{q}/100",
              help=f"{qlabel} — 0-100 composite score (v3): AI conf + score + regime + breakout + multi-bagger + news")
    c6.metric("Regime",        _REGIME_EMOJI.get(res["regime"], res["regime"]),
              help=f"Score threshold today: ELITE ≥{res['regime_thresholds']['elite']}  ·  STRONG BUY ≥{res['regime_thresholds']['strong_buy']}")
    c7.metric("Expected R",    f"{res.get('expected_r', 0):.2f}R",
              help="Expected R-multiple (reward:risk). Must be > 0 for an alert.")

    st.markdown(
        f"### <span style='color:{res['color']}'>{res['label']}</span>"
        + (f" &nbsp;<span style='color:gray;font-size:0.8em'>"
           f"⏱ {PREDICTION_DAYS}-day window · 🎯 +{TARGET_RETURN*100:.0f}% target"
           f" · Quality {q}/100 {qlabel}</span>"
           if res["signal"] not in ("GATED", "IGNORE") else ""),
        unsafe_allow_html=True,
    )

    # Multi-bagger callout
    mb = res.get("multibagger", {})
    if mb:
        st.success(
            f"🚀 **Multi-bagger setup detected** — {mb.get('consolidation_days')}d base  ·  "
            f"{mb.get('vol_expansion', 0):.1f}× volume  ·  "
            f"**{mb.get('upside_category', '')} measured-move target**  ·  "
            f"est. hold {mb.get('holding_period_days', '?')}d  ·  "
            f"range {mb.get('range_pct', 0):.1f}% tight"
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
            st.write("**Score breakdown (raw)**")
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
            st.write(f"**Regime:** {_REGIME_EMOJI.get(res['regime'], res['regime'])}  |  "
                     f"ELITE threshold: **{res['regime_thresholds']['elite']}**  ·  "
                     f"STRONG BUY threshold: **{res['regime_thresholds']['strong_buy']}**")
            if mb:
                st.write(f"**Multi-bagger:** {mb.get('consolidation_days')}d base · "
                         f"{mb.get('upside_category','')} target · "
                         f"score bonus +{2 if mb.get('consolidation_days', 0) >= 60 else 1}")

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

# ── Watchlist scan (ranked — v3) ───────────────────────────────────────────────
if run_scan:
    st.divider()
    st.subheader("🔍 Scan Results — Ranked by Quality")
    rows, prog = [], st.progress(0)
    candidates = []

    # Pass 1: collect all data + decisions
    for i, ticker in enumerate(WATCHLIST):
        try:
            d  = get_data(ticker, "6mo")
            if d.empty:
                continue
            r  = decide(ticker, d, model)
            ll = d.iloc[-1]
            q  = r.get("quality_score", quality_score_100(r))
            mb = r.get("multibagger", {})

            rows.append({
                "Ticker":    ticker,
                "Price":     round(ll["Close"], 2),
                "Quality":   q if r["signal"] != "GATED" else "—",
                "AI %":      f"{r['prob']*100:.1f}%",
                "RSI":       round(ll["rsi"], 1),
                "Vol ×":     round(ll["vol_ratio"], 2),
                "Regime":    r.get("regime", "—"),
                "Multi-bag": f"🚀 {mb.get('upside_category','')} ({mb.get('consolidation_days')}d)" if mb else "—",
                "Signal":    r["label"],
                "Alert":     "—",
                "_score":    r["score"] if r["signal"] != "GATED" else -1,
                "_q":        q,
            })

            if r["alert"]:
                candidates.append({"ticker": ticker, "res": r, "price": float(ll["Close"]),
                                   "df": d, "group_id": None})
        except Exception:
            pass
        prog.progress((i + 1) / len(WATCHLIST))

    # Pass 2: rank and alert top MAX_ALERTS
    ranked = rank_opportunities(candidates)
    fired  = 0
    alerted_tickers = set()
    for c in ranked:
        if fired >= MAX_ALERTS:
            break
        alerted = send_alert(c["ticker"], c["res"], c["price"])
        if alerted:
            mark_alerted(c["ticker"])
            log_signal(c["ticker"], c["price"], c["res"]["signal"])
            alerted_tickers.add(c["ticker"])
            fired += 1

    # Mark alerts in rows
    for row_d in rows:
        if row_d["Ticker"] in alerted_tickers:
            row_d["Alert"] = "📣 Sent"
        elif row_d.get("_score", -1) > 0 and row_d["Signal"] not in ("🚫 GATED", "⛔ IGNORE"):
            row_d["Alert"] = "⏳ Qualified"

    prog.empty()
    if rows:
        out = pd.DataFrame(rows).drop(columns=["_score", "_q"])
        st.dataframe(out.sort_values("Quality", ascending=False,
                                      key=lambda s: pd.to_numeric(s, errors="coerce").fillna(-1))
                        .reset_index(drop=True),
                     use_container_width=True)
        if fired == 0:
            st.warning("⛔ No ELITE or STRONG BUY setups found — conditions don't meet quality bar right now. Hold cash.")
        else:
            st.caption(f"Alerted {fired}/{MAX_ALERTS} ranked candidates.")
