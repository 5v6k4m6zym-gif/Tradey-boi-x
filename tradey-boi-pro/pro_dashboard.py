"""
Tradey Boi Pro — Control Centre
Run with: streamlit run pro_dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

import db.database as db
import config.settings as cfg
from engine.signal_bridge import get_pending_signals, format_signal_display
from engine.risk import performance_metrics, current_exposure, circuit_breaker_active

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tradey Boi Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Init DB + defaults ───────────────────────────────────────────────────────
db.init_db()
cfg.ensure_defaults()

# ── Session state — broker + bot ─────────────────────────────────────────────
if "broker" not in st.session_state:
    from broker.ibkr_client import IBKRClient
    st.session_state.broker = IBKRClient()

if "bot" not in st.session_state:
    from engine.bot_runner import BotRunner
    st.session_state.bot = BotRunner(st.session_state.broker)

broker = st.session_state.broker
bot    = st.session_state.bot

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Tradey Boi Pro")
    mode = cfg.get("mode") or "PAPER"
    mode_color = "🟡" if mode == "PAPER" else "🔴"
    st.markdown(f"**Mode:** {mode_color} {mode}")

    conn_color = "🟢" if broker.connected else "🔴"
    conn_text  = "Connected" if broker.connected else "Disconnected"
    st.markdown(f"**IBKR:** {conn_color} {conn_text}")

    if broker.connected:
        st.metric("Account Value",  f"${broker.get_account_value():,.0f}")
        st.metric("Cash Available", f"${broker.get_cash():,.0f}")
        exp = current_exposure(broker.get_account_value())
        st.metric("Exposure",       f"{exp:.1f}%")

    st.divider()

    # ── Bot start/stop ───────────────────────────────────────────────────────
    if bot.is_running():
        if st.button("⏹ Stop Bot", use_container_width=True, type="secondary"):
            bot.stop()
            cfg.set("bot_enabled", False)
            st.rerun()
        st.success("Bot is RUNNING")
    else:
        if broker.connected:
            if st.button("▶ Start Bot", use_container_width=True, type="primary"):
                bot.start()
                cfg.set("bot_enabled", True)
                st.rerun()
        else:
            st.button("▶ Start Bot", use_container_width=True,
                      disabled=True, help="Connect to IBKR first")

    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_main, tab_positions, tab_performance, tab_health, tab_settings = st.tabs([
    "📊 Dashboard", "📋 Positions", "📈 Performance", "🔧 Health", "⚙️ Settings"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_main:
    # ── Connection setup wizard ───────────────────────────────────────────────
    if not broker.connected:
        st.header("🔌 Connect to Interactive Brokers")
        st.info(
            "**Before connecting:**\n"
            "1. Download & install [IB Gateway](https://www.interactivebrokers.com.au/en/trading/ibgateway.php) "
            "or Trader Workstation (TWS)\n"
            "2. Log in to your paper or live account\n"
            "3. Enable API: File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients\n"
            "4. Set Socket port: **7497** (paper) or **7496** (live)\n"
            "5. Click Connect below"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            host = st.text_input("IB Gateway Host", value=cfg.get("ibkr_host") or "127.0.0.1")
        with col2:
            mode_sel = st.selectbox("Mode", ["Paper Trading", "Live Trading"],
                                    index=0 if (cfg.get("mode") or "PAPER") == "PAPER" else 1)
            port = 7497 if mode_sel == "Paper Trading" else 7496
        with col3:
            client_id = st.number_input("Client ID", value=int(cfg.get("ibkr_client_id") or 1),
                                        min_value=1, max_value=99)

        if st.button("🔌 Connect", type="primary", use_container_width=True):
            mode_val = "PAPER" if mode_sel == "Paper Trading" else "LIVE"
            cfg.set("ibkr_host",      host)
            cfg.set("ibkr_port",      port)
            cfg.set("ibkr_client_id", client_id)
            cfg.set("mode",           mode_val)

            with st.spinner("Connecting to IB Gateway…"):
                ok = broker.connect(host, port, client_id)

            if ok:
                st.success("✅ Connected!")
                st.rerun()
            else:
                st.error(f"❌ Connection failed: {broker.error}\n\n"
                         "Make sure IB Gateway is running and API is enabled.")
        st.stop()

    # ── Market status ─────────────────────────────────────────────────────────
    st.header("📊 Dashboard")
    import pytz
    aest_now = datetime.now(pytz.timezone("Australia/Sydney"))
    et_now   = datetime.now(pytz.timezone("America/New_York"))
    asx_open = 10 <= aest_now.hour < 16 and aest_now.weekday() < 5
    us_open  = 9 <= et_now.hour < 16   and et_now.weekday() < 5

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🇦🇺 ASX",    "OPEN" if asx_open  else "CLOSED",
                aest_now.strftime("%H:%M AEST"))
    col2.metric("🇺🇸 US",     "OPEN" if us_open   else "CLOSED",
                et_now.strftime("%H:%M ET"))
    col3.metric("Circuit Breaker", "ACTIVE ⚠️" if circuit_breaker_active() else "Clear ✅")
    col4.metric("Bot Status", "RUNNING 🟢" if bot.is_running() else "STOPPED 🔴")

    st.divider()

    # ── Account summary ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    account_val  = broker.get_account_value()
    open_pos     = db.open_positions()
    metrics      = performance_metrics()

    col1.metric("Account Value",   f"${account_val:,.0f}")
    col2.metric("Open Positions",  len(open_pos),
                f"of {cfg.get('max_positions') or 5} max")
    col3.metric("Total P&L",       f"${metrics['total_pnl']:+,.0f}")
    col4.metric("Win Rate",        f"{metrics['win_rate']*100:.0f}%"
                if metrics["trade_count"] else "—")

    st.divider()

    # ── Top opportunities ─────────────────────────────────────────────────────
    col_sig, col_log = st.columns([1, 1])

    with col_sig:
        st.subheader("🎯 Top Signals (from Tradey Boi X)")
        signals = get_pending_signals(lookback_hours=48)
        if signals:
            for sig in signals[:5]:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{sig['ticker']}** — {sig['tier']}")
                    c1.caption(
                        f"Score {sig['score']:.0f}  ·  "
                        f"Prob {sig['prob']*100:.0f}%  ·  "
                        f"Entry ${sig['entry_price']:.3f}  ·  "
                        f"ATR {sig['atr_pct']:.1f}%"
                    )
                    if bot.is_running():
                        if c2.button("Trade", key=f"trade_{sig['ticker']}",
                                     type="primary", use_container_width=True):
                            from engine.executor import execute_signal
                            result = execute_signal(sig, broker)
                            if result["ok"]:
                                st.success(f"✅ Order placed: {sig['ticker']}")
                            else:
                                st.error(f"❌ {result['reason']}")
                            st.rerun()
                    else:
                        c2.caption("Start bot to trade")
        else:
            st.info("No pending signals right now. "
                    "Signals come from Tradey Boi X scanner (GitHub Actions).")

    with col_log:
        st.subheader("📝 Bot Activity Log")
        if bot.scan_log:
            for entry in bot.scan_log[:15]:
                st.caption(entry)
        else:
            st.caption("No activity yet. Start the bot to begin scanning.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.header("📋 Open Positions")
    open_pos = db.open_positions()

    if not open_pos:
        st.info("No open positions.")
    else:
        for pos in open_pos:
            # Try to get current price
            curr_price = broker.get_current_price(
                pos["ticker"], pos["exchange"],
                "AUD" if pos["exchange"] == "ASX" else "USD"
            ) or pos["entry_price"]

            unreal_pnl  = (curr_price - pos["entry_price"]) * pos["quantity"]
            unreal_pct  = (curr_price - pos["entry_price"]) / pos["entry_price"] * 100
            entry_dt    = datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d")
            days_held   = (datetime.utcnow() - entry_dt).days
            days_left   = max(0, (pos.get("max_hold_days") or 15) - days_held)
            pnl_color   = "🟢" if unreal_pnl >= 0 else "🔴"

            with st.container(border=True):
                h1, h2 = st.columns([4, 1])
                h1.markdown(
                    f"**{pos['ticker']}**  ·  {pos['exchange']}  ·  "
                    f"{pnl_color} **{unreal_pct:+.1f}%**  (${unreal_pnl:+.0f})"
                )

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Entry",    f"${pos['entry_price']:.3f}")
                c2.metric("Current",  f"${curr_price:.3f}")
                c3.metric("Stop",     f"${pos['stop_price']:.3f}")
                c4.metric("Target",   f"${pos['target_price']:.3f}")
                c5.metric("Qty",      f"{pos['quantity']:.0f}")
                c6.metric("Days",     f"{days_held}d  ({days_left}d left)")

                if pos.get("signal_score"):
                    st.caption(
                        f"Signal: score={pos['signal_score']:.0f}  "
                        f"prob={pos.get('signal_prob',0)*100:.0f}%  "
                        f"atr={pos.get('atr_pct',0):.1f}%"
                    )

                # Manual close
                if h2.button("Close", key=f"close_{pos['id']}",
                             type="secondary", use_container_width=True):
                    from engine.executor import manual_close
                    result = manual_close(pos["id"], broker)
                    if result["ok"]:
                        st.success(f"✅ Closed {pos['ticker']} @ ${result['exit_price']:.3f}")
                    else:
                        st.error(f"❌ {result['reason']}")
                    st.rerun()

    st.divider()
    st.subheader("📜 Recent Closed Trades")
    trades = db.all_trades(limit=50)
    if trades:
        df = pd.DataFrame(trades)[
            ["ticker","entry_date","exit_date","entry_price","exit_price",
             "quantity","pnl","pnl_pct","outcome","exit_reason","mode"]
        ]
        df["pnl_pct"] = (df["pnl_pct"] * 100).round(1)
        df["pnl"]     = df["pnl"].round(2)
        st.dataframe(
            df.rename(columns={
                "pnl_pct": "P&L %", "pnl": "P&L $",
                "entry_price": "Entry", "exit_price": "Exit",
                "exit_reason": "Reason"
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No closed trades yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_performance:
    st.header("📈 Performance")
    metrics = performance_metrics()

    if metrics["trade_count"] == 0:
        st.info("No completed trades yet. Performance metrics will appear after the first trade closes.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades",        metrics["trade_count"])
        c2.metric("Win Rate",      f"{metrics['win_rate']*100:.1f}%")
        c3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        c4.metric("Total P&L",     f"${metrics['total_pnl']:+,.0f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Win",       f"${metrics['avg_win']:,.0f}")
        c2.metric("Avg Loss",      f"${metrics['avg_loss']:,.0f}")
        c3.metric("Max Drawdown",  f"{metrics['max_drawdown']*100:.1f}%")
        c4.metric("Sharpe Ratio",  f"{metrics['sharpe']:.2f}")

        st.divider()

        # ── Equity curve ─────────────────────────────────────────────────────
        trades = db.all_trades(limit=500)
        if len(trades) >= 2:
            df = pd.DataFrame(trades)
            df["exit_date"] = pd.to_datetime(df["exit_date"])
            df = df.sort_values("exit_date")
            df["cum_pnl"] = df["pnl"].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["exit_date"], y=df["cum_pnl"],
                mode="lines+markers",
                line=dict(color="#00d4aa", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.1)",
                name="Cumulative P&L"
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Equity Curve (Cumulative P&L)",
                xaxis_title="Date", yaxis_title="P&L ($)",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Win/Loss distribution ────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                outcome_counts = df["outcome"].value_counts()
                fig2 = px.pie(
                    values=outcome_counts.values,
                    names=outcome_counts.index,
                    title="Trade Outcomes",
                    color_discrete_map={"WIN": "#00d4aa", "LOSS": "#ff4b4b",
                                        "STOP": "#ffa500"},
                )
                fig2.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), height=300,
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                fig3 = px.histogram(
                    df, x="pnl_pct", nbins=20,
                    title="P&L Distribution (%)",
                    color_discrete_sequence=["#00d4aa"],
                )
                fig3.add_vline(x=0, line_dash="dash", line_color="gray")
                fig3.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), height=300,
                )
                st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_health:
    st.header("🔧 System Health")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Connection Status")
        with st.container(border=True):
            st.markdown(f"**IBKR Gateway:** {'🟢 Connected' if broker.connected else '🔴 Disconnected'}")
            if broker.error:
                st.error(broker.error)
            if broker._last_ping:
                ago = (datetime.utcnow() - broker._last_ping).seconds
                st.caption(f"Last ping: {ago}s ago")

            st.markdown(f"**Bot Runner:** {'🟢 Running' if bot.is_running() else '⚫ Stopped'}")
            if bot.last_scan:
                ago = int((datetime.utcnow() - bot.last_scan).total_seconds())
                st.caption(f"Last scan: {ago}s ago")

            from engine.position_manager import PositionManager
            pm = bot._pm
            st.markdown(f"**Position Manager:** {'🟢 Running' if pm.is_running() else '⚫ Stopped'}")
            if pm.last_run:
                ago = int((datetime.utcnow() - pm.last_run).total_seconds())
                st.caption(f"Last check: {ago}s ago")

        st.subheader("Risk Controls")
        with st.container(border=True):
            cb = circuit_breaker_active()
            st.markdown(f"**Circuit Breaker:** {'⚠️ ACTIVE' if cb else '✅ Clear'}")

            account_val = broker.get_account_value()
            exp = current_exposure(account_val) if account_val > 0 else 0
            max_exp = cfg.get("max_exposure_pct") or 30.0
            st.markdown(f"**Exposure:** {exp:.1f}% / {max_exp:.0f}% max")
            st.progress(min(exp / max_exp, 1.0))

            open_count = len(db.open_positions())
            max_pos    = cfg.get("max_positions") or 5
            st.markdown(f"**Positions:** {open_count} / {max_pos} max")
            st.progress(min(open_count / max_pos, 1.0))

    with col2:
        st.subheader("Signal Source")
        with st.container(border=True):
            from engine.signal_bridge import _signal_log_path
            log_path = _signal_log_path()
            exists   = log_path.exists()
            st.markdown(f"**Signal Log:** {'🟢 Found' if exists else '🔴 Not found'}")
            st.caption(str(log_path))
            if exists:
                import os
                mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
                age   = (datetime.utcnow() - mtime).total_seconds() / 3600
                st.caption(f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M')} UTC ({age:.1f}h ago)")

        st.subheader("Recent Errors")
        errors = db.recent_errors(limit=10)
        if errors:
            for e in errors:
                level_icon = "🔴" if e["level"] == "ERROR" else "🟡"
                st.caption(f"{level_icon} [{e['logged_at'][:16]}] {e['source']}: {e['message']}")
        else:
            st.success("No errors logged.")

        st.subheader("Position Manager Activity")
        if pm.last_actions:
            for action in pm.last_actions:
                st.caption(f"• {action}")
        else:
            st.caption("No recent actions.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.header("⚙️ Settings")
    st.info("All settings are saved automatically. Bot restart required for some changes.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Management")
        risk_pct = st.slider(
            "Risk per trade (%)", 0.5, 5.0,
            float(cfg.get("risk_pct") or 2.0), step=0.1,
            help="% of account value risked on each trade"
        )
        max_pos = st.slider(
            "Max open positions", 1, 20,
            int(cfg.get("max_positions") or 5),
            help="Maximum simultaneous open trades"
        )
        max_exp = st.slider(
            "Max total exposure (%)", 10.0, 100.0,
            float(cfg.get("max_exposure_pct") or 30.0), step=5.0,
            help="Max % of account in open positions at once"
        )
        max_daily_loss = st.slider(
            "Daily loss limit (%)", 1.0, 10.0,
            float(cfg.get("max_daily_loss_pct") or 3.0), step=0.5,
            help="Pause trading if daily loss exceeds this % of account"
        )
        hold_days = st.slider(
            "Max hold days", 5, 30,
            int(cfg.get("hold_days") or 15),
            help="Automatically exit positions after this many days"
        )

        st.subheader("Circuit Breaker")
        cb_losses = st.slider(
            "Consecutive losses to trigger", 2, 6,
            int(cfg.get("cb_consecutive_losses") or 3)
        )
        cb_pause = st.slider(
            "Pause days after trigger", 1, 14,
            int(cfg.get("cb_pause_days") or 7)
        )

    with col2:
        st.subheader("Signal Quality Gates")
        min_prob = st.slider(
            "Minimum signal probability", 0.50, 0.70,
            float(cfg.get("min_prob") or 0.53), step=0.01,
            help="Minimum ML probability to consider a signal"
        )
        min_score = st.slider(
            "Minimum signal score", 5, 10,
            int(cfg.get("min_score") or 7),
            help="Minimum score gate (same as Tradey Boi X)"
        )

        st.subheader("Stop Loss Multipliers (× ATR)")
        sl_hi  = st.slider("High-vol stops  (ATR ≥ 3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_hi")  or 1.2), step=0.1)
        sl_mid = st.slider("Mid-vol stops   (ATR 1.5–3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_mid") or 1.0), step=0.1)
        sl_lo  = st.slider("Low-vol stops   (ATR < 1.5%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_lo")  or 0.8), step=0.1)

        st.subheader("Profit Targets (%)")
        tgt_hi  = st.slider("High-vol target",  5.0, 25.0,
                             float(cfg.get("target_hi")  or 12.0), step=1.0)
        tgt_mid = st.slider("Mid-vol target",   3.0, 20.0,
                             float(cfg.get("target_mid") or 8.0),  step=1.0)
        tgt_lo  = st.slider("Low-vol target",   2.0, 15.0,
                             float(cfg.get("target_lo")  or 5.0),  step=1.0)

        st.subheader("Scanner")
        scan_interval = st.slider(
            "Scan interval (minutes)", 15, 240,
            int(cfg.get("scan_interval_mins") or 60), step=15
        )

    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        cfg.set("risk_pct",              risk_pct)
        cfg.set("max_positions",         max_pos)
        cfg.set("max_exposure_pct",      max_exp)
        cfg.set("max_daily_loss_pct",    max_daily_loss)
        cfg.set("hold_days",             hold_days)
        cfg.set("cb_consecutive_losses", cb_losses)
        cfg.set("cb_pause_days",         cb_pause)
        cfg.set("min_prob",              min_prob)
        cfg.set("min_score",             min_score)
        cfg.set("sl_mult_hi",            sl_hi)
        cfg.set("sl_mult_mid",           sl_mid)
        cfg.set("sl_mult_lo",            sl_lo)
        cfg.set("target_hi",             tgt_hi)
        cfg.set("target_mid",            tgt_mid)
        cfg.set("target_lo",             tgt_lo)
        cfg.set("scan_interval_mins",    scan_interval)
        st.success("✅ Settings saved!")

    st.divider()
    st.subheader("⚠️ Danger Zone")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Disconnect IBKR", type="secondary"):
            broker.disconnect()
            st.rerun()
    with col2:
        if st.button("🔌 Reconnect IBKR", type="secondary"):
            host = cfg.get("ibkr_host") or "127.0.0.1"
            port = int(cfg.get("ibkr_port") or 7497)
            cid  = int(cfg.get("ibkr_client_id") or 1)
            broker.disconnect()
            import time; time.sleep(1)
            ok = broker.connect(host, port, cid)
            st.success("Reconnected ✅") if ok else st.error("Failed ❌")
            st.rerun()

# ── Auto-refresh every 60s while bot is running ───────────────────────────────
if bot.is_running():
    import time
    time.sleep(0.1)
    st.markdown(
        "<meta http-equiv='refresh' content='60'>",
        unsafe_allow_html=True
    )
