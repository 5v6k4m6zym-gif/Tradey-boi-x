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
from datetime import datetime

import db.database as db
import config.settings as cfg
from engine.risk import performance_metrics, current_exposure, circuit_breaker_active

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tradey Boi Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Init DB + defaults ────────────────────────────────────────────────────────
db.init_db()
cfg.ensure_defaults()

# ── Session state ─────────────────────────────────────────────────────────────
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
    mode       = cfg.get("mode") or "PAPER"
    mode_color = "🟡" if mode == "PAPER" else "🔴"
    st.markdown(f"**Mode:** {mode_color} {mode}")

    conn_color = "🟢" if broker.connected else "🔴"
    st.markdown(f"**IBKR:** {conn_color} {'Connected' if broker.connected else 'Disconnected'}")

    if broker.connected:
        acct = broker.get_account_value()
        st.metric("Account",      f"${acct:,.0f}")
        st.metric("Cash",         f"${broker.get_cash():,.0f}")
        exp = current_exposure(acct)
        st.metric("Exposure",     f"{exp:.1f}%")

    st.divider()

    # ── Scanner status ────────────────────────────────────────────────────────
    scanner = bot.scanner
    if bot.is_running():
        if scanner.is_scanning:
            done, total = scanner.progress
            st.info(f"🔍 Scanning… {done}/{total}")
        else:
            sig_count = len(scanner.signals)
            st.success(f"🟢 Bot RUNNING\n\n{sig_count} signal(s) ready")
        if st.button("⏹ Stop Bot", use_container_width=True, type="secondary"):
            bot.stop()
            cfg.set("bot_enabled", False)
            st.rerun()
        if st.button("🔍 Scan Now", use_container_width=True):
            bot.force_scan()
            st.toast("Scan triggered!")
    else:
        if broker.connected:
            if st.button("▶ Start Bot", use_container_width=True, type="primary"):
                bot.start()
                cfg.set("bot_enabled", True)
                st.rerun()
        else:
            st.button("▶ Start Bot", disabled=True, use_container_width=True,
                      help="Connect to IBKR first")

    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    if scanner.last_scan:
        ago = int((datetime.utcnow() - scanner.last_scan).total_seconds())
        st.caption(f"Last scan: {ago//60}m {ago%60}s ago  ·  #{scanner.scan_count}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_scan, tab_pos, tab_perf, tab_health, tab_settings = st.tabs([
    "📊 Dashboard", "🔍 Scanner", "📋 Positions",
    "📈 Performance", "🔧 Health", "⚙️ Settings"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    if not broker.connected:
        # ── Connection wizard ────────────────────────────────────────────────
        st.header("🔌 Connect to Interactive Brokers")
        st.info(
            "**Before connecting:**\n"
            "1. Install & open [IB Gateway](https://www.interactivebrokers.com.au/en/trading/ibgateway.php) "
            "or Trader Workstation (TWS)\n"
            "2. Enable API: Configure → Settings → API → Enable ActiveX and Socket Clients ✅\n"
            "3. Socket port: **7497** (paper) or **7496** (live)\n"
            "4. Click Connect"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            host = st.text_input("Host", value=cfg.get("ibkr_host") or "127.0.0.1")
        with col2:
            mode_sel = st.selectbox("Mode", ["Paper Trading", "Live Trading"],
                index=0 if (cfg.get("mode") or "PAPER") == "PAPER" else 1)
            port = 7497 if mode_sel == "Paper Trading" else 7496
        with col3:
            cid = st.number_input("Client ID", value=int(cfg.get("ibkr_client_id") or 1),
                                   min_value=1, max_value=99)
        if st.button("🔌 Connect", type="primary", use_container_width=True):
            mode_val = "PAPER" if mode_sel == "Paper Trading" else "LIVE"
            cfg.set("ibkr_host", host); cfg.set("ibkr_port", port)
            cfg.set("ibkr_client_id", cid); cfg.set("mode", mode_val)
            with st.spinner("Connecting to IB Gateway…"):
                ok = broker.connect(host, port, cid)
            if ok:
                st.success("✅ Connected!")
                st.rerun()
            else:
                st.error(f"❌ Failed: {broker.error}")
        st.stop()

    # ── Market status ─────────────────────────────────────────────────────────
    st.header("📊 Dashboard")
    import pytz
    aest_now = datetime.now(pytz.timezone("Australia/Sydney"))
    et_now   = datetime.now(pytz.timezone("America/New_York"))
    asx_open = 10 <= aest_now.hour < 16 and aest_now.weekday() < 5
    us_open  = (et_now.weekday() < 5 and
                ((et_now.hour == 9 and et_now.minute >= 30) or 10 <= et_now.hour < 16))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🇦🇺 ASX",    "OPEN ✅" if asx_open else "CLOSED",
              aest_now.strftime("%H:%M AEST"))
    c2.metric("🇺🇸 US",     "OPEN ✅" if us_open  else "CLOSED",
              et_now.strftime("%H:%M ET"))
    c3.metric("Scanner",    scanner.status[:30] if scanner.status else "Idle")
    c4.metric("CB",         "⚠️ ACTIVE" if circuit_breaker_active() else "Clear ✅")
    c5.metric("Bot",        "🟢 RUNNING" if bot.is_running() else "⚫ STOPPED")

    st.divider()

    # ── Account + key stats ──────────────────────────────────────────────────
    acct_val = broker.get_account_value()
    metrics  = performance_metrics()
    open_pos = db.open_positions()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Account Value",   f"${acct_val:,.0f}")
    c2.metric("Open Positions",  f"{len(open_pos)} / {cfg.get('max_positions') or 5}")
    c3.metric("Signals Ready",   len(scanner.signals))
    c4.metric("Total P&L",       f"${metrics['total_pnl']:+,.0f}")
    c5.metric("Win Rate",        f"{metrics['win_rate']*100:.0f}%" if metrics["trade_count"] else "—")

    st.divider()

    # ── Top signals + activity log ───────────────────────────────────────────
    col_sig, col_log = st.columns([1, 1])

    with col_sig:
        st.subheader("🎯 Top Signals")
        from engine.signal_bridge import get_pending_signals
        pending = get_pending_signals(scanner_signals=scanner.signals)

        if pending:
            for sig in pending[:6]:
                src_icon = "🔍" if sig.get("source") == "pro_scanner" else "📡"
                tier_color = "🟢" if sig.get("tier") == "STRONG BUY" else "🔵"
                with st.container(border=True):
                    h1, h2 = st.columns([3, 1])
                    h1.markdown(
                        f"**{sig['ticker']}**  {tier_color} {sig['tier']}  {src_icon}"
                    )
                    h1.caption(
                        f"Score **{sig['score']:.0f}**  ·  "
                        f"Prob **{sig['prob']*100:.0f}%**  ·  "
                        f"Entry **${sig['entry_price']:.3f}**  ·  "
                        f"ATR {sig['atr_pct']:.1f}%"
                        + (f"  ·  RSI {sig['rsi']:.0f}" if "rsi" in sig else "")
                    )
                    if bot.is_running():
                        if h2.button("Trade", key=f"t_{sig['ticker']}",
                                     type="primary", use_container_width=True):
                            from engine.executor import execute_signal
                            result = execute_signal(sig, broker)
                            if result["ok"]:
                                st.success(f"✅ {sig['ticker']} order placed")
                            else:
                                st.error(f"❌ {result['reason']}")
                            st.rerun()
                    else:
                        h2.caption("Start bot")
        else:
            st.info(
                "No qualifying signals yet.\n\n"
                "Start the bot to begin scanning. Pro scans ASX + US stocks "
                "every 15 minutes during market hours."
            )

    with col_log:
        st.subheader("📝 Activity Log")
        if bot.scan_log:
            for entry in bot.scan_log[:20]:
                st.caption(entry)
        else:
            st.caption("No activity yet. Start the bot to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.header("🔍 Live Scanner")

    # ── Scanner controls ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    from scanner.watchlist_manager import (
        get_all_active_tickers, get_watchlist, set_watchlist,
        add_tickers, remove_tickers, set_enabled_markets,
    )

    total_tickers = len(get_all_active_tickers())
    col1.metric("Watchlist Size",  f"{total_tickers} tickers")
    col2.metric("Scanner Status",  scanner.status[:40] if scanner.status else "Idle")
    col3.metric("Scans Completed", scanner.scan_count)
    if scanner.last_scan:
        ago = int((datetime.utcnow() - scanner.last_scan).total_seconds())
        col4.metric("Last Scan",   f"{ago//60}m {ago%60}s ago")
    else:
        col4.metric("Last Scan",   "Never")

    # Progress bar when scanning
    if scanner.is_scanning:
        done, total = scanner.progress
        if total > 0:
            st.progress(done / total, text=f"Scanning {done}/{total}…")

    col_a, col_b = st.columns(2)
    with col_a:
        if bot.is_running():
            if st.button("🔍 Force Scan Now", type="primary", use_container_width=True):
                bot.force_scan()
                st.toast("Scan triggered — results will update shortly")
        else:
            st.warning("Start the bot to enable scanning.")

    with col_b:
        interval = st.selectbox(
            "Scan every…",
            [5, 10, 15, 30, 60, 120],
            index=[5,10,15,30,60,120].index(int(cfg.get("scan_interval_mins") or 15)),
            format_func=lambda x: f"{x} minutes"
        )
        if st.button("Apply Interval", use_container_width=True):
            cfg.set("scan_interval_mins", interval)
            st.success(f"✅ Scan interval set to {interval} minutes")

    st.divider()

    # ── Live scanner results ──────────────────────────────────────────────────
    st.subheader("📋 Latest Scan Results")
    all_signals = scanner.signals

    if not all_signals:
        st.info("No results yet — scanner hasn't run or found no setups meeting quality gates.")
    else:
        # Quick filters
        fcol1, fcol2 = st.columns(2)
        min_score_filter = fcol1.slider("Min score filter", 1, 10, int(cfg.get("min_score") or 7))
        show_asx = fcol2.checkbox("ASX only", value=False)

        filtered = [s for s in all_signals if s["score"] >= min_score_filter]
        if show_asx:
            filtered = [s for s in filtered if s.get("exchange") == "ASX"]

        st.caption(f"Showing {len(filtered)} of {len(all_signals)} signals")

        if filtered:
            df = pd.DataFrame(filtered)[[
                "ticker","tier","score","prob","entry_price",
                "atr_pct","exchange","signal_date","source"
            ]].copy()
            df["prob"]       = (df["prob"] * 100).round(1).astype(str) + "%"
            df["entry_price"]= df["entry_price"].round(3)
            df["atr_pct"]    = df["atr_pct"].round(1)
            st.dataframe(
                df.rename(columns={
                    "entry_price": "Entry $", "atr_pct": "ATR %",
                    "signal_date": "Found", "source": "Source"
                }),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Watchlist management ──────────────────────────────────────────────────
    st.subheader("📝 Watchlist Management")

    enabled_markets = cfg.get("enabled_markets") or ["ASX", "US"]
    new_markets = st.multiselect(
        "Markets to scan",
        ["ASX", "US"],
        default=enabled_markets,
        help="Pro scans ASX and/or US markets. Deselect to disable."
    )
    if st.button("Save Markets", use_container_width=False):
        cfg.set("enabled_markets", new_markets)
        set_enabled_markets(new_markets)
        st.success("✅ Saved")

    wl_tab1, wl_tab2, wl_tab3 = st.tabs(["ASX Watchlist", "US Watchlist", "Custom Tickers"])

    for market, tab in [("ASX", wl_tab1), ("US", wl_tab2), ("CUSTOM", wl_tab3)]:
        with tab:
            tickers = get_watchlist(market)
            st.caption(f"{len(tickers)} tickers in {market} watchlist")

            # Add
            new_raw = st.text_input(
                f"Add tickers (comma-separated)",
                key=f"add_{market}",
                placeholder="e.g. CBA.AX, WBC.AX" if market == "ASX" else "e.g. AAPL, MSFT"
            )
            if st.button(f"Add to {market}", key=f"addbtn_{market}"):
                if new_raw.strip():
                    new_list = [t.strip().upper() for t in new_raw.split(",") if t.strip()]
                    add_tickers(market, new_list)
                    st.success(f"Added {len(new_list)} ticker(s)")
                    st.rerun()

            # Remove
            remove_raw = st.text_input(
                f"Remove tickers (comma-separated)",
                key=f"rm_{market}",
            )
            if st.button(f"Remove from {market}", key=f"rmbtn_{market}"):
                if remove_raw.strip():
                    rm_list = [t.strip().upper() for t in remove_raw.split(",") if t.strip()]
                    remove_tickers(market, rm_list)
                    st.success(f"Removed {len(rm_list)} ticker(s)")
                    st.rerun()

            # View
            with st.expander(f"View {market} watchlist ({len(tickers)})"):
                st.text(", ".join(tickers))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pos:
    st.header("📋 Open Positions")
    open_pos = db.open_positions()

    if not open_pos:
        st.info("No open positions.")
    else:
        for pos in open_pos:
            curr_price = broker.get_current_price(
                pos["ticker"], pos["exchange"],
                "AUD" if pos["exchange"] == "ASX" else "USD"
            ) or pos["entry_price"]

            unreal_pnl  = (curr_price - pos["entry_price"]) * pos["quantity"]
            unreal_pct  = (curr_price - pos["entry_price"]) / pos["entry_price"] * 100
            entry_dt    = datetime.strptime(pos["entry_date"][:10], "%Y-%m-%d")
            days_held   = (datetime.utcnow() - entry_dt).days
            days_left   = max(0, (pos.get("max_hold_days") or 15) - days_held)
            pnl_icon    = "🟢" if unreal_pnl >= 0 else "🔴"

            with st.container(border=True):
                h1, h2 = st.columns([4, 1])
                h1.markdown(
                    f"**{pos['ticker']}**  ·  {pos['exchange']}  "
                    f"{pnl_icon} **{unreal_pct:+.1f}%**  (${unreal_pnl:+,.0f})"
                )
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Entry",   f"${pos['entry_price']:.3f}")
                c2.metric("Current", f"${curr_price:.3f}")
                c3.metric("Stop",    f"${pos['stop_price']:.3f}")
                c4.metric("Target",  f"${pos['target_price']:.3f}")
                c5.metric("Qty",     f"{pos['quantity']:.0f}")
                c6.metric("Days",    f"{days_held}d  ({days_left}d left)")
                if pos.get("signal_score"):
                    src = pos.get("notes", "")
                    st.caption(
                        f"score={pos['signal_score']:.0f}  "
                        f"prob={pos.get('signal_prob',0)*100:.0f}%  "
                        f"atr={pos.get('atr_pct',0):.1f}%  ·  {src}"
                    )
                if h2.button("Close", key=f"cl_{pos['id']}",
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
    trades = db.all_trades(limit=100)
    if trades:
        df = pd.DataFrame(trades)[[
            "ticker","entry_date","exit_date","entry_price","exit_price",
            "quantity","pnl","pnl_pct","outcome","exit_reason","mode"
        ]].copy()
        df["pnl_pct"] = (df["pnl_pct"] * 100).round(1)
        df["pnl"]     = df["pnl"].round(2)
        st.dataframe(
            df.rename(columns={"pnl_pct": "P&L %", "pnl": "P&L $",
                                "entry_price": "Entry", "exit_price": "Exit",
                                "exit_reason": "Reason"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No closed trades yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.header("📈 Performance")
    metrics = performance_metrics()

    if metrics["trade_count"] == 0:
        st.info("No completed trades yet.")
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
        c4.metric("Sharpe",        f"{metrics['sharpe']:.2f}")

        st.divider()
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
                fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
                name="Cumulative P&L"
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Equity Curve", xaxis_title="Date", yaxis_title="P&L ($)",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                oc = df["outcome"].value_counts()
                fig2 = px.pie(values=oc.values, names=oc.index, title="Outcomes",
                              color_discrete_map={"WIN":"#00d4aa","LOSS":"#ff4b4b"})
                fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                   font=dict(color="white"), height=300)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.histogram(df, x="pnl_pct", nbins=20, title="P&L Distribution",
                                    color_discrete_sequence=["#00d4aa"])
                fig3.add_vline(x=0, line_dash="dash", line_color="gray")
                fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                   font=dict(color="white"), height=300)
                st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_health:
    st.header("🔧 System Health")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Services")
        with st.container(border=True):
            st.markdown(f"**IBKR Gateway:** {'🟢 Connected' if broker.connected else '🔴 Disconnected'}")
            if broker.error:
                st.error(broker.error)
            if broker._last_ping:
                ago = (datetime.utcnow() - broker._last_ping).seconds
                st.caption(f"Last ping: {ago}s ago")
            st.markdown(f"**Bot Runner:** {'🟢 Running' if bot.is_running() else '⚫ Stopped'}")
            st.markdown(f"**Scanner:** {'🟢 Running' if scanner.is_running() else '⚫ Stopped'}")
            if scanner.last_scan:
                ago = int((datetime.utcnow() - scanner.last_scan).total_seconds())
                st.caption(f"Last scan: {ago//60}m ago · {scanner.scan_count} total scans")
            pm = bot.position_manager
            st.markdown(f"**Position Manager:** {'🟢 Running' if pm.is_running() else '⚫ Stopped'}")
            if pm.last_run:
                ago = int((datetime.utcnow() - pm.last_run).total_seconds())
                st.caption(f"Last check: {ago}s ago")

        st.subheader("Risk Controls")
        with st.container(border=True):
            cb = circuit_breaker_active()
            st.markdown(f"**Circuit Breaker:** {'⚠️ ACTIVE' if cb else '✅ Clear'}")
            acct_v = broker.get_account_value()
            exp    = current_exposure(acct_v) if acct_v > 0 else 0.0
            max_e  = cfg.get("max_exposure_pct") or 30.0
            st.markdown(f"**Exposure:** {exp:.1f}% / {max_e:.0f}% max")
            st.progress(min(exp / max_e, 1.0))
            op     = len(db.open_positions())
            mx     = cfg.get("max_positions") or 5
            st.markdown(f"**Positions:** {op} / {mx}")
            st.progress(min(op / mx, 1.0))

    with col2:
        st.subheader("Watchlist & Signals")
        with st.container(border=True):
            tickers = get_all_active_tickers()
            enabled_mkt = cfg.get("enabled_markets") or ["ASX", "US"]
            st.markdown(f"**Active markets:** {', '.join(enabled_mkt)}")
            st.markdown(f"**Total tickers:** {len(tickers)}")
            from scanner.watchlist_manager import get_watchlist
            for mkt in enabled_mkt:
                st.caption(f"  {mkt}: {len(get_watchlist(mkt))} tickers")
            custom = get_watchlist("CUSTOM")
            if custom:
                st.caption(f"  CUSTOM: {len(custom)} tickers")
            st.markdown(f"**Live signals:** {len(scanner.signals)}")

        st.subheader("Recent Errors")
        errors = db.recent_errors(limit=10)
        if errors:
            for e in errors:
                lvl = "🔴" if e["level"] == "ERROR" else "🟡"
                st.caption(f"{lvl} [{e['logged_at'][:16]}] {e['source']}: {e['message']}")
        else:
            st.success("No errors.")

        if pm.last_actions:
            st.subheader("Recent Position Actions")
            for a in pm.last_actions:
                st.caption(f"• {a}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.header("⚙️ Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Management")
        risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0,
                              float(cfg.get("risk_pct") or 2.0), step=0.1)
        max_pos  = st.slider("Max open positions", 1, 20,
                              int(cfg.get("max_positions") or 5))
        max_exp  = st.slider("Max total exposure (%)", 10.0, 100.0,
                              float(cfg.get("max_exposure_pct") or 30.0), step=5.0)
        max_dl   = st.slider("Daily loss limit (%)", 1.0, 10.0,
                              float(cfg.get("max_daily_loss_pct") or 3.0), step=0.5)
        hold_d   = st.slider("Max hold days", 5, 30, int(cfg.get("hold_days") or 15))

        st.subheader("Circuit Breaker")
        cb_losses = st.slider("Losses to trigger", 2, 6,
                               int(cfg.get("cb_consecutive_losses") or 3))
        cb_pause  = st.slider("Pause days", 1, 14, int(cfg.get("cb_pause_days") or 7))

    with col2:
        st.subheader("Signal Quality Gates")
        min_prob  = st.slider("Min probability", 0.50, 0.75,
                               float(cfg.get("min_prob") or 0.53), step=0.01)
        min_score = st.slider("Min score", 5, 10, int(cfg.get("min_score") or 7))

        st.subheader("Stop Loss (× ATR)")
        sl_hi  = st.slider("High-vol  (ATR ≥ 3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_hi")  or 1.2), step=0.1)
        sl_mid = st.slider("Mid-vol   (1.5–3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_mid") or 1.0), step=0.1)
        sl_lo  = st.slider("Low-vol   (< 1.5%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_lo")  or 0.8), step=0.1)

        st.subheader("Profit Targets (%)")
        tgt_hi  = st.slider("High-vol target",  5.0, 25.0,
                             float(cfg.get("target_hi")  or 12.0), step=1.0)
        tgt_mid = st.slider("Mid-vol target",   3.0, 20.0,
                             float(cfg.get("target_mid") or 8.0), step=1.0)
        tgt_lo  = st.slider("Low-vol target",   2.0, 15.0,
                             float(cfg.get("target_lo")  or 5.0), step=1.0)

    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        cfg.set("risk_pct",              risk_pct)
        cfg.set("max_positions",         max_pos)
        cfg.set("max_exposure_pct",      max_exp)
        cfg.set("max_daily_loss_pct",    max_dl)
        cfg.set("hold_days",             hold_d)
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
        st.success("✅ Settings saved!")

    st.divider()
    st.subheader("⚠️ IBKR Connection")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Disconnect", type="secondary"):
            broker.disconnect()
            st.rerun()
    with c2:
        if st.button("🔌 Reconnect", type="secondary"):
            host = cfg.get("ibkr_host") or "127.0.0.1"
            port = int(cfg.get("ibkr_port") or 7497)
            cid  = int(cfg.get("ibkr_client_id") or 1)
            broker.disconnect()
            import time; time.sleep(1)
            ok = broker.connect(host, port, cid)
            st.success("Reconnected ✅") if ok else st.error("Failed ❌")
            st.rerun()

# ── Auto-refresh every 30s while bot is running ───────────────────────────────
if bot.is_running():
    st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)
