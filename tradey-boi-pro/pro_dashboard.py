"""
Tradey Boi Pro — Control Centre
Run with: streamlit run pro_dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import asyncio, sys as _sys
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

import db.database as db
import config.settings as cfg
from engine.risk import performance_metrics, current_exposure, circuit_breaker_active
from scanner.monitor import TIER1_INTERVAL, TIER2_INTERVAL, TIER3_INTERVAL

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

    # ── Regime summary ────────────────────────────────────────────────────────
    _sr = scanner.regimes
    if _sr:
        st.divider()
        _regime_icons = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
        for _mkt, _rd in _sr.items():
            _icon = _regime_icons.get(_rd.regime.value, "⚪")
            st.caption(f"{_icon} {_mkt} {_rd.regime.value}  conf {_rd.confidence:.0%}")

    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    _last_t1 = scanner.last_scans.get("tier1")
    if _last_t1:
        _ago = int((datetime.utcnow() - _last_t1).total_seconds())
        st.caption(f"Last scan: {_ago//60}m {_ago%60}s ago  ·  #{scanner.scan_count}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_scan, tab_pos, tab_perf, tab_health, tab_settings, tab_bt, tab_analysis = st.tabs([
    "📊 Dashboard", "🔍 Scanner", "📋 Positions",
    "📈 Performance", "🔧 Health", "⚙️ Settings", "🧪 Backtest", "🔭 Stock Analysis"
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
            port = 4002 if mode_sel == "Paper Trading" else 4001
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
    c3.metric("ELITE Signals",  len(scanner.elite_signals))
    c4.metric("Total P&L",       f"${metrics['total_pnl']:+,.0f}")
    c5.metric("Win Rate",        f"{metrics['win_rate']*100:.0f}%" if metrics["trade_count"] else "—")

    st.divider()

    # ── Top signals + activity log ───────────────────────────────────────────
    col_sig, col_log = st.columns([1, 1])

    with col_sig:
        st.subheader("🎯 Top Signals")
        from engine.signal_bridge import get_pending_signals
        pending = get_pending_signals(scanner_signals=scanner.actionable_signals)

        if pending:
            for sig in pending[:6]:
                src_icon   = "🔍" if sig.get("source") == "pro_scanner" else "📡"
                tier_icons = {"ELITE": "⭐", "STRONG BUY": "🟢", "BUY": "🔵"}
                tier_icon  = tier_icons.get(sig.get("tier", ""), "⚪")
                csc        = sig.get("composite_score", sig.get("score", 0))
                ai_conf    = sig.get("ai_confidence",   sig.get("prob",  0))
                with st.container(border=True):
                    h1, h2 = st.columns([3, 1])
                    h1.markdown(
                        f"**{sig['ticker']}**  {tier_icon} {sig['tier']}  {src_icon}"
                    )
                    h1.caption(
                        f"Score **{csc:.1f}**  ·  "
                        f"AI conf **{ai_conf*100:.0f}%**  ·  "
                        f"Entry **${sig['entry_price']:.3f}**  ·  "
                        f"R/R {sig.get('risk_reward', '?')}  ·  "
                        f"Regime {sig.get('regime_alignment', '?')}"
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
                "continuously (Tier 1 every 60 min, Tier 2 every 15 min, Tier 3 every 5 min)."
            )

    with col_log:
        st.subheader("📝 Activity Log")
        if bot.scan_log:
            for entry in bot.scan_log[:20]:
                st.caption(entry)
        else:
            st.caption("No activity yet. Start the bot to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCANNER  (Tiered Continuous Monitor)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.header("🔍 Continuous Scanner")
    st.caption(
        "Tier 1 · Full universe every **60 min**  ·  "
        "Tier 2 · Top-50 refresh every **15 min**  ·  "
        "Tier 3 · Top-20 deep watch every **5 min**"
    )

    from scanner.universe import ASX_UNIVERSE, US_UNIVERSE
    from scanner.market_regime import regime_summary

    # ── Market regime cards ───────────────────────────────────────────────────
    st.subheader("🌐 Market Regime")
    live_regimes = scanner.regimes
    rc1, rc2, rc3 = st.columns(3)

    def _regime_card(col, label, rd):
        if rd is None:
            col.metric(label, "—")
            return
        icons = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
        icon  = icons.get(rd.regime.value, "⚪")
        col.metric(
            label,
            f"{icon} {rd.regime.value}",
            delta=f"conf {rd.confidence:.0%}",
            delta_color="off",
        )
        col.caption(
            (f"50EMA {rd.index_pct_50ema:+.1f}%  " if rd.index_pct_50ema else "") +
            (f"VIX {rd.vix}" if rd.vix else "")
        )

    _regime_card(rc1, "🇦🇺 ASX",    live_regimes.get("ASX"))
    _regime_card(rc2, "🇺🇸 US",     live_regimes.get("US"))
    with rc3:
        univ_n = scanner.universe_size if scanner.universe_size > 0 else len(ASX_UNIVERSE) + len(US_UNIVERSE)
        st.metric("Universe Size", f"{univ_n} tickers")
        st.caption(f"ASX {len(ASX_UNIVERSE)}  ·  US {len(US_UNIVERSE)}")

    st.divider()

    # ── Scan tier status ──────────────────────────────────────────────────────
    st.subheader("⏱ Scan Tier Status")
    tc1, tc2, tc3, tc4 = st.columns(4)

    def _ago(dt):
        if dt is None:
            return "Never"
        secs = int((datetime.utcnow() - dt).total_seconds())
        return f"{secs//60}m {secs%60}s ago"

    def _next_in(last_dt, interval_secs, fallback_anchor=None):
        """Return human-readable time until next run of a tier."""
        now = datetime.utcnow()
        if last_dt is not None:
            elapsed   = (now - last_dt).total_seconds()
            remaining = max(0.0, interval_secs - elapsed)
        elif fallback_anchor is not None:
            # Tier hasn't run yet — count from when Tier 1 finished + interval
            elapsed   = (now - fallback_anchor).total_seconds()
            remaining = max(0.0, interval_secs - elapsed)
        else:
            return "Waiting…"
        if remaining < 5:
            return "Any moment…"
        m, s = int(remaining) // 60, int(remaining) % 60
        return f"{m}m {s:02d}s"

    last_scans = scanner.last_scans
    _t1_last   = last_scans.get("tier1")
    _t2_last   = last_scans.get("tier2")
    _t3_last   = last_scans.get("tier3")

    tc1.metric(
        "Tier 1 (60m — full)",
        _ago(_t1_last),
        delta=f"next in {_next_in(_t1_last, TIER1_INTERVAL)}",
    )
    tc2.metric(
        "Tier 2 (15m — top 50)",
        _ago(_t2_last),
        delta=f"next in {_next_in(_t2_last, TIER2_INTERVAL, _t1_last)}",
    )
    tc3.metric(
        "Tier 3  (5m — top 20)",
        _ago(_t3_last),
        delta=f"next in {_next_in(_t3_last, TIER3_INTERVAL, _t1_last)}",
    )
    tc4.metric("Scans Run", scanner.scan_count)

    # Progress bar while scanning
    if scanner.is_scanning:
        done, total = scanner.progress
        if total > 0:
            st.progress(done / total, text=scanner.status)
    else:
        st.caption(scanner.status or "Idle")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if bot.is_running():
            if st.button("⚡ Force Tier 1 Scan Now", type="primary", use_container_width=True):
                bot.force_scan()
                st.toast("Full universe scan triggered — results update in ~60s")
        else:
            st.warning("Start the bot to enable continuous scanning.")

    st.divider()

    # ── Signal results by tier ────────────────────────────────────────────────
    st.subheader("📋 Ranked Opportunities")

    all_signals = scanner.signals

    if not all_signals:
        st.info(
            "No results yet.\n\n"
            "Start the bot — the first Tier 1 scan runs immediately, covering the full "
            f"~{len(ASX_UNIVERSE)+len(US_UNIVERSE)}-ticker universe."
        )
    else:
        # Tier counts
        tier_counts = {t: 0 for t in ["ELITE", "STRONG BUY", "BUY", "WATCH"]}
        for s in all_signals:
            tier_counts[s.get("tier", "WATCH")] = tier_counts.get(s.get("tier", "WATCH"), 0) + 1

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("⭐ ELITE",       tier_counts["ELITE"],       help="Composite ≥ 8.5 — trade immediately")
        kc2.metric("🟢 STRONG BUY", tier_counts["STRONG BUY"], help="Composite ≥ 7.0 — trade next open")
        kc3.metric("🔵 BUY",        tier_counts["BUY"],         help="Composite ≥ 5.5 — watch only")
        kc4.metric("⚪ WATCH",      tier_counts["WATCH"],        help="Below threshold — monitor")

        # Tier filter
        show_tiers = st.multiselect(
            "Show tiers",
            ["ELITE", "STRONG BUY", "BUY", "WATCH"],
            default=["ELITE", "STRONG BUY"],
        )
        show_market = st.selectbox("Market", ["All", "ASX", "US"], index=0)

        filtered = [
            s for s in all_signals
            if s.get("tier", "WATCH") in show_tiers
            and (show_market == "All"
                 or (show_market == "ASX" and s.get("ticker","").endswith(".AX"))
                 or (show_market == "US"  and not s.get("ticker","").endswith(".AX")))
        ]

        st.caption(f"Showing {len(filtered)} of {len(all_signals)} signals")

        if filtered:
            rows = []
            for s in filtered:
                rows.append({
                    "Rank":       s.get("rank", "—"),
                    "Ticker":     s.get("ticker"),
                    "Tier":       s.get("tier", "?"),
                    "Score":      round(float(s.get("composite_score", s.get("score", 0))), 1),
                    "AI Conf":    f"{float(s.get('ai_confidence', s.get('prob', 0)))*100:.0f}%",
                    "R/R":        s.get("risk_reward", "?"),
                    "Entry $":    round(float(s.get("entry_price", 0)), 3),
                    "Stop $":     round(float(s.get("stop_price", 0)), 3),
                    "Target $":   round(float(s.get("target_price", 0)), 3),
                    "ATR %":      round(float(s.get("atr_pct", 0)), 1),
                    "Regime":     s.get("regime_alignment", "?"),
                    "RSI":        round(float(s.get("rsi", 0)), 0) if s.get("rsi") else "—",
                    "Vol Ratio":  round(float(s.get("vol_ratio", 0)), 1) if s.get("vol_ratio") else "—",
                    "Source":     s.get("source", "pro"),
                    "Found":      s.get("signal_date", "")[:16],
                })
            df_sigs = pd.DataFrame(rows)
            st.dataframe(df_sigs, use_container_width=True, hide_index=True)

            # Factor breakdown for top signal
            if filtered and filtered[0].get("ranked_factors"):
                with st.expander(f"🔬 Factor breakdown — #{filtered[0]['ticker']}"):
                    factors = filtered[0]["ranked_factors"]
                    factor_df = pd.DataFrame([
                        {"Factor": k, "Score (0–1)": round(v, 3), "Weight": w}
                        for (k, v), w in zip(
                            factors.items(),
                            [0.30, 0.20, 0.15, 0.20, 0.10, 0.05]
                        )
                    ])
                    st.dataframe(factor_df, hide_index=True, use_container_width=True)

            # Manual trade buttons for ELITE signals
            elite = [s for s in filtered if s.get("tier") == "ELITE"]
            if elite and bot.is_running():
                st.subheader("⭐ ELITE — Trade Now")
                for sig in elite[:3]:
                    with st.container(border=True):
                        ec1, ec2, ec3 = st.columns([3, 2, 1])
                        ec1.markdown(f"**{sig['ticker']}**  ⭐ ELITE  — Score {sig.get('composite_score',0):.1f}")
                        ec2.caption(
                            f"Entry ${sig.get('entry_price',0):.3f}  ·  "
                            f"Stop ${sig.get('stop_price',0):.3f}  ·  "
                            f"Target ${sig.get('target_price',0):.3f}  ·  "
                            f"R/R {sig.get('risk_reward','?')}"
                        )
                        if ec3.button("Trade", key=f"elite_{sig['ticker']}", type="primary"):
                            from engine.executor import execute_signal
                            res = execute_signal(sig, broker)
                            if res["ok"]:
                                st.success(f"✅ {sig['ticker']} order placed")
                            else:
                                st.error(f"❌ {res['reason']}")
                            st.rerun()

    st.divider()

    # ── Universe management ───────────────────────────────────────────────────
    st.subheader("📝 Universe & Markets")

    from scanner.watchlist_manager import (
        get_all_active_tickers, get_watchlist, set_watchlist,
        add_tickers, remove_tickers, set_enabled_markets,
    )

    enabled_markets = cfg.get("enabled_markets") or ["ASX", "US"]
    new_markets = st.multiselect(
        "Markets to scan",
        ["ASX", "US"],
        default=enabled_markets,
        help=f"Pro built-in universe: ASX {len(ASX_UNIVERSE)} + US {len(US_UNIVERSE)} tickers"
    )
    if st.button("Save Markets", use_container_width=False):
        cfg.set("enabled_markets", new_markets)
        set_enabled_markets(new_markets)
        st.success("✅ Saved — new universe applies on next Tier 1 scan")

    wl_tab1, wl_tab2, wl_tab3 = st.tabs(["ASX Additions", "US Additions", "Custom Tickers"])

    for market, tab, placeholder in [
        ("ASX",    wl_tab1, "e.g. CBA.AX, WBC.AX"),
        ("US",     wl_tab2, "e.g. AAPL, MSFT"),
        ("CUSTOM", wl_tab3, "Any ticker"),
    ]:
        with tab:
            tickers = get_watchlist(market)
            if market != "CUSTOM":
                st.caption(
                    f"**{len(tickers)}** custom additions to the built-in "
                    f"{'ASX' if market=='ASX' else 'US'} universe"
                )
            else:
                st.caption(f"{len(tickers)} custom tickers")

            new_raw = st.text_input(
                "Add tickers (comma-separated)",
                key=f"add_{market}", placeholder=placeholder
            )
            if st.button(f"Add to {market}", key=f"addbtn_{market}"):
                if new_raw.strip():
                    new_list = [t.strip().upper() for t in new_raw.split(",") if t.strip()]
                    add_tickers(market, new_list)
                    st.success(f"Added {len(new_list)} ticker(s)")
                    st.rerun()

            remove_raw = st.text_input(
                "Remove tickers (comma-separated)", key=f"rm_{market}"
            )
            if st.button(f"Remove from {market}", key=f"rmbtn_{market}"):
                if remove_raw.strip():
                    rm_list = [t.strip().upper() for t in remove_raw.split(",") if t.strip()]
                    remove_tickers(market, rm_list)
                    st.success(f"Removed {len(rm_list)} ticker(s)")
                    st.rerun()

            if tickers:
                with st.expander(f"View ({len(tickers)})"):
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
        st.info(
            "No completed trades yet.\n\n"
            "Pro records every trade automatically — once the bot executes "
            "and the position closes (stop hit, target hit, or max hold), "
            "full institutional metrics appear here."
        )
    else:
        # ── Row 1: Primary stats ──────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades",            metrics["trade_count"])
        c2.metric("Win Rate",          f"{metrics['win_rate']*100:.1f}%")
        c3.metric("Profit Factor",     f"{metrics['profit_factor']:.3f}",
                  help="Gross profit ÷ gross loss. >1.2 = good, >1.5 = strong")
        c4.metric("Expectancy R",      f"{metrics['expectancy_r']:+.3f}",
                  help="Expected profit per $1 risked. Positive = edge exists")
        c5.metric("Total P&L",         f"${metrics['total_pnl']:+,.0f}")
        c6.metric("Annualised",        f"{metrics['annualised_return_pct']:+.1f}%",
                  help="Return annualised by hold-day weighting")

        # ── Row 2: Risk-adjusted stats ────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Sharpe",            f"{metrics['sharpe']:.2f}",
                  help="Risk-adjusted return (penalises all volatility)")
        c2.metric("Sortino",           f"{metrics['sortino']:.2f}",
                  help="Like Sharpe but only penalises downside volatility")
        c3.metric("Max Drawdown",      f"{metrics['max_drawdown']*100:.1f}%")
        c4.metric("Avg Win",           f"${metrics['avg_win']:,.0f}  ({metrics['avg_gain_pct']:+.1f}%)")
        c5.metric("Avg Loss",          f"${metrics['avg_loss']:,.0f}  ({metrics['avg_loss_pct']:.1f}%)")
        c6.metric("Avg Hold",          f"{metrics['avg_hold_days']:.0f}d")

        # ── Row 3: Streak stats ───────────────────────────────────────────────
        c1, c2, _, _, _, _ = st.columns(6)
        c1.metric("Win Streak (max)",  metrics["win_streak"],
                  help="Longest consecutive winning run")
        c2.metric("Loss Streak (max)", metrics["loss_streak"],
                  help="Longest consecutive losing run — circuit breaker fires at "
                       f"{int(metrics.get('cb_threshold', 3))}")

        st.divider()

        # ── Equity curve ─────────────────────────────────────────────────────
        perf_trades = db.all_trades(limit=500)
        if len(perf_trades) >= 2:
            df = pd.DataFrame(perf_trades)
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
                font=dict(color="white"), height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                oc   = df["outcome"].value_counts()
                fig2 = px.pie(values=oc.values, names=oc.index, title="Outcomes",
                              color_discrete_map={"WIN": "#00d4aa", "LOSS": "#ff4b4b"})
                fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                   font=dict(color="white"), height=280)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.histogram(df, x="pnl_pct", nbins=20,
                                    title="P&L % Distribution",
                                    color_discrete_sequence=["#00d4aa"])
                fig3.add_vline(x=0, line_dash="dash", line_color="gray")
                fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                   font=dict(color="white"), height=280)
                st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ── Monte Carlo simulation ────────────────────────────────────────────
        with st.expander("🎲 Monte Carlo Risk Analysis", expanded=False):
            st.caption(
                "Resamples your actual trade returns 1 000 times to estimate "
                "the range of possible outcomes and risk of ruin."
            )
            import random as _rand

            _rets = [t["pnl_pct"] for t in perf_trades]
            if len(_rets) >= 5:
                N_SIMS   = 1000
                N_TRADES = len(_rets)
                _ruin_thresh = 0.50   # account drops to ≤50%
                _sim_finals  = []
                _sim_maxdds  = []
                for _ in range(N_SIMS):
                    _sample  = _rand.choices(_rets, k=N_TRADES)
                    _eq      = 1.0; _pk = 1.0; _mdd = 0.0
                    for _r in _sample:
                        _eq *= (1 + _r)
                        if _eq > _pk: _pk = _eq
                        _dd = (_pk - _eq) / _pk if _pk > 0 else 0.0
                        if _dd > _mdd: _mdd = _dd
                    _sim_finals.append(_eq)
                    _sim_maxdds.append(_mdd)

                _sorted_f = sorted(_sim_finals)
                _sorted_d = sorted(_sim_maxdds)
                _n = len(_sorted_f)
                _ror = sum(1 for f in _sim_finals if f <= _ruin_thresh) / N_SIMS * 100

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Return — median",
                           f"{(_sorted_f[_n//2]-1)*100:+.1f}%",
                           help=f"p5: {(_sorted_f[_n//20]-1)*100:+.1f}%  ·  "
                                f"p95: {(_sorted_f[_n*19//20]-1)*100:+.1f}%")
                mc2.metric("Max Drawdown — median",
                           f"{_sorted_d[_n//2]*100:.1f}%",
                           help=f"p95 worst-case: {_sorted_d[_n*19//20]*100:.1f}%")
                mc3.metric("Risk of Ruin",
                           f"{_ror:.1f}%",
                           help="Probability account halves over this trade count")

                _mc_caption = (
                    f"{N_SIMS:,} simulations  ·  {N_TRADES} trades resampled per sim  ·  "
                    "Ruin = account ≤ 50%"
                )
                st.caption(_mc_caption)
            else:
                st.info("Need at least 5 closed trades for Monte Carlo.")


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
            _t1_last = scanner.last_scans.get("tier1")
            if _t1_last:
                _t1_ago = int((datetime.utcnow() - _t1_last).total_seconds())
                st.caption(
                    f"Tier 1: {_t1_ago//60}m ago  ·  "
                    f"Tier 2: {_ago(scanner.last_scans.get('tier2'))}  ·  "
                    f"{scanner.scan_count} scans total"
                )
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
        st.subheader("Universe & Signals")
        with st.container(border=True):
            from scanner.universe import ASX_UNIVERSE as _ASX_U, US_UNIVERSE as _US_U
            from scanner.watchlist_manager import get_watchlist as _gwl
            enabled_mkt = cfg.get("enabled_markets") or ["ASX", "US"]
            st.markdown(f"**Active markets:** {', '.join(enabled_mkt)}")
            _built_in = (len(_ASX_U) if "ASX" in enabled_mkt else 0) + \
                        (len(_US_U)  if "US"  in enabled_mkt else 0)
            st.markdown(f"**Built-in universe:** {_built_in} tickers")
            for mkt in enabled_mkt:
                _add = _gwl(mkt)
                if _add:
                    st.caption(f"  + {len(_add)} custom {mkt} additions")
            _custom = _gwl("CUSTOM")
            if _custom:
                st.caption(f"  + {len(_custom)} custom tickers")
            st.divider()
            _sigs    = scanner.signals
            _elite   = len([s for s in _sigs if s.get("tier") == "ELITE"])
            _sbuy    = len([s for s in _sigs if s.get("tier") == "STRONG BUY"])
            st.markdown(f"**Live signals:** {len(_sigs)}  ·  ⭐ {_elite}  🟢 {_sbuy}")

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
                              float(cfg.get("risk_pct") or 2.0), step=0.1, key="set_risk_pct")
        max_pos  = st.slider("Max open positions", 1, 20,
                              int(cfg.get("max_positions") or 5), key="set_max_pos")
        max_exp  = st.slider("Max total exposure (%)", 10.0, 100.0,
                              float(cfg.get("max_exposure_pct") or 30.0), step=5.0, key="set_max_exp")
        max_dl   = st.slider("Daily loss limit (%)", 1.0, 10.0,
                              float(cfg.get("max_daily_loss_pct") or 3.0), step=0.5, key="set_max_dl")
        hold_d   = st.slider("Max hold days", 5, 30, int(cfg.get("hold_days") or 15), key="set_hold_d")

        st.subheader("Circuit Breaker")
        cb_losses = st.slider("Losses to trigger", 2, 6,
                               int(cfg.get("cb_consecutive_losses") or 3), key="set_cb_losses")
        cb_pause  = st.slider("Pause days", 1, 14, int(cfg.get("cb_pause_days") or 7), key="set_cb_pause")

    with col2:
        st.subheader("Signal Quality Gates")
        min_prob      = st.slider("Min probability", 0.50, 0.75,
                                   float(cfg.get("min_prob") or 0.53), step=0.01, key="set_min_prob")
        min_score     = st.slider("Min score (X-style 0–10)", 5, 10,
                                   int(cfg.get("min_score") or 7), key="set_min_score")
        min_composite = st.slider("Min composite score (Pro ranking)", 5.0, 9.5,
                                   float(cfg.get("min_composite") or 7.0), step=0.5,
                                   help="Only STRONG BUY (≥7.0) and ELITE (≥8.5) are traded",
                                   key="set_min_composite")

        st.subheader("Stop Loss (× ATR)")
        sl_hi  = st.slider("High-vol  (ATR ≥ 3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_hi")  or 1.2), step=0.1, key="set_sl_hi")
        sl_mid = st.slider("Mid-vol   (1.5–3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_mid") or 1.0), step=0.1, key="set_sl_mid")
        sl_lo  = st.slider("Low-vol   (< 1.5%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_lo")  or 0.8), step=0.1, key="set_sl_lo")

        st.subheader("Profit Targets (%)")
        tgt_hi  = st.slider("High-vol target",  5.0, 25.0,
                             float(cfg.get("target_hi")  or 12.0), step=1.0, key="set_tgt_hi")
        tgt_mid = st.slider("Mid-vol target",   3.0, 20.0,
                             float(cfg.get("target_mid") or 8.0), step=1.0, key="set_tgt_mid")
        tgt_lo  = st.slider("Low-vol target",   2.0, 15.0,
                             float(cfg.get("target_lo")  or 5.0), step=1.0, key="set_tgt_lo")

        st.subheader("Brokerage")
        brokerage = st.number_input(
            "Brokerage per side ($)", value=float(cfg.get("brokerage") or 2.0),
            min_value=0.0, step=0.5, key="set_brokerage",
            help="Applied to both entry and exit in backtests and P&L calculations"
        )

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
        cfg.set("min_composite",         min_composite)
        cfg.set("sl_mult_hi",            sl_hi)
        cfg.set("sl_mult_mid",           sl_mid)
        cfg.set("sl_mult_lo",            sl_lo)
        cfg.set("target_hi",             tgt_hi)
        cfg.set("target_mid",            tgt_mid)
        cfg.set("target_lo",             tgt_lo)
        cfg.set("brokerage",             brokerage)
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.header("🧪 Walk-Forward Backtest")
    st.caption(
        "Simulates the Pro scanner day-by-day on real historical data. "
        "No lookahead bias — each day's signal only uses data available at that point. "
        "Entry at next day's open. Exits at stop, target, or max hold days."
    )

    # ── Configuration ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Backtest Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        from datetime import date, timedelta
        today = date.today()

        with col1:
            st.markdown("**Date Range**")
            period_choice = st.selectbox(
                "Test period",
                ["1 Month", "3 Months", "6 Months", "12 Months", "Custom"],
                index=2,
            )
            if period_choice == "Custom":
                bt_start = st.date_input("Start date", value=today - timedelta(days=180))
                bt_end   = st.date_input("End date",   value=today - timedelta(days=1))
            else:
                days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "12 Months": 365}
                bt_start = today - timedelta(days=days_map[period_choice])
                bt_end   = today - timedelta(days=1)
                st.caption(f"{bt_start}  →  {bt_end}")

            initial_cap = st.number_input(
                "Starting capital ($)",
                value=10_000, min_value=1_000, step=1_000
            )

        with col2:
            st.markdown("**Markets to test**")
            from scanner.universe import ASX_UNIVERSE as _BT_ASX, US_UNIVERSE as _BT_US
            from scanner.watchlist_manager import get_watchlist as _bt_gwl
            bt_markets = st.multiselect(
                "Markets", ["ASX", "US", "Custom"],
                default=["ASX", "US"],
            )
            st.caption(
                f"ASX: {len(_BT_ASX)} tickers  ·  "
                f"US: {len(_BT_US)} tickers"
            )
            st.markdown("**Quality gates**")
            bt_min_score = st.slider("Min score",       5, 10,  int(cfg.get("min_score") or 7), key="bt_min_score")
            bt_min_prob  = st.slider("Min probability", 0.50, 0.75,
                                     float(cfg.get("min_prob") or 0.53), step=0.01, key="bt_min_prob")

        with col3:
            st.markdown("**Risk parameters**")
            bt_risk_pct  = st.slider("Risk per trade (%)", 0.5, 5.0,
                                     float(cfg.get("risk_pct") or 2.0), step=0.1, key="bt_risk_pct")
            bt_max_pos   = st.slider("Max positions",  1, 10,
                                     int(cfg.get("max_positions") or 5), key="bt_max_pos")
            bt_hold      = st.slider("Max hold days",  5, 30,
                                     int(cfg.get("hold_days") or 15), key="bt_hold")
            bt_brokerage = st.number_input("Brokerage per side ($)",
                                           value=float(cfg.get("brokerage") or 2.0),
                                           step=0.5, key="bt_brokerage")

    # ── Run button ────────────────────────────────────────────────────────────
    run_col, clear_col = st.columns([2, 1])
    run_bt  = run_col.button("▶ Run Backtest", type="primary", use_container_width=True)
    clear_bt = clear_col.button("🗑 Clear Results", use_container_width=True)

    if clear_bt:
        st.session_state.pop("bt_results", None)
        st.rerun()

    if run_bt:
        # Build ticker list from the full Pro universe (not just watchlist additions)
        from scanner.universe import build_universe
        selected_mkts = [m for m in bt_markets if m in ("ASX", "US")]
        bt_tickers: list[str] = build_universe(markets=selected_mkts, apply_liquidity=False)
        if "Custom" in bt_markets:
            bt_tickers.extend(_bt_gwl("CUSTOM"))
        bt_tickers = list(dict.fromkeys(bt_tickers))

        if not bt_tickers:
            st.error("No tickers in selected markets.")
        else:
            bt_params = {
                "min_score":      bt_min_score,
                "min_prob":       bt_min_prob,
                "risk_pct":       bt_risk_pct,
                "max_positions":  bt_max_pos,
                "hold_days":      bt_hold,
                "brokerage":      bt_brokerage,
                "sl_mult_hi":     float(cfg.get("sl_mult_hi")  or 1.2),
                "sl_mult_mid":    float(cfg.get("sl_mult_mid") or 1.0),
                "sl_mult_lo":     float(cfg.get("sl_mult_lo")  or 0.8),
                "target_hi":      float(cfg.get("target_hi")   or 12.0),
                "target_mid":     float(cfg.get("target_mid")  or 8.0),
                "target_lo":      float(cfg.get("target_lo")   or 5.0),
                "cb_consecutive_losses": int(cfg.get("cb_consecutive_losses") or 3),
                "cb_pause_days":  int(cfg.get("cb_pause_days") or 7),
            }

            st.info(
                f"Running backtest on **{len(bt_tickers)} tickers** "
                f"from **{bt_start}** to **{bt_end}**…  "
                f"(this takes 1–3 minutes while data downloads)"
            )

            progress_bar  = st.progress(0.0)
            status_text   = st.empty()

            def _progress(done: int, total: int, msg: str):
                if total > 0:
                    progress_bar.progress(min(done / total, 1.0))
                status_text.caption(msg)

            # Clear previous results immediately so stale data isn't shown during the run
            st.session_state.pop("bt_results", None)

            from backtest.engine import run_backtest
            try:
                with st.spinner(""):
                    results = run_backtest(
                        tickers         = bt_tickers,
                        test_start      = bt_start,
                        test_end        = bt_end,
                        initial_capital = float(initial_cap),
                        params          = bt_params,
                        progress_cb     = _progress,
                    )
                progress_bar.progress(1.0)
                status_text.caption("✅ Backtest complete")
                st.session_state["bt_results"] = results
                st.rerun()
            except Exception as _bt_err:
                progress_bar.progress(1.0)
                st.error(
                    f"❌ **Backtest failed:** {_bt_err}\n\n"
                    "Check your internet connection (yfinance downloads are required) "
                    "and try again. If the error persists, try a shorter date range or "
                    "fewer markets."
                )

    # ── Display results ───────────────────────────────────────────────────────
    results = st.session_state.get("bt_results")
    if results:
        m  = results["metrics"]
        pf = m["profit_factor"]
        wr = m["win_rate"] * 100

        st.divider()

        # ── Pass / Fail banner ────────────────────────────────────────────────
        if pf >= 1.2 and wr >= 45:
            st.success(f"✅ **PASSES** — Profit Factor {pf:.2f}  ·  Win Rate {wr:.0f}%  ·  ROI {m['roi_pct']:+.1f}%")
        elif pf >= 1.0:
            st.warning(f"⚠️ **MARGINAL** — Profit Factor {pf:.2f}  ·  Win Rate {wr:.0f}%  ·  ROI {m['roi_pct']:+.1f}%")
        else:
            st.error(f"❌ **FAILS** — Profit Factor {pf:.2f}  ·  Win Rate {wr:.0f}%  ·  ROI {m['roi_pct']:+.1f}%")

        # ── Key metrics ───────────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades",          m["trade_count"])
        c2.metric("Win Rate",        f"{wr:.1f}%")
        c3.metric("Profit Factor",   f"{pf:.2f}")
        c4.metric("ROI",             f"{m['roi_pct']:+.1f}%")
        c5.metric("Max Drawdown",    f"{m['max_drawdown']*100:.1f}%")
        c6.metric("Sharpe",          f"{m['sharpe']:.2f}")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total P&L",       f"${m['total_pnl']:+,.0f}")
        c2.metric("Avg Win",         f"${m['avg_win']:,.0f}")
        c3.metric("Avg Loss",        f"${m['avg_loss']:,.0f}")
        c4.metric("Avg Hold",        f"{m['avg_hold_days']:.0f}d")
        c5.metric("Tickers Scanned", results.get("tickers_scanned", "—"))
        c6.metric("Trading Days",    results.get("trading_days", "—"))

        _skipped = results.get("tickers_skipped", 0)
        if _skipped:
            st.caption(
                f"ℹ️ {_skipped} ticker{'s' if _skipped != 1 else ''} skipped — "
                "no price data available (likely delisted or not yet listed in the test period). "
                "This is normal and doesn't affect the backtest accuracy."
            )

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        trades = results["trades"]
        eq_crv = results["equity_curve"]

        if not trades:
            st.warning(
                "⚠️ **No trades were generated in this backtest period.**\n\n"
                "The scanner found no tickers that met all conditions simultaneously "
                "(confirmed uptrend, positive MACD, breakout setup, and minimum score). "
                "Try one or more of these:\n"
                "- Lower **Min score** to 5 or 6\n"
                "- Lower **Min probability** to 0.50\n"
                "- Extend the **date range** to 12 months\n"
                "- Add more markets (ASX + US)"
            )

        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            if eq_crv:
                df_eq = pd.DataFrame(eq_crv)
                df_eq["date"] = pd.to_datetime(df_eq["date"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_eq["date"], y=df_eq["equity"],
                    mode="lines", name="Equity",
                    line=dict(color="#00d4aa", width=2),
                    fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
                ))
                fig.add_hline(y=float(initial_cap), line_dash="dash",
                              line_color="gray", annotation_text="Starting capital")
                fig.update_layout(
                    title="Equity Curve (Account + Open P&L)",
                    xaxis_title="Date", yaxis_title="Equity ($)",
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            if trades:
                # Exit reasons pie
                reasons = m.get("exit_reasons", {})
                colour_map = {
                    "TARGET_HIT":   "#00d4aa",
                    "STOP_HIT":     "#ff4b4b",
                    "MAX_HOLD":     "#ffa500",
                    "END_OF_TEST":  "#888888",
                }
                fig2 = px.pie(
                    values=list(reasons.values()),
                    names=list(reasons.keys()),
                    title="Exit Reasons",
                    color=list(reasons.keys()),
                    color_discrete_map=colour_map,
                )
                fig2.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), height=380,
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Monthly P&L bar chart ─────────────────────────────────────────────
        if trades:
            df_t = pd.DataFrame([{
                "exit_date":  t.exit_date,
                "pnl":        t.pnl,
                "pnl_pct":    t.pnl_pct * 100,
                "ticker":     t.ticker,
                "outcome":    t.outcome,
                "exit_reason": t.exit_reason,
                "score":      t.score,
                "prob":       t.prob,
                "hold_days":  t.hold_days,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
            } for t in trades])
            df_t["exit_date"] = pd.to_datetime(df_t["exit_date"])
            df_t["month"] = df_t["exit_date"].dt.to_period("M").astype(str)

            monthly = df_t.groupby("month")["pnl"].sum().reset_index()
            monthly["color"] = monthly["pnl"].apply(lambda x: "#00d4aa" if x >= 0 else "#ff4b4b")

            fig3 = go.Figure(go.Bar(
                x=monthly["month"], y=monthly["pnl"],
                marker_color=monthly["color"],
                name="Monthly P&L"
            ))
            fig3.update_layout(
                title="Monthly P&L ($)",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), height=280,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ── Best / Worst trades ───────────────────────────────────────────────
        if trades:
            best_col, worst_col = st.columns(2)
            df_t_sorted_win  = df_t.sort_values("pnl", ascending=False).head(5)
            df_t_sorted_loss = df_t.sort_values("pnl", ascending=True).head(5)

            with best_col:
                st.subheader("🏆 Best Trades")
                st.dataframe(
                    df_t_sorted_win[["ticker","exit_date","pnl","pnl_pct","hold_days","exit_reason"]].rename(
                        columns={"pnl": "P&L $", "pnl_pct": "P&L %", "hold_days": "Days",
                                 "exit_date": "Date", "exit_reason": "Exit"}
                    ).round(2),
                    hide_index=True, use_container_width=True
                )

            with worst_col:
                st.subheader("📉 Worst Trades")
                st.dataframe(
                    df_t_sorted_loss[["ticker","exit_date","pnl","pnl_pct","hold_days","exit_reason"]].rename(
                        columns={"pnl": "P&L $", "pnl_pct": "P&L %", "hold_days": "Days",
                                 "exit_date": "Date", "exit_reason": "Exit"}
                    ).round(2),
                    hide_index=True, use_container_width=True
                )

        # ── Full trade table ──────────────────────────────────────────────────
        st.subheader("📋 All Trades")
        if trades:
            display_cols = ["ticker","exit_date","pnl","pnl_pct","hold_days",
                            "outcome","exit_reason","score","prob"]
            st.dataframe(
                df_t[display_cols].rename(columns={
                    "pnl": "P&L $", "pnl_pct": "P&L %",
                    "hold_days": "Days", "exit_date": "Date",
                    "exit_reason": "Exit",
                }).round(3).sort_values("Date", ascending=False),
                use_container_width=True, hide_index=True,
            )

            # Download button
            csv = df_t.to_csv(index=False)
            st.download_button(
                "⬇️ Download trades as CSV",
                data=csv,
                file_name=f"backtest_trades_{date.today()}.csv",
                mime="text/csv",
            )

    # ════════════════════════════════════════════════════════════════════════════
    # ROI PROJECTION — Monte Carlo forward simulation
    # Always visible; uses real backtest stats when available, optimised defaults
    # when no backtest has been run yet.
    # ════════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📊 Forward ROI Projection")

    import numpy as _np

    _bt  = st.session_state.get("bt_results")
    _rng = _np.random.default_rng(42)
    _SIMS, _MTHS = 1_000, 12

    if _bt and _bt.get("trades"):
        _bm         = _bt["metrics"]
        _td         = _bt.get("trading_days", 126)
        _cap        = float(_bm["initial_capital"]) or float(initial_cap)
        _pm         = max(_td / 21.0, 0.5)
        _trades_raw = _bt["trades"]

        # ── Build actual monthly portfolio returns from individual trades ──────
        _monthly_map: dict = {}
        for _t in _trades_raw:
            _mo = str(_t.exit_date)[:7]
            _monthly_map[_mo] = _monthly_map.get(_mo, 0.0) + _t.pnl / max(_cap, 1)

        _mo_rets = list(_monthly_map.values())

        if len(_mo_rets) >= 2:
            _monthly_r   = float(_np.mean(_mo_rets))
            _monthly_std = float(_np.std(_mo_rets, ddof=1))
        else:
            _total_roi   = _bm["roi_pct"] / 100.0
            _monthly_r   = (1 + _total_roi) ** (1 / _pm) - 1
            _monthly_std = max(_bm["max_drawdown"] * 0.35, 0.015)

        _ann_roi_bt = ((1 + _monthly_r) ** 12 - 1) * 100
        _tpm        = _bm["trade_count"] / max(_pm, 0.5)

        st.info(
            f"📈 **Your backtest implies {_ann_roi_bt:+.1f}% annualised ROI** "
            f"({_bm['trade_count']} trades · {_bm['win_rate']*100:.0f}% win rate · "
            f"PF {_bm['profit_factor']:.2f} · avg hold {_bm['avg_hold_days']:.0f}d).  "
            f"The fan chart below shows the probability distribution of outcomes "
            f"if that same monthly return / volatility persists forward."
        )

        # ── Vectorised Monte Carlo from actual monthly return distribution ────
        _paths = _np.ones((_SIMS, _MTHS + 1))
        for _mi in range(_MTHS):
            _draws = _rng.normal(_monthly_r, max(_monthly_std, 0.001), _SIMS)
            _paths[:, _mi + 1] = _paths[:, _mi] * (1 + _draws)

    else:
        # ── No usable trades — show estimated projection with note ────────────
        _cap = float(initial_cap)
        _monthly_r   = 0.018
        _monthly_std = 0.045
        _ann_roi_bt  = None
        _tpm         = 9.0

        if _bt:
            # Backtest ran but produced 0 trades
            st.warning(
                "⚠️ **Backtest ran but found 0 trades — showing estimated projection.**  "
                "The scanner's conditions weren't triggered for any ticker in this period. "
                "Try lowering **Min score** to 5 or 6, lowering **Min probability** to 0.50, "
                "or extending the date range to 12 months, then re-run."
            )
        else:
            # No backtest run yet
            st.warning(
                "⚠️ **No backtest run yet — showing estimated projection only.**  "
                "Run a backtest above to replace these estimates with your real stats. "
                "The numbers below assume PF 1.15, 53% win rate, ~9 trades/month."
            )

        # ── Vectorised Monte Carlo ────────────────────────────────────────────
        _paths = _np.ones((_SIMS, _MTHS + 1))
        for _mi in range(_MTHS):
            _draws = _rng.normal(_monthly_r, _monthly_std, _SIMS)
            _paths[:, _mi + 1] = _paths[:, _mi] * (1 + _draws)

    _p10 = _np.percentile(_paths, 10, axis=0)
    _p50 = _np.percentile(_paths, 50, axis=0)
    _p90 = _np.percentile(_paths, 90, axis=0)
    _ax  = list(range(_MTHS + 1))

    # ── Fan chart ─────────────────────────────────────────────────────────────
    _fig_mc = go.Figure()
    _fig_mc.add_trace(go.Scatter(
        x=_ax, y=(_p90 - 1) * 100, fill=None, mode="lines",
        line=dict(color="rgba(0,212,170,0.35)", width=1), name="Optimistic (P90)",
    ))
    _fig_mc.add_trace(go.Scatter(
        x=_ax, y=(_p10 - 1) * 100, fill="tonexty", mode="lines",
        line=dict(color="rgba(255,75,75,0.35)", width=1),
        fillcolor="rgba(100,200,170,0.10)", name="Conservative (P10)",
    ))
    _fig_mc.add_trace(go.Scatter(
        x=_ax, y=(_p50 - 1) * 100, mode="lines",
        line=dict(color="#00d4aa", width=2.5), name="Expected (P50)",
    ))
    _fig_mc.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="Break-even")
    _fig_mc.update_layout(
        title=f"Monte Carlo ROI Projection — 1,000 simulations  ·  ${_cap:,.0f} starting capital",
        xaxis_title="Months", yaxis_title="Portfolio ROI (%)",
        xaxis=dict(tickmode="linear", dtick=1),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(_fig_mc, use_container_width=True)

    # ── Summary table + KPIs ──────────────────────────────────────────────────
    _tbl_col, _kpi_col = st.columns([3, 1])
    with _tbl_col:
        _rows = []
        for _mth in [3, 6, 12]:
            _r10 = (_p10[_mth] - 1) * 100
            _r50 = (_p50[_mth] - 1) * 100
            _r90 = (_p90[_mth] - 1) * 100
            _rows.append({
                "Horizon":              f"{_mth} month{'s' if _mth > 1 else ''}",
                "Conservative (P10)":   f"{_r10:+.1f}%   (${_r10/100*_cap:+,.0f})",
                "Expected (P50)":       f"{_r50:+.1f}%   (${_r50/100*_cap:+,.0f})",
                "Optimistic (P90)":     f"{_r90:+.1f}%   (${_r90/100*_cap:+,.0f})",
            })
        st.dataframe(pd.DataFrame(_rows).set_index("Horizon"), use_container_width=True)

    with _kpi_col:
        _ann_roi   = (_p50[12] - 1) * 100
        _loss_prob = float(_np.mean(_paths[:, 6] < 1.0)) * 100
        _ruin_prob = float(_np.mean(_paths[:, 12] < 0.8)) * 100   # >20% drawdown by year end
        st.metric("Expected 12-month ROI",  f"{_ann_roi:+.1f}%")
        st.metric("Prob. of loss at 6 mo.", f"{_loss_prob:.0f}%")
        st.metric("Prob. >20% loss at 12m", f"{_ruin_prob:.0f}%")
        st.metric("Trades/month",           f"{_tpm:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — STOCK ANALYSIS  (same deep-dive as Tradey Boi X, bot executes instead)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("🔭 Stock Analysis")
    st.caption(
        "Deep-dive any ticker using the same scoring engine the scanner runs. "
        "When ELITE or STRONG BUY — the bot executes automatically. "
        "No manual alerts needed."
    )

    from scanner.universe import ASX_UNIVERSE as _AU, US_UNIVERSE as _UU
    from scanner.market_scanner import _score_signal, _download_batch, _default_params

    _all_tickers = sorted(_AU + _UU)

    col_sel, col_period, col_btn = st.columns([4, 1, 1])
    with col_sel:
        _sel = st.selectbox(
            "Ticker", _all_tickers,
            help="Type to search. ASX tickers end in .AX"
        )
    with col_period:
        _period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1, key="analysis_period")
    with col_btn:
        st.write("")   # vertical align
        _run_analysis = st.button("🔬 Analyse", type="primary", use_container_width=True)

    if _run_analysis or st.session_state.get("_analysis_ticker") == _sel:
        st.session_state["_analysis_ticker"] = _sel
        with st.spinner(f"Fetching {_sel}…"):
            _data = _download_batch([_sel], period=_period)
        _df = _data.get(_sel)

        if _df is None or _df.empty:
            st.error(f"No data for {_sel}. It may be delisted or the ticker is wrong.")
        else:
            _params = _default_params()
            _params["min_score"] = 0      # lower gate so we always get a result
            _params["min_prob"]  = 0.0    # same
            _sig = _score_signal(_df, _sel, _params)

            _row   = _df.iloc[-1]
            _prev  = _df.iloc[-2]
            _close = float(_row["Close"])
            _chg   = (_close - float(_prev["Close"])) / float(_prev["Close"]) * 100

            # ── Indicator math (independent of signal gate) ──────────────────
            import numpy as _np
            _vol   = _df["Volume"].squeeze().dropna()
            _cls   = _df["Close"].squeeze().dropna()
            _hi    = _df["High"].squeeze().dropna()
            _lo    = _df["Low"].squeeze().dropna()
            _avg_vol20 = float(_vol.iloc[-21:-1].mean())
            _vr    = float(_vol.iloc[-1]) / _avg_vol20 if _avg_vol20 > 0 else 1.0
            _ema20 = float(_cls.ewm(span=20, adjust=False).mean().iloc[-1])
            _ema50 = float(_cls.ewm(span=50, adjust=False).mean().iloc[-1])
            _delta = _cls.diff()
            _avg_g = _delta.clip(lower=0).rolling(14).mean().iloc[-1]
            _avg_l = (-_delta).clip(lower=0).rolling(14).mean().iloc[-1]
            _rsi   = 100 - 100 / (1 + _avg_g / _avg_l) if _avg_l > 0 else 100
            _tr    = [max(float(_hi.iloc[i]) - float(_lo.iloc[i]),
                          abs(float(_hi.iloc[i]) - float(_cls.iloc[i-1])),
                          abs(float(_lo.iloc[i]) - float(_cls.iloc[i-1])))
                      for i in range(-15, 0)]
            _atr   = float(_np.mean(_tr))
            _atr_pct = _atr / _close * 100
            _high20  = float(_hi.iloc[-21:-1].max())
            _bp      = (_close - _high20) / _high20 * 100 if _high20 > 0 else 0

            # ── Score reconstruction (for display even when filtered) ────────
            if _sig:
                _score = _sig["score"]
                _prob  = _sig["prob"]
                _tier  = _sig["tier"]
            else:
                _score = 0
                _prob  = min(0.50 + _score * 0.025, 0.82)
                _tier  = "GATED"

            _quality = min(int((_score / 10) * 100), 100)

            # ── Regime for this market ────────────────────────────────────────
            _market      = "ASX" if _sel.endswith(".AX") else "US"
            _live_regime = scanner.regimes.get(_market)
            _reg_icons   = {"BULL": "📈", "NEUTRAL": "↔️", "BEAR": "📉"}
            _reg_label   = (
                f"{_reg_icons.get(_live_regime.regime.value, '⚪')} {_live_regime.regime.value}"
                if _live_regime else "—"
            )
            _rr = (
                round((_sig["target_price"] - _sig["entry_price"]) /
                      max(_sig["entry_price"] - _sig["stop_price"], 0.0001), 2)
                if _sig else "—"
            )

            # ── 7 key metrics (same as X's layout) ───────────────────────────
            mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
            mc1.metric("Price",          f"${_close:.3f}", f"{_chg:+.2f}%")
            mc2.metric("AI Confidence",  f"{_prob*100:.1f}%")
            mc3.metric("RSI",            f"{_rsi:.1f}")
            mc4.metric("Vol Ratio",      f"{_vr:.2f}×")
            mc5.metric("Quality Score",  f"{_quality}/100",
                       help="0-100 composite: breakout strength, volume, trend, RSI, momentum")
            mc6.metric("Regime",         _reg_label)
            mc7.metric("R/R",            f"{_rr}R" if isinstance(_rr, float) else _rr,
                       help="Reward-to-risk ratio")

            # ── Signal label ─────────────────────────────────────────────────
            _tier_styles = {
                "ELITE":      ("🟡 ELITE",      "success"),
                "STRONG BUY": ("🟢 STRONG BUY", "success"),
                "BUY":        ("🔵 BUY",         "info"),
                "GATED":      ("🚫 GATED",        "warning"),
            }
            _tier_label, _tier_type = _tier_styles.get(_tier, (f"⚪ {_tier}", "info"))

            if _tier in ("ELITE", "STRONG BUY"):
                st.success(
                    f"### {_tier_label}  —  Score **{_score}/10**  ·  Quality **{_quality}/100**\n\n"
                    f"🤖 **Bot will auto-execute this signal** — "
                    f"Entry ${_sig['entry_price']:.3f}  ·  "
                    f"Stop ${_sig['stop_price']:.3f}  ·  "
                    f"Target ${_sig['target_price']:.3f}  ·  "
                    f"R/R {_rr}  ·  ATR {_atr_pct:.1f}%"
                )
            elif _tier == "BUY":
                st.info(
                    f"### {_tier_label}  —  Score **{_score}/10**  ·  Quality **{_quality}/100**\n\n"
                    f"Qualifies as BUY tier — bot only executes ELITE & STRONG BUY. "
                    f"Entry ${_sig['entry_price']:.3f}  ·  Stop ${_sig['stop_price']:.3f}  ·  "
                    f"Target ${_sig['target_price']:.3f}"
                )
            else:
                st.warning(
                    f"### {_tier_label}  —  Score **{_score}/10**\n\n"
                    f"Does not meet hard filters — no position will be opened."
                )

            # ── Chart: Candlestick + RSI + MACD ──────────────────────────────
            from plotly.subplots import make_subplots

            _df2 = _df.copy()
            _df2.columns = _df2.columns.get_level_values(0) if hasattr(_df2.columns, 'get_level_values') else _df2.columns
            _c   = _df2["Close"].squeeze()
            _fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
                subplot_titles=(f"{_sel} Price", "RSI (14)", "MACD"),
            )
            _fig.add_trace(go.Candlestick(
                x=_df2.index,
                open=_df2["Open"].squeeze(), high=_df2["High"].squeeze(),
                low=_df2["Low"].squeeze(),   close=_c,
            ), row=1, col=1)
            _ema20s = _c.ewm(span=20, adjust=False).mean()
            _ema50s = _c.ewm(span=50, adjust=False).mean()
            for _y, _nm, _col in [(_ema20s, "EMA20", "orange"), (_ema50s, "EMA50", "royalblue")]:
                _fig.add_trace(go.Scatter(x=_df2.index, y=_y, name=_nm,
                                          line=dict(color=_col, width=1)), row=1, col=1)
            _bb_mid = _c.rolling(20).mean()
            _bb_std = _c.rolling(20).std()
            _fig.add_trace(go.Scatter(x=_df2.index, y=_bb_mid + 2 * _bb_std,
                                       line=dict(color="gray", dash="dot", width=1)), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_df2.index, y=_bb_mid - 2 * _bb_std,
                                       line=dict(color="gray", dash="dot", width=1),
                                       fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)
            _dlt  = _c.diff()
            _ag   = _dlt.clip(lower=0).rolling(14).mean()
            _al   = (-_dlt).clip(lower=0).rolling(14).mean()
            _rsi_s = 100 - 100 / (1 + _ag / _al.replace(0, float("nan")))
            _fig.add_trace(go.Scatter(x=_df2.index, y=_rsi_s,
                                       line=dict(color="purple", width=1.5)), row=2, col=1)
            for _lvl, _dash in [(70, "dash"), (30, "dash"), (65, "dot"), (35, "dot")]:
                _fig.add_hline(y=_lvl, line_dash=_dash,
                               line_color="red" if _lvl >= 65 else "green", row=2, col=1)
            _macd   = _c.ewm(span=12, adjust=False).mean() - _c.ewm(span=26, adjust=False).mean()
            _macd_s = _macd.ewm(span=9, adjust=False).mean()
            _macd_h = _macd - _macd_s
            _colors = ["green" if v >= 0 else "red" for v in _macd_h]
            _fig.add_trace(go.Bar(x=_df2.index, y=_macd_h, marker_color=_colors), row=3, col=1)
            _fig.add_trace(go.Scatter(x=_df2.index, y=_macd,   line=dict(color="royalblue", width=1)), row=3, col=1)
            _fig.add_trace(go.Scatter(x=_df2.index, y=_macd_s, line=dict(color="orange",    width=1)), row=3, col=1)
            _fig.update_layout(
                height=600, showlegend=False, xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"),
            )
            st.plotly_chart(_fig, use_container_width=True)

            # ── Filter & Score breakdown (same as X) ─────────────────────────
            with st.expander("🔍 Filter & Score Breakdown"):
                st.write("**Hard Filters**")
                _filters = [
                    ("20-day high breakout",      _close > _high20 * 1.001),
                    ("Above 50-day EMA (≥ 97%)",  _close >= _ema50 * 0.97),
                    ("RSI not overbought (< 80)",  _rsi < 80),
                    ("Volume confirmation (≥ 1.1×)", _vr >= 1.1),
                    ("ATR sufficient (≥ 0.3%)",   _atr_pct >= 0.3),
                ]
                for _fname, _passed in _filters:
                    st.write(("✅ " if _passed else "❌ ") + _fname)

                if _tier != "GATED":
                    st.write("**Score breakdown (0–10)**")
                    _score_items = [
                        (2 if _bp > 3 else 1 if _bp > 1 else 0,
                         f"Breakout strength ({_bp:+.1f}% above 20d high)", _bp > 0),
                        (2 if _vr > 3 else 1 if _vr > 2 else 0,
                         f"Volume surge ({_vr:.1f}×)", _vr > 2),
                        (2 if _close > _ema20 > _ema50 else 1 if _close > _ema50 else 0,
                         f"EMA uptrend (EMA20 {'>' if _ema20 > _ema50 else '<'} EMA50)", _close > _ema50),
                        (2 if 55 <= _rsi <= 70 else 1 if 50 <= _rsi < 55 else 0,
                         f"RSI sweet spot ({_rsi:.1f} → ideal 55–70)", 50 <= _rsi <= 70),
                        (1 if (_close - float(_prev["Close"])) / float(_prev["Close"]) * 100 > 2 else 0,
                         f"Day momentum ({(_close - float(_prev['Close'])) / float(_prev['Close']) * 100:+.1f}%)", False),
                    ]
                    for _pts, _name, _met in _score_items:
                        _mk = "✅" if _pts > 0 else "—"
                        st.write(f"{_mk} `+{_pts}` {_name}")
                    st.write(f"**Total: {_score}/10  ·  AI confidence: {_prob*100:.1f}%  ·  Quality: {_quality}/100**")
                    st.write(f"**Regime:** {_reg_label}")

    else:
        st.info(
            "Select a ticker above and click **Analyse** to see the full breakdown — "
            "the same analysis the scanner uses 24/7, including chart, "
            "score breakdown, and whether the bot would execute this signal."
        )


# ── Auto-refresh every 30s while bot is running ───────────────────────────────
# Uses streamlit-autorefresh which refreshes via JS without killing the session.
# The old <meta http-equiv='refresh'> caused a full page reload that wiped
# st.session_state and disconnected the broker every 30 seconds.
if bot.is_running():
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, key="dashboard_autorefresh")
