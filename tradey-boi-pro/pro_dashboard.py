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
import threading, time as _time_mod

# ── Background backtest state ─────────────────────────────────────────────────
# Stored in a separate module (bt_state.py) so Python's import cache (sys.modules)
# keeps the same dict object alive across every Streamlit rerun.  A module-level
# variable in this file is NOT safe — Streamlit reimports this script fresh on
# each rerun, resetting any plain assignment.
import bt_state as _bts
_BT_STATE = _bts.STATE   # local alias for backwards-compat references below

import db.database as db
import config.settings as cfg
from engine.risk import performance_metrics, current_exposure, circuit_breaker_active
from scanner.monitor import TIER1_INTERVAL, TIER2_INTERVAL, TIER3_INTERVAL

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tradey Boi Pro v1.1",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Init DB + defaults ────────────────────────────────────────────────────────
db.init_db()
cfg.ensure_defaults()
cfg.migrate_settings()   # force-write tuned params over any stale DB values

# ── Backtest lock helpers (time-based auto-expiry) ────────────────────────────
_BT_LOCK_TTL = 20 * 60  # seconds — auto-expire if process was killed

def _is_bt_running() -> bool:
    import time as _t
    if not st.session_state.get("_bt_running", False):
        return False
    age = _t.time() - st.session_state.get("_bt_start_time", 0)
    if age > _BT_LOCK_TTL:
        st.session_state["_bt_running"]   = False   # auto-clear stale lock
        st.session_state["_bt_start_time"] = 0
        return False
    return True

def _set_bt_lock():
    import time as _t
    st.session_state["_bt_running"]    = True
    st.session_state["_bt_start_time"] = _t.time()

def _clear_bt_lock():
    st.session_state["_bt_running"]    = False
    st.session_state["_bt_start_time"] = 0
    # Also clear the background thread state so the polling loop stops
    _BT_STATE["running"] = False
    _BT_STATE["done"]    = False

def _bt_lock_age_str() -> str:
    import time as _t
    secs = int(_t.time() - st.session_state.get("_bt_start_time", _t.time()))
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


# ── Session state ─────────────────────────────────────────────────────────────
if "broker" not in st.session_state:
    from broker.ibkr_client import IBKRClient
    st.session_state.broker = IBKRClient()
    # Auto-connect on first load if settings are already saved
    _b = st.session_state.broker
    _saved_host = cfg.get("ibkr_host")
    _saved_port = cfg.get("ibkr_port")
    if _saved_host and _saved_port:
        _b.connect_async(
            str(_saved_host),
            int(_saved_port),
            int(cfg.get("ibkr_client_id") or 1),
        )

if "bot" not in st.session_state:
    from engine.bot_runner import BotRunner
    st.session_state.bot = BotRunner(st.session_state.broker)
    # Auto-start bot if it was running before a page reload / Streamlit restart.
    # The bot's _trade_cycle skips safely when broker isn't connected yet,
    # so it's safe to start before the IBKR handshake completes.
    if cfg.get("bot_enabled"):
        st.session_state.bot.start()

broker = st.session_state.broker
bot    = st.session_state.bot

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Tradey Boi Pro v1.1")
    mode       = cfg.get("mode") or "PAPER"
    mode_color = "🟡" if mode == "PAPER" else "🔴"
    st.markdown(f"**Mode:** {mode_color} {mode}")

    _conn_icon  = "🟢" if broker.connected else ("🟡" if broker.is_connecting else "🔴")
    _conn_label = "Connected" if broker.connected else ("Reconnecting…" if broker.is_connecting else "Disconnected")
    st.markdown(f"**IBKR:** {_conn_icon} {_conn_label}")

    if broker.connected:
        # Throttle: refresh IBKR account values at most once every 30 s
        import time as _t
        _last_acct_refresh = st.session_state.get("_last_acct_refresh", 0)
        if _t.time() - _last_acct_refresh > 30:
            broker.refresh_account_summary()
            st.session_state["_last_acct_refresh"] = _t.time()
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
        _bt_busy = _is_bt_running()
        if st.button("🔍 Scan Now", use_container_width=True,
                     disabled=_bt_busy,
                     help="Disabled while backtest is running" if _bt_busy else None):
            bot.force_scan()
            st.toast("Scan triggered!")
        if _bt_busy:
            if st.button("🔓 Unlock Scanner", use_container_width=True,
                         help="Force-clear the backtest lock if it stalled"):
                _clear_bt_lock()
                st.rerun()
    else:
        if broker.connected:
            if st.button("▶ Start Bot", use_container_width=True, type="primary"):
                bot.start()
                cfg.set("bot_enabled", True)
                st.rerun()
        elif broker.is_connecting:
            st.button("▶ Start Bot", disabled=True, use_container_width=True,
                      help="Waiting for IBKR connection…")
        else:
            _sb_c1, _sb_c2 = st.columns(2)
            with _sb_c1:
                st.button("▶ Start Bot", disabled=True, use_container_width=True,
                          help="IBKR not connected")
            with _sb_c2:
                if st.button("🔌", use_container_width=True, help="Retry IBKR connection"):
                    broker.connect_async(
                        str(cfg.get("ibkr_host") or "127.0.0.1"),
                        int(cfg.get("ibkr_port") or 4002),
                        int(cfg.get("ibkr_client_id") or 1),
                    )
                    st.rerun()

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
        _has_saved = bool(cfg.get("ibkr_host") and cfg.get("ibkr_port"))

        if _has_saved and broker.is_connecting:
            # Auto-reconnect in progress — show status and let the page continue
            st.info(
                f"🔄 Reconnecting to IB Gateway ({cfg.get('ibkr_host')}:{cfg.get('ibkr_port')})…  "
                "The dashboard will update automatically once connected.",
                icon="🔄",
            )
            # Offer a manual override in case Gateway needs restarting
            if st.button("⚙️ Change connection settings", use_container_width=False):
                cfg.set("ibkr_host", None)   # clear so wizard shows next refresh
                st.rerun()

        elif _has_saved and not broker.is_connecting:
            # Saved settings exist but not currently trying — kick off a fresh attempt
            st.warning(
                f"⚠️ Not connected to IBKR ({cfg.get('ibkr_host')}:{cfg.get('ibkr_port')}).  "
                "Auto-reconnect is active — retrying in the background.",
                icon="⚠️",
            )
            rcol1, rcol2 = st.columns([1, 4])
            with rcol1:
                if st.button("🔌 Reconnect now", type="primary", use_container_width=True):
                    broker.connect_async(
                        str(cfg.get("ibkr_host")),
                        int(cfg.get("ibkr_port")),
                        int(cfg.get("ibkr_client_id") or 1),
                    )
                    st.rerun()
            with rcol2:
                if st.button("⚙️ Change connection settings", use_container_width=True):
                    cfg.set("ibkr_host", None)
                    st.rerun()

        else:
            # ── First-time setup wizard (no saved settings) ──────────────────
            st.header("🔌 Connect to Interactive Brokers")
            st.info(
                "**Before connecting:**\n"
                "1. Install & open [IB Gateway](https://www.interactivebrokers.com.au/en/trading/ibgateway.php) "
                "or Trader Workstation (TWS)\n"
                "2. Enable API: Configure → Settings → API → Enable ActiveX and Socket Clients ✅\n"
                "3. Socket port: **4002** (Gateway paper) or **4001** (Gateway live)\n"
                "4. Click Connect — settings are saved and auto-reconnect activates from then on"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                host = st.text_input("Host", value="127.0.0.1")
            with col2:
                mode_sel = st.selectbox("Mode", ["Paper Trading", "Live Trading"],
                    index=0 if (cfg.get("mode") or "PAPER") == "PAPER" else 1)
                port = 4002 if mode_sel == "Paper Trading" else 4001
            with col3:
                cid = st.number_input("Client ID", value=1, min_value=1, max_value=99)
            if st.button("🔌 Connect", type="primary", use_container_width=True):
                mode_val = "PAPER" if mode_sel == "Paper Trading" else "LIVE"
                cfg.set("ibkr_host", host); cfg.set("ibkr_port", port)
                cfg.set("ibkr_client_id", cid); cfg.set("mode", mode_val)
                with st.spinner("Connecting to IB Gateway…"):
                    ok = broker.connect(host, port, cid)
                if ok:
                    st.success("✅ Connected! Settings saved — auto-reconnect is now active.")
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
    # account_summary is refreshed by the sidebar (30s throttle); just read it here
    acct_val = broker.get_account_value()
    metrics  = performance_metrics()
    open_pos = db.open_positions()

    # Unrealised P&L from open positions (uses price cache populated by Positions tab)
    _acct_cache = st.session_state.get("_pos_price_cache", {})
    _unreal_pnl = sum(
        (_acct_cache[p["ticker"]][1] - p["entry_price"]) * p["quantity"]
        for p in open_pos
        if p["ticker"] in _acct_cache and len(_acct_cache[p["ticker"]]) >= 2
    )

    # If IBKR offline, estimate account value: starting capital + realised + unrealised
    if acct_val == 0:
        starting_cap = float(cfg.get("starting_capital") or cfg.get("initial_capital") or 10000)
        acct_val     = starting_cap + metrics["total_pnl"] + _unreal_pnl

    # Total P&L = closed trades (realised) + open positions (unrealised)
    _total_pnl_display = metrics["total_pnl"] + _unreal_pnl

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Account Value",   f"${acct_val:,.0f}")
    c2.metric("Open Positions",  f"{len(open_pos)} / {cfg.get('max_positions') or 5}")
    c3.metric("ELITE Signals",  len(scanner.elite_signals))
    c4.metric("Total P&L",       f"${_total_pnl_display:+,.0f}",
              delta=f"Realised ${metrics['total_pnl']:+,.0f}  ·  Open ${_unreal_pnl:+,.0f}",
              delta_color="normal" if _total_pnl_display >= 0 else "inverse",
              help="Realised P&L from closed trades + unrealised P&L from open positions")
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
            if bot.is_running():
                if scanner.scan_count == 0:
                    st.info("🔍 First scan in progress — signals will appear here shortly.")
                else:
                    st.info(
                        f"📭 No qualifying signals from the last scan ({scanner.scan_count} scan(s) completed).\n\n"
                        "Markets may be quiet or no setups met the ELITE / STRONG BUY criteria. "
                        "The scanner will keep checking — next update within the tier interval."
                    )
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
        pts   = (f"  {rd.bull_points}🟢/{rd.bear_points}🔴"
                 if rd.bull_points is not None else "")
        col.metric(
            label,
            f"{icon} {rd.regime.value}",
            delta=f"conf {rd.confidence:.0%}{pts}",
            delta_color="off",
        )
        lines = []
        if rd.index_pct_50ema  is not None: lines.append(f"50EMA {rd.index_pct_50ema:+.1f}%")
        if rd.index_pct_200ema is not None: lines.append(f"200EMA {rd.index_pct_200ema:+.1f}%")
        if rd.vix               is not None: lines.append(f"VIX {rd.vix:.1f}")
        if rd.rsi               is not None: lines.append(f"RSI {rd.rsi:.0f}")
        if rd.roc10             is not None: lines.append(f"10d {rd.roc10:+.1f}%")
        if rd.roc50             is not None: lines.append(f"50d {rd.roc50:+.1f}%")
        col.caption("  ·  ".join(lines))

    _regime_card(rc1, "🇦🇺 ASX",    live_regimes.get("ASX"))
    _regime_card(rc2, "🇺🇸 US",     live_regimes.get("US"))
    with rc3:
        univ_n = scanner.universe_size if scanner.universe_size > 0 else len(ASX_UNIVERSE) + len(US_UNIVERSE)
        st.metric("Universe Size", f"{univ_n} tickers")
        st.caption(f"ASX {len(ASX_UNIVERSE)}  ·  US {len(US_UNIVERSE)}")
        if st.button("🔄 Refresh Regime", use_container_width=True,
                     help="Force re-fetch regime data from Yahoo Finance (bypasses 4-hour cache)"):
            from scanner.market_regime import clear_cache
            clear_cache()
            st.toast("Regime cache cleared — fetching fresh data…")
            st.rerun()

    # ── Factor breakdown ──────────────────────────────────────────────────────
    from scanner.market_regime import get_etf_movers
    _fb_asx = live_regimes.get("ASX")
    _fb_us  = live_regimes.get("US")

    if _fb_asx or _fb_us:
        with st.expander("🔬 Regime Factor Breakdown", expanded=False):
            fb_c1, fb_c2 = st.columns(2)
            for _col, _rd, _label in [
                (fb_c1, _fb_asx, "🇦🇺 ASX"),
                (fb_c2, _fb_us,  "🇺🇸 US"),
            ]:
                with _col:
                    st.markdown(f"**{_label}**")
                    if _rd and _rd.factors:
                        _rows = []
                        for f in _rd.factors:
                            _b, _br = f["bull"], f["bear"]
                            if _b > 0:
                                _sign = f"🟢 +{_b} bull"
                            elif _br > 0:
                                _sign = f"🔴 +{_br} bear"
                            else:
                                _sign = "⚪ neutral"
                            _rows.append({
                                "Factor": f["factor"],
                                "Score":  _sign,
                                "Detail": f["note"],
                            })
                        st.dataframe(
                            pd.DataFrame(_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.caption(
                            f"Total: {_rd.bull_points}🟢 bull  /  {_rd.bear_points}🔴 bear  "
                            f"→ threshold: bull ≥ 1.5× bear for BULL"
                        )
                    elif _rd:
                        st.caption("No factor data — refresh regime to populate.")
                    else:
                        st.caption("Regime not yet loaded.")

    # ── Top ETF movers ────────────────────────────────────────────────────────
    st.subheader("📊 Top ETF Movers (Today)")
    etf_c1, etf_c2 = st.columns(2)

    for _col, _market, _label in [(etf_c1, "ASX", "🇦🇺 ASX"), (etf_c2, "US", "🇺🇸 US")]:
        with _col:
            st.markdown(f"**{_label}**")
            _movers = get_etf_movers(_market, n=3)
            if _movers:
                for _m in _movers:
                    _arrow = "▲" if _m["direction"] == "up" else "▼"
                    _clr   = "green" if _m["direction"] == "up" else "red"
                    st.markdown(
                        f":{_clr}[{_arrow} **{_m['ticker']}** ({_m['name']})  "
                        f"{_m['change_pct']:+.2f}%]  `${_m['price']:.2f}`"
                    )
            else:
                st.caption("No data — market may be closed or data unavailable.")

    st.divider()

    # ── Sector performance ────────────────────────────────────────────────────
    from scanner.market_regime import get_sector_performance
    st.subheader("🏭 Sectors")
    sec_c1, sec_c2 = st.columns(2)

    for _col, _market, _label in [(sec_c1, "ASX", "🇦🇺 ASX"), (sec_c2, "US", "🇺🇸 US")]:
        with _col:
            st.markdown(f"**{_label}**")
            _sectors = get_sector_performance(_market)
            if _sectors:
                _rows = []
                for _s in _sectors:
                    _d  = _s["change_1d"]
                    _w  = _s["change_1w"]
                    _d_str = f"{'▲' if _d >= 0 else '▼'} {_d:+.2f}%"
                    _w_str = f"{_w:+.2f}%" if _w is not None else "—"
                    _rows.append({
                        "Sector":  _s["sector"],
                        "Today":   _d_str,
                        "1 Week":  _w_str,
                    })
                _df_sec = pd.DataFrame(_rows)

                def _colour_today(val):
                    colour = "color: #22c55e" if "▲" in str(val) else "color: #ef4444"
                    return colour

                def _colour_week(val):
                    try:
                        v = float(str(val).replace("%", "").replace("+", ""))
                        return "color: #22c55e" if v >= 0 else "color: #ef4444"
                    except Exception:
                        return ""

                st.dataframe(
                    _df_sec.style
                        .map(_colour_today, subset=["Today"])
                        .map(_colour_week,  subset=["1 Week"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No sector data — market may be closed.")

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
            _bt_busy2 = _is_bt_running()
            if _bt_busy2:
                st.warning(f"⏸ Backtest is running ({_bt_lock_age_str()}) — scanner paused to avoid conflicts.")
                if st.button("🔓 Unlock Scanner", key="unlock_scan_tab",
                             help="Force-clear the backtest lock if it stalled"):
                    _clear_bt_lock()
                    st.rerun()
            elif st.button("⚡ Force Tier 1 Scan Now", type="primary", use_container_width=True):
                bot.force_scan()
                st.toast("Full universe scan triggered — results update in ~60s")
        else:
            st.warning("Start the bot to enable continuous scanning.")

    st.divider()

    # ── Signal results by tier ────────────────────────────────────────────────
    st.subheader("📋 Ranked Opportunities")

    all_signals   = scanner.signals       # qualifying signals only (ELITE / STRONG BUY)
    top_scanned   = scanner.top_scanned   # ALL scored tickers including WATCH

    # Use top_scanned as the display pool — always show best N even if nothing qualifies
    display_pool  = top_scanned if top_scanned else all_signals

    if not display_pool:
        if bot.is_running():
            if scanner.scan_count == 0:
                st.info("🔍 First Tier 1 scan in progress — results will appear here once complete.")
            else:
                st.info(
                    f"📭 Scan complete ({scanner.scan_count} run(s)) — no tickers passed the hard filters.\n\n"
                    "Hard filters require: EMA20 > EMA50, MACD positive, RSI 38–72, volume ratio ≥ 1.5×, "
                    "price rising. The bot will rescan on the next tier interval."
                )
        else:
            st.info(
                "No results yet.\n\n"
                "Start the bot — the first Tier 1 scan runs immediately, covering the full "
                f"~{len(ASX_UNIVERSE)+len(US_UNIVERSE)}-ticker universe."
            )
    else:
        # ── Tier summary counts (from qualifying signals) ─────────────────────
        tier_counts = {t: 0 for t in ["ELITE", "STRONG BUY", "BUY", "WATCH"]}
        for s in display_pool:
            t = s.get("tier", "WATCH")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("⭐ ELITE",       tier_counts["ELITE"],       help="Score ≥ 8.5 — bot auto-executes")
        kc2.metric("🟢 STRONG BUY", tier_counts["STRONG BUY"], help="Score ≥ 7.0 — bot auto-executes")
        kc3.metric("🔵 BUY",        tier_counts["BUY"],         help="Score ≥ 5.5 — does not qualify yet")
        kc4.metric("⚪ WATCH",      tier_counts["WATCH"],        help="Score < 5.5 — does not qualify")

        # ── Market filter + top-N selector ────────────────────────────────────
        _fc1, _fc2 = st.columns([2, 1])
        show_market = _fc1.selectbox("Market", ["All", "ASX", "US"], index=0, key="scan_mkt")
        top_n       = _fc2.selectbox("Show top", [10, 20, 50], index=0, key="scan_topn")

        market_filtered = [
            s for s in display_pool
            if show_market == "All"
            or (show_market == "ASX" and s.get("ticker", "").endswith(".AX"))
            or (show_market == "US"  and not s.get("ticker", "").endswith(".AX"))
        ]

        top_signals = market_filtered[:top_n]

        # ── Real execution queue (signals that passed ALL hard filters) ───────
        # Use scanner.actionable_signals (live hard-filter pass only) as the
        # source of truth for badges — NOT get_pending_signals, which also
        # pulls X signal_log.json and inflates the set.
        _live_actioned  = scanner.actionable_signals   # ELITE/STRONG BUY from live scan
        _queued_tickers = {s["ticker"] for s in _live_actioned}
        # Full bridge output (including X signals) is still used for the banner
        from engine.signal_bridge import get_pending_signals as _get_pending
        _real_queue = _get_pending(scanner_signals=_live_actioned)

        # ── Helper: auto-execute status badge ────────────────────────────────
        def _auto_badge(tier: str, ticker: str = "") -> str:
            if ticker in _queued_tickers:
                return "🤖 Auto-executing"
            if tier in ("ELITE", "STRONG BUY"):
                return "❌ Not ELITE/STRONG BUY in live scan"
            if tier == "BUY":
                return "❌ Not ELITE/STRONG BUY in live scan"
            return "❌ Does not qualify"

        total_scanned = len(display_pool)
        qualifying    = tier_counts["ELITE"] + tier_counts["STRONG BUY"]
        st.caption(
            f"Top {len(top_signals)} of {total_scanned} scored tickers — "
            f"{qualifying} qualify for auto-execution · bot auto-executes ELITE and STRONG BUY"
        )

        rows = []
        for i, s in enumerate(top_signals, start=1):
            tier = s.get("tier", "WATCH")
            rows.append({
                "#":            i,
                "Ticker":       s.get("ticker"),
                "Score /10":    round(float(s.get("composite_score", s.get("score", 0))), 2),
                "Tier":         tier,
                "Auto-Execute": _auto_badge(tier, s.get("ticker", "")),
                "AI Conf":      f"{float(s.get('ai_confidence', s.get('prob', 0)))*100:.0f}%",
                "R/R":          s.get("risk_reward", "—"),
                "Entry $":      round(float(s.get("entry_price", 0)), 3),
                "Stop $":       round(float(s.get("stop_price", 0)), 3),
                "Target $":     round(float(s.get("target_price", 0)), 3),
                "RSI":          round(float(s.get("rsi", 0)), 0) if s.get("rsi") else "—",
                "Vol Ratio":    round(float(s.get("vol_ratio", 0)), 1) if s.get("vol_ratio") else "—",
                "Found":        s.get("signal_date", "")[:16],
            })

        df_sigs = pd.DataFrame(rows)
        st.dataframe(df_sigs, use_container_width=True, hide_index=True)

        # ── Auto-executing now banners (only real queue — passed ALL hard filters) ──
        # _real_queue is already fetched above; filter to what's visible in top_signals
        visible_tickers = {s["ticker"] for s in top_signals}
        auto_sigs = [s for s in _real_queue if s.get("ticker") in visible_tickers]
        if auto_sigs:
            for sig in auto_sigs[:5]:
                tier_icon = "⭐" if sig.get("tier") == "ELITE" else "🟢"
                st.success(
                    f"{tier_icon} **{sig['ticker']}** — {sig['tier']}  ·  "
                    f"Score **{sig.get('composite_score', sig.get('score', 0)):.1f}/10**  ·  "
                    f"Entry **${sig.get('entry_price', 0):.3f}**  ·  "
                    f"Stop **${sig.get('stop_price', 0):.3f}**  ·  "
                    f"Target **${sig.get('target_price', 0):.3f}**  ·  "
                    f"R/R **{sig.get('risk_reward', '?')}**  ·  "
                    f"AI **{float(sig.get('ai_confidence', sig.get('prob', 0)))*100:.0f}%**  ·  "
                    f"🤖 Bot has queued this for execution"
                )

        # ── Factor breakdown for #1 signal ────────────────────────────────────
        if top_signals and top_signals[0].get("ranked_factors"):
            with st.expander(f"🔬 Factor breakdown — {top_signals[0]['ticker']} (rank #1)"):
                factors = top_signals[0]["ranked_factors"]
                factor_df = pd.DataFrame([
                    {"Factor": k, "Score (0–1)": round(v, 3), "Weight": w}
                    for (k, v), w in zip(
                        factors.items(),
                        [0.30, 0.20, 0.15, 0.20, 0.10, 0.05]
                    )
                ])
                st.dataframe(factor_df, hide_index=True, use_container_width=True)

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

    # ── Price fetch with yfinance fallback ────────────────────────────────────
    # IBKR price preferred; when gateway is offline fall back to yfinance
    # (delayed ~15 min) so P&L still updates without a live connection.
    import time as _time
    _price_cache: dict = st.session_state.setdefault("_pos_price_cache", {})

    # Fetch portfolio prices once per render from IBKR portfolio() cache —
    # no market-data subscription needed, TWS pushes this automatically.
    _portfolio_px: dict = broker.get_portfolio_prices()

    def _live_price(ticker: str, exchange: str, entry_price: float) -> tuple[float, str]:
        """Returns (price, source) where source is 'IBKR (portfolio)', 'IBKR', 'Yahoo', or 'Entry'."""
        # 1. IBKR portfolio() — best source, no subscription, always current
        port = _portfolio_px.get(ticker) or _portfolio_px.get(ticker.replace(".AX", ""))
        if port:
            px = port["market_price"]
            _price_cache[ticker] = (_time.time(), px, "IBKR")
            return px, "IBKR"

        # 2. IBKR reqMktData — fallback if portfolio() has no entry for this ticker
        ibkr_px = broker.get_current_price(
            ticker, exchange, "AUD" if exchange == "ASX" else "USD"
        )
        if ibkr_px:
            _price_cache[ticker] = (_time.time(), ibkr_px, "IBKR")
            return ibkr_px, "IBKR"

        # 3. Cache (5-min TTL) — avoids hammering Yahoo on every render
        cached = _price_cache.get(ticker)
        if cached and _time.time() - cached[0] < 300:
            return cached[1], cached[2]

        # 4. Yahoo Finance — delayed ~15 min, works when IBKR offline
        try:
            import yfinance as _yf, warnings as _w, io as _io, contextlib as _cl
            _sink = _io.StringIO()
            with _cl.redirect_stderr(_sink), _w.catch_warnings():
                _w.simplefilter("ignore")
                _raw = _yf.download(ticker, period="5d", interval="1d",
                                    auto_adjust=True, progress=False, threads=False)
            if not _raw.empty:
                # Handle both flat and MultiIndex columns (yfinance version differences)
                cols = list(_raw.columns)
                close_col = next(
                    (c for c in cols if (c[0] if isinstance(c, tuple) else c) == "Close"),
                    None
                )
                if close_col is not None:
                    series = _raw[close_col]
                    if hasattr(series, "squeeze"):
                        series = series.squeeze()
                    px = float(series.dropna().iloc[-1])
                    if px > 0:
                        _price_cache[ticker] = (_time.time(), px, "Yahoo")
                        return px, "Yahoo"
        except Exception:
            pass

        return entry_price, "Entry"

    if not open_pos:
        st.info("No open positions.")
    else:
        for pos in open_pos:
            curr_price, price_src = _live_price(
                pos["ticker"], pos["exchange"], pos["entry_price"]
            )

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
                src_label = {"IBKR": "Current (live)", "Yahoo": "Current (delayed)", "Entry": "Current (no data)"}.get(price_src, "Current")
                c1.metric("Entry",      f"${pos['entry_price']:.3f}")
                c2.metric(src_label,    f"${curr_price:.3f}")
                c3.metric("Stop",       f"${pos['stop_price']:.3f}")
                c4.metric("Target",     f"${pos['target_price']:.3f}")
                c5.metric("Qty",        f"{pos['quantity']:.0f}")
                c6.metric("Days",       f"{days_held}d  ({days_left}d left)")
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
    with st.expander("➕ Manually add a position (e.g. to recover after reinstall)"):
        st.caption(
            "Use this to re-enter a position that exists in IBKR but is missing from the bot's "
            "database — for example after reinstalling. The bot will then track its stop, target, "
            "and hold timer as normal."
        )
        mc1, mc2, mc3 = st.columns(3)
        m_ticker   = mc1.text_input("Ticker (e.g. CBA.AX or AAPL)", key="m_ticker").strip().upper()
        m_exchange = mc2.selectbox("Exchange", ["ASX", "NASDAQ", "NYSE", "CBOE", "OTHER"], key="m_exchange")
        m_entry_dt = mc3.date_input("Entry date", key="m_entry_dt")

        mc4, mc5, mc6, mc7 = st.columns(4)
        m_entry_px = mc4.number_input("Entry price ($)", min_value=0.001, step=0.01, format="%.3f", key="m_entry_px")
        m_qty      = mc5.number_input("Quantity (shares)", min_value=1, step=1, key="m_qty")
        m_stop     = mc6.number_input("Stop price ($)",   min_value=0.001, step=0.01, format="%.3f", key="m_stop")
        m_target   = mc7.number_input("Target price ($)", min_value=0.001, step=0.01, format="%.3f", key="m_target")

        mc8, mc9 = st.columns(2)
        m_hold     = mc8.number_input("Max hold days", min_value=1, max_value=60, value=10, key="m_hold")
        m_notes    = mc9.text_input("Notes (optional)", value="manual entry", key="m_notes")

        if st.button("Add Position to Database", type="primary", key="m_add_pos"):
            if not m_ticker:
                st.error("Ticker is required.")
            elif m_entry_px <= 0 or m_stop <= 0 or m_target <= 0 or m_qty <= 0:
                st.error("Entry price, stop, target and quantity must all be greater than 0.")
            elif m_stop >= m_entry_px:
                st.error("Stop price must be below entry price.")
            elif m_target <= m_entry_px:
                st.error("Target price must be above entry price.")
            else:
                db.upsert_position({
                    "ticker":        m_ticker,
                    "exchange":      m_exchange,
                    "entry_price":   float(m_entry_px),
                    "stop_price":    float(m_stop),
                    "target_price":  float(m_target),
                    "quantity":      float(m_qty),
                    "entry_date":    m_entry_dt.isoformat(),
                    "max_hold_days": int(m_hold),
                    "status":        "OPEN",
                    "notes":         m_notes or "manual entry",
                })
                st.success(f"✅ {m_ticker} added — bot will now track stop/target/hold for this position.")
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
        hold_d   = st.slider("Max hold days", 5, 30, int(cfg.get("hold_days") or 10), key="set_hold_d")

        st.subheader("Circuit Breaker")
        cb_losses = st.slider("Losses to trigger", 2, 6,
                               int(cfg.get("cb_consecutive_losses") or 3), key="set_cb_losses")
        cb_pause  = st.slider("Pause days", 1, 14, int(cfg.get("cb_pause_days") or 7), key="set_cb_pause")

    with col2:
        st.subheader("Signal Quality Gates")
        min_prob      = st.slider("Min probability", 0.50, 0.75,
                                   float(cfg.get("min_prob") or 0.53), step=0.01, key="set_min_prob")
        min_score     = st.slider("Min score (X-style 0–10)", 5, 10,
                                   int(cfg.get("min_score") or 6), key="set_min_score")
        min_composite = st.slider("Min composite score (Pro ranking)", 5.0, 9.5,
                                   float(cfg.get("min_composite") or 7.0), step=0.5,
                                   help="Only STRONG BUY (≥7.0) and ELITE (≥8.5) are traded",
                                   key="set_min_composite")

        st.subheader("Stop Loss (× ATR)")
        sl_hi  = st.slider("High-vol  (ATR ≥ 3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_hi")  or 0.8), step=0.1, key="set_sl_hi")
        sl_mid = st.slider("Mid-vol   (1.5–3%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_mid") or 0.6), step=0.1, key="set_sl_mid")
        sl_lo  = st.slider("Low-vol   (< 1.5%)", 0.5, 2.5,
                            float(cfg.get("sl_mult_lo")  or 0.5), step=0.1, key="set_sl_lo")

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
        "This IS the bot simulation — it runs the same scanner, same filters, and same "
        "exit logic on the full live universe day-by-day. "
        "No lookahead bias. Entry at next day's open. Exits at stop, target, or max hold days."
    )

    from datetime import date, timedelta
    today = date.today()

    # ── Bot Simulation Preset ──────────────────────────────────────────────────
    st.info(
        "💡 **To simulate the bot's real performance:** use the **🤖 Bot Simulation Preset** below — "
        "it locks in the full live universe and your current saved settings. "
        "Expect ~10–25 min for 12 months (934 tickers).",
        icon=None,
    )
    if st.button("🤖 Load Bot Simulation Preset", use_container_width=False):
        st.session_state["_bt_preset_period"]    = "12 Months"
        st.session_state["_bt_preset_markets"]   = ["ASX", "US"]
        st.session_state["_bt_preset_min_score"] = int(cfg.get("min_score") or 5)
        st.session_state["_bt_preset_min_prob"]  = float(cfg.get("min_prob") or 0.50)
        st.session_state["_bt_preset_risk_pct"]  = float(cfg.get("risk_pct") or 2.0)
        st.session_state["_bt_preset_max_pos"]   = int(cfg.get("max_positions") or 5)
        st.session_state["_bt_preset_hold"]      = int(cfg.get("hold_days") or 15)
        st.rerun()

    # ── Configuration ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Backtest Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Date Range**")
            _period_options = ["1 Month", "3 Months", "6 Months", "12 Months", "18 Months", "Custom"]
            _period_default = st.session_state.get("_bt_preset_period", "12 Months")
            _period_idx     = _period_options.index(_period_default) if _period_default in _period_options else 3
            period_choice = st.selectbox(
                "Test period",
                _period_options,
                index=_period_idx,
                key="bt_period_choice",
            )
            if period_choice == "Custom":
                bt_start = st.date_input("Start date", value=today - timedelta(days=365))
                bt_end   = st.date_input("End date",   value=today - timedelta(days=1))
            else:
                days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180,
                            "12 Months": 365, "18 Months": 548}
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
            _mkt_default = st.session_state.get("_bt_preset_markets", ["ASX", "US"])
            bt_markets = st.multiselect(
                "Markets", ["ASX", "US", "Custom"],
                default=_mkt_default,
                key="bt_markets_sel",
            )
            st.caption(
                f"ASX: {len(_BT_ASX)} tickers  ·  "
                f"US: {len(_BT_US)} tickers  ·  "
                f"**Total: {len(_BT_ASX)+len(_BT_US)}**"
            )
            st.markdown("**Quality gates**")
            bt_min_score = st.slider("Min score",       1, 10,
                                     st.session_state.get("_bt_preset_min_score",
                                         int(cfg.get("min_score") or 5)),
                                     key="bt_min_score")
            bt_min_prob  = st.slider("Min probability", 0.50, 0.75,
                                     st.session_state.get("_bt_preset_min_prob",
                                         float(cfg.get("min_prob") or 0.50)),
                                     step=0.01, key="bt_min_prob")
            bt_regime    = st.checkbox(
                "Market regime filter",
                value=True,
                key="bt_regime",
                help=(
                    "When ON, skips all long trades on days when the broad market index "
                    "(XJO for ASX, S&P 500 for US) is below its 200-day moving average. "
                    "Prevents trading into bear markets where most stocks fall regardless of signal quality."
                ),
            )
            st.caption(
                "ℹ️ Score and probability here **must match your Settings tab** to simulate the live bot accurately. "
                "Click 🤖 Bot Simulation Preset above to auto-fill them from your saved settings."
            )

        with col3:
            st.markdown("**Risk parameters**")
            # Show the live-bot exit params that will be used (not adjustable here — set in Settings tab)
            _sl_hi  = float(cfg.get("sl_mult_hi")  or 0.8)
            _sl_mid = float(cfg.get("sl_mult_mid") or 0.6)
            _sl_lo  = float(cfg.get("sl_mult_lo")  or 0.5)
            _tg_hi  = float(cfg.get("target_hi")   or 15.0)
            _tg_mid = float(cfg.get("target_mid")  or 10.0)
            _tg_lo  = float(cfg.get("target_lo")   or 7.0)
            _be_r   = float(cfg.get("be_trigger_r")    or 0.5)
            _tr_r   = float(cfg.get("trail_trigger_r") or 1.5)
            _td_r   = float(cfg.get("trail_dist_r")    or 0.7)
            st.info(
                f"📌 **Live bot exit params** (from Settings tab)\n\n"
                f"Stops: {_sl_hi}× / {_sl_mid}× / {_sl_lo}× ATR  \n"
                f"Targets: {_tg_hi:.0f}% / {_tg_mid:.0f}% / {_tg_lo:.0f}%  \n"
                f"BE stop at +{_be_r}R  ·  Trail at +{_tr_r}R  ·  Trail dist {_td_r}R",
                icon=None,
            )
            bt_risk_pct  = st.slider("Risk per trade (%)", 0.5, 5.0,
                                     st.session_state.get("_bt_preset_risk_pct",
                                         float(cfg.get("risk_pct") or 2.0)),
                                     step=0.1, key="bt_risk_pct")
            bt_max_pos   = st.slider("Max positions",  1, 10,
                                     st.session_state.get("_bt_preset_max_pos",
                                         int(cfg.get("max_positions") or 5)),
                                     key="bt_max_pos")
            bt_hold      = st.slider("Max hold days",  5, 30,
                                     st.session_state.get("_bt_preset_hold",
                                         int(cfg.get("hold_days") or 15)),
                                     key="bt_hold")
            bt_min_hold  = st.slider(
                "Min hold days (stop grace period)", 0, 5,
                int(cfg.get("min_hold_days") or 2), key="bt_min_hold",
                help="Stop loss cannot trigger during the first N days after entry. "
                     "Prevents immediate stop-outs from entry-day spread or gap noise.",
            )
            bt_brokerage = st.number_input("Brokerage per side ($)",
                                           value=float(cfg.get("brokerage") or 2.0),
                                           step=0.5, key="bt_brokerage")

    # ── Run button ────────────────────────────────────────────────────────────
    _scan_active = scanner.is_scanning
    _bt_stuck    = _is_bt_running()
    run_col, clear_col, unlock_col = st.columns([2, 1, 1])
    run_bt  = run_col.button(
        "▶ Run Backtest", type="primary", use_container_width=True,
        disabled=_scan_active or _bt_stuck,
        help=(
            "Wait for the active scan to finish first" if _scan_active
            else f"Backtest lock is stuck ({_bt_lock_age_str()}) — click Unlock first" if _bt_stuck
            else None
        ),
    )
    clear_bt  = clear_col.button("🗑 Clear Results", use_container_width=True)
    unlock_bt = unlock_col.button(
        "🔓 Unlock", use_container_width=True,
        disabled=not _bt_stuck,
        help=f"Lock active for {_bt_lock_age_str()} — click to force-clear" if _bt_stuck else "No lock active",
    )

    if unlock_bt:
        _clear_bt_lock()
        st.toast("Backtest lock cleared — scanner is now available.")
        st.rerun()

    if _scan_active:
        st.warning("⏸ Scanner is currently running — wait for it to finish before starting a backtest.")
    if _bt_stuck:
        st.warning("⏸ Backtest lock is active. If the backtest stalled, click **🔓 Unlock** to re-enable the scanner.")

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
                "min_score":         bt_min_score,
                "min_prob":          bt_min_prob,
                "risk_pct":          bt_risk_pct,
                "max_positions":     bt_max_pos,
                "hold_days":         bt_hold,
                "min_hold_days":     bt_min_hold,
                "brokerage":         bt_brokerage,
                # These always pull from live bot settings so backtest = same system
                # Fallbacks match pro-sweep winner params (PF=2.248)
                "sl_mult_hi":        float(cfg.get("sl_mult_hi")  or 0.8),
                "sl_mult_mid":       float(cfg.get("sl_mult_mid") or 0.6),
                "sl_mult_lo":        float(cfg.get("sl_mult_lo")  or 0.5),
                "target_hi":         float(cfg.get("target_hi")   or 15.0),
                "target_mid":        float(cfg.get("target_mid")  or 10.0),
                "target_lo":         float(cfg.get("target_lo")   or 7.0),
                "be_trigger_r":      float(cfg.get("be_trigger_r")    or 0.5),
                "trail_trigger_r":   float(cfg.get("trail_trigger_r") or 1.5),
                "trail_dist_r":      float(cfg.get("trail_dist_r")    or 0.7),
                "cb_consecutive_losses": int(cfg.get("cb_consecutive_losses") or 3),
                "cb_pause_days":     int(cfg.get("cb_pause_days") or 7),
                "use_regime_filter": bt_regime,
            }

            _n_days = (bt_end - bt_start).days
            _est_mins = max(1, round(len(bt_tickers) * _n_days / 25_000))
            if len(bt_tickers) > 80:
                st.warning(
                    f"⚠️ **Large backtest: {len(bt_tickers)} tickers × {_n_days} days** — "
                    f"estimated {_est_mins}–{_est_mins*2} min. "
                    f"Consider selecting one market (ASX or US only) to keep it under 2 minutes."
                )
            else:
                st.info(
                    f"Running backtest on **{len(bt_tickers)} tickers** "
                    f"from **{bt_start}** to **{bt_end}**… "
                    f"(~{_est_mins} min)"
                )

            # ── Launch backtest in a background thread ────────────────────────
            # IMPORTANT: set _BT_STATE["running"] = True HERE in the main thread
            # before st.rerun(), so the polling loop sees it immediately on the
            # next render — not inside the thread (race condition: thread may not
            # start before Streamlit re-renders).
            # The thread ONLY writes to _BT_STATE, never to st.session_state
            # (background threads cannot safely write to Streamlit session state).
            st.session_state.pop("bt_results", None)
            _set_bt_lock()
            _bts.reset()   # sets running=True, clears result/error/progress in sys.modules singleton

            def _bt_thread_fn(tickers, start, end, cap, params):
                import traceback as _tb2, pathlib as _pl2
                import bt_state as _bts2   # import directly — do NOT close over dashboard's _bts alias
                try:
                    from backtest.engine import run_backtest as _rbt
                    res = _rbt(tickers=tickers, test_start=start, test_end=end,
                               initial_capital=float(cap), params=params,
                               progress_cb=_bts2.set_progress)
                    _bts2.finish_ok(res)
                except Exception as _ex:
                    tb = _tb2.format_exc()
                    try:
                        _pl2.Path(__file__).parent.joinpath("bt_error.log").write_text(
                            tb, encoding="utf-8")
                    except Exception:
                        pass
                    _bts2.finish_err(str(_ex), tb)

            _t = threading.Thread(
                target=_bt_thread_fn,
                args=(bt_tickers, bt_start, bt_end, initial_cap, bt_params),
                daemon=True,
            )
            _t.start()
            st.session_state["_bt_thread"] = _t   # store so we can check is_alive()
            st.rerun()   # immediately re-render into the polling UI

    # ── Poll backtest thread progress / harvest results ───────────────────────
    _bt_thread = st.session_state.get("_bt_thread")
    _thread_alive = _bt_thread is not None and _bt_thread.is_alive()

    if _BT_STATE["running"] or _thread_alive:
        _done_u, _total_u, _msg_u = _BT_STATE["progress"]
        _pct = min(_done_u / max(_total_u, 1), 0.99)
        st.progress(_pct, text=f"⏳ {_msg_u}")
        st.caption("Backtest running in background — you can switch tabs freely.")
        _time_mod.sleep(1)
        st.rerun()

    if _BT_STATE["done"] or (not _thread_alive and _bt_thread is not None
                              and not _BT_STATE["running"]):
        # Harvest in main thread (safe to touch session_state here)
        _BT_STATE["done"] = False
        st.session_state.pop("_bt_thread", None)
        _clear_bt_lock()
        if _BT_STATE["error"]:
            st.error(f"❌ **Backtest crashed:** {_BT_STATE['error']}\n\nSaved to `bt_error.log`")
            if _BT_STATE["traceback"]:
                st.code(_BT_STATE["traceback"], language="text")
        elif _BT_STATE["result"] is not None:
            _raw_res = _BT_STATE["result"]
            # Pop heavy preloaded artifacts into their own session keys so they
            # don't travel with every bt_results reference, then reuse them in
            # the stop-parameter sweep without re-downloading anything.
            st.session_state["_bt_preloaded_data"]      = _raw_res.pop("_preloaded_data", None)
            st.session_state["_bt_preloaded_regimes"]   = _raw_res.pop("_preloaded_regimes", None)
            st.session_state["_bt_precomputed_signals"] = _raw_res.pop("_precomputed_signals", None)
            st.session_state["bt_results"] = _raw_res
            _BT_STATE["result"] = None
            st.rerun()
        else:
            # Thread finished but neither result nor error set — show bt_error.log if present
            import pathlib as _pl3
            _elog = _pl3.Path(__file__).parent / "bt_error.log"
            if _elog.exists():
                st.error("❌ **Backtest failed.** Error from `bt_error.log`:")
                st.code(_elog.read_text(encoding="utf-8", errors="replace"), language="text")
            else:
                st.warning("⚠️ Backtest finished with no results — no trades were generated. "
                           "Try widening the date range or lowering the min score.")

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

        _scanned = results.get("tickers_scanned", 0)
        if not trades:
            if _scanned == 0:
                st.error(
                    "❌ **No price data downloaded — possible internet or firewall issue.**\n\n"
                    "yfinance couldn't fetch any historical data. Try:\n"
                    "- Check your internet connection\n"
                    "- Disable VPN or firewall temporarily\n"
                    "- Try selecting only **ASX** market and running again\n"
                    "- Restart the dashboard and try with a shorter date range (3 months)"
                )
            else:
                st.warning(
                    f"⚠️ **Backtest ran on {_scanned} tickers but found 0 qualifying trades.**\n\n"
                    "The signal score threshold is too strict for this period. Fix:\n"
                    "- **Lower Min score** — try 3 or 4 (currently set above the results)\n"
                    "- **Lower Min probability** to 0.50\n"
                    "- **Extend the date range** to 12 months to capture more setups\n"
                    "- Try a different period — some market conditions produce fewer breakouts"
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
                reasons_raw = m.get("exit_reasons", {})
                label_map = {
                    "STOP_HIT":     "Hard Stop (Loss)",
                    "BE_STOP":      "BE Stop (Breakeven)",
                    "TRAIL_STOP":   "Trail Stop (Win)",
                    "TARGET_HIT":   "Hit Target (Win)",
                    "ABOVE_TARGET": "Ran Above Target (Win)",
                    "MAX_HOLD":     "Ran Out of Time",
                    "END_OF_TEST":  "Open at End",
                }
                colour_map = {
                    "Hard Stop (Loss)":         "#ff4b4b",
                    "BE Stop (Breakeven)":       "#ffa500",
                    "Trail Stop (Win)":          "#00d4aa",
                    "Hit Target (Win)":          "#00aaff",
                    "Ran Above Target (Win)":    "#7c5cbf",
                    "Ran Out of Time":           "#888888",
                    "Open at End":               "#444444",
                }
                labels = [label_map.get(k, k) for k in reasons_raw]
                values = list(reasons_raw.values())
                fig2 = px.pie(
                    values=values,
                    names=labels,
                    title="How Trades Exited",
                    color=labels,
                    color_discrete_map=colour_map,
                )
                fig2.update_traces(textinfo="label+percent")
                fig2.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="white"), height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
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

        # ── Exit reason breakdown (diagnostic) ───────────────────────────────
        if trades:
            _label_map = {
                "STOP_HIT":     "Stop Hit",
                "TARGET_HIT":   "Target Hit",
                "ABOVE_TARGET": "Ran Past Target",
                "MAX_HOLD":     "Max Hold",
                "END_OF_TEST":  "Open at End",
                "ELITE_BUMP":   "Elite Bump",
            }
            _er_rows = []
            for _reason, _cnt in sorted(m.get("exit_reasons", {}).items(),
                                        key=lambda x: -x[1]):
                _subset = [t for t in trades if t.exit_reason == _reason]
                _pnl_avg = sum(t.pnl for t in _subset) / len(_subset) if _subset else 0
                _wins    = sum(1 for t in _subset if t.pnl >= 0)
                _er_rows.append({
                    "Exit Reason": _label_map.get(_reason, _reason),
                    "Count":       _cnt,
                    "Win %":       f"{_wins/_cnt*100:.0f}%" if _cnt else "—",
                    "Avg P&L":     f"${_pnl_avg:+,.0f}",
                })
            _er_df = pd.DataFrame(_er_rows)
            st.caption("**Exit breakdown** — use this to diagnose what's driving losses:")
            st.dataframe(_er_df, hide_index=True, use_container_width=True)

        # ── Stop parameter sweep ──────────────────────────────────────────────
        if trades:
            with st.expander("🔍 Find Optimal Stop Parameters — test 27 combinations automatically"):
                st.caption(
                    "Reuses already-downloaded market data (no re-download). "
                    "Tests all combinations of **Break-Even** (0.5 / 1.0 / 1.5 R) × "
                    "**Trail Trigger** (1.5 / 2.0 / 2.5 R) × "
                    "**Trail Distance** (0.5 / 0.7 / 1.0 R). "
                    "Takes ~1-3 minutes."
                )
                _pre_data = st.session_state.get("_bt_preloaded_data")
                _pre_reg  = st.session_state.get("_bt_preloaded_regimes")
                _pre_sig  = st.session_state.get("_bt_precomputed_signals")

                if _pre_data is None:
                    st.info("Preloaded data not found — run the backtest once to enable the sweep.")
                else:
                    if st.button("▶ Run Stop Sweep (27 combos)", key="stop_sweep_btn"):
                        st.session_state.pop("sweep_results", None)
                        _base_p = dict(results["params_used"])
                        _sweep_combos = [
                            {**_base_p,
                             "be_trigger_r":    _be,
                             "trail_trigger_r": _tt,
                             "trail_dist_r":    _td}
                            for _be in [0.5, 1.0, 1.5]
                            for _tt in [1.5, 2.0, 2.5]
                            for _td in [0.5, 0.7, 1.0]
                        ]
                        _sw_bar = st.progress(0.0, text="Starting sweep…")

                        def _sw_cb(done, total, msg):
                            _sw_bar.progress(min(done / max(total, 1), 0.99), text=msg)

                        from backtest.engine import parameter_sweep as _sweep_fn
                        _sw_out = _sweep_fn(
                            tickers=[],
                            test_start=bt_start,
                            test_end=bt_end,
                            sweep=_sweep_combos,
                            initial_capital=float(initial_cap),
                            progress_cb=_sw_cb,
                            preloaded_data=_pre_data,
                            preloaded_regimes=_pre_reg,
                            precomputed_signals=_pre_sig,
                        )
                        _sw_bar.progress(1.0, text="Done!")
                        st.session_state["sweep_results"] = _sw_out
                        st.rerun()

                    _sw_res = st.session_state.get("sweep_results")
                    if _sw_res:
                        _best_s = _sw_res[0]
                        _best_p = _best_s["params"]
                        _best_m = _best_s["metrics"]
                        st.success(
                            f"🏆 Best: BE = **{_best_p['be_trigger_r']}R**, "
                            f"Trail trigger = **{_best_p['trail_trigger_r']}R**, "
                            f"Trail dist = **{_best_p['trail_dist_r']}R**  "
                            f"→ PF **{_best_m['profit_factor']:.3f}** · "
                            f"Win {_best_m['win_rate']*100:.0f}% · "
                            f"ROI {_best_m['roi_pct']:+.1f}%"
                        )
                        _sw_table = []
                        for _rank, _s in enumerate(_sw_res[:15], 1):
                            _sp, _sm = _s["params"], _s["metrics"]
                            _sw_table.append({
                                "#":             _rank,
                                "BE (R)":        _sp.get("be_trigger_r"),
                                "Trail Trig (R)": _sp.get("trail_trigger_r"),
                                "Trail Dist (R)": _sp.get("trail_dist_r"),
                                "PF":            f"{_sm['profit_factor']:.3f}",
                                "Win %":         f"{_sm['win_rate']*100:.0f}%",
                                "Avg Win $":     f"${_sm['avg_win']:,.0f}",
                                "Avg Loss $":    f"${_sm['avg_loss']:,.0f}",
                                "ROI %":         f"{_sm['roi_pct']:+.1f}%",
                                "Trades":        _sm['trade_count'],
                            })
                        st.dataframe(pd.DataFrame(_sw_table), hide_index=True, use_container_width=True)

                        if st.button("✅ Apply Best Parameters to Bot", key="apply_best_stops"):
                            from config import settings as _cfg_sw
                            _cfg_sw.set("be_trigger_r",    _best_p["be_trigger_r"])
                            _cfg_sw.set("trail_trigger_r", _best_p["trail_trigger_r"])
                            _cfg_sw.set("trail_dist_r",    _best_p["trail_dist_r"])
                            st.success(
                                f"✅ Applied to bot — BE={_best_p['be_trigger_r']}R, "
                                f"Trail={_best_p['trail_trigger_r']}R / {_best_p['trail_dist_r']}R"
                            )

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
                "Try lowering **Min score** to 3, lowering **Min probability** to 0.50, "
                "or extending the date range to 12 months, then re-run."
            )
            _rr = _bt.get("rejection_reasons", {})
            if _rr:
                import pandas as _pd_rr
                st.markdown("**Why signals were rejected** (top filters across all tickers × days):")
                _rr_df = _pd_rr.DataFrame(
                    [{"Filter": k, "Rejections": v} for k, v in list(_rr.items())[:10]],
                )
                st.dataframe(_rr_df, use_container_width=True, hide_index=True)
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
