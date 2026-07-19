"""
Background scanner — runs automatically every hour during market hours.
Skips weekends and sleeps until the next market open when called outside hours.
Covers both US (NYSE/NASDAQ) and Australian (ASX) markets.
"""
import time
from datetime import datetime, timedelta

import pytz

import engine
from engine import (
    WATCHLIST, MAX_ALERTS, CORRELATION_GROUPS,
    get_data, train_model, decide, send_alert,
    log_signal, mark_alerted, update_ticker_performance,
    big_mover_check, send_mover_alert, resolve_outcomes,
    send_morning_brief, rank_opportunities, send_no_elite_setups_alert,
    market_regime,
)
try:
    from market_open import send_open_report as _send_open_report
    _OPEN_REPORT_AVAILABLE = True
except ImportError:
    _OPEN_REPORT_AVAILABLE = False
try:
    from opportunity import run_opportunity_pass, refresh_regime, wrap_run_scan
    _OPP_AVAILABLE = True
except ImportError:
    _OPP_AVAILABLE = False
    def wrap_run_scan(fn): return fn   # no-op shim when package unavailable

try:
    from diagnostics import trace_logger as _trace
    _DIAGNOSTICS_AVAILABLE = True
except ImportError:
    _DIAGNOSTICS_AVAILABLE = False

try:
    from opportunity.trade_evaluator import process_trade_signal
    from opportunity.config import ENABLE_TRADE_EVALUATOR, SHADOW_MODE
    _TRADE_EVAL_AVAILABLE = True
except ImportError:
    _TRADE_EVAL_AVAILABLE = False
    ENABLE_TRADE_EVALUATOR = False
    SHADOW_MODE = True

try:
    from opportunity.adaptive_core import process_trade_signal as process_adaptive_trade_signal
    from opportunity.config import ENABLE_ADAPTIVE_CORE
    _ADAPTIVE_CORE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_CORE_AVAILABLE = False
    ENABLE_ADAPTIVE_CORE = False

try:
    from opportunity.audit_engine import audit_trade
    from opportunity.config import ENABLE_AUDIT_ENGINE
    _AUDIT_ENGINE_AVAILABLE = True
except ImportError:
    _AUDIT_ENGINE_AVAILABLE = False
    ENABLE_AUDIT_ENGINE = False

try:
    from opportunity.strategy_optimizer import process_trade_signal as process_strategy_signal
    from opportunity.config import ENABLE_STRATEGY_OPTIMIZER
    _STRATEGY_OPTIMIZER_AVAILABLE = True
except ImportError:
    _STRATEGY_OPTIMIZER_AVAILABLE = False
    ENABLE_STRATEGY_OPTIMIZER = False

SCAN_INTERVAL_SECONDS = 3600   # scan every hour while markets are open

# ─── MARKET HOURS ────────────────────────────────────────────────────────────
# US:  Mon–Fri  09:30–16:00 ET
# ASX: Mon–Fri  10:00–16:00 AEST
US_TZ  = pytz.timezone("America/New_York")
ASX_TZ = pytz.timezone("Australia/Sydney")

def _market_open(tz, open_h, open_m, close_h, close_m) -> tuple[bool, datetime]:
    """Returns (is_open, next_open_utc)."""
    now_local = datetime.now(tz)
    weekday   = now_local.weekday()          # 0=Mon … 6=Sun

    open_today  = now_local.replace(hour=open_h,  minute=open_m,  second=0, microsecond=0)
    close_today = now_local.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    is_open = weekday < 5 and open_today <= now_local < close_today

    # Calculate next open
    next_open = open_today
    if weekday >= 5 or now_local >= close_today:
        # Push to next Monday (or next day if only Sat)
        days_ahead = (7 - weekday) % 7 or 7
        if weekday == 5:   days_ahead = 2   # Sat → Mon
        elif weekday == 6: days_ahead = 1   # Sun → Mon
        elif now_local >= close_today:       # past close on weekday → tomorrow
            days_ahead = 1
            if weekday == 4:                 # Friday past close → Monday
                days_ahead = 3
        next_open = (now_local + timedelta(days=days_ahead)).replace(
            hour=open_h, minute=open_m, second=0, microsecond=0)

    return is_open, next_open.astimezone(pytz.utc)

def markets_open() -> bool:
    us_open,  _ = _market_open(US_TZ,  9, 30, 16, 0)
    asx_open, _ = _market_open(ASX_TZ, 10, 0,  16, 0)
    return us_open or asx_open

def seconds_until_next_open() -> int:
    """How many seconds until the earlier of the next US or ASX market open."""
    _, us_next  = _market_open(US_TZ,  9, 30, 16, 0)
    _, asx_next = _market_open(ASX_TZ, 10, 0,  16, 0)
    next_open   = min(us_next, asx_next)
    delta       = (next_open - datetime.now(pytz.utc)).total_seconds()
    return max(int(delta), 0)

# ─── SCAN ────────────────────────────────────────────────────────────────────
def _corr_group(ticker: str) -> int | None:
    """Return the index of this ticker's correlation group, or None if ungrouped."""
    for i, group in enumerate(CORRELATION_GROUPS):
        if ticker in group:
            return i
    return None


def _apply_alert_filters(ticker: str, res: dict, price: float, df) -> bool:
    """
    Run all additive filter layers (trade evaluator, adaptive core, strategy
    optimiser, audit engine) for a single candidate.

    Returns True if the alert should proceed, False if any live (non-shadow)
    layer rejects it. The audit engine always runs regardless of the return
    value — it is logging-only and never gates.
    """
    if ENABLE_TRADE_EVALUATOR:
        try:
            params = engine._trade_params(ticker, res, price, df)
            trade  = {
                "ticker":      ticker,
                "direction":   "LONG",
                "entry":       price,
                "stop_loss":   params["stop_loss"],
                "take_profit": params["target_price"],
                "probability": res.get("prob", 0.0),
                "expected_r":  res.get("expected_r"),
            }
            approved = process_trade_signal(trade, df)
            if not SHADOW_MODE and approved is None:
                print(f"  🧪 {ticker}: rejected by trade evaluator")
                return False
        except Exception as _te:
            print(f"  ⚠️  {ticker}: trade evaluator error ({_te}) — proceeding")

    if ENABLE_ADAPTIVE_CORE:
        try:
            params = engine._trade_params(ticker, res, price, df)
            trade  = {
                "ticker":      ticker,
                "direction":   "LONG",
                "entry":       price,
                "stop_loss":   params["stop_loss"],
                "take_profit": params["target_price"],
                "probability": res.get("prob", 0.0),
                "expected_r":  res.get("expected_r"),
            }
            adaptive_approved = process_adaptive_trade_signal(trade, df)
            if not SHADOW_MODE and adaptive_approved is None:
                print(f"  🧬 {ticker}: rejected by adaptive core")
                return False
        except Exception as _ac:
            print(f"  ⚠️  {ticker}: adaptive core error ({_ac}) — proceeding")

    if ENABLE_STRATEGY_OPTIMIZER:
        try:
            params = engine._trade_params(ticker, res, price, df)
            trade  = {
                "ticker":      ticker,
                "direction":   "LONG",
                "entry":       price,
                "stop_loss":   params["stop_loss"],
                "take_profit": params["target_price"],
                "probability": res.get("prob", 0.0),
                "expected_r":  res.get("expected_r"),
                "why":         res.get("why", []),
                "rsi":         res.get("rsi"),
                "edge_score":  res.get("prob", 0.0),
            }
            strategy_approved = process_strategy_signal(trade, df)
            if not SHADOW_MODE and strategy_approved is None:
                print(f"  🧭 {ticker}: rejected by strategy optimiser")
                return False
        except Exception as _so:
            print(f"  ⚠️  {ticker}: strategy optimiser error ({_so}) — proceeding")

    if ENABLE_AUDIT_ENGINE:
        try:
            params = engine._trade_params(ticker, res, price, df)
            trade  = {
                "ticker":      ticker,
                "direction":   "LONG",
                "entry":       price,
                "stop_loss":   params["stop_loss"],
                "take_profit": params["target_price"],
                "probability": res.get("prob", 0.0),
                "expected_r":  res.get("expected_r"),
            }
            audit_trade(trade, df)
        except Exception as _ae:
            print(f"  ⚠️  {ticker}: audit engine error ({_ae}) — proceeding")

    return True


def run_scan(model) -> int:
    """
    Two-pass scan with Opportunity Ranking (v3):

    Pass 1 — collect all qualifying candidates without alerting, run big-mover
             checks on non-qualifying tickers, gather _scan_data for the
             opportunity engine second pass.

    Pass 2 — rank all candidates by composite quality score (ELITE tier +
             decide() score + AI prob + expected-R + multi-bagger bonus),
             apply the additive filter layers in ranked order, and alert the
             top MAX_ALERTS setups. This ensures the BEST opportunities are
             alerted, not just the first ones found in watchlist order.

    Returns the number of alerts actually sent.
    """
    try:
        entries  = resolve_outcomes()
        resolved = [e for e in entries if e.get("outcome")]
        pending  = [e for e in entries if not e.get("outcome")]
        if resolved:
            wins   = sum(1 for e in resolved if e["outcome"] in ("WIN", "HIT_TARGET", "EXPIRED_GAIN"))
            losses = len(resolved) - wins
            rate   = round(wins / len(resolved) * 100) if resolved else 0
            print(f"  📊 Signal log: {len(resolved)} resolved ({wins}W/{losses}L, {rate}% win rate) | {len(pending)} pending")
    except Exception as _e:
        print(f"  ⚠️  resolve_outcomes error: {_e}")

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Scanning {len(WATCHLIST)} tickers…")

    candidates: list = []   # will be ranked before alerting
    _scan_data: list = []   # for opportunity engine second-pass

    # ── Pass 1: collect all qualifying candidates ────────────────────────────
    for ticker in WATCHLIST:
        try:
            df  = get_data(ticker, "6mo")
            if df.empty:
                continue
            _scan_data.append((ticker, df))
            res = decide(ticker, df, model)

            if res["alert"]:
                price    = float(df.iloc[-1]["Close"])
                group_id = _corr_group(ticker)
                candidates.append({
                    "ticker":   ticker,
                    "res":      res,
                    "price":    price,
                    "df":       df,
                    "group_id": group_id,
                })
                rank = len(candidates)
                print(f"  🎯 {ticker}: {res['label']} (score {res['score']}, prob {res['prob']*100:.0f}%) — candidate #{rank}")
            else:
                status = res["label"] if res["signal"] != "GATED" else "🚫 GATED"
                print(f"  — {ticker}: {status}")

                # Big Mover check on non-qualifying tickers
                mover = big_mover_check(ticker, df, model=model)
                if mover:
                    tier = mover["tier"]
                    sent = send_mover_alert(ticker, mover, df=df)
                    if tier == "ACTIVE":
                        detail = f"+{mover['daily_ret']*100:.1f}% | {mover['vol_r']:.1f}× vol"
                    else:
                        detail = f"ai={mover.get('ai_prob',0)*100:.0f}% | adx={mover['adx']:.0f} | obv={mover['obv_r']:.1f}"
                    print(f"  [{tier}] {ticker}: {detail} — {'alert sent ✅' if sent else 'cooldown active'}")

        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    # ── Pass 2: rank candidates and alert top MAX_ALERTS ────────────────────
    if candidates:
        ranked = rank_opportunities(candidates)
        print(f"\n  📊 Ranking {len(ranked)} candidate(s) by quality score…")
        for i, c in enumerate(ranked):
            t = c["ticker"]; r = c["res"]
            print(f"    #{i+1}  {t}  {r['label']}  score={r['score']}  prob={r['prob']*100:.0f}%  R={r.get('expected_r',0):.2f}")

    fired          = 0
    alerted_groups: set = set()

    for c in (rank_opportunities(candidates) if candidates else []):
        if fired >= MAX_ALERTS:
            break
        ticker   = c["ticker"]
        res      = c["res"]
        price    = c["price"]
        df       = c["df"]
        group_id = c["group_id"]

        # Correlation guard
        if group_id is not None and group_id in alerted_groups:
            print(f"  ⏭ {ticker}: correlation guard — similar ticker already alerted")
            continue

        # Additive filter layers (trade evaluator, adaptive core, strategy optimizer, audit)
        if not _apply_alert_filters(ticker, res, price, df):
            continue

        sent = send_alert(ticker, res, price, df)
        if sent:
            mark_alerted(ticker)
            log_signal(ticker, price, res["signal"],
                       score=res.get("score", 0),
                       prob=res.get("prob", 0.0),
                       features={
                           "regime":        res.get("regime", ""),
                           "quality_score": res.get("quality_score", 0),
                           "rsi":           res.get("rsi", 0),
                           "multibagger":   bool(res.get("multibagger")),
                       })
            if group_id is not None:
                alerted_groups.add(group_id)
            print(f"  ✅ Alert sent: {ticker} {res['label']} (quality {res.get('quality_score',0)}/100, rank #{fired+1})")
            fired += 1

    update_ticker_performance()
    print(f"Scan done. {fired} alert(s) sent ({len(candidates)} candidate(s) found, top {min(fired, MAX_ALERTS)} alerted).")

    # ── Opportunity Engine second pass (additive) ────────────────────────────
    if _OPP_AVAILABLE:
        try:
            run_opportunity_pass(_scan_data)
        except Exception as _e:
            print(f"  ⚠️  Opportunity engine error: {_e}")

    return fired

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
def main(budget_seconds: float | None = None):
    """Run the scan loop. If budget_seconds is set, return cleanly (instead of
    looping forever) once that much wall-clock time has elapsed — this lets a
    single GitHub Actions job scan continuously for its whole time slice and
    still exit normally so the log-commit step afterwards actually runs,
    rather than being hard-killed by the job timeout mid-scan."""
    print("=" * 50)
    print("  TRADEY BOI X — Auto Scanner")
    print(f"  Interval: every {SCAN_INTERVAL_SECONDS // 60} min during market hours")
    print(f"  Markets: US (09:30–16:00 ET) | ASX (10:00–16:00 AEST)")
    print(f"  Watchlist: {len(WATCHLIST)} tickers")
    if budget_seconds:
        print(f"  Time budget: {budget_seconds // 60:.0f} min (will exit cleanly after)")
    print("=" * 50)

    print("\nTraining AI model…")
    model = train_model()
    print("Model ready.\n")

    _start = time.monotonic()

    # ── Dedup state is persisted to disk (scanner_state.json) so a process
    # restart mid-session does not defeat the once-per-calendar-day gate and
    # re-send the morning brief. This is scheduling bookkeeping only — it does
    # not touch the (read-only) morning-brief sender's internals or any
    # trading/signal logic. See diagnostics/ for the audit that found this. ──
    _brief_sent_date: str | None = None
    _asx_report_sent_date: str | None = None
    _us_report_sent_date:  str | None = None
    _no_elite_sent_date:   str | None = None   # v3: send "NO ELITE SETUPS" once per day
    if _DIAGNOSTICS_AVAILABLE:
        _state = _trace.load_scanner_state()
        _brief_sent_date      = _state.get("brief_sent_date")
        _asx_report_sent_date = _state.get("asx_report_sent_date")
        _us_report_sent_date  = _state.get("us_report_sent_date")

    while True:
        if budget_seconds is not None and (time.monotonic() - _start) >= budget_seconds:
            print(f"[{datetime.now().strftime('%H:%M')}] Time budget reached — exiting cleanly.")
            return

        if markets_open():
            # ── Morning brief — once per calendar day when ASX is open ──
            asx_open, _ = _market_open(ASX_TZ, 10, 0, 16, 0)
            today = datetime.now(ASX_TZ).strftime("%Y-%m-%d")
            is_dup = asx_open and _brief_sent_date == today
            if _DIAGNOSTICS_AVAILABLE:
                _trace.log_trace(
                    "morning_brief_check", trigger_source="scheduler_loop",
                    asx_open=asx_open, today=today,
                    brief_sent_date_before=_brief_sent_date,
                    duplication_flag=is_dup,
                )
            if asx_open and _brief_sent_date != today:
                print(f"[{datetime.now().strftime('%H:%M')}] Sending morning brief…")
                ok = send_morning_brief()
                _brief_sent_date = today
                if _DIAGNOSTICS_AVAILABLE:
                    _trace.save_scanner_state({"brief_sent_date": _brief_sent_date})
                    _trace.log_trace(
                        "morning_brief_sent", trigger_source="scheduler_loop",
                        success=ok, brief_sent_date_after=_brief_sent_date,
                        duplication_flag=False,
                    )
                print(f"  Morning brief {'sent ✅' if ok else 'failed ⚠️  (Discord unreachable)'}")

            # ── Open reports — once per session, keyed by local market date ──
            # Fired from inside the scanner loop so they always send at the
            # actual market open (not hours late via a standalone GH cron job).
            today_asx = datetime.now(ASX_TZ).strftime("%Y-%m-%d")
            today_us  = datetime.now(US_TZ).strftime("%Y-%m-%d")
            us_open,  _ = _market_open(US_TZ, 9, 30, 16, 0)

            if _OPEN_REPORT_AVAILABLE and asx_open and _asx_report_sent_date != today_asx:
                print(f"[{datetime.now().strftime('%H:%M')}] Sending ASX open report…")
                ok = _send_open_report("ASX")
                _asx_report_sent_date = today_asx
                if _DIAGNOSTICS_AVAILABLE:
                    _trace.save_scanner_state({
                        "brief_sent_date": _brief_sent_date,
                        "asx_report_sent_date": _asx_report_sent_date,
                        "us_report_sent_date":  _us_report_sent_date,
                    })
                print(f"  ASX open report {'sent ✅' if ok else 'failed ⚠️  (Discord unreachable)'}")

            if _OPEN_REPORT_AVAILABLE and us_open and _us_report_sent_date != today_us:
                print(f"[{datetime.now().strftime('%H:%M')}] Sending US open report…")
                ok = _send_open_report("US")
                _us_report_sent_date = today_us
                if _DIAGNOSTICS_AVAILABLE:
                    _trace.save_scanner_state({
                        "brief_sent_date": _brief_sent_date,
                        "asx_report_sent_date": _asx_report_sent_date,
                        "us_report_sent_date":  _us_report_sent_date,
                    })
                print(f"  US open report {'sent ✅' if ok else 'failed ⚠️  (Discord unreachable)'}")

            fired = wrap_run_scan(run_scan)(model)

            # v3: "NO ELITE SETUPS — HOLD CASH" — once per calendar day when
            # a market-hours scan completes with zero alerts sent.
            if fired == 0 and _no_elite_sent_date != today:
                try:
                    regime = market_regime("SPY")   # broad regime (US/global)
                    ok = send_no_elite_setups_alert(regime)
                    if ok:
                        _no_elite_sent_date = today
                        print(f"  📭 No elite setups — 'HOLD CASH' message sent (regime: {regime})")
                except Exception as _ne:
                    print(f"  ⚠️  No-elite-setups alert error: {_ne}")

            if budget_seconds is not None:
                remaining = budget_seconds - (time.monotonic() - _start)
                if remaining <= 0:
                    print(f"[{datetime.now().strftime('%H:%M')}] Time budget reached — exiting cleanly.")
                    return
                sleep_for = min(SCAN_INTERVAL_SECONDS, remaining)
            else:
                sleep_for = SCAN_INTERVAL_SECONDS
            print(f"Next scan in {sleep_for // 60:.0f} min.\n")
            time.sleep(sleep_for)
        else:
            wait = seconds_until_next_open()
            if budget_seconds is not None:
                remaining = budget_seconds - (time.monotonic() - _start)
                if remaining <= 0:
                    print(f"[{datetime.now().strftime('%H:%M')}] Time budget reached — exiting cleanly.")
                    return
                wait = min(wait, remaining)
            wake = datetime.now() + timedelta(seconds=wait)
            print(f"[{datetime.now().strftime('%H:%M')}] Markets closed. "
                  f"Sleeping until {wake.strftime('%Y-%m-%d %H:%M')} "
                  f"({wait // 3600}h {(wait % 3600) // 60}m).")
            time.sleep(wait)

if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        # Single-run mode for GitHub Actions / cron jobs
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Single scan starting…")
        if markets_open():
            m = train_model()
            run_scan(m)
        else:
            print("Markets closed — skipping scan.")
    elif "--minutes" in sys.argv:
        # Bounded continuous-loop mode — scans every SCAN_INTERVAL_SECONDS for
        # up to N minutes, then exits cleanly. Used by GitHub Actions so a
        # single job covers a whole market session instead of depending on
        # GitHub's cron scheduler to fire reliably every hour (it does not,
        # under load — see market_open scheduling notes).
        _idx = sys.argv.index("--minutes")
        _n_minutes = float(sys.argv[_idx + 1])
        main(budget_seconds=_n_minutes * 60)
    else:
        main()
