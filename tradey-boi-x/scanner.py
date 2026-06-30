"""
Background scanner — runs automatically every hour during market hours.
Skips weekends and sleeps until the next market open when called outside hours.
Covers both US (NYSE/NASDAQ) and Australian (ASX) markets.
"""
import time
from datetime import datetime, timedelta

import pytz

from engine import (
    WATCHLIST, MAX_ALERTS, CORRELATION_GROUPS,
    get_data, train_model, decide, send_alert,
    log_signal, mark_alerted, update_ticker_performance,
    big_mover_check, send_mover_alert,
)

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


def run_scan(model) -> int:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Scanning {len(WATCHLIST)} tickers…")
    fired          = 0
    alerted_groups: set = set()   # correlation guard — one alert per group per scan

    for ticker in WATCHLIST:
        try:
            df  = get_data(ticker, "6mo")
            if df.empty:
                continue
            res = decide(ticker, df, model)

            if res["alert"] and fired < MAX_ALERTS:
                # Correlation guard — skip if a correlated ticker already alerted this scan
                group_id = _corr_group(ticker)
                if group_id is not None and group_id in alerted_groups:
                    print(f"  ⏭ {ticker}: correlation guard — similar ticker already alerted")
                    continue

                price = float(df.iloc[-1]["Close"])
                sent  = send_alert(ticker, res, price, df)
                if sent:
                    mark_alerted(ticker)
                    log_signal(ticker, price, res["signal"],
                               score=res.get("score", 0),
                               prob=res.get("prob", 0.0))
                    if group_id is not None:
                        alerted_groups.add(group_id)
                    print(f"  ✅ Alert sent: {ticker} {res['label']} (score {res['score']}/14)")
                    fired += 1
            else:
                status = res["label"] if res["signal"] != "GATED" else "🚫 GATED"
                print(f"  — {ticker}: {status}")

                # ── Big Mover check — runs on every non-alerted ticker ────────
                # Catches large moves in progress even when standard gates fail.
                # Passes the trained model so the SETUP tier can use AI probability
                # to reject false positives before sending a Discord alert.
                mover = big_mover_check(ticker, df, model=model)
                if mover:
                    sent = send_mover_alert(ticker, mover, df=df)
                    tier = mover["tier"]
                    if tier == "ACTIVE":
                        detail = f"+{mover['daily_ret']*100:.1f}% | {mover['vol_r']:.1f}× vol"
                    else:
                        detail = f"ai={mover.get('ai_prob',0)*100:.0f}% | adx={mover['adx']:.0f} | obv={mover['obv_r']:.1f}"
                    flag = "alert sent ✅" if sent else "cooldown active"
                    print(f"  [{tier}] {ticker}: {detail} — {flag}")

        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    update_ticker_performance()
    print(f"Scan done. {fired} alert(s) sent.")
    return fired

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  TRADEY BOI X — Auto Scanner")
    print(f"  Interval: every {SCAN_INTERVAL_SECONDS // 60} min during market hours")
    print(f"  Markets: US (09:30–16:00 ET) | ASX (10:00–16:00 AEST)")
    print(f"  Watchlist: {', '.join(WATCHLIST)}")
    print("=" * 50)

    print("\nTraining AI model…")
    model = train_model()
    print("Model ready.\n")

    while True:
        if markets_open():
            run_scan(model)
            print(f"Next scan in {SCAN_INTERVAL_SECONDS // 60} min.\n")
            time.sleep(SCAN_INTERVAL_SECONDS)
        else:
            wait = seconds_until_next_open()
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
    else:
        main()
