"""
Manual full simulation of every alert type / branch in the system.
Not part of pytest — run directly: python3 tests/manual_full_alert_sim.py

Mocks:
  - requests.post / urllib.request.urlopen -> captured, never actually sent
  - datetime.now(tz) inside engine.py -> controlled "fake now" per scenario

Prints every generated message plus a PASS/FAIL verdict per scenario based on
sanity checks (no crash, contains expected markers, no leftover placeholders).
"""
import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz
from datetime import datetime, timedelta

import engine

RESULTS = []

def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"    [{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))


class FakeDatetime(datetime):
    """Patch engine.datetime so datetime.now(tz) returns a fixed instant."""
    _fixed = None

    @classmethod
    def now(cls, tz=None):
        base = cls._fixed
        if tz is not None:
            return base.astimezone(tz)
        return base


def set_fake_now(aest_naive_str):
    """aest_naive_str like '2026-07-06 11:00:00' interpreted as AEST wall clock."""
    aest = pytz.timezone("Australia/Sydney")
    dt = aest.localize(datetime.strptime(aest_naive_str, "%Y-%m-%d %H:%M:%S"))
    FakeDatetime._fixed = dt
    engine.datetime = FakeDatetime


captured_posts = []
def fake_requests_post(url, json=None, timeout=5, **kw):
    captured_posts.append(json["content"] if json else None)
    class R:
        status_code = 200
    return R()

engine.requests.post = fake_requests_post
engine.DISCORD = "http://fake-webhook"
engine._guard_ok = lambda t: True


def get_df(ticker):
    return engine.get_data(ticker, "6mo")


def make_result(why, score=12, prob=0.55):
    return {
        "signal": "GOOD BUY", "label": "✅ GOOD BUY", "color": "#0f0",
        "alert": True, "prob": prob, "score": score,
        "why": why, "filters": [], "adj": 0, "rsi": 50.0,
    }


SWING_WHY = ["RSI in ideal zone (35-65)", "EMA uptrend confirmed", "Breaking 20-day resistance"]
INTRADAY_WHY = ["VWAP cross-above on volume surge — institutional repositioning long (VWAP $100.00)",
                "Gap-up on institutional volume — strong overnight buying"]

# Monday 6 Jul 2026 dates for deterministic weekday control
DATES = {
    "asx_open_first30":   "2026-07-06 10:15:00",   # Mon, ASX opening window
    "asx_open_mid":       "2026-07-06 13:00:00",   # Mon, ASX mid-session
    "asx_closed":         "2026-07-06 20:00:00",   # Mon, ASX closed (after 4pm)
    "us_open_first30":    "2026-07-06 23:45:00",   # Mon night AEST = US opening window
    "us_open_mid":        "2026-07-07 02:00:00",   # Tue AEST = US mid-session
    "us_closed":          "2026-07-06 12:00:00",   # Mon midday AEST = US closed
}


def scenario(label, ticker, when_key, why, section):
    print(f"\n--- {section}: {label} ({ticker}, {when_key}) ---")
    set_fake_now(DATES[when_key])
    df = get_df(ticker)
    price = float(df.iloc[-1]["Close"])
    result = make_result(why)
    captured_posts.clear()
    try:
        ok = engine.send_alert(ticker, result, price, df)
        msg = captured_posts[-1] if captured_posts else None
        print(msg)
        check(f"{section}/{label}: send returned True", ok is True)
        check(f"{section}/{label}: message captured", msg is not None)
        if msg:
            check(f"{section}/{label}: no crash placeholders", "None" not in msg and "Traceback" not in msg)
    except Exception as e:
        print("EXCEPTION:", e)
        traceback.print_exc()
        check(f"{section}/{label}: no exception", False, str(e))


print("=" * 70)
print("SECTION 1: send_alert() — Opening window (buy now)")
print("=" * 70)
scenario("ASX opening window", "CBA.AX", "asx_open_first30", SWING_WHY, "send_alert")
scenario("US opening window",  "AAPL",   "us_open_first30",  SWING_WHY, "send_alert")

print("\n" + "=" * 70)
print("SECTION 2: send_alert() — Mid-session open (RSI-based entry)")
print("=" * 70)
scenario("ASX mid-session", "CBA.AX", "asx_open_mid", SWING_WHY, "send_alert")
scenario("US mid-session",  "AAPL",   "us_open_mid",  SWING_WHY, "send_alert")

print("\n" + "=" * 70)
print("SECTION 3: send_alert() — Intraday signal, market closed")
print("=" * 70)
scenario("ASX intraday+closed", "CBA.AX", "asx_closed", INTRADAY_WHY, "send_alert")
scenario("US intraday+closed",  "AAPL",   "us_closed",  INTRADAY_WHY, "send_alert")

print("\n" + "=" * 70)
print("SECTION 4: send_alert() — Swing signal, market closed")
print("=" * 70)
scenario("ASX swing+closed", "CBA.AX", "asx_closed", SWING_WHY, "send_alert")
scenario("US swing+closed",  "AAPL",   "us_closed",  SWING_WHY, "send_alert")

print("\n" + "=" * 70)
print("SECTION 5: send_alert() — Mixed swing+intraday reasons (regression check)")
print("=" * 70)
scenario("ASX mixed reasons, closed", "CBA.AX", "asx_closed", SWING_WHY + [
    "Trading above VWAP $100.00 — bullish positioning",
    "1-hour trend aligned with daily signal (EMA + MACD bullish)",
], "send_alert")


def mover_scenario(label, ticker, when_key, tier, section):
    print(f"\n--- {section}: {label} ({ticker}, {when_key}, tier={tier}) ---")
    set_fake_now(DATES[when_key])
    df = get_df(ticker)
    price = float(df.iloc[-1]["Close"])
    mover = {
        "tier": tier, "price": price, "rsi": 55, "daily_ret": 0.045,
        "vol_r": 2.1, "atr_exp": 1.8, "watch_level": price * 1.02,
        "ai_prob": 0.6, "adx": 28, "obv_r": 1.4,
        "_cd_key": f"{ticker}_{tier}_{label}",
    }
    captured_posts.clear()
    try:
        ok = engine.send_mover_alert(ticker, mover, df)
        msg = captured_posts[-1] if captured_posts else None
        print(msg)
        check(f"{section}/{label}: send returned True", ok is True)
        check(f"{section}/{label}: message captured", msg is not None)
    except Exception as e:
        print("EXCEPTION:", e)
        traceback.print_exc()
        check(f"{section}/{label}: no exception", False, str(e))


print("\n" + "=" * 70)
print("SECTION 6: send_mover_alert() — ACTIVE tier (Big Mover, breakout confirmed)")
print("=" * 70)
mover_scenario("ASX ACTIVE market open",   "CBA.AX", "asx_open_mid", "ACTIVE", "send_mover_alert")
mover_scenario("ASX ACTIVE market closed", "CBA.AX", "asx_closed",   "ACTIVE", "send_mover_alert")
mover_scenario("US ACTIVE market open",    "AAPL",   "us_open_mid",  "ACTIVE", "send_mover_alert")
mover_scenario("US ACTIVE market closed",  "AAPL",   "us_closed",    "ACTIVE", "send_mover_alert")

print("\n" + "=" * 70)
print("SECTION 7: send_mover_alert() — SETUP tier (incoming breakout watch)")
print("=" * 70)
mover_scenario("ASX SETUP", "CBA.AX", "asx_closed", "SETUP", "send_mover_alert")
mover_scenario("US SETUP",  "AAPL",   "us_closed",  "SETUP", "send_mover_alert")


print("\n" + "=" * 70)
print("SECTION 8: send_morning_brief()")
print("=" * 70)
set_fake_now(DATES["asx_open_first30"])
engine._pytz = pytz
captured_posts.clear()
try:
    ok = engine.send_morning_brief()
    msg = captured_posts[-1] if captured_posts else None
    print(msg)
    check("morning_brief: returned True", ok is True)
    check("morning_brief: message captured", msg is not None)
except Exception as e:
    print("EXCEPTION:", e)
    traceback.print_exc()
    check("morning_brief: no exception", False, str(e))


print("\n" + "=" * 70)
print("SECTION 9: opportunity/alerts.py — format_opportunity_alert / format_outcome_alert")
print("=" * 70)
from opportunity import alerts as opp_alerts

sample_opp = {
    "ticker": "NVDA", "opportunity_score": 82, "confidence": 0.71,
    "expected_upside_pct": 0.09, "expected_downside_pct": 0.04,
    "risk_level": "MEDIUM", "rr_ratio": 2.1, "entry_zone": [120.5, 122.0],
    "stop_loss": 115.0, "take_profit": [130.0, 135.0, 140.0],
    "trailing_stop_pct": 0.03, "est_holding_days": 8, "regime": "BULLISH",
    "reasons_for": ["Strong volume", "Sector leadership"],
    "reasons_against": ["High valuation"],
    "technical_summary": "Bullish structure", "momentum_summary": "Accelerating",
}
try:
    msg = opp_alerts.format_opportunity_alert(sample_opp)
    print(msg)
    check("opportunity_alert: formatted", "NVDA" in msg and "None" not in msg)
except Exception as e:
    traceback.print_exc()
    check("opportunity_alert: no exception", False, str(e))

for outcome, pct in [("WIN", 0.08), ("LOSS", -0.03)]:
    sample_trade = {
        "ticker": "NVDA", "outcome": outcome, "entry_price": 120.0,
        "exit_price": 120.0 * (1 + pct), "actual_pct": pct,
        "signal_date": "2026-06-20", "opportunity_score": 82, "confidence": 0.71,
        "stop_price": 115.0, "target_price": 130.0,
    }
    try:
        msg = opp_alerts.format_outcome_alert(sample_trade)
        print(msg)
        check(f"outcome_alert[{outcome}]: formatted", "NVDA" in msg and "None" not in msg)
    except Exception as e:
        traceback.print_exc()
        check(f"outcome_alert[{outcome}]: no exception", False, str(e))


print("\n" + "=" * 70)
print("SECTION 10: opportunity/backtester.py — send_backtest_discord")
print("=" * 70)
from opportunity import backtester as opp_bt
os.environ.setdefault("Discordwebhook", "http://fake-webhook")

captured_urllib = []
import urllib.request as _ur
class _FakeCtx:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
def fake_urlopen(req, timeout=10):
    captured_urllib.append(req.data)
    return _FakeCtx()
_ur.urlopen = fake_urlopen

sample_results = {"summary": {
    "trade_count": 42, "win_rate": 0.6, "avg_gain_pct": 7.2, "avg_loss_pct": 3.1,
    "profit_factor": 1.9, "max_drawdown_pct": 12.0, "sharpe_ratio": 1.4,
    "sortino_ratio": 1.8, "expectancy_r": 0.35, "annualised_return_pct": 22.0,
    "avg_hold_days": 8.2, "winning_streak": 5, "losing_streak": 2,
}}
try:
    ok = opp_bt.send_backtest_discord(sample_results, "swing")
    print(captured_urllib[-1].decode() if captured_urllib else None)
    check("backtest_discord: returned True", ok is True)
except Exception as e:
    traceback.print_exc()
    check("backtest_discord: no exception", False, str(e))


print("\n" + "=" * 70)
print("SECTION 11: opportunity/health.py — send_weekly_health_report")
print("=" * 70)
from opportunity import health as opp_health
opp_health.ENABLE_SYSTEM_HEALTH = True
captured_urllib.clear()
try:
    ok = opp_health.send_weekly_health_report()
    print(captured_urllib[-1].decode() if captured_urllib else "(no records / no message)")
    check("health_report: no exception", True)
except Exception as e:
    traceback.print_exc()
    check("health_report: no exception", False, str(e))


print("\n" + "=" * 70)
print("SECTION 12: opportunity/performance.py — send_weekly_performance_report")
print("=" * 70)
from opportunity import performance as opp_perf
opp_perf.ENABLE_PERFORMANCE_ANALYTICS = True
captured_urllib.clear()
try:
    ok = opp_perf.send_weekly_performance_report()
    print(captured_urllib[-1].decode() if captured_urllib else "(no resolved trades / no message)")
    check("performance_report: no exception", True)
except Exception as e:
    traceback.print_exc()
    check("performance_report: no exception", False, str(e))


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = [r for r in RESULTS if not r[1]]
print(f"{passed}/{len(RESULTS)} checks passed")
if failed:
    print("FAILURES:")
    for name, ok, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
