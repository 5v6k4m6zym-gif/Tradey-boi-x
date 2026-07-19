"""
close_report.py — Market Close Report for Tradey Boi X.

Two separate scheduled jobs call this with an explicit market argument —
same convention as market_open.py, so there is no time-based guessing:

  python close_report.py ASX   — ~4:20pm AEST (ASX close)
  python close_report.py US    — ~6:35am AEST (US close)

Same simplified format as the open report (all sectors + overall market
condition) PLUS:

  • Today's top performers on that market's watchlist tickers
  • Why the AI did/didn't flag each one as a signal

No swing-setup entry levels here — that's evening_scan.py's job.
This is purely a market recap.
"""

import os, sys, json, datetime
sys.path.insert(0, os.path.dirname(__file__))

from market_open import (
    analyse_market, build_market_report, send_discord, _rating,
)
from engine import WATCHLIST, get_data, train_model, decide, explain_filter_plain

TOP_N = 5

_LOG_FILE = os.path.join(os.path.dirname(__file__), "signal_log.json")

def _alerted_today(ticker: str) -> bool:
    """Return True if the scanner actually sent a Discord alert for this ticker today."""
    today = datetime.date.today().isoformat()
    try:
        with open(_LOG_FILE) as f:
            entries = json.load(f)
        return any(e.get("ticker") == ticker and e.get("signal_date") == today
                   for e in entries)
    except Exception:
        return False


def _market_tickers(market: str) -> list[str]:
    if market == "ASX":
        return [t for t in WATCHLIST if t.endswith(".AX")]
    return [t for t in WATCHLIST if not t.endswith(".AX")]


def _today_change(ticker: str) -> tuple[float, object] | None:
    try:
        df = get_data(ticker, "6mo")
        if df is None or len(df) < 60:
            return None
        chg = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
        return chg, df
    except Exception:
        return None


def _why_not_flagged(ticker: str, df, model) -> str:
    """Explain, in plain terms, why the AI did or didn't send an alert for this ticker."""
    try:
        result = decide(ticker, df, model)
    except Exception as e:
        return f"couldn't evaluate ({e})"

    # Check the signal log — only say "alerted" if a Discord alert was actually sent today
    if _alerted_today(ticker):
        return f"✅ Scanner sent a buy alert for this today — {result['signal']} (score {result['score']}, {result['prob']*100:.0f}% confidence)"

    # Qualifies at close time but was never alerted (conditions developed late in the session,
    # or it was gated by a filter when the scanner ran earlier)
    if result["signal"] in ("ELITE", "STRONG BUY") and result.get("alert"):
        return (f"📊 Rates {result['signal']} at close (score {result['score']}, {result['prob']*100:.0f}% confidence) "
                f"— conditions developed after today's scans, no alert was sent")

    if result["signal"] == "GATED":
        failed = [name for name, ok in result.get("filters", []) if not ok]
        if failed:
            return (f"❌ Blocked by filter: {failed[0]}\n"
                     f"    → Why that's a bad buy: {explain_filter_plain(failed[0])}")
        return "❌ Blocked by a safety filter"

    # Passed filters but scored too low for STRONG BUY/ELITE
    return (f"⛔ Scored {result['score']} with {result['prob']*100:.0f}% AI confidence — "
            f"below the STRONG BUY threshold (needs score ≥7, confidence ≥53%, positive expected value)")


def build_top_performers_section(tickers: list[str], model) -> list[str]:
    print(f"  Scanning {len(tickers)} tickers for today's top performers...")
    movers = []
    for ticker in tickers:
        r = _today_change(ticker)
        if r is None:
            continue
        chg, df = r
        movers.append((ticker, chg, df))

    movers.sort(key=lambda x: x[1], reverse=True)
    top = movers[:TOP_N]

    lines = ["", "**🚀 Today's top performers:**"]
    for ticker, chg, df in top:
        reason = _why_not_flagged(ticker, df, model)
        lines.append(f"  **{ticker}** {chg*100:+.2f}%")
        lines.append(f"    {reason}")
    return lines


def run():
    if len(sys.argv) < 2 or sys.argv[1].upper() not in ("ASX", "US"):
        print("Usage: python close_report.py [ASX|US]")
        sys.exit(1)

    market = sys.argv[1].upper()
    title  = "CLOSE REPORT"
    print(f"=== {market} {title} ===")

    analysis = analyse_market(market)
    rating, advice = _rating(analysis["score"])
    print(f"  Score: {analysis['score']:+d} | {rating}")

    content = build_market_report(analysis, title)

    tickers = _market_tickers(market)
    print("  Training model for signal explanations...")
    try:
        model = train_model()
        top_lines = build_top_performers_section(tickers, model)
    except Exception as e:
        print(f"  Could not build top performers section: {e}")
        top_lines = []

    # Insert top-performer lines before the closing divider/timestamp block.
    if top_lines:
        content_lines = content.split("\n")
        # last two lines are ["", divider, timestamp] appended by build_market_report
        closing = content_lines[-2:]
        body = content_lines[:-2]
        content = "\n".join(body + top_lines + closing)[:2000]

    sent = send_discord(content)
    print(f"  Discord sent: {sent}")


if __name__ == "__main__":
    run()
