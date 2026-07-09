"""
eod_report.py — End-of-Day Market Report for Tradey Boi X.

Runs after ASX close (~4:20pm AEST). Same simplified format as
market_open.py (all sectors + overall market condition) PLUS:

  • Today's top performers (biggest gainers on the watchlist)
  • Why the AI did/didn't flag each one as a signal

No swing-setup entry levels here — that's evening_scan.py's job.
This is purely a market recap.
"""

import os, sys, datetime, pytz, requests
sys.path.insert(0, os.path.dirname(__file__))

from market_open import (
    analyse_market, build_market_report, send_discord, _rating,
)
from engine import WATCHLIST, get_data, train_model, decide, explain_filter_plain

AEST = pytz.timezone("Australia/Sydney")
TOP_N = 5


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
    """Explain, in plain terms, why the AI did or didn't suggest this ticker."""
    try:
        result = decide(ticker, df, model)
    except Exception as e:
        return f"couldn't evaluate ({e})"

    if result.get("alert"):
        return f"✅ AI already flagged this — {result['signal']} (score {result['score']}, {result['prob']*100:.0f}% confidence)"

    if result["signal"] == "GATED":
        failed = [name for name, ok in result.get("filters", []) if not ok]
        if failed:
            return (f"❌ Blocked by filter: {failed[0]}\n"
                     f"    → Why that's a bad buy: {explain_filter_plain(failed[0])}")
        return "❌ Blocked by a safety filter"

    # Passed filters but scored too low for STRONG BUY/ELITE
    return (f"⛔ Scored {result['score']} with {result['prob']*100:.0f}% AI confidence — "
            f"below the STRONG BUY threshold (needs score ≥6, confidence ≥50%, positive expected value)")


def build_top_performers_section(model) -> list[str]:
    print(f"  Scanning {len(WATCHLIST)} tickers for today's top performers...")
    movers = []
    for ticker in WATCHLIST:
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
    market = "ASX"
    title  = "END OF DAY REPORT"
    print(f"=== ASX {title} ===")

    analysis = analyse_market(market)
    rating, advice = _rating(analysis["score"])
    print(f"  Score: {analysis['score']:+d} | {rating}")

    content = build_market_report(analysis, title)

    print("  Training model for signal explanations...")
    try:
        model = train_model()
        top_lines = build_top_performers_section(model)
    except Exception as e:
        print(f"  Could not build top performers section: {e}")
        top_lines = []

    # Insert top-performer lines before the closing divider/timestamp block.
    if top_lines:
        content_lines = content.split("\n")
        divider = content_lines[0]
        # last two lines are ["", divider, timestamp] appended by build_market_report
        closing = content_lines[-2:]
        body = content_lines[:-2]
        content = "\n".join(body + top_lines + closing)[:2000]

    sent = send_discord(content)
    print(f"  Discord sent: {sent}")


if __name__ == "__main__":
    run()
