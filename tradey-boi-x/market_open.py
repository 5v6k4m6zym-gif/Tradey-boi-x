"""
market_open.py — Simplified Market Open Report for Tradey Boi X.

Two separate scheduled jobs call this with an explicit market argument —
there is no time-based guessing, so the report can never be mislabeled
even if the GitHub Actions runner is delayed:

  python market_open.py ASX   — 10:00am AEST (ASX open)
  python market_open.py US    — 11:30pm AEST (US open)

No individual stock picks. Just: all sectors + performance, and an
overall market health summary.
"""

import os, sys, datetime, pytz, requests
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf

DISCORD_URL = os.environ.get("Discordwebhook", "")
AEST        = pytz.timezone("Australia/Sydney")

# ─── SECTORS (all GICS sectors, one liquid proxy each) ───────────────────────
US_SECTORS = {
    "Technology":             "XLK",
    "Financials":             "XLF",
    "Health Care":            "XLV",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}
ASX_SECTORS = {
    "Financials":             "CBA.AX",
    "Materials":              "BHP.AX",
    "Energy":                 "WDS.AX",
    "Health Care":            "CSL.AX",
    "Consumer Discretionary": "WES.AX",
    "Consumer Staples":       "WOW.AX",
    "Industrials":            "TCL.AX",
    "Real Estate":            "GMG.AX",
    "Utilities":              "AGL.AX",
    "Information Technology": "XRO.AX",
    "Communication Services": "TLS.AX",
}
INDEX = {"ASX": "^AXJO", "US": "SPY"}


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _day_change(ticker: str) -> float | None:
    """Most recent session's % change vs the prior close. None on error.

    Uses a 5-day lookback (not 2-day) because right at/after market open —
    especially after a weekend — Yahoo may not have posted today's daily bar
    yet, and a 2-day window collapses to a single row, silently dropping
    this factor.
    """
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d")
        if len(df) < 2:
            return None
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
    except Exception:
        return None


def _index_gap(ticker: str) -> float | None:
    """Gap between the most recent session's open and the prior close."""
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d")
        if len(df) < 2:
            return None
        today_open = float(df["Open"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        return (today_open / prev_close) - 1
    except Exception:
        return None


def _vix() -> float | None:
    try:
        return float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
    except Exception:
        return None


# ─── ANALYSIS (shared logic for both markets) ────────────────────────────────
def analyse_market(market: str) -> dict:
    """market: 'ASX' or 'US'."""
    sectors_map = ASX_SECTORS if market == "ASX" else US_SECTORS
    index_ticker = INDEX[market]
    index_name   = "ASX 200" if market == "ASX" else "S&P 500 (SPY)"

    index_gap = _index_gap(index_ticker)
    vix       = _vix()

    sector_rows = []
    greens = reds = 0
    for name, tkr in sectors_map.items():
        chg = _day_change(tkr)
        if chg is None:
            continue
        sector_rows.append((name, chg))
        if chg > 0:
            greens += 1
        else:
            reds += 1
    sector_rows.sort(key=lambda x: x[1], reverse=True)

    # ── Overall market health score (breadth + VIX + index gap) ────────────
    score = 0
    total = greens + reds
    if total > 0:
        breadth_pct = greens / total
        if breadth_pct >= 0.67:
            score += 2
        elif breadth_pct >= 0.50:
            score += 1
        elif breadth_pct <= 0.33:
            score -= 2
        else:
            score -= 1

    if vix is not None:
        if vix < 15:
            score += 2
        elif vix < 18:
            score += 1
        elif vix >= 25:
            score -= 2

    if index_gap is not None:
        if index_gap > 0.003:
            score += 1
        elif index_gap < -0.003:
            score -= 1

    return {
        "market": market,
        "index_name": index_name,
        "index_gap": index_gap,
        "vix": vix,
        "sector_rows": sector_rows,
        "greens": greens,
        "reds": reds,
        "score": score,
    }


# ─── RATING ───────────────────────────────────────────────────────────────────
def _rating(score: int) -> tuple[str, str]:
    if score >= 4:
        return ("💪 STRONG", "Broad-based strength across sectors and low volatility.")
    elif score >= 2:
        return ("📈 LEANING STRONG", "Generally healthy conditions.")
    elif score >= 0:
        return ("➡️ NEUTRAL", "Mixed conditions — no clear market-wide direction.")
    elif score >= -2:
        return ("📉 LEANING WEAK", "Broad-based softness. Trade cautiously.")
    else:
        return ("⚠️ WEAK", "Poor conditions — high fear and/or widespread sector weakness.")


# ─── DISCORD ALERT ───────────────────────────────────────────────────────────
def build_market_report(analysis: dict, title: str) -> str:
    now_str  = datetime.datetime.now(AEST).strftime("%a %d %b %Y %I:%M %p AEST")
    market   = analysis["market"]
    rating, advice = _rating(analysis["score"])
    divider  = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    lines = [
        divider,
        f"**🇦🇺 ASX REPORT**" if market == "ASX" else f"**🇺🇸 US REPORT**",
        f"**{title}**",
        divider,
        "",
        f"**Overall market condition: {rating}**  (score: {analysis['score']:+d})",
        f"_{advice}_",
        "",
    ]

    if analysis["index_gap"] is not None:
        lines.append(f"**{analysis['index_name']}:** {analysis['index_gap']*100:+.2f}%")
    if analysis["vix"] is not None:
        lines.append(f"**VIX (fear index):** {analysis['vix']:.1f}")

    total = analysis["greens"] + analysis["reds"]
    if total:
        lines.append(f"**Sector breadth:** {analysis['greens']} green / {analysis['reds']} red (of {total})")

    if analysis["sector_rows"]:
        lines += ["", "**All sectors:**"]
        for name, chg in analysis["sector_rows"]:
            emoji = "🟢" if chg > 0 else "🔴"
            lines.append(f"  {emoji} {name}: {chg*100:+.2f}%")

    lines += ["", divider, f"_{now_str}_"]
    return "\n".join(lines)[:2000]


def send_discord(content: str) -> bool:
    if not DISCORD_URL:
        print("No Discord webhook configured — skipping send.")
        return False
    try:
        r = requests.post(DISCORD_URL, json={"content": content}, timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f"Discord error: {e}")
        return False


def send_open_report(market: str) -> bool:
    """Send the open report for the given market ('ASX' or 'US') to Discord.
    Returns True on success. Safe to call from scanner.py at session start."""
    market   = market.upper()
    title    = "ASX OPEN REPORT" if market == "ASX" else "US OPEN REPORT"
    analysis = analyse_market(market)
    content  = build_market_report(analysis, title)
    return send_discord(content)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    if len(sys.argv) < 2 or sys.argv[1].upper() not in ("ASX", "US"):
        print("Usage: python market_open.py [ASX|US]")
        sys.exit(1)

    market = sys.argv[1].upper()
    title  = "ASX OPEN REPORT" if market == "ASX" else "US OPEN REPORT"

    print(f"=== {title} ===")
    analysis = analyse_market(market)

    rating, advice = _rating(analysis["score"])
    print(f"  Market: {market}  |  Score: {analysis['score']:+d}  |  {rating}")
    print(f"  Sectors: {analysis['greens']} green / {analysis['reds']} red")

    content = build_market_report(analysis, title)
    sent = send_discord(content)
    print(f"  Discord sent: {sent}")


if __name__ == "__main__":
    run()
