"""
market_open.py — Market Open Strength Report for Tradey Boi X.

Fires at:
  • 10:00am AEST  (ASX open)   — analyses ASX 200 + key sectors
  • 11:30pm AEST  (US open)    — analyses SPY/QQQ + all 6 GICS sectors

Sends one clean Discord message: "Here's what the market looks like today."
Tells you whether conditions favour trading or sitting out.
"""

import os, sys, datetime, pytz, requests
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd

DISCORD_URL = os.environ.get("Discordwebhook", "")
AEST        = pytz.timezone("Australia/Sydney")

# ─── SECTOR ETFs ──────────────────────────────────────────────────────────────
US_SECTORS = {
    "Tech":        "XLK",
    "Finance":     "XLF",
    "Health":      "XLV",
    "Energy":      "XLE",
    "Industrials": "XLI",
    "Comms":       "XLC",
}
ASX_PROXIES = {
    "Miners":  "BHP.AX",
    "Banks":   "CBA.AX",
    "Healthcare": "CSL.AX",
    "Energy":  "WDS.AX",
    "Mining2": "FMG.AX",
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _day_change(ticker: str) -> float | None:
    """Return most recent session's % change vs the prior close. None on error.

    Uses a 5-day lookback (not 2-day) because right at/after market open —
    especially after a weekend — Yahoo may not have posted today's daily bar
    yet, and a 2-day window collapses to a single row (Friday's), silently
    dropping this factor. A wider window guarantees at least two valid rows.
    """
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d")
        if len(df) < 2:
            return None
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
    except Exception:
        return None


def _vix() -> float | None:
    try:
        return float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
    except Exception:
        return None


def _index_gap(ticker: str) -> float | None:
    """Gap between the most recent session's open and the prior close.

    See _day_change for why a 5-day (not 2-day) lookback is required.
    """
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d")
        if len(df) < 2:
            return None
        today_open = float(df["Open"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        return (today_open / prev_close) - 1
    except Exception:
        return None


def _momentum(ticker: str, days: int = 5) -> float | None:
    try:
        df = yf.Ticker(ticker).history(period="1mo")
        if len(df) < days:
            return None
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-days] - 1)
    except Exception:
        return None


# ─── ASX ANALYSIS ────────────────────────────────────────────────────────────
def analyse_asx() -> dict:
    index_gap = _index_gap("^AXJO")
    index_chg = _day_change("^AXJO")
    vix        = _vix()
    mom5       = _momentum("^AXJO", 5)

    score   = 0
    factors = []

    # Index gap
    if index_gap is not None:
        if index_gap >  0.003:
            score += 2; factors.append(f"📈 ASX 200 gapping up {index_gap*100:+.2f}% at open")
        elif index_gap > 0:
            score += 1; factors.append(f"📈 ASX 200 slightly positive at open ({index_gap*100:+.2f}%)")
        elif index_gap < -0.003:
            score -= 2; factors.append(f"📉 ASX 200 gapping DOWN {index_gap*100:+.2f}% at open")
        else:
            factors.append(f"➡️ ASX 200 flat open ({index_gap*100:+.2f}%)")

    # VIX
    if vix is not None:
        if vix < 15:
            score += 2; factors.append(f"😌 VIX {vix:.1f} — very low fear, calm conditions")
        elif vix < 18:
            score += 1; factors.append(f"✅ VIX {vix:.1f} — low fear")
        elif vix < 25:
            factors.append(f"⚠️ VIX {vix:.1f} — elevated, trade cautiously")
        else:
            score -= 2; factors.append(f"🚨 VIX {vix:.1f} — HIGH FEAR, consider sitting out")

    # 5-day momentum
    if mom5 is not None:
        if mom5 > 0.01:
            score += 1; factors.append(f"📊 ASX 200 up {mom5*100:+.1f}% over 5 days — uptrend intact")
        elif mom5 < -0.01:
            score -= 1; factors.append(f"📊 ASX 200 down {mom5*100:+.1f}% over 5 days — weak trend")

    # Sector breadth
    greens = 0; reds = 0; sector_lines = []
    for name, tkr in ASX_PROXIES.items():
        chg = _day_change(tkr)
        if chg is None:
            continue
        emoji = "🟢" if chg > 0 else "🔴"
        sector_lines.append(f"{emoji} {name} {chg*100:+.1f}%")
        if chg > 0: greens += 1
        else: reds += 1

    if greens + reds > 0:
        if greens > reds:
            score += 1; factors.append(f"Sectors: {greens} green / {reds} red")
        elif reds > greens:
            score -= 1; factors.append(f"Sectors: {greens} green / {reds} red")

    return {"score": score, "factors": factors,
            "sectors": sector_lines, "vix": vix,
            "index_gap": index_gap, "market": "ASX"}


# ─── US ANALYSIS ─────────────────────────────────────────────────────────────
def analyse_us() -> dict:
    spy_gap  = _index_gap("SPY")
    qqq_gap  = _index_gap("QQQ")
    vix      = _vix()
    spy_mom  = _momentum("SPY", 5)

    score   = 0
    factors = []

    # SPY gap
    if spy_gap is not None:
        if spy_gap > 0.004:
            score += 2; factors.append(f"📈 SPY gapping up {spy_gap*100:+.2f}% — strong open")
        elif spy_gap > 0.001:
            score += 1; factors.append(f"📈 SPY slightly positive at open ({spy_gap*100:+.2f}%)")
        elif spy_gap < -0.004:
            score -= 2; factors.append(f"📉 SPY gapping DOWN {spy_gap*100:+.2f}% — weak open")
        elif spy_gap < -0.001:
            score -= 1; factors.append(f"📉 SPY negative at open ({spy_gap*100:+.2f}%)")
        else:
            factors.append(f"➡️ SPY flat open ({spy_gap*100:+.2f}%)")

    # QQQ (tech leadership matters for NVDA, AMD, META etc.)
    if qqq_gap is not None:
        if qqq_gap > 0.005:
            score += 1; factors.append(f"💻 QQQ (tech) up {qqq_gap*100:+.2f}% — tech leading")
        elif qqq_gap < -0.005:
            score -= 1; factors.append(f"💻 QQQ (tech) down {qqq_gap*100:+.2f}% — tech lagging")

    # VIX
    if vix is not None:
        if vix < 15:
            score += 2; factors.append(f"😌 VIX {vix:.1f} — very low fear, ideal conditions")
        elif vix < 18:
            score += 1; factors.append(f"✅ VIX {vix:.1f} — low fear, good conditions")
        elif vix < 25:
            factors.append(f"⚠️ VIX {vix:.1f} — elevated caution, size down")
        else:
            score -= 2; factors.append(f"🚨 VIX {vix:.1f} — HIGH FEAR, consider sitting out")

    # SPY 5-day momentum
    if spy_mom is not None:
        if spy_mom > 0.01:
            score += 1; factors.append(f"📊 SPY up {spy_mom*100:+.1f}% over 5 days — uptrend")
        elif spy_mom < -0.01:
            score -= 1; factors.append(f"📊 SPY down {spy_mom*100:+.1f}% over 5 days — downtrend")

    # Sector breadth
    greens = 0; reds = 0; sector_lines = []
    for name, etf in US_SECTORS.items():
        chg = _day_change(etf)
        if chg is None:
            continue
        emoji = "🟢" if chg > 0 else "🔴"
        sector_lines.append(f"{emoji} {name} {chg*100:+.1f}%")
        if chg > 0: greens += 1
        else: reds += 1

    total = greens + reds
    if total > 0:
        breadth_pct = greens / total
        if breadth_pct >= 0.67:
            score += 2; factors.append(f"Broad strength: {greens}/{total} sectors green")
        elif breadth_pct >= 0.50:
            score += 1; factors.append(f"Mixed breadth: {greens}/{total} sectors green")
        elif breadth_pct <= 0.33:
            score -= 2; factors.append(f"Broad weakness: only {greens}/{total} sectors green")
        else:
            score -= 1; factors.append(f"Weak breadth: {greens}/{total} sectors green")

    return {"score": score, "factors": factors,
            "sectors": sector_lines, "vix": vix,
            "index_gap": spy_gap, "market": "US"}


# ─── RATING ───────────────────────────────────────────────────────────────────
def _rating(score: int) -> tuple[str, str]:
    """Return (emoji_label, trading_advice) based on score."""
    if score >= 4:
        return ("💪 STRONG",
                "Excellent conditions. High-conviction signals are worth acting on.")
    elif score >= 2:
        return ("📈 LEANING STRONG",
                "Good conditions. Stick to ELITE and STRONG BUY signals.")
    elif score >= 0:
        return ("➡️ NEUTRAL",
                "Mixed conditions. Only act on ELITE signals. Size down slightly.")
    elif score >= -2:
        return ("📉 LEANING WEAK",
                "Tough conditions. Consider waiting for better setups.")
    else:
        return ("⚠️ WEAK",
                "Poor conditions. High risk of false signals. Best to sit out today.")


# ─── DISCORD ALERT ───────────────────────────────────────────────────────────
def send_market_alert(analysis: dict) -> bool:
    if not DISCORD_URL:
        return False

    now_str  = datetime.datetime.now(AEST).strftime("%a %d %b %Y %I:%M %p AEST")
    market   = analysis["market"]
    score    = analysis["score"]
    rating, advice = _rating(score)
    divider  = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    lines = [
        divider,
        f"**TRADEY BOI X  |  {market} OPEN REPORT**",
        divider,
        f"**{rating}**  (score: {score:+d})",
        f"_{advice}_",
        "",
    ]

    for f in analysis["factors"]:
        lines.append(f"  • {f}")

    if analysis["sectors"]:
        lines += ["", "**Sectors:**", "  " + "   ".join(analysis["sectors"])]

    if analysis.get("vix"):
        lines.append(f"\n_VIX: {analysis['vix']:.1f}_")

    lines += ["", divider, f"_{now_str}_"]

    payload = {"content": "\n".join(lines)[:2000]}
    try:
        r = requests.post(DISCORD_URL, json=payload, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    now_aest = datetime.datetime.now(AEST)
    h, wd    = now_aest.hour, now_aest.weekday()

    # Detect which market is opening
    # ASX open = 10am AEST (cron triggers at 00:00 UTC)
    # US open  = 11:30pm AEST (cron triggers at 13:30 UTC)
    if 9 <= h <= 11:
        print("=== ASX Market Open Report ===")
        analysis = analyse_asx()
    else:
        print("=== US Market Open Report ===")
        analysis = analyse_us()

    rating, advice = _rating(analysis["score"])
    print(f"  Market: {analysis['market']}  |  Score: {analysis['score']:+d}  |  {rating}")
    for f in analysis["factors"]:
        print(f"  • {f}")

    sent = send_market_alert(analysis)
    print(f"  Discord alert sent: {sent}")


if __name__ == "__main__":
    run()
