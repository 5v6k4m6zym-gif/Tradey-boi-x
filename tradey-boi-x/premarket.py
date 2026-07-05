"""
premarket.py — Pre-market opportunity scanner for Tradey Boi X.

Runs 30–60 min before market open to catch:
  • Gap-ups on overnight news / futures movement
  • Unusual pre-market volume vs 20-day average
  • Positive news sentiment since prior close
  • Price above VWAP of pre-market session

Sends a lighter-weight Discord alert; does NOT require AI model (too slow
for pre-market window). Flags tickers for the main scanner to confirm at open.
"""

import os, sys, time, pytz, datetime, requests
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd
from engine import (
    WATCHLIST, _signal_cached, _signal_store,
    news_sentiment as get_news_sentiment, vwap_signal, fear_greed_signal,
)

DISCORD_URL = os.environ.get("Discordwebhook", "")
GAP_THRESHOLD  = 0.015   # 1.5% gap to flag (lower than engine's 2%)
VOL_THRESHOLD  = 1.3     # pre-market volume 1.3× avg 20-day pre-market


# ─── PRE-MARKET DATA ──────────────────────────────────────────────────────────
def get_premarket(ticker: str) -> dict | None:
    """
    Fetch overnight / pre-market data.
    Returns None if no pre-market data available.
    """
    try:
        t   = yf.Ticker(ticker)
        df  = t.history(period="2d", interval="1h", prepost=True)
        if len(df) < 4:
            return None
        info   = t.fast_info
        prev_close = float(df["Close"].dropna().iloc[-4]) if len(df) >= 4 else None
        last       = df.iloc[-1]
        premarket_price = float(last["Close"])
        premarket_vol   = float(last["Volume"])

        # 20-day average hourly volume (regular hours only)
        reg = yf.Ticker(ticker).history(period="1mo", interval="1h")
        avg_vol = float(reg["Volume"].mean()) if len(reg) > 0 else premarket_vol

        gap = (premarket_price / prev_close - 1) if prev_close else 0
        vol_ratio = premarket_vol / avg_vol if avg_vol > 0 else 0

        return {
            "price":       premarket_price,
            "prev_close":  prev_close,
            "gap":         gap,
            "vol_ratio":   vol_ratio,
            "ticker":      ticker,
        }
    except Exception:
        return None


# ─── PRE-MARKET VWAP ─────────────────────────────────────────────────────────
def premarket_vwap(ticker: str) -> tuple[float | None, bool]:
    """Return (vwap, price_above_vwap) for the pre-market session."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="5m", prepost=True)
        if len(df) < 5:
            return None, False
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap    = float((typical * df["Volume"]).cumsum().iloc[-1] /
                        df["Volume"].cumsum().iloc[-1])
        price   = float(df["Close"].iloc[-1])
        return vwap, price > vwap
    except Exception:
        return None, False


# ─── DISCORD ALERT ───────────────────────────────────────────────────────────
def send_premarket_alert(results: list[dict]) -> bool:
    if not DISCORD_URL or not results:
        return False
    now_str = datetime.datetime.now(pytz.timezone("Australia/Sydney")).strftime("%a %d %b %Y %I:%M %p AEST")
    lines   = [f"# 🌅 Pre-Market Scan — {now_str}", ""]

    for r in results:
        gap_emoji  = "⬆️" if r["gap"] > 0 else "⬇️"
        vol_label  = f"🔥 {r['vol_ratio']:.1f}× avg vol" if r["vol_ratio"] > 1.5 else f"{r['vol_ratio']:.1f}× avg vol"
        vwap_label = f"✅ above pre-market VWAP ${r['vwap']:.2f}" if r["above_vwap"] and r["vwap"] else ""
        news_label = f"📰 {r['news_label']} sentiment ({r['news_count']} headlines)" if r["news_count"] else ""

        lines += [
            f"**{r['ticker']}** — ${r['price']:.3f}",
            f"  {gap_emoji} Gap: **{r['gap']*100:+.1f}%**  vs prior close ${r['prev_close']:.3f}",
            f"  📊 {vol_label}",
        ]
        if vwap_label: lines.append(f"  {vwap_label}")
        if news_label: lines.append(f"  {news_label}")
        lines.append(f"  ⚠️ _Watch at open — confirm with main scanner signal before trading_")
        lines.append("")

    payload = {"content": "\n".join(lines)[:2000]}
    try:
        r = requests.post(DISCORD_URL, json=payload, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ─── MAIN SCAN ───────────────────────────────────────────────────────────────
def run_premarket_scan():
    print("=== Tradey Boi X — Pre-Market Scan ===")
    now_sydney = datetime.datetime.now(pytz.timezone("Australia/Sydney"))
    print(f"  Time: {now_sydney.strftime('%H:%M AEST')}  |  Scanning {len(WATCHLIST)} tickers")

    fg_adj, fg_why = fear_greed_signal()
    macro_ok = fg_adj >= 0   # skip if fear is elevated

    flagged = []
    for ticker in WATCHLIST:
        try:
            data = get_premarket(ticker)
            if data is None:
                continue

            gap       = data["gap"]
            vol_ratio = data["vol_ratio"]

            # Must have a meaningful gap OR volume surge to be worth flagging
            if abs(gap) < GAP_THRESHOLD and vol_ratio < VOL_THRESHOLD:
                continue

            # Only flag gap-ups in a fear environment if really large
            if fg_adj < 0 and gap < 0.03:
                continue

            vwap, above_vwap = premarket_vwap(ticker)
            news             = get_news_sentiment(ticker)
            news_label       = news.get("label", "NEUTRAL")
            news_count       = news.get("count", 0)

            # Skip on strong negative news
            if news_label == "NEGATIVE" and news.get("compound", 0) < -0.3:
                print(f"  {ticker}: skipped — negative news")
                continue

            score = 0
            if gap > 0.02:            score += 2
            elif gap > GAP_THRESHOLD: score += 1
            if vol_ratio > 1.5:       score += 2
            elif vol_ratio > VOL_THRESHOLD: score += 1
            if above_vwap:            score += 1
            if news_label == "POSITIVE": score += 1

            if score >= 3:
                flagged.append({
                    **data,
                    "vwap":       vwap,
                    "above_vwap": above_vwap,
                    "news_label": news_label,
                    "news_count": news_count,
                    "score":      score,
                })
                print(f"  ✅ {ticker}: gap {gap*100:+.1f}% | vol {vol_ratio:.1f}× | score {score}")
            else:
                print(f"  — {ticker}: gap {gap*100:+.1f}% | vol {vol_ratio:.1f}× | score {score} (below threshold)")

        except Exception as e:
            print(f"  {ticker}: error — {e}")

    if flagged:
        flagged.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n  Sending pre-market alert for {len(flagged)} ticker(s)...")
        sent = send_premarket_alert(flagged)
        print(f"  Alert sent: {sent}")
    else:
        print("\n  No pre-market setups meet threshold today.")


if __name__ == "__main__":
    run_premarket_scan()
