"""
evening_scan.py — End-of-day "tomorrow's game plan" scanner.

Runs at 5:00pm AEST (after ASX closes, before you've sat down for dinner).
Only surfaces SWING setups — multi-day holds where the entry is genuinely
valid at tomorrow's open. Intraday-only signals (gap-up, VWAP cross) are
filtered out because they expire at today's close.

Sends one consolidated Discord message: "Here's what to watch tomorrow."
"""

import os, sys, datetime, pytz
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd
from engine import (
    WATCHLIST, FEATURES, get_data, train_model, decide,
    fear_greed_signal, _next_trading_day,
)

DISCORD_URL = os.environ.get("Discordwebhook", "")
AEST        = pytz.timezone("Australia/Sydney")

# Signals that are only valid during market hours — don't defer these
INTRADAY_KEYWORDS = (
    "vwap cross-above",
    "gap-up on institutional",
    "gap-up detected",
    "intraday",
)

def _is_intraday_signal(why: list[str]) -> bool:
    return any(any(kw in w.lower() for kw in INTRADAY_KEYWORDS) for w in why)

def _is_swing_signal(why: list[str]) -> bool:
    """True if the setup has substance beyond intraday momentum."""
    swing_keywords = (
        "ema", "uptrend", "rsi", "support", "resistance", "macd",
        "breakout", "obv", "relative strength", "sector", "fundamental",
        "squeeze", "multi-timeframe", "oversold", "volume surge",
    )
    return any(any(kw in w.lower() for kw in swing_keywords) for w in why)

def _entry_tomorrow(price: float, rsi: float, ticker: str) -> dict:
    """
    Build a tomorrow-open entry suggestion with a staleness guard.
    Only buy if price at open is within the validity band.
    """
    is_asx = ticker.endswith(".AX")
    open_time = "10:00am AEST" if is_asx else "11:30pm AEST tonight"

    if rsi < 40:
        low, high = price * 0.995, price * 1.025
        note = f"RSI oversold — buy at open ({open_time}). Good value even with a small gap-up."
    elif rsi < 55:
        low, high = price * 0.99, price * 1.015
        note = f"Buy at open ({open_time}) if price is within ${low:.3f}–${high:.3f}. Skip if it gaps up more than 1.5%."
    else:
        low, high = price * 0.985, price * 1.01
        note = f"RSI elevated — only buy at open ({open_time}) if price pulls back to ${low:.3f}–${high:.3f}. Don't chase."

    return {
        "entry_low":  low,
        "entry_high": high,
        "note":       note,
        "open_time":  open_time,
    }

def send_evening_alert(results: list[dict]) -> bool:
    if not DISCORD_URL or not results:
        return False

    now_aest = datetime.datetime.now(AEST)
    tomorrow = _next_trading_day(now_aest)
    date_str = tomorrow.strftime("%A %d %b %Y")
    hdr = (
        f"# 📋 Tomorrow's Game Plan — {date_str}\n"
        f"_Swing setups only — entries valid at tomorrow's open_\n"
    )
    lines = [hdr]

    fg_adj, fg_why = fear_greed_signal()
    if fg_adj < 0:
        lines.append(f"⚠️ **Macro caution:** {fg_why} — size positions smaller than usual.\n")
    elif fg_adj > 0:
        lines.append(f"✅ **Macro tailwind:** {fg_why}\n")

    for r in results:
        e     = r["entry"]
        score = r["result"]["score"]
        prob  = r["result"]["prob"]
        label = r["result"]["label"]
        lines += [
            f"## {label}  {r['ticker']} — ${r['price']:.3f}",
            f"  📊 Score **{score}** | AI confidence **{prob*100:.0f}%**",
            f"  🟢 **Entry:** {e['note']}",
            f"  💰 **Target:** ${r['params']['target_price']:.3f} (+{r['params']['target_pct']*100:.0f}%)",
            f"  🛑 **Stop-loss:** ${r['params']['stop_loss']:.3f} ({r['params']['stop_loss_pct']:.1f}%)",
            f"  ⚖️ **R:R:** {abs(r['params']['target_pct']/r['params']['stop_loss_pct']):.1f}:1",
            f"  📌 _Why: {'; '.join(r['result']['why'][:3])}_",
            "",
        ]

    lines.append("_Set your orders tonight. If price gaps past the validity band — wait for a pullback._")
    payload = {"content": "\n".join(lines)[:2000]}
    try:
        import requests
        resp = requests.post(DISCORD_URL, json=payload, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False


def run_evening_scan():
    print("=== Tradey Boi X — Evening Scan ===")
    now_aest = datetime.datetime.now(AEST)
    print(f"  Time: {now_aest.strftime('%H:%M AEST')} | Scanning {len(WATCHLIST)} tickers for tomorrow")

    print("  Training ensemble model...")
    model = train_model()

    qualifying = []
    for ticker in WATCHLIST:
        try:
            df = get_data(ticker, "6mo")
            if df.empty or len(df) < 20:
                continue
            price  = float(df["Close"].iloc[-1])
            result = decide(ticker, df, model)

            # Only include STRONG BUY or ELITE
            if result["signal"] not in ("STRONG BUY", "ELITE"):
                continue

            why = result.get("why", [])

            # Skip pure intraday setups — not valid at tomorrow's open
            if _is_intraday_signal(why) and not _is_swing_signal(why):
                print(f"  {ticker}: skipped — intraday-only signal (not valid at open)")
                continue

            from engine import _dynamic_trade_params
            params = _dynamic_trade_params(price, df, result["signal"])
            entry  = _entry_tomorrow(price, result.get("rsi", 50), ticker)

            qualifying.append({
                "ticker": ticker,
                "price":  price,
                "result": result,
                "params": params,
                "entry":  entry,
            })
            print(f"  ✅ {ticker}: {result['signal']} | score {result['score']} | R:R {abs(params['target_pct']/params['stop_loss_pct']):.1f}:1")

        except Exception as e:
            print(f"  {ticker}: error — {e}")

    if qualifying:
        qualifying.sort(key=lambda x: x["result"]["score"], reverse=True)
        top = qualifying[:5]   # cap at 5 — don't overwhelm
        print(f"\n  Sending evening summary for {len(top)} setup(s)...")
        sent = send_evening_alert(top)
        print(f"  Sent: {sent}")
    else:
        print("\n  No qualifying swing setups for tomorrow. No alert sent.")


if __name__ == "__main__":
    run_evening_scan()
