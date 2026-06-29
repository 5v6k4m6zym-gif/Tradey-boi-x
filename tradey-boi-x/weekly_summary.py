"""
Weekly performance summary — sent to Discord every Sunday.
Reports win rate, top/bottom tickers, and weekly signal count.
Also shows which tickers have earned score adjustments from learning.
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import requests

DISCORD  = os.getenv("Discordwebhook", "") or os.getenv("discordwebhook", "")
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "signal_log.json"


def load_log() -> list:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            return []
    return []


def send_weekly_summary():
    if not DISCORD:
        print("No Discord webhook — skipping.")
        return

    entries  = load_log()
    resolved = [e for e in entries if e["outcome"] is not None]

    # ── This week's signals ───────────────────────────────────────────────────
    week_ago  = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    this_week = [e for e in entries if e["signal_date"] >= week_ago]
    week_wins = [e for e in this_week if e.get("outcome") == "WIN"]
    week_loss = [e for e in this_week if e.get("outcome") == "LOSS"]

    # ── All-time stats ────────────────────────────────────────────────────────
    total_wins   = sum(1 for e in resolved if e["outcome"] == "WIN")
    total_losses = len(resolved) - total_wins
    overall_wr   = (total_wins / len(resolved) * 100) if resolved else 0
    avg_return   = (sum(e["actual_pct"] for e in resolved) / len(resolved) * 100) if resolved else 0

    # ── Per-ticker win rates ──────────────────────────────────────────────────
    bucket: dict = defaultdict(list)
    for e in resolved:
        bucket[e["ticker"]].append(e["outcome"] == "WIN")

    ticker_stats = [
        (t, sum(r)/len(r)*100, len(r))
        for t, r in bucket.items() if len(r) >= 2
    ]
    ticker_stats.sort(key=lambda x: x[1], reverse=True)

    best   = ticker_stats[:3]
    worst  = ticker_stats[-3:][::-1]

    # ── Adaptive adjustments ─────────────────────────────────────────────────
    boosted  = [t for t, wr, n in ticker_stats if wr >= 65]
    penalised= [t for t, wr, n in ticker_stats if wr <= 35]

    # ── Build message ─────────────────────────────────────────────────────────
    lines = [
        f"**📊 TRADEY BOI X — Weekly Report** | {datetime.now().strftime('%d %b %Y')}",
        "",
        f"**This week:** {len(this_week)} signal(s) fired"
        + (f" · {len(week_wins)}W / {len(week_loss)}L" if this_week else ""),
        "",
        "**All-time performance:**",
        f"• Win rate: **{overall_wr:.1f}%** ({total_wins}W / {total_losses}L of {len(resolved)} resolved)",
        f"• Avg return per signal: **{avg_return:+.2f}%**",
    ]

    if best:
        lines += ["", "**🏆 Best tickers:**"]
        for t, wr, n in best:
            lines.append(f"  {t}: {wr:.0f}% win rate ({n} signals)")

    if worst and len(ticker_stats) > 3:
        lines += ["", "**⚠️ Weakest tickers:**"]
        for t, wr, n in worst:
            lines.append(f"  {t}: {wr:.0f}% win rate ({n} signals)")

    if boosted:
        lines += ["", f"**🧠 AI score boost applied to:** {', '.join(boosted)}"]
    if penalised:
        lines += [f"**🧠 AI score penalty applied to:** {', '.join(penalised)}"]

    if not resolved:
        lines += ["", "_No resolved signals yet — check back after the first 10 trading days._"]

    lines.append(f"\n_Next scan: hourly Mon–Fri during market hours_")

    msg = "\n".join(lines)

    try:
        r = requests.post(DISCORD, json={"content": msg}, timeout=5)
        print(f"Discord status: {r.status_code}")
    except Exception as e:
        print(f"Discord error: {e}")


if __name__ == "__main__":
    send_weekly_summary()
