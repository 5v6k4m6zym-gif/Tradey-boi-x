"""
Weekly performance digest — sends a Discord summary every Sunday night (AEST).
Called by GitHub Actions on a cron schedule.
Reads signal_log.json to summarise the past 7 days of signals.
"""
import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_DIR    = Path(__file__).parent
LOG_FILE    = BASE_DIR / "signal_log.json"
DISCORD     = os.getenv("Discordwebhook", "") or os.getenv("discordwebhook", "")
LOOKBACK    = int(os.getenv("DIGEST_DAYS", "7"))


def _load_entries(days: int) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).date()
    out = []
    for line in LOG_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            ds = e.get("date", "")[:10]
            if ds and datetime.strptime(ds, "%Y-%m-%d").date() >= cutoff:
                out.append(e)
        except Exception:
            pass
    return out


def _pnl_estimate(e: dict) -> float | None:
    """Rough P&L estimate from outcome and actual_pct if available."""
    actual = e.get("actual_pct")
    if actual is not None:
        # 2% risk per trade, scale by actual_pct / stop_loss_pct
        entry  = e.get("entry_price", 0)
        stop   = e.get("stop_loss", 0)
        if entry and stop and entry > stop > 0:
            risk_pct = (entry - stop) / entry
            r_mult   = actual / risk_pct if risk_pct else 0
            return round(r_mult * 0.02 * 1000, 2)  # $1k base
    return None


def build_digest(entries: list[dict]) -> str:
    total      = len(entries)
    resolved   = [e for e in entries if e.get("resolved") and e.get("outcome")]
    unresolved = total - len(resolved)

    wins  = [e for e in resolved if e.get("outcome", "").upper()
             not in {"HIT_STOP", "EXPIRED_LOSS", "LOSS", "STOP"}]
    stops = [e for e in resolved if e.get("outcome", "").upper()
             in {"HIT_STOP", "EXPIRED_LOSS", "LOSS", "STOP"}]
    wr    = len(wins) / len(resolved) * 100 if resolved else 0

    elite_n = sum(1 for e in entries if e.get("tier") in {"ELITE BUY", "ELITE"})
    sb_n    = sum(1 for e in entries if e.get("tier") in {"STRONG BUY"})

    tickers_w = ", ".join(e.get("ticker", "") for e in wins[:5])  or "—"
    tickers_l = ", ".join(e.get("ticker", "") for e in stops[:5]) or "—"

    pnl_vals = [p for e in resolved if (p := _pnl_estimate(e)) is not None]
    pnl_str  = f"${sum(pnl_vals):+,.0f} est." if pnl_vals else "pending resolution"

    streak = 0
    for e in reversed(resolved):
        if e.get("outcome", "").upper() in {"HIT_STOP", "EXPIRED_LOSS", "LOSS", "STOP"}:
            streak -= 1
        else:
            break
    for e in reversed(resolved):
        if e.get("outcome", "").upper() not in {"HIT_STOP", "EXPIRED_LOSS", "LOSS", "STOP"}:
            streak += 1
        else:
            break

    streak_str = (f"🔥 {abs(streak)}-win streak" if streak > 0
                  else (f"⚠️ {abs(streak)}-loss streak" if streak < 0 else "—"))

    divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    week_end = datetime.now().strftime("%a %d %b %Y")

    lines = [
        divider,
        f"📊  **TRADEY BOI X — Weekly Digest**  ·  w/e {week_end}",
        divider,
        "",
        f"**Signals sent this week:** {total}  ({elite_n} ELITE  ·  {sb_n} STRONG BUY)",
        f"**Resolved:**  {len(resolved)}  |  **Pending:**  {unresolved}",
        "",
    ]

    if resolved:
        lines += [
            f"**Win rate:**  {wr:.0f}%  ({len(wins)} wins  ·  {len(stops)} stops)",
            f"**Est. P&L:**  {pnl_str}",
            f"**Streak:**    {streak_str}",
            "",
            f"✅ Winners: _{tickers_w}_",
            f"🛑 Stopped: _{tickers_l}_",
        ]
    else:
        lines.append("_No resolved trades yet this week — positions still open._")

    lines += [
        "",
        divider,
        f"_{datetime.now().strftime('%a %d %b %Y %I:%M %p AEST')} — auto digest_",
    ]
    return "\n".join(lines)


def send_digest() -> bool:
    if not DISCORD:
        print("No Discord webhook configured — skipping digest.")
        return False
    entries = _load_entries(LOOKBACK)
    if not entries:
        print(f"No signals in the last {LOOKBACK} days — skipping digest.")
        return False
    msg = build_digest(entries)
    print(msg)
    try:
        r = requests.post(DISCORD, json={"content": msg}, timeout=10)
        ok = r.status_code in (200, 204)
        print(f"Discord: {'✅ sent' if ok else f'❌ {r.status_code}'}")
        return ok
    except Exception as e:
        print(f"Discord error: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if send_digest() else 1)
