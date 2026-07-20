"""
Weekly performance digest — sends a Discord summary every Sunday night (AEST).
Called by GitHub Actions on a cron schedule.
Reads signal_log.json to summarise the past 7 days of signals,
plus a full 14-metric performance dashboard (v4.0).
"""
import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "signal_log.json"
DISCORD  = os.getenv("Discordwebhook", "") or os.getenv("discordwebhook", "")
LOOKBACK = int(os.getenv("DIGEST_DAYS", "7"))

WIN_OUTCOMES = {"HIT_STOP", "EXPIRED_LOSS", "LOSS", "STOP"}


def _load_all_resolved() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        entries = json.loads(LOG_FILE.read_text())
    except Exception:
        return []
    return [e for e in entries if e.get("outcome") and e.get("actual_pct") is not None]


def _load_entries(days: int) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        entries = json.loads(LOG_FILE.read_text())
    except Exception:
        return []
    cutoff = (datetime.now() - timedelta(days=days)).date()
    out = []
    for e in entries:
        ds = (e.get("signal_date") or e.get("date") or "")[:10]
        try:
            if datetime.strptime(ds, "%Y-%m-%d").date() >= cutoff:
                out.append(e)
        except Exception:
            pass
    return out


def _pnl_estimate(e: dict) -> float | None:
    actual = e.get("actual_pct")
    if actual is not None:
        entry = e.get("entry_price", 0)
        stop  = e.get("stop_loss",  0)
        if entry and stop and entry > stop > 0:
            risk_pct = (entry - stop) / entry
            r_mult   = actual / risk_pct if risk_pct else 0
            return round(r_mult * 0.02 * 1000, 2)
    return None


def _trend(current: float, previous: float, *, higher_is_better: bool = True) -> str:
    if previous == 0:
        return ""
    delta = current - previous
    if abs(delta) < 0.001:
        return "→"
    improving = delta > 0 if higher_is_better else delta < 0
    return "↑" if improving else "↓"


def _full_metrics_section(all_resolved: list[dict], prev_resolved: list[dict]) -> list[str]:
    """Build the 14-metric dashboard section for the Discord message."""
    try:
        from opportunity.metrics import compute_metrics
    except Exception:
        return []

    if not all_resolved:
        return []

    m  = compute_metrics(all_resolved)
    mp = compute_metrics(prev_resolved) if prev_resolved else {}

    def _fmt_pct(v: float) -> str:
        return f"{v*100:+.1f}%"

    def _row(label: str, val: str, trend: str = "") -> str:
        return f"  {label:<22} {val:>10}  {trend}"

    pf_trend  = _trend(m["profit_factor"],   mp.get("profit_factor",  0))
    exp_trend = _trend(m["expectancy"],      mp.get("expectancy",     0))
    dd_trend  = _trend(m["max_drawdown"],    mp.get("max_drawdown",   0), higher_is_better=False)
    sh_trend  = _trend(m["sharpe"],          mp.get("sharpe",         0))
    so_trend  = _trend(m["sortino"],         mp.get("sortino",        0))
    ca_trend  = _trend(m["calmar"],          mp.get("calmar",         0))
    wr_trend  = _trend(m["win_rate"],        mp.get("win_rate",       0))
    eq_trend  = _trend(m["equity_stability"],mp.get("equity_stability",0))

    lines = [
        "",
        "**📈 Performance Dashboard (all-time · v4.0)**",
        "```",
        f"  {'Metric':<22} {'Value':>10}  Trend",
        f"  {'─'*22} {'─'*10}  {'─'*5}",
        _row("Profit Factor",    f"{m['profit_factor']:.2f}",              pf_trend),
        _row("Expectancy",       f"{m['expectancy']*100:+.2f}%/trade",     exp_trend),
        _row("Win Rate",         f"{m['win_rate']*100:.1f}%",              wr_trend),
        _row("Avg Winner",       _fmt_pct(m["avg_win"])),
        _row("Avg Loser",        _fmt_pct(m["avg_loss"])),
        _row("R:R Ratio",        f"{m['rr_ratio']:.2f}"),
        _row("Max Drawdown",     _fmt_pct(m["max_drawdown"]),              dd_trend),
        _row("CAGR (est.)",      _fmt_pct(m["cagr"])),
        _row("Sharpe Ratio",     f"{m['sharpe']:.2f}",                    sh_trend),
        _row("Sortino Ratio",    f"{m['sortino']:.2f}",                   so_trend),
        _row("Calmar Ratio",     f"{m['calmar']:.2f}",                    ca_trend),
        _row("Avg Hold",         f"{m['avg_hold_days']:.0f} days"),
        _row("Trade Freq",       f"{m['trade_freq_month']:.1f}/month"),
        _row("Equity Stability", f"{m['equity_stability']:.2f}  (R²)",    eq_trend),
        f"  {'─'*22} {'─'*10}",
        f"  {'Trades analysed':<22} {m['n']:>10}",
        "```",
    ]
    return lines


def build_digest(entries: list[dict]) -> str:
    total      = len(entries)
    resolved   = [e for e in entries if e.get("resolved") and e.get("outcome")]
    unresolved = total - len(resolved)

    wins  = [e for e in resolved if e.get("outcome", "").upper() not in WIN_OUTCOMES]
    stops = [e for e in resolved if e.get("outcome", "").upper() in WIN_OUTCOMES]
    wr    = len(wins) / len(resolved) * 100 if resolved else 0

    elite_n = sum(1 for e in entries if e.get("tier") in {"ELITE BUY", "ELITE"})
    sb_n    = sum(1 for e in entries if e.get("tier") in {"STRONG BUY"})

    tickers_w = ", ".join(e.get("ticker", "") for e in wins[:5])  or "—"
    tickers_l = ", ".join(e.get("ticker", "") for e in stops[:5]) or "—"

    pnl_vals = [p for e in resolved if (p := _pnl_estimate(e)) is not None]
    pnl_str  = f"${sum(pnl_vals):+,.0f} est." if pnl_vals else "pending resolution"

    streak = 0
    for e in reversed(resolved):
        if e.get("outcome", "").upper() in WIN_OUTCOMES:
            streak -= 1
        else:
            break
    for e in reversed(resolved):
        if e.get("outcome", "").upper() not in WIN_OUTCOMES:
            streak += 1
        else:
            break

    streak_str = (f"🔥 {abs(streak)}-win streak" if streak > 0
                  else (f"⚠️ {abs(streak)}-loss streak" if streak < 0 else "—"))

    divider  = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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

    all_resolved  = _load_all_resolved()
    prev_resolved = [e for e in all_resolved if e not in resolved]
    dashboard     = _full_metrics_section(all_resolved, prev_resolved)
    if dashboard:
        lines += dashboard

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
        r  = requests.post(DISCORD, json={"content": msg}, timeout=10)
        ok = r.status_code in (200, 204)
        print(f"Discord: {'✅ sent' if ok else f'❌ {r.status_code}'}")
        return ok
    except Exception as e:
        print(f"Discord error: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if send_digest() else 1)
