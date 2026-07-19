"""
Feature 8 — Self-Review Engine (v3)

Computes a comprehensive weekly/monthly performance review from signal_log.json
and posts it to Discord. Designed to run as a scheduled GitHub Actions job
(Sunday night AEST) with no interactive I/O.

Usage:
    python tradey-boi-x/weekly_review.py [--lookback DAYS]

Default lookback: 7 days (weekly). Pass --lookback 30 for monthly.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "signal_log.json"

WIN_OUTCOMES = ("WIN", "HIT_TARGET", "EXPIRED_GAIN")
CALIB_BUCKETS = [
    (0.50, 0.60, "50–60%"),
    (0.60, 0.70, "60–70%"),
    (0.70, 0.80, "70–80%"),
    (0.80, 1.01, "80%+"),
]


# ─── Data helpers ──────────────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return []


def _resolved(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("outcome") is not None]


# ─── Analytics ─────────────────────────────────────────────────────────────────

def _calibration(entries: list[dict]) -> list[str]:
    lines = []
    for lo, hi, label in CALIB_BUCKETS:
        bucket = [e for e in entries if lo <= float(e.get("prob") or 0) < hi]
        if not bucket:
            continue
        wins   = sum(1 for e in bucket if e.get("outcome") in WIN_OUTCOMES)
        actual = wins / len(bucket)
        mid    = (lo + hi) / 2.0
        diff   = actual - mid
        icon   = "✅" if abs(diff) <= 0.08 else ("🟢" if diff > 0 else "⚠️")
        lines.append(f"  {label}: predicted {mid*100:.0f}% → actual {actual*100:.0f}% ({len(bucket)} trades) {icon}")
    return lines


def _regime_breakdown(entries: list[dict]) -> list[str]:
    regimes: dict[str, list] = {}
    for e in entries:
        r = (e.get("features") or {}).get("regime") or "unknown"
        regimes.setdefault(r, []).append(e)
    rows = []
    for r, trades in sorted(regimes.items(), key=lambda x: -len(x[1])):
        wins = sum(1 for t in trades if t.get("outcome") in WIN_OUTCOMES)
        pcts = [float(t.get("actual_pct") or 0) for t in trades]
        avg  = statistics.mean(pcts) * 100 if pcts else 0
        rows.append(f"  {r}: {wins}/{len(trades)} wins · {avg:+.1f}% avg")
    return rows


def _tier_breakdown(entries: list[dict]) -> list[str]:
    tiers: dict[str, list] = {}
    for e in entries:
        t = e.get("tier", "?")
        tiers.setdefault(t, []).append(e)
    rows = []
    for t, trades in sorted(tiers.items()):
        wins = sum(1 for x in trades if x.get("outcome") in WIN_OUTCOMES)
        pcts = [float(x.get("actual_pct") or 0) for x in trades]
        avg  = statistics.mean(pcts) * 100 if pcts else 0
        rows.append(f"  {t}: {wins}/{len(trades)} wins · {avg:+.1f}% avg")
    return rows


def _top_performers(entries: list[dict], n: int = 3) -> list[str]:
    winners = sorted(
        [e for e in entries if e.get("outcome") in WIN_OUTCOMES],
        key=lambda e: float(e.get("actual_pct") or 0),
        reverse=True,
    )[:n]
    return [
        f"  {e['ticker']}: {float(e.get('actual_pct') or 0)*100:+.1f}%"
        for e in winners
    ]


def _worst_performers(entries: list[dict], n: int = 3) -> list[str]:
    losers = sorted(
        [e for e in entries if e.get("outcome") is not None],
        key=lambda e: float(e.get("actual_pct") or 0),
    )[:n]
    return [
        f"  {e['ticker']}: {float(e.get('actual_pct') or 0)*100:+.1f}%"
        for e in losers
    ]


def build_report(lookback_days: int = 7) -> str:
    """Compile the full Discord report string."""
    all_entries  = _load_log()
    resolved_all = _resolved(all_entries)

    cutoff  = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent  = [e for e in resolved_all if (e.get("signal_date") or "") >= cutoff]

    now_str = datetime.utcnow().strftime("%d %b %Y")
    period  = "WEEKLY" if lookback_days <= 7 else "MONTHLY"

    lines: list[str] = [
        f"─────────────────────────────",
        f"📊 **TRADEY BOI X — {period} REVIEW** ({now_str})",
        f"Lookback: last {lookback_days} days  ·  Resolved: {len(recent)}  ·  All-time: {len(resolved_all)}",
        f"─────────────────────────────",
    ]

    if not recent:
        lines.append("No resolved trades in this period — check back after signals close out.")
        lines.append(f"─────────────────────────────")
        return "\n".join(lines)

    wins     = sum(1 for e in recent if e.get("outcome") in WIN_OUTCOMES)
    pcts     = [float(e.get("actual_pct") or 0) for e in recent]
    win_rate = wins / len(recent)
    avg_ret  = statistics.mean(pcts) * 100 if pcts else 0
    total_r  = sum(pcts) * 100

    gains = [p for p in pcts if p >= 0]
    losses= [abs(p) for p in pcts if p < 0]
    avg_g = statistics.mean(gains)  if gains  else 0.0
    avg_l = statistics.mean(losses) if losses else 1.0
    r_unit = avg_l if avg_l > 0 else 1.0
    expectancy = (win_rate * avg_g - (1 - win_rate) * avg_l) / r_unit

    lines += [
        f"**Win rate:** {win_rate*100:.0f}% ({wins}W / {len(recent)-wins}L)",
        f"**Avg return:** {avg_ret:+.2f}%  ·  **Total:** {total_r:+.1f}%",
        f"**Expectancy:** {expectancy:+.2f}R",
        "",
    ]

    # Tier breakdown
    tier_rows = _tier_breakdown(recent)
    if tier_rows:
        lines.append("**By tier:**")
        lines += tier_rows
        lines.append("")

    # Regime breakdown
    regime_rows = _regime_breakdown(recent)
    if regime_rows:
        lines.append("**By regime:**")
        lines += regime_rows
        lines.append("")

    # Top / worst performers
    best = _top_performers(recent)
    if best:
        lines.append("**Best trades:**")
        lines += best
    worst = _worst_performers(recent)
    if worst:
        lines.append("**Worst trades:**")
        lines += worst
    if best or worst:
        lines.append("")

    # Calibration (uses all-time resolved for stability)
    calib_rows = _calibration(resolved_all)
    if calib_rows:
        lines.append("**Confidence calibration (all-time):**")
        lines += calib_rows
        lines.append("")

    # Self-review verdict
    if win_rate >= 0.60 and expectancy >= 0.30:
        verdict = "✅ System performing well — continue scanning."
    elif win_rate >= 0.50 and expectancy >= 0.0:
        verdict = "👀 Marginal performance — review recent false positives."
    else:
        verdict = "⚠️ Performance below target — review thresholds and regime gating."

    lines += [
        f"**Auto-review:** {verdict}",
        f"─────────────────────────────",
        f"_{datetime.utcnow().strftime('%a %d %b %Y %H:%M UTC')}_",
    ]

    return "\n".join(lines)


def post_to_discord(message: str) -> bool:
    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook") or os.getenv("DISCORDWEBHOOK")
    if not webhook:
        print("No Discord webhook configured — skipping Discord post.")
        return False
    try:
        data = json.dumps({"content": message}).encode()
        req  = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        return True
    except Exception as exc:
        print(f"Discord post failed: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tradey Boi X — Weekly Self-Review Engine")
    parser.add_argument("--lookback", type=int, default=7,
                        help="Lookback window in days (default 7 = weekly, 30 = monthly)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report to stdout only; do not post to Discord")
    args = parser.parse_args()

    report = build_report(args.lookback)
    print(report)

    if not args.dry_run:
        ok = post_to_discord(report)
        if ok:
            print("\n✅ Report posted to Discord.")
        else:
            print("\n⚠️  Discord post failed or skipped.")
