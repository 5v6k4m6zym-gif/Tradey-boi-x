"""
Enhanced Alert Formatter — Phase 3
Sends a second, richer Discord message for high-conviction Opportunity candidates.
Controlled by ENABLE_ENHANCED_ALERTS feature flag.

IMPORTANT: This does NOT replace or modify any existing alert from engine.py.
           It is purely additive — a second message that fires alongside the
           existing alert when the Opportunity Engine approves the trade.
"""
from __future__ import annotations
import json
import os
import urllib.request

from opportunity.config import ENABLE_ENHANCED_ALERTS

_RISK_EMOJI: dict[str, str] = {
    "LOW":    "🟢",
    "MEDIUM": "🟡",
    "HIGH":   "🔴",
}

_REGIME_EMOJI: dict[str, str] = {
    "BULLISH":  "🟢",
    "BEARISH":  "🔴",
    "SIDEWAYS": "🟡",
    "HIGH_VOL": "⚡",
    "LOW_VOL":  "😴",
}


# ─── Formatters ───────────────────────────────────────────────────────────────

def format_opportunity_alert(opp: dict) -> str:
    """
    Build the Discord message text for an Opportunity Alert.

    Parameters
    ----------
    opp : dict returned by opportunity.scoring.score_opportunity()

    Returns
    -------
    str  — message text, capped at 2000 chars.
    """
    ticker  = opp.get("ticker", "???")
    score   = opp.get("opportunity_score", 0)
    conf    = (opp.get("confidence", 0) or 0) * 100
    upside  = (opp.get("expected_upside_pct",  0) or 0) * 100
    dn_pct  = (opp.get("expected_downside_pct", 0) or 0) * 100
    risk    = opp.get("risk_level", "MEDIUM")
    rr      = opp.get("rr_ratio", 0.0) or 0.0
    entry   = opp.get("entry_zone", [0, 0])
    stop    = opp.get("stop_loss", 0.0) or 0.0
    tps     = opp.get("take_profit", [0, 0, 0])
    hold    = opp.get("est_holding_days", 0)
    trail   = (opp.get("trailing_stop_pct", 0) or 0) * 100
    regime  = opp.get("regime", "")
    risk_em = _RISK_EMOJI.get(risk, "⚪")
    reg_em  = _REGIME_EMOJI.get(regime, "")

    lines: list[str] = [
        f"🎯 **OPPORTUNITY ALERT — {ticker}**",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"**Opportunity Score:** {score:.0f}/100  |  **Confidence:** {conf:.0f}%",
        f"**Expected Return:** +{upside:.1f}%  |  "
        f"**Risk:** {risk_em} {risk}  |  **R:R** {rr:.2f}:1",
        "",
        f"**Entry Zone:**    ${entry[0]:.3f} – ${entry[1]:.3f}",
        f"**Stop Loss:**     ${stop:.3f}  (-{dn_pct:.1f}%)",
        f"**Take Profit:**   TP1 ${tps[0]:.3f}  |  TP2 ${tps[1]:.3f}  |  TP3 ${tps[2]:.3f}",
        f"**Trailing Stop:** {trail:.1f}% from peak",
        f"**Est. Hold:**     ~{hold} days",
    ]

    if regime:
        lines += ["", f"**Market Regime:** {reg_em} {regime}"]

    reasons_for     = opp.get("reasons_for", [])
    reasons_against = opp.get("reasons_against", [])

    if reasons_for:
        lines += ["", "**Why this trade:**"]
        lines += [f"• {r}" for r in reasons_for[:4]]

    if reasons_against:
        lines += ["", "**Risks:**"]
        lines += [f"• {r}" for r in reasons_against[:3]]

    tech = opp.get("technical_summary", "")
    mom  = opp.get("momentum_summary",  "")
    if tech or mom:
        lines += ["", f"_{tech}  |  {mom}_"]

    return "\n".join(lines)[:2000]


def format_outcome_alert(trade: dict) -> str:
    """
    Build a Discord message for a resolved trade outcome.

    Parameters
    ----------
    trade : dict
        A resolved signal log entry (has 'outcome', 'actual_pct', etc.)

    Returns
    -------
    str  — message text, capped at 2000 chars.
    """
    ticker     = trade.get("ticker", "???")
    outcome    = trade.get("outcome", "UNKNOWN")
    entry      = trade.get("entry_price", 0.0) or 0.0
    exit_p     = trade.get("exit_price",  0.0) or 0.0
    actual_pct = (trade.get("actual_pct", 0.0) or 0.0) * 100
    signal_dt  = trade.get("signal_date", "")
    score      = trade.get("opportunity_score", None)
    conf       = trade.get("confidence", None)
    stop_p     = trade.get("stop_price",   None)
    target_p   = trade.get("target_price", None)

    _WIN_OUTCOMES = ("WIN", "HIT_TARGET", "EXPIRED_GAIN")
    is_win = outcome in _WIN_OUTCOMES
    icon   = "✅" if is_win else "❌"
    pl_sign = "+" if actual_pct >= 0 else ""

    lines = [
        f"{icon} **TRADE CLOSED — {ticker}**  [{outcome}]",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"**P&L:** {pl_sign}{actual_pct:.1f}%",
        f"**Entry:** ${entry:.4f}  →  **Exit:** ${exit_p:.4f}",
        f"**Signal date:** {signal_dt}",
    ]

    if stop_p and target_p:
        lines.append(
            f"**Stop:** ${stop_p:.3f}  |  **Target:** ${target_p:.3f}"
        )

    if score is not None and conf is not None:
        lines.append(
            f"**Original prediction:** Score {score}/100  |  Conf {conf*100:.0f}%"
        )

    if is_win:
        lines.append("\n**Suggested next action:** Review sector for continuation plays.")
    else:
        lines.append("\n**Suggested next action:** Check stop level — consider reducing size.")

    return "\n".join(lines)[:2000]


# ─── Discord dispatch ──────────────────────────────────────────────────────────

def send_opportunity_alert(opp: dict) -> bool:
    """
    Format and post an Opportunity Alert to Discord.

    Returns True if delivered (HTTP 200/204), False otherwise.
    """
    if not ENABLE_ENHANCED_ALERTS:
        return False

    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook")
    if not webhook or not opp:
        return False

    try:
        msg  = format_opportunity_alert(opp)
        data = json.dumps({"content": msg}).encode()
        req  = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def send_outcome_alert(trade: dict) -> bool:
    """
    Format and post a Trade Outcome Alert to Discord.

    Returns True if delivered, False otherwise.
    """
    if not ENABLE_ENHANCED_ALERTS:
        return False

    webhook = os.getenv("Discordwebhook") or os.getenv("discordwebhook")
    if not webhook or not trade:
        return False

    try:
        msg  = format_outcome_alert(trade)
        data = json.dumps({"content": msg}).encode()
        req  = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False
