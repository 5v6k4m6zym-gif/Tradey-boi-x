"""
Enhanced Alert Formatter — Phase 3
Sends a second, richer Discord message for high-conviction Opportunity candidates.
Controlled by ENABLE_ENHANCED_ALERTS feature flag.

IMPORTANT: This does NOT replace or modify any existing alert from engine.py.
           It is purely additive — a second message that fires alongside the
           existing alert when the Opportunity Engine approves the trade.
"""
from __future__ import annotations
import os
import requests
from opportunity.config import ENABLE_ENHANCED_ALERTS

_DISCORD = os.getenv("Discordwebhook", "") or os.getenv("discordwebhook", "")

_RISK_EMOJI: dict[str, str] = {
    "LOW":    "🟢",
    "MEDIUM": "🟡",
    "HIGH":   "🔴",
}


def send_opportunity_alert(opp: dict) -> bool:
    """
    Send an enhanced Opportunity Alert to Discord.

    Parameters
    ----------
    opp : dict
        Dict returned by opportunity.scoring.score_opportunity().

    Returns
    -------
    bool
        True if the message was delivered (HTTP 200 or 204).
    """
    if not ENABLE_ENHANCED_ALERTS:
        return False
    if not _DISCORD or not opp:
        return False

    ticker  = opp["ticker"]
    score   = opp["opportunity_score"]
    conf    = opp["confidence"] * 100
    upside  = opp["expected_upside_pct"]
    risk    = opp["risk_level"]
    rr      = opp["rr_ratio"]
    entry   = opp["entry_zone"]
    stop    = opp["stop_loss"]
    tps     = opp["take_profit"]
    hold    = opp["est_holding_days"]
    dn_pct  = opp["expected_downside_pct"]
    trail   = opp["trailing_stop_pct"] * 100
    regime_note  = opp.get("regime_note", "")
    risk_em      = _RISK_EMOJI.get(risk, "⚪")

    lines: list[str] = [
        f"🎯 **OPPORTUNITY ALERT — {ticker}**",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"**Opportunity Score:** {score:.0f}/100  |  **Confidence:** {conf:.0f}%",
        f"**Expected Return:** +{upside:.1f}%  |  **Risk:** {risk_em} {risk}  |  **R:R** {rr:.1f}:1",
        "",
        f"**Entry Zone:**   ${entry[0]:.3f} – ${entry[1]:.3f}",
        f"**Stop Loss:**    ${stop:.3f}  (-{dn_pct:.1f}%)",
        f"**Take Profit:**  TP1 ${tps[0]:.3f}  |  TP2 ${tps[1]:.3f}  |  TP3 ${tps[2]:.3f}",
        f"**Trailing Stop:** {trail:.1f}% from peak",
        f"**Est. Hold:**    ~{hold} days",
    ]

    if regime_note:
        lines += ["", regime_note]

    if opp.get("reasons_for"):
        lines += ["", "**Why this trade:**"]
        lines += [f"• {r}" for r in opp["reasons_for"][:4]]

    if opp.get("reasons_against"):
        lines += ["", "**Risks:**"]
        lines += [f"• {r}" for r in opp["reasons_against"][:3]]

    lines += [
        "",
        f"_{opp.get('technical_summary', '')}  |  {opp.get('momentum_summary', '')}_",
    ]

    try:
        r = requests.post(
            _DISCORD,
            json={"content": "\n".join(lines)[:2000]},
            timeout=10,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False
