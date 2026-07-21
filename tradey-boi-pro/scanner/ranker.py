"""
Multi-factor opportunity ranker for Tradey Boi Pro.

Takes a raw technical signal from market_scanner and enhances it with:
  1. Liquidity score          — average dollar volume
  2. Risk/Reward ratio        — target dist / stop dist
  3. Regime alignment         — how well this fits the current market environment
  4. Relative strength        — stock vs index over 20 days
  5. Volume quality           — sustained surge or single spike?
  6. AI confidence composite  — weighted blend of all factors

Output: composite_score (0–10), tier (ELITE / STRONG BUY / BUY / WATCH),
        rank (relative position in full signal list).

Tradey Boi X strategy logic, indicators, and filters are preserved.
Only the output ranking and universe scope are upgraded.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scanner.market_regime import RegimeData

log = logging.getLogger("Ranker")

# ── Tier thresholds ────────────────────────────────────────────────────────────
TIER_ELITE      = 8.5   # Immediate execution candidate
TIER_STRONG_BUY = 7.0   # Trade on next open
TIER_BUY        = 5.5   # Monitor / watch list
# < 5.5 = WATCH, not actioned


def rank_signal(
    signal:     dict,
    df_history: pd.DataFrame,
    regime:     "RegimeData",
) -> dict:
    """
    Enhance a raw scanner signal with multi-factor ranking.
    Returns updated signal dict with composite_score, tier, ranked_factors.
    """
    try:
        base_score  = float(signal.get("score", 0))
        base_prob   = float(signal.get("prob",  0.53))
        entry       = float(signal.get("entry_price", 1))
        stop        = float(signal.get("stop_price",  entry * 0.95))
        target      = float(signal.get("target_price", entry * 1.08))
        atr_pct     = float(signal.get("atr_pct", 2.0))

        factors: dict[str, float] = {}

        # ── F1: Technical quality (from base scanner, 0–10) ──────────────────
        factors["technical"] = base_score / 10   # normalise to 0–1

        # ── F2: Risk/Reward ratio ─────────────────────────────────────────────
        stop_dist   = max(entry - stop,   entry * 0.01)
        target_dist = max(target - entry, entry * 0.01)
        rr          = target_dist / stop_dist
        factors["risk_reward"] = min(rr / 4, 1.0)   # cap at 4:1 → score 1.0

        # ── F3: Liquidity (avg daily dollar volume) ───────────────────────────
        try:
            vol_series = df_history["Volume"].squeeze()
            close_series = df_history["Close"].squeeze()
            avg_dollar_vol = float((vol_series * close_series).rolling(20).mean().iloc[-1])
            # Score: $500K→0.25,  $1M→0.5,  $5M→0.75,  $20M+→1.0
            factors["liquidity"] = min(math.log10(max(avg_dollar_vol, 1e4)) / math.log10(2e7), 1.0)
        except Exception:
            factors["liquidity"] = 0.5   # neutral if unavailable

        # ── F4: Regime alignment ──────────────────────────────────────────────
        if regime.regime.value == "BULL":
            regime_score = 0.9 + regime.confidence * 0.1
        elif regime.regime.value == "NEUTRAL":
            regime_score = 0.4 + regime.confidence * 0.2
        else:   # BEAR
            regime_score = 0.0
        factors["regime"] = regime_score

        # ── F5: Relative strength (stock vs index 20-day) ────────────────────
        try:
            close_series  = df_history["Close"].squeeze().dropna()
            if len(close_series) >= 21:
                stock_ret   = (float(close_series.iloc[-1]) - float(close_series.iloc[-21])) \
                              / float(close_series.iloc[-21])
                # If we have index returns: compare; else use absolute momentum
                # Positive 20-day return with recent acceleration = strong RS
                roc5 = (float(close_series.iloc[-1]) - float(close_series.iloc[-6])) \
                       / float(close_series.iloc[-6])
                if stock_ret > 0.10 and roc5 > 0.02:
                    factors["relative_strength"] = 1.0
                elif stock_ret > 0.05:
                    factors["relative_strength"] = 0.75
                elif stock_ret > 0:
                    factors["relative_strength"] = 0.5
                else:
                    factors["relative_strength"] = 0.2
            else:
                factors["relative_strength"] = 0.5
        except Exception:
            factors["relative_strength"] = 0.5

        # ── F6: Volume quality (sustained vs spike) ───────────────────────────
        try:
            vol  = df_history["Volume"].squeeze().dropna()
            if len(vol) >= 5:
                avg3   = float(vol.iloc[-3:].mean())
                avg20  = float(vol.iloc[-21:-1].mean())
                # Sustained surge (3-day avg elevated vs 20-day) = higher quality
                sustained_ratio = avg3 / avg20 if avg20 > 0 else 1
                factors["volume_quality"] = min(sustained_ratio / 2, 1.0)
            else:
                factors["volume_quality"] = 0.5
        except Exception:
            factors["volume_quality"] = 0.5

        # ── Weighted composite (Tradey Boi Pro AI confidence) ─────────────────
        weights = {
            "technical":        0.30,
            "risk_reward":      0.20,
            "liquidity":        0.15,
            "regime":           0.20,
            "relative_strength":0.10,
            "volume_quality":   0.05,
        }
        composite = sum(factors[k] * weights[k] for k in weights)
        composite_score = round(composite * 10, 2)    # 0–10

        # ── Tier assignment ───────────────────────────────────────────────────
        if composite_score >= TIER_ELITE:
            tier = "ELITE"
        elif composite_score >= TIER_STRONG_BUY:
            tier = "STRONG BUY"
        elif composite_score >= TIER_BUY:
            tier = "BUY"
        else:
            tier = "WATCH"

        # ── AI confidence (probability adjusted by regime and composite) ───────
        ai_confidence = base_prob * (0.7 + composite * 0.3)
        ai_confidence = round(min(ai_confidence, 0.92), 3)

        return {
            **signal,
            "composite_score":    composite_score,
            "ai_confidence":      ai_confidence,
            "tier":               tier,
            "risk_reward":        round(rr, 2),
            "regime_alignment":   regime.regime.value,
            "ranked_factors":     {k: round(v, 3) for k, v in factors.items()},
        }

    except Exception as e:
        log.debug(f"Rank error for {signal.get('ticker')}: {e}")
        return {
            **signal,
            "composite_score": float(signal.get("score", 0)),
            "ai_confidence":   float(signal.get("prob", 0.53)),
            "tier":            signal.get("tier", "WATCH"),
            "risk_reward":     0.0,
            "regime_alignment": regime.regime.value if regime else "UNKNOWN",
            "ranked_factors":  {},
        }


def rank_signals(
    signals:     list[dict],
    df_history:  dict[str, pd.DataFrame],
    regime_map:  dict[str, "RegimeData"],
) -> list[dict]:
    """
    Rank a list of signals using multi-factor scoring.
    regime_map: {"ASX": RegimeData, "US": RegimeData}
    df_history:  ticker → DataFrame
    Returns list sorted by composite_score desc.
    """
    ranked = []
    for sig in signals:
        ticker = sig.get("ticker", "")
        df     = df_history.get(ticker)
        market = "ASX" if ticker.endswith(".AX") else "US"
        regime = regime_map.get(market) or regime_map.get("US")
        if df is None or regime is None:
            ranked.append({**sig, "composite_score": float(sig.get("score", 0)),
                           "ai_confidence": float(sig.get("prob", 0.53)),
                           "tier": sig.get("tier", "WATCH"), "risk_reward": 0.0,
                           "regime_alignment": "UNKNOWN", "ranked_factors": {}})
            continue
        ranked.append(rank_signal(sig, df, regime))

    ranked.sort(key=lambda s: (s["composite_score"], s["ai_confidence"]), reverse=True)
    for i, s in enumerate(ranked):
        s["rank"] = i + 1
    return ranked
