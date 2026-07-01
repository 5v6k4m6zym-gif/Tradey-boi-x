"""
Opportunity Scoring Engine — Phase 2
Runs a multi-factor weighted analysis on a ticker to identify high-return,
high-conviction setups (20–100%+ expected moves).

Controlled by ENABLE_OPPORTUNITY_ENGINE feature flag — returns None when off.
Never modifies engine.py logic or any existing signal.
"""
from __future__ import annotations
import pandas as pd
from opportunity.config import ENABLE_OPPORTUNITY_ENGINE, WEIGHTS, FILTERS


# ── Component scorers (0–100 each) ───────────────────────────────────────────

def _score_expected_return(
    df: pd.DataFrame,
    atr_multiplier: float = 4.0,
) -> tuple[float, float, float]:
    """
    Estimate expected return using ATR-scaled projection.

    High-return focus: score of 100 = ≥50% expected upside.
    Returns (score_0_100, expected_upside_pct, expected_downside_pct).
    """
    close = float(df["Close"].iloc[-1])
    atr   = float(df["atr"].iloc[-1])
    adx   = float(df["adx"].iloc[-1])

    trend_mult   = 1.0 + max(0.0, (adx - 20.0) / 30.0)
    upside_abs   = atr * atr_multiplier * trend_mult
    upside_pct   = upside_abs / close if close > 0 else 0.0
    downside_pct = (atr * 1.5) / close if close > 0 else 0.0

    score = min(100.0, max(0.0, (upside_pct - 0.05) / 0.45 * 100.0))
    return round(score, 1), round(upside_pct, 4), round(downside_pct, 4)


def _score_technical_strength(df: pd.DataFrame) -> float:
    """Score 0–100 based on technical indicator alignment."""
    row   = df.iloc[-1]
    score = 0.0

    if row.get("ema20", 0) > row.get("ema50", 1): score += 25
    if 45 <= float(row.get("rsi",    50)) <= 70:  score += 20
    if float(row.get("adx",  0)) > 25:            score += 20
    if row.get("breakout", 0):                     score += 20
    if float(row.get("macd_diff", 0)) > 0:        score += 15

    return min(100.0, score)


def _score_volume_expansion(df: pd.DataFrame) -> float:
    """Score 0–100 based on volume surge relative to average."""
    vol_ratio = float(df.iloc[-1].get("vol_ratio", 1.0))
    if vol_ratio <= 1.0: return 0.0
    if vol_ratio >= 5.0: return 100.0
    return min(100.0, (vol_ratio - 1.0) / 4.0 * 100.0)


def _score_momentum(df: pd.DataFrame) -> float:
    """Score 0–100 based on short and medium-term price momentum."""
    row   = df.iloc[-1]
    ret5  = float(row.get("ret_5",  0))
    ret20 = float(row.get("ret_20", 0))
    score = 0.0

    if   ret5  > 0.05:  score += 40
    elif ret5  > 0.02:  score += 20
    elif ret5  > 0:     score += 10

    if   ret20 > 0.10:  score += 40
    elif ret20 > 0.05:  score += 25
    elif ret20 > 0:     score += 10

    if float(row.get("macd_diff", 0)) > 0: score += 20

    return min(100.0, score)


def _score_news_catalyst(df: pd.DataFrame) -> float:
    """Score 0–100 using OBV ratio as a news/catalyst activity proxy."""
    obv_ratio = float(df.iloc[-1].get("obv_ratio", 1.0))
    if obv_ratio >= 2.0: return 90.0
    if obv_ratio >= 1.5: return 70.0
    if obv_ratio >= 1.2: return 50.0
    if obv_ratio >= 1.0: return 30.0
    return 10.0


def _score_institutional(df: pd.DataFrame) -> float:
    """Score 0–100 as institutional buying proxy using OBV trend slope."""
    obv_ratio = float(df.iloc[-1].get("obv_ratio", 1.0))
    if obv_ratio >= 2.5: return 100.0
    if obv_ratio >= 2.0: return 85.0
    if obv_ratio >= 1.5: return 65.0
    if obv_ratio >= 1.0: return 40.0
    return 10.0


def _score_risk_reward(upside_pct: float, downside_pct: float) -> tuple[float, float]:
    """Score 0–100 and R:R ratio. Score of 100 = 10:1 or better."""
    if downside_pct <= 0:
        return 0.0, 0.0
    rr    = round(upside_pct / downside_pct, 2)
    score = min(100.0, max(0.0, (rr - 1.0) / 9.0 * 100.0))
    return round(score, 1), rr


# ── Supporting helpers ────────────────────────────────────────────────────────

def _estimate_holding_period(df: pd.DataFrame, upside_pct: float) -> int:
    """Estimate holding period in calendar days."""
    close = float(df["Close"].iloc[-1])
    atr   = float(df["atr"].iloc[-1])
    daily_move = atr / close if close > 0 else 0.01
    return min(180, max(5, int(upside_pct / (daily_move + 1e-9) * 0.6)))


def _reasons(
    row: pd.Series,
    upside_pct: float,
    rr: float,
) -> tuple[list[str], list[str]]:
    """Generate bullet-point reasons for and against the trade."""
    for_r: list[str] = []
    against_r: list[str] = []

    if row.get("breakout", 0):
        for_r.append("Confirmed price breakout")
    if float(row.get("ema20", 0)) > float(row.get("ema50", 1)):
        for_r.append("EMA 20 > 50 — established uptrend")
    if float(row.get("adx", 0)) > 25:
        for_r.append(f"Strong trend (ADX {float(row.get('adx', 0)):.0f})")
    if float(row.get("vol_ratio", 1)) > 1.5:
        for_r.append(f"Volume surge ({float(row.get('vol_ratio', 1)):.1f}× average)")
    if float(row.get("ret_20", 0)) > 0.05:
        for_r.append(f"Strong 20-day momentum (+{float(row.get('ret_20', 0)) * 100:.1f}%)")
    if rr >= 3:
        for_r.append(f"Attractive risk/reward ({rr:.1f}:1)")
    if upside_pct >= 0.30:
        for_r.append(f"High expected return (+{upside_pct * 100:.0f}%)")

    if float(row.get("rsi", 50)) > 70:
        against_r.append(f"RSI overbought ({float(row.get('rsi', 50)):.0f})")
    if float(row.get("rsi", 50)) < 40:
        against_r.append(f"RSI weak ({float(row.get('rsi', 50)):.0f})")
    if float(row.get("adx", 0)) < 20:
        against_r.append("Trend not yet established (ADX < 20)")
    if float(row.get("vol_ratio", 1)) < 1.0:
        against_r.append("Below-average volume")
    if rr < 2:
        against_r.append(f"Risk/reward below 2:1 ({rr:.1f}:1)")

    return for_r, against_r


# ── Public API ────────────────────────────────────────────────────────────────

def score_opportunity(
    ticker: str,
    df: pd.DataFrame,
    regime: dict | None = None,
) -> dict | None:
    """
    Run the full Opportunity Scoring Engine on a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    df : pd.DataFrame
        Output from engine.get_data() — must include standard feature columns.
    regime : dict | None
        Output from opportunity.regime.detect_regime() — optional regime context.

    Returns
    -------
    dict or None
        Full opportunity analysis dict, or None if flag is off or filters fail.
    """
    if not ENABLE_OPPORTUNITY_ENGINE:
        return None
    if df is None or df.empty or len(df) < 30:
        return None

    try:
        row = df.iloc[-1]

        # ── Component scores ─────────────────────────────────────────────────
        ret_score, upside_pct, downside_pct = _score_expected_return(df)
        tech_score  = _score_technical_strength(df)
        vol_score   = _score_volume_expansion(df)
        mom_score   = _score_momentum(df)
        news_score  = _score_news_catalyst(df)
        inst_score  = _score_institutional(df)
        rr_score, rr = _score_risk_reward(upside_pct, downside_pct)

        # ── Weighted composite score ──────────────────────────────────────────
        w = WEIGHTS
        opp_score = (
            ret_score  * w["expected_return"]    +
            tech_score * w["technical_strength"] +
            vol_score  * w["volume_expansion"]   +
            mom_score  * w["momentum"]           +
            news_score * w["news_catalyst"]      +
            inst_score * w["institutional"]      +
            rr_score   * w["risk_reward"]
        )

        confidence = min(0.95, max(0.05, opp_score / 100 * 0.9 + 0.05))

        # ── Filter gate ───────────────────────────────────────────────────────
        f = FILTERS
        if opp_score    < f["min_opportunity_score"]: return None
        if confidence   < f["min_confidence"]:         return None
        if upside_pct   < f["min_expected_upside"]:   return None
        if rr           < f["min_rr_ratio"]:           return None
        if downside_pct > f["max_downside"]:            return None

        # ── Derived trade plan ────────────────────────────────────────────────
        close        = float(row["Close"])
        stop         = round(close * (1.0 - downside_pct), 3)
        tp1          = round(close * (1.0 + upside_pct * 0.40), 3)
        tp2          = round(close * (1.0 + upside_pct * 0.70), 3)
        tp3          = round(close * (1.0 + upside_pct), 3)
        entry_lo     = round(close * 0.99, 3)
        entry_hi     = round(close * 1.01, 3)
        hold_days    = _estimate_holding_period(df, upside_pct)
        trailing_pct = round(downside_pct * 0.80, 3)

        risk_level = (
            "LOW"    if downside_pct < 0.04 else
            "MEDIUM" if downside_pct < 0.07 else
            "HIGH"
        )

        # ── Regime adjustment ─────────────────────────────────────────────────
        regime_note = ""
        if regime:
            reg = regime.get("regime", "")
            if reg == "BEARISH":
                confidence  = round(confidence * 0.85, 3)
                regime_note = "⚠️ Bearish regime — reduce position size"
            elif reg == "BULLISH":
                regime_note = "✅ Bullish regime — conditions favour the trade"
            elif reg == "HIGH_VOL":
                trailing_pct = round(trailing_pct * 1.30, 3)
                regime_note  = "⚡ High volatility — stops widened automatically"

        reasons_for, reasons_against = _reasons(row, upside_pct, rr)

        return {
            "ticker":                ticker,
            "opportunity_score":     round(opp_score, 1),
            "confidence":            round(confidence, 3),
            "expected_upside_pct":   round(upside_pct * 100, 1),
            "expected_downside_pct": round(downside_pct * 100, 1),
            "est_holding_days":      hold_days,
            "prob_target_hit":       round(confidence * 0.90, 3),
            "prob_stop_hit":         round((1.0 - confidence) * 0.70, 3),
            "risk_level":            risk_level,
            "rr_ratio":              rr,
            "entry_zone":            [entry_lo, entry_hi],
            "stop_loss":             stop,
            "take_profit":           [tp1, tp2, tp3],
            "trailing_stop_pct":     trailing_pct,
            "regime":                regime.get("regime") if regime else None,
            "regime_note":           regime_note,
            "reasons_for":           reasons_for,
            "reasons_against":       reasons_against,
            "technical_summary":     (
                f"RSI {float(row.get('rsi', 0)):.0f} | "
                f"ADX {float(row.get('adx', 0)):.0f} | "
                f"{'Breakout ✅' if row.get('breakout') else 'No breakout'}"
            ),
            "momentum_summary": (
                f"5d: {float(row.get('ret_5', 0)) * 100:+.1f}% | "
                f"20d: {float(row.get('ret_20', 0)) * 100:+.1f}%"
            ),
            "component_scores": {
                "expected_return":    round(ret_score,  1),
                "technical_strength": round(tech_score, 1),
                "volume_expansion":   round(vol_score,  1),
                "momentum":           round(mom_score,  1),
                "news_catalyst":      round(news_score, 1),
                "institutional":      round(inst_score, 1),
                "risk_reward":        round(rr_score,   1),
            },
        }

    except Exception:
        return None
