"""
Trade Evaluation & Filtering Layer (Phase 8)
=============================================
A SAFE INSTRUMENTATION + FILTER LAYER on top of the existing bot. It does NOT
modify the prediction model, does NOT change how trade signals are generated,
and does NOT alter execution logic — it only wraps it.

Controlled by two independent switches in opportunity/config.py:
  - ENABLE_TRADE_EVALUATOR: master on/off for this whole layer. When False,
    `process_trade_signal()` is a no-op passthrough (returns the trade
    unchanged) so nothing about the existing flow changes.
  - SHADOW_MODE (default True): when True, decisions are computed and logged
    but the trade is NEVER allowed through (returns None) — pure observation,
    zero behavioural impact on live trading. Set SHADOW_MODE=false only once
    shadow-mode logs have been reviewed and the filter is trusted.

Public API
----------
TradeEvaluator            — stateless evaluator: computes edge/predictability/
                             noise/risk-reward and applies the pass/fail rules.
process_trade_signal(...) — wrapper: evaluate -> log -> gate -> return trade
                             or None. This is the only integration point other
                             modules should call.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from opportunity.config import (
    ENABLE_TRADE_EVALUATOR,
    SHADOW_MODE,
    TRADE_EVAL_THRESHOLDS,
    TRADE_EVAL_LOG_PATH,
)


# ─── Config wrapper (also importable as a class, per the spec) ───────────────
class Config:
    """Thin object-style view over the module-level flags in opportunity.config.

    Kept as a class (per spec) for callers that prefer `Config.SHADOW_MODE`
    over importing the constant directly; both stay in sync since this reads
    the same underlying values.
    """
    ENABLE_TRADE_EVALUATOR = ENABLE_TRADE_EVALUATOR
    SHADOW_MODE            = SHADOW_MODE
    THRESHOLDS             = TRADE_EVAL_THRESHOLDS
    LOG_PATH               = TRADE_EVAL_LOG_PATH


# ─── Evaluation result ────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    edge_score: float
    predictability_score: float
    noise_index: float
    risk_reward: float
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "edge_score":            self.edge_score,
            "predictability_score":  self.predictability_score,
            "noise_index":           self.noise_index,
            "risk_reward":           self.risk_reward,
            "passed":                self.passed,
            "rejection_reasons":     self.rejection_reasons,
        }


# ─── TradeEvaluator ───────────────────────────────────────────────────────────
class TradeEvaluator:
    """
    Evaluates a proposed trade's quality using data already available from the
    existing engine — does not call out to the model or execution system.

    `trade` is expected to be a dict-like signal with (at minimum):
        ticker / symbol   : str
        direction         : str, e.g. "LONG"           (optional, default LONG)
        entry             : float — planned/actual entry price
        stop_loss         : float — planned stop price
        take_profit       : float — planned target price
        probability       : float — model win-probability (0-1), e.g. res["prob"]
        expected_r         : float — optional, ATR-implied expected R multiple
                              (engine.expected_value_r output), used to enrich
                              the edge score if present.

    `market_data` is the OHLCV DataFrame already fetched for that ticker
    (same shape as engine.get_data()'s output — needs High/Low/Open/Close).
    """

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = thresholds or TRADE_EVAL_THRESHOLDS

    # ── Component calculations ────────────────────────────────────────────
    @staticmethod
    def compute_risk_reward(entry: float, stop_loss: float, take_profit: float) -> float:
        """RR = |take_profit - entry| / |entry - stop_loss|."""
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return 0.0
        reward = abs(take_profit - entry)
        return round(reward / risk, 3)

    @staticmethod
    def compute_noise_index(market_data: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
        """
        Noise Index — higher = worse (more random/unstable).

        Blends three components, each a well-understood technical proxy so
        this stays independent of (and doesn't require changes to) the
        existing prediction model:
          - choppiness   : 1 - Kaufman Efficiency Ratio (net move / path length)
                           over `lookback` bars. 0 = perfectly trending,
                           1 = pure random walk.
          - volatility   : stdev of daily returns over the window (raw, not
                           annualised — keeps it on a comparable small scale).
          - wick_instability: average fraction of each bar's range that is
                           "wick" rather than body — large wicks relative to
                           the close-to-open move indicate indecision/noise.

        Returns (noise_index, efficiency_ratio) — the ER is reused by
        compute_predictability_score() so it's only computed once.
        """
        recent = market_data.tail(lookback + 1)
        if len(recent) < 3:
            return 1.5, 0.0   # too little data — treat as high-noise/unknown

        closes = recent["Close"]
        net_change   = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))
        path_length  = float(closes.diff().abs().sum())
        efficiency_ratio = net_change / path_length if path_length > 0 else 0.0

        returns    = closes.pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 1 else 0.0

        bars       = recent.iloc[1:]
        bar_range  = (bars["High"] - bars["Low"]).replace(0, pd.NA)
        body       = (bars["Close"] - bars["Open"]).abs()
        wick_ratio = ((bar_range - body) / bar_range).dropna()
        wick_instability = float(wick_ratio.mean()) if len(wick_ratio) else 0.5

        choppiness  = 1.0 - efficiency_ratio
        noise_index = (choppiness * 1.0) + (volatility * 20.0) + (wick_instability * 0.5)

        return round(float(noise_index), 3), round(float(efficiency_ratio), 3)

    @staticmethod
    def compute_predictability_score(
        market_data: pd.DataFrame, efficiency_ratio: float, lookback: int = 20
    ) -> float:
        """
        Predictability Score (0-1, higher = better) — market structure quality.
        Blends trend efficiency (how directly price is moving) with trend
        alignment/strength from indicators already computed elsewhere in the
        pipeline when present (ema20/ema50), falling back to neutral values
        when those columns aren't available.
        """
        recent = market_data.tail(lookback)
        if "ema20" in recent.columns and "ema50" in recent.columns and len(recent) > 1:
            ema20_last = float(recent["ema20"].iloc[-1])
            ema50_last = float(recent["ema50"].iloc[-1])
            ema50_first = float(recent["ema50"].iloc[0])
            trend_alignment = 1.0 if ema20_last > ema50_last else 0.0
            ema_slope = (ema50_last - ema50_first) / ema50_first if ema50_first else 0.0
            trend_strength = min(abs(ema_slope) * 10.0, 1.0)
        else:
            trend_alignment, trend_strength = 0.5, 0.5

        score = 0.5 * efficiency_ratio + 0.3 * trend_strength + 0.2 * trend_alignment
        return round(float(min(max(score, 0.0), 1.0)), 3)

    @staticmethod
    def compute_edge_score(probability: float, risk_reward: float, expected_r: float | None) -> float:
        """
        Edge Score (0-1, higher = better) — weighted blend of:
          - probability (from the existing model, unchanged)
          - risk/reward ratio
          - expected-return quality (uses expected_r if the caller supplied
            it, e.g. from engine.expected_value_r; otherwise falls back to a
            risk/reward-only proxy so this still works standalone).
        """
        prob_component = min(max((probability - 0.5) / 0.5, 0.0), 1.0)
        rr_component   = min(max(risk_reward / 5.0, 0.0), 1.0)
        if expected_r is not None:
            er_component = min(max(expected_r / 2.0, 0.0), 1.0)
        else:
            er_component = rr_component

        score = 0.40 * prob_component + 0.35 * rr_component + 0.25 * er_component
        return round(float(score), 3)

    # ── Full evaluation ────────────────────────────────────────────────────
    def evaluate(self, trade: dict[str, Any], market_data: pd.DataFrame) -> EvaluationResult:
        entry       = float(trade["entry"])
        stop_loss   = float(trade["stop_loss"])
        take_profit = float(trade["take_profit"])
        probability = float(trade.get("probability", trade.get("prob", 0.5)))
        expected_r  = trade.get("expected_r")

        risk_reward = self.compute_risk_reward(entry, stop_loss, take_profit)
        noise_index, efficiency_ratio = self.compute_noise_index(market_data)
        predictability = self.compute_predictability_score(market_data, efficiency_ratio)
        edge_score = self.compute_edge_score(probability, risk_reward, expected_r)

        reasons: list[str] = []
        if edge_score < self.thresholds["min_edge_score"]:
            reasons.append(f"edge_score {edge_score:.2f} < {self.thresholds['min_edge_score']:.2f}")
        if predictability < self.thresholds["min_predictability_score"]:
            reasons.append(f"predictability_score {predictability:.2f} < {self.thresholds['min_predictability_score']:.2f}")
        if risk_reward < self.thresholds["min_risk_reward"]:
            reasons.append(f"risk_reward {risk_reward:.2f} < {self.thresholds['min_risk_reward']:.2f}")
        if noise_index > self.thresholds["max_noise_index"]:
            reasons.append(f"noise_index {noise_index:.2f} > {self.thresholds['max_noise_index']:.2f}")

        return EvaluationResult(
            edge_score=edge_score,
            predictability_score=predictability,
            noise_index=noise_index,
            risk_reward=risk_reward,
            passed=(len(reasons) == 0),
            rejection_reasons=reasons,
        )


# ─── Logging (append-only JSONL) ──────────────────────────────────────────────
def _log_path() -> Path:
    p = Path(TRADE_EVAL_LOG_PATH)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return p


def log_trade_decision(trade: dict[str, Any], evaluation: EvaluationResult) -> None:
    """Append one JSONL record for this trade decision. Never raises — a
    logging failure must not be able to break the scan/trade flow."""
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "symbol":               trade.get("ticker", trade.get("symbol")),
            "direction":            trade.get("direction", "LONG"),
            "probability":          trade.get("probability", trade.get("prob")),
            "edge_score":           evaluation.edge_score,
            "predictability_score": evaluation.predictability_score,
            "noise_index":          evaluation.noise_index,
            "risk_reward":          evaluation.risk_reward,
            "passed":               evaluation.passed,
            "rejection_reasons":    evaluation.rejection_reasons,
            "shadow_mode":          SHADOW_MODE,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"  \u26a0\ufe0f  trade_evaluator: failed to log decision ({e})")


# ─── Wrapper ───────────────────────────────────────────────────────────────
_evaluator = TradeEvaluator()


def process_trade_signal(trade: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any] | None:
    """
    The single integration point for this layer.

    1. If the layer is disabled, passes `trade` through completely unchanged
       (existing behaviour, zero impact).
    2. Otherwise evaluates the trade, logs the decision (always, pass or
       fail), then:
       - SHADOW_MODE=True  -> always returns None (never allow execution,
         even for trades that pass — shadow mode only observes).
       - SHADOW_MODE=False -> returns `trade` unchanged if it passed, else
         None. The caller's existing execution/alerting logic is untouched;
         this function only decides whether to call it.
    """
    if not ENABLE_TRADE_EVALUATOR:
        return trade

    evaluation = _evaluator.evaluate(trade, market_data)
    log_trade_decision(trade, evaluation)

    if SHADOW_MODE:
        return None

    return trade if evaluation.passed else None
