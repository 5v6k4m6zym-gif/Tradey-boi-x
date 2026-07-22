"""
Adaptive Learning Prototype v4 — STANDALONE TEST ONLY
=======================================================
Adds regime awareness to v3.

  Regime detector  — classifies market into BULL / BEAR / SIDEWAYS
                     using a rolling 20-bar window of synthetic market returns
  Per-regime models — three independent logistic regression instances;
                      each learns which features matter in its regime
  Warm transfer    — when regime flips, new model inherits old weights at
                     50% blend so it isn't starting from scratch
  All v3 safeguards retained (L2, clamp, calibration, circuit breaker)

Compares:
  A) Single model (v3, no regime awareness)
  B) Regime-routed model (v4)
"""

import random
import math
import json
from collections import deque

SEED = 42

# ── Config ────────────────────────────────────────────────────────────────────
N_WARMUP   = 60
N_LEARN    = 120
N_VALIDATE = 120
TOTAL      = N_WARMUP + N_LEARN + N_VALIDATE

BASE_RISK    = 200
MAX_MULT     = 2.0
MIN_MULT     = 0.5
RR           = 2.0
ROLL_WIN     = 60
LR           = 0.08
L2_DECAY     = 0.002
THRESHOLD    = 0.54
PRED_MIN     = 0.20
PRED_MAX     = 0.85
CALIB_WIN    = 50      # v3 lesson: wider window avoids noise-triggered flags
CALIB_TOL    = 0.18
CB_LOSSES    = 4       # v3 lesson: slightly higher before CB trips
CB_COOLDOWN  = 10

REGIME_WIN   = 20      # bars to classify regime
BULL_THRESH  =  0.003  # avg daily return above this = BULL
BEAR_THRESH  = -0.003  # avg daily return below this = BEAR
WARM_BLEND   = 0.50    # weight of inherited model on regime transition

FEATURE_NAMES = [
    "RSI", "Volume ratio", "Breakout", "Trend days",
    "ATR pct", "Vol×Breakout", "RSI momentum",
]
REGIMES = ["BULL", "BEAR", "SIDEWAYS"]


# ── Synthetic market ──────────────────────────────────────────────────────────
def build_market(rng, n):
    """
    Generate a synthetic daily return series with three distinct regimes.
    Transitions are abrupt so regime detection has something real to catch.
    Returns list of (daily_return, regime_label) pairs.
    """
    series = []
    segments = [
        (n // 3,       "BULL",    0.006, 0.012),   # trending up
        (n // 4,       "BEAR",   -0.005, 0.015),   # trending down
        (n - n//3 - n//4, "SIDEWAYS", 0.000, 0.010),  # choppy
    ]
    for length, label, mu, sigma in segments:
        for _ in range(length):
            ret = rng.gauss(mu, sigma)
            series.append((ret, label))
    return series


def generate_signal(rng):
    return {
        "rsi":        rng.gauss(52, 12),
        "vol_ratio":  rng.gauss(2.0, 0.8),
        "atr_pct":    rng.uniform(0.8, 4.5),
        "breakout":   rng.random() < 0.45,
        "trend_days": int(rng.uniform(0, 30)),
    }


def true_win_prob(sig, regime):
    """Win probability varies meaningfully by regime."""
    score = 0.0

    # Volume is always important but LESS in bear markets
    vol_contrib = min(sig["vol_ratio"] / 2.5, 1.0) * 2.5
    if regime == "BEAR":
        vol_contrib *= 0.6   # volume breakouts often fail in downtrends

    score += vol_contrib

    # Breakout is powerful in BULL, dangerous in BEAR
    if sig["breakout"]:
        if regime == "BULL":     score += 2.0
        elif regime == "BEAR":   score += 0.3   # mostly false breakouts
        else:                    score += 1.0

    # RSI matters most in SIDEWAYS (mean reversion) and BEAR (oversold bounces)
    rsi_score = 1.0 - abs(sig["rsi"] - 52) / 30
    if regime == "SIDEWAYS":  score += max(0, rsi_score) * 2.2
    elif regime == "BEAR":    score += max(0, rsi_score) * 1.6
    else:                     score += max(0, rsi_score) * 1.2

    # Trend days: valuable in BULL, harmful in BEAR (exhausted)
    trend_val = 1.0 - abs(sig["trend_days"] - 13) / 15
    if regime == "BULL":      score += max(0, trend_val) * 1.5
    elif regime == "BEAR":    score += max(0, trend_val) * 0.3
    else:                     score += max(0, trend_val) * 0.8

    # ATR: mid-range best in all regimes
    atr_val = 1.0 - abs(sig["atr_pct"] - 2.2) / 2.5
    score += max(0, atr_val) * 0.8

    # Base rate is lower in BEAR/SIDEWAYS
    base_bias = {"BULL": 3.2, "BEAR": 4.2, "SIDEWAYS": 3.7}
    return 1 / (1 + math.exp(-score + base_bias[regime]))


def simulate_outcome(sig, rng, regime):
    return rng.random() < true_win_prob(sig, regime)


def extract(sig):
    vol_norm = min(sig["vol_ratio"] / 3.0, 1.5)
    brk      = 1.0 if sig["breakout"] else 0.0
    return [
        (sig["rsi"] - 38) / 40,
        vol_norm,
        brk,
        sig["trend_days"] / 30,
        sig["atr_pct"] / 5.0,
        vol_norm * brk,
        (sig["rsi"] - 50) / 50,
    ]


def static_passes(sig):
    score = 0
    if 38 <= sig["rsi"] <= 75:       score += 2
    if sig["vol_ratio"] >= 1.5:      score += 2
    if sig["breakout"]:               score += 2
    if sig["trend_days"] >= 5:        score += 1
    if 1.0 <= sig["atr_pct"] <= 4.0: score += 1
    return score >= 5


def size_mult(prob):
    m = MIN_MULT + (MAX_MULT - MIN_MULT) * (prob - THRESHOLD) / (0.70 - THRESHOLD)
    return max(MIN_MULT, min(MAX_MULT, m))


# ── Regime detector ───────────────────────────────────────────────────────────
class RegimeDetector:
    def __init__(self, window=REGIME_WIN):
        self.returns = deque(maxlen=window)
        self.current = "SIDEWAYS"
        self.history = []

    def update(self, daily_return, trade_num):
        self.returns.append(daily_return)
        if len(self.returns) >= REGIME_WIN // 2:
            avg = sum(self.returns) / len(self.returns)
            prev = self.current
            if avg > BULL_THRESH:
                self.current = "BULL"
            elif avg < BEAR_THRESH:
                self.current = "BEAR"
            else:
                self.current = "SIDEWAYS"
            if self.current != prev:
                self.history.append((trade_num, prev, self.current))
        return self.current


# ── Logistic regression (single instance) ────────────────────────────────────
class LogReg:
    def __init__(self, n=len(FEATURE_NAMES), lr=LR):
        self.w  = [0.0] * n
        self.b  = 0.0
        self.lr = lr

    def _sig(self, x):
        return 1 / (1 + math.exp(-max(-20, min(20, x))))

    def predict(self, sig):
        f = extract(sig)
        z = self.b + sum(w * x for w, x in zip(self.w, f))
        p = self._sig(z)
        return max(PRED_MIN, min(PRED_MAX, p))

    def update(self, sig, outcome, recency=1.0):
        f   = extract(sig)
        p   = self._sig(self.b + sum(w * x for w, x in zip(self.w, f)))
        err = (p - outcome) * recency
        self.b -= self.lr * err
        for i in range(len(self.w)):
            self.w[i] -= self.lr * err * f[i]
            self.w[i] *= (1 - L2_DECAY)

    def blend_from(self, other, alpha=WARM_BLEND):
        """Inherit `alpha` fraction of another model's weights (warm transfer)."""
        for i in range(len(self.w)):
            self.w[i] = alpha * other.w[i] + (1 - alpha) * self.w[i]
        self.b = alpha * other.b + (1 - alpha) * self.b


# ── Regime-aware model ensemble ───────────────────────────────────────────────
class RegimeModel:
    def __init__(self):
        self.models   = {r: LogReg() for r in REGIMES}
        self.detector = RegimeDetector()
        self.regime   = "SIDEWAYS"

        # Calibration + CB per-regime
        self.calib_preds    = {r: deque(maxlen=CALIB_WIN) for r in REGIMES}
        self.calib_outcomes = {r: deque(maxlen=CALIB_WIN) for r in REGIMES}
        self.miscalibrated  = {r: False for r in REGIMES}
        self.consec_losses  = {r: 0     for r in REGIMES}
        self.cb_cooldown    = {r: 0     for r in REGIMES}
        self.cb_total       = 0
        self.calib_total    = 0

    def step(self, daily_return, trade_num):
        prev   = self.regime
        new    = self.detector.update(daily_return, trade_num)
        if new != prev:
            # Warm-transfer: new regime inherits half the old regime's knowledge
            self.models[new].blend_from(self.models[prev])
        self.regime = new
        return new

    def predict(self, sig):
        return self.models[self.regime].predict(sig)

    def update(self, sig, outcome, recency=1.0):
        self.models[self.regime].update(sig, outcome, recency)

        # Calibration monitor
        p = self.predict(sig)
        self.calib_preds[self.regime].append(p)
        self.calib_outcomes[self.regime].append(outcome)
        if len(self.calib_outcomes[self.regime]) >= CALIB_WIN:
            actual  = sum(self.calib_outcomes[self.regime]) / CALIB_WIN
            avg_p   = sum(self.calib_preds[self.regime]) / CALIB_WIN
            was     = self.miscalibrated[self.regime]
            self.miscalibrated[self.regime] = abs(avg_p - actual) > CALIB_TOL
            if not was and self.miscalibrated[self.regime]:
                self.calib_total += 1

    def record_loss(self, won):
        r = self.regime
        if self.cb_cooldown[r] > 0:
            self.cb_cooldown[r] -= 1
            return
        if won:
            self.consec_losses[r] = 0
        else:
            self.consec_losses[r] += 1
            if self.consec_losses[r] >= CB_LOSSES:
                self.cb_cooldown[r]   = CB_COOLDOWN
                self.consec_losses[r] = 0
                self.cb_total        += 1

    def effective_mult(self, prob):
        r = self.regime
        if self.cb_cooldown[r] > 0 or self.miscalibrated[r]:
            return 1.0
        return size_mult(prob)

    def should_trade(self, sig, prob):
        if self.cb_cooldown[self.regime] > 0:
            return static_passes(sig)
        return prob >= THRESHOLD


# ── Single model (v3 baseline) ────────────────────────────────────────────────
class SingleModel:
    def __init__(self):
        self.m              = LogReg()
        self.calib_preds    = deque(maxlen=CALIB_WIN)
        self.calib_outcomes = deque(maxlen=CALIB_WIN)
        self.miscalibrated  = False
        self.consec_losses  = 0
        self.cb_cooldown    = 0
        self.cb_total       = 0
        self.calib_total    = 0

    def predict(self, sig):
        return self.m.predict(sig)

    def update(self, sig, outcome, recency=1.0):
        self.m.update(sig, outcome, recency)
        p = self.m.predict(sig)
        self.calib_preds.append(p)
        self.calib_outcomes.append(1 if outcome else 0)
        if len(self.calib_outcomes) >= CALIB_WIN:
            actual = sum(self.calib_outcomes) / CALIB_WIN
            avg_p  = sum(self.calib_preds)    / CALIB_WIN
            was    = self.miscalibrated
            self.miscalibrated = abs(avg_p - actual) > CALIB_TOL
            if not was and self.miscalibrated:
                self.calib_total += 1

    def record_loss(self, won):
        if self.cb_cooldown > 0:
            self.cb_cooldown -= 1
            return
        self.consec_losses = 0 if won else self.consec_losses + 1
        if self.consec_losses >= CB_LOSSES:
            self.cb_cooldown   = CB_COOLDOWN
            self.consec_losses = 0
            self.cb_total     += 1

    def effective_mult(self, prob):
        if self.cb_cooldown > 0 or self.miscalibrated:
            return 1.0
        return size_mult(prob)

    def should_trade(self, sig, prob):
        return (static_passes(sig) if self.cb_cooldown > 0
                else prob >= THRESHOLD)


# ── Simulate one run ──────────────────────────────────────────────────────────
def run_sim(model_cls, market, rng_sig, rng_out):
    model = model_cls()
    phases = {ph: {"taken": 0, "won": 0, "pnl": 0.0,
                   "by_regime": {r: {"taken":0,"won":0,"pnl":0.0}
                                 for r in REGIMES}}
              for ph in ["warmup", "learn", "validate"]}

    roll_sigs, roll_out = deque(maxlen=ROLL_WIN), deque(maxlen=ROLL_WIN)
    regime_seq = []

    for i in range(TOTAL):
        mkt_ret, true_regime = market[i]
        sig  = generate_signal(rng_sig)
        won  = simulate_outcome(sig, rng_out, true_regime)

        phase = ("warmup"   if i < N_WARMUP else
                 "learn"    if i < N_WARMUP + N_LEARN else
                 "validate")

        # Advance regime detector (regime-model uses this; single model ignores it)
        if isinstance(model, RegimeModel):
            detected = model.step(mkt_ret, i)
        else:
            detected = true_regime   # single model doesn't detect — use true label
        regime_seq.append(detected)

        prob = model.predict(sig)

        take = (phase == "warmup") or model.should_trade(sig, prob)

        if take:
            mult = 1.0 if phase == "warmup" else model.effective_mult(prob)
            risk = BASE_RISK * mult
            pnl  = risk * RR if won else -risk
            ph   = phases[phase]
            ph["taken"] += 1
            ph["pnl"]   += pnl
            if won: ph["won"] += 1
            ph["by_regime"][detected]["taken"] += 1
            ph["by_regime"][detected]["pnl"]   += pnl
            if won: ph["by_regime"][detected]["won"] += 1

            if isinstance(model, RegimeModel):
                model.record_loss(won)
            else:
                model.record_loss(won)

        if phase != "validate":
            roll_sigs.append(sig)
            roll_out.append(1 if won else 0)
            rec = 0.3 + 0.7 * min(1, (i + 1) / ROLL_WIN)
            model.update(sig, 1 if won else 0, recency=rec)

    # Extract final weights
    if isinstance(model, RegimeModel):
        weights = {r: dict(zip(FEATURE_NAMES, model.models[r].w)) for r in REGIMES}
        transitions = model.detector.history
        cb_total    = model.cb_total
        calib_total = model.calib_total
    else:
        weights     = {"ALL": dict(zip(FEATURE_NAMES, model.m.w))}
        transitions = []
        cb_total    = model.cb_total
        calib_total = model.calib_total

    return phases, weights, transitions, regime_seq, cb_total, calib_total


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    rng_mkt = random.Random(SEED)
    market  = build_market(rng_mkt, TOTAL)

    print("=" * 72)
    print("  ADAPTIVE LEARNING PROTOTYPE v4 — REGIME AWARENESS")
    print("=" * 72)

    # Show regime composition
    regime_counts = {}
    for _, r in market:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"\n  Market composition across {TOTAL} trades:")
    for r, c in regime_counts.items():
        bar = "█" * (c // 5)
        print(f"    {r:<10}  {c:>4} trades  {bar}")

    results = {}
    for label, cls in [("Single model (v3)", SingleModel),
                        ("Regime-aware (v4)", RegimeModel)]:
        rng_s = random.Random(SEED + 10)
        rng_o = random.Random(SEED + 20)
        phases, weights, transitions, regime_seq, cb, calib = \
            run_sim(cls, market, rng_s, rng_o)
        results[label] = dict(phases=phases, weights=weights,
                              transitions=transitions, regime_seq=regime_seq,
                              cb=cb, calib=calib)

    # ── Phase comparison ──────────────────────────────────────────────────────
    print()
    print("  PHASE COMPARISON")
    print("  " + "─" * 68)
    print(f"  {'Phase':<12}  {'Metric':<12}  {'Single model':>16}  {'Regime-aware':>16}")
    print("  " + "─" * 68)
    for phase in ["warmup", "learn", "validate"]:
        tag = "  ← OOS" if phase == "validate" else ""
        for metric in ["Win rate", "PF", "P&L"]:
            nm = results["Single model (v3)"]["phases"][phase]
            ra = results["Regime-aware (v4)"]["phases"][phase]
            def fmt(ph, m):
                t, w, p = ph["taken"], ph["won"], ph["pnl"]
                if not t: return "—"
                if m == "Win rate": return f"{w/t*100:.1f}%"
                if m == "PF":
                    l = t - w
                    return f"{(w*RR/l):.2f}" if l > 0 else "∞"
                return f"${p:>+,.0f}"
            row_label = phase.capitalize() + tag if metric == "Win rate" else ""
            print(f"  {row_label:<12}  {metric:<12}  {fmt(nm,metric):>16}  {fmt(ra,metric):>16}")
        print()

    # ── Per-regime breakdown (validate phase) ─────────────────────────────────
    print("  VALIDATE PHASE — BREAKDOWN BY DETECTED REGIME")
    print("  " + "─" * 68)
    print(f"  {'Regime':<12}  {'Model':<20}  {'Trades':>8}  {'Win rate':>10}  {'P&L':>10}")
    print("  " + "─" * 68)
    for label in ["Single model (v3)", "Regime-aware (v4)"]:
        for regime in REGIMES:
            br = results[label]["phases"]["validate"]["by_regime"][regime]
            t, w, p = br["taken"], br["won"], br["pnl"]
            if t == 0:
                print(f"  {regime:<12}  {label:<20}  {'—':>8}")
                continue
            wr = w / t * 100
            print(f"  {regime:<12}  {label:<20}  {t:>8}  {wr:>9.1f}%  ${p:>+9,.0f}")
        print()

    # ── Regime transitions detected ───────────────────────────────────────────
    print("  REGIME TRANSITIONS DETECTED (v4 only)")
    print("  " + "─" * 68)
    transitions = results["Regime-aware (v4)"]["transitions"]
    if transitions:
        for trade_num, prev, new in transitions:
            phase = ("warmup"   if trade_num < N_WARMUP else
                     "learn"    if trade_num < N_WARMUP + N_LEARN else "validate")
            print(f"    Trade {trade_num:>3}  [{phase}]  {prev} → {new}  "
                  f"(weights warm-transferred at {WARM_BLEND:.0%} blend)")
    else:
        print("    No transitions detected in this run")

    # ── Per-regime weights ────────────────────────────────────────────────────
    print()
    print("  LEARNED WEIGHTS PER REGIME (v4) vs SINGLE MODEL (v3)")
    print("  " + "─" * 68)
    single_w = results["Single model (v3)"]["weights"]["ALL"]
    regime_w = results["Regime-aware (v4)"]["weights"]
    print(f"  {'Feature':<20}  {'Single':>8}", end="")
    for r in REGIMES:
        print(f"  {r:>10}", end="")
    print()
    print("  " + "─" * 68)
    for fname in FEATURE_NAMES:
        print(f"  {fname:<20}  {single_w[fname]:>+8.3f}", end="")
        for r in REGIMES:
            diff = regime_w[r][fname] - single_w[fname]
            marker = " ↑" if diff > 0.05 else (" ↓" if diff < -0.05 else "  ")
            print(f"  {regime_w[r][fname]:>+8.3f}{marker}", end="")
        print()

    print()
    print("  ↑/↓ = regime weight diverged >0.05 from single model")
    print()
    print("  WHAT DIVERGENCES MEAN:")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  BULL model learning higher breakout weight → breakouts more reliable")
    print("  BEAR model learning lower breakout/trend weight → fade breakouts in downtrends")
    print("  SIDEWAYS model learning higher RSI weight → mean reversion rules in ranges")

    # ── Safeguard events ──────────────────────────────────────────────────────
    print()
    print("  SAFEGUARD ACTIVATIONS")
    print("  " + "─" * 68)
    for label in ["Single model (v3)", "Regime-aware (v4)"]:
        cb    = results[label]["cb"]
        calib = results[label]["calib"]
        print(f"  {label:<24}  CB trips={cb:>3}  Calib flags={calib:>3}")

    # ── Overall summary ───────────────────────────────────────────────────────
    print()
    print("  OVERALL SUMMARY")
    print("  " + "─" * 68)
    for label in ["Single model (v3)", "Regime-aware (v4)"]:
        phases      = results[label]["phases"]
        total_pnl   = sum(p["pnl"]   for p in phases.values())
        total_won   = sum(p["won"]   for p in phases.values())
        total_taken = sum(p["taken"] for p in phases.values())
        total_l     = total_taken - total_won
        pf          = (total_won * RR) / total_l if total_l > 0 else float("inf")
        print(f"  {label:<24}  trades={total_taken}  "
              f"WR={total_won/total_taken*100:.1f}%  "
              f"PF={pf:.2f}  P&L=${total_pnl:>+,.0f}")

    with open("prototype-adaptive/results_v4.json", "w") as f:
        json.dump({k: {
            "phases": v["phases"],
            "weights": v["weights"],
            "transitions": [(t, p, n) for t, p, n in v["transitions"]],
            "cb": v["cb"], "calib": v["calib"],
        } for k, v in results.items()}, f, indent=2, default=str)

    print()
    print("  Full results → prototype-adaptive/results_v4.json")
    print()


if __name__ == "__main__":
    run()
