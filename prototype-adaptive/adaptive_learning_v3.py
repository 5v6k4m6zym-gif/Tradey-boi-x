"""
Adaptive Learning Prototype v3 — STANDALONE TEST ONLY
=======================================================
Builds on v2. Adds four overconfidence safeguards:

  1. L2 regularisation   — weight decay on every update; stops weights exploding
  2. Prediction clamp    — predicted probability capped at [0.20, 0.85]
  3. Calibration monitor — rolling actual-WR vs predicted-prob; if gap > 15pp
                           the model is flagged as miscalibrated and sizing reverts
                           to 1× until it recalibrates over the next 20 trades
  4. Consecutive-loss CB — 3 adaptive losses in a row → pause adaptive,
                           fall back to static rules for 10 trades

Runs the same simulation twice (without/with safeguards) so the difference
is directly comparable on identical data.
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

BASE_RISK   = 200
MAX_MULT    = 2.0
MIN_MULT    = 0.5
RR          = 2.0
ROLL_WIN    = 60
LR          = 0.08
L2_DECAY    = 0.002   # weight shrinkage per update (L2 regularisation)
THRESHOLD   = 0.54
PRED_MIN    = 0.20    # never predict below this
PRED_MAX    = 0.85    # never predict above this (prevents extreme sizing)
CALIB_WIN   = 20      # rolling window to compare predicted vs actual WR
CALIB_TOL   = 0.15    # max allowed gap between predicted prob and actual WR
CB_LOSSES   = 3       # consecutive losses before circuit breaker trips
CB_COOLDOWN = 10      # trades to fall back to static after CB trips

FEATURE_NAMES = [
    "RSI", "Volume ratio", "Breakout", "Trend days",
    "ATR pct", "Vol×Breakout", "RSI momentum",
]


# ── Simulation helpers ────────────────────────────────────────────────────────
def generate_signal(rng):
    return {
        "rsi":        rng.gauss(52, 12),
        "vol_ratio":  rng.gauss(2.0, 0.8),
        "atr_pct":    rng.uniform(0.8, 4.5),
        "breakout":   rng.random() < 0.45,
        "trend_days": int(rng.uniform(0, 30)),
    }


def true_win_prob(sig, regime=1.0):
    """regime < 1.0 simulates a bear/choppy market shift."""
    score = 0.0
    rsi_score = 1.0 - abs(sig["rsi"] - 52) / 30
    score += max(0, rsi_score) * 1.8
    score += min(sig["vol_ratio"] / 2.5, 1.0) * 2.5
    if sig["breakout"]:
        score += 1.5
    trend_val = 1.0 - abs(sig["trend_days"] - 13) / 15
    score += max(0, trend_val) * 1.2
    atr_val = 1.0 - abs(sig["atr_pct"] - 2.2) / 2.5
    score += max(0, atr_val) * 0.8
    base = 1 / (1 + math.exp(-score + 3.5))
    return base * regime   # regime shift degrades all signals


def simulate_outcome(sig, rng, regime=1.0):
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


# ── Model ─────────────────────────────────────────────────────────────────────
class Model:
    def __init__(self, use_safeguards=False):
        self.w              = [0.0] * len(FEATURE_NAMES)
        self.b              = 0.0
        self.safeguards     = use_safeguards

        # Calibration monitor
        self.calib_preds    = deque(maxlen=CALIB_WIN)   # predicted probs
        self.calib_outcomes = deque(maxlen=CALIB_WIN)   # actual outcomes
        self.miscalibrated  = False

        # Consecutive-loss circuit breaker
        self.consec_losses  = 0
        self.cb_cooldown    = 0   # trades remaining in static-fallback mode

    def _sig(self, x):
        return 1 / (1 + math.exp(-max(-20, min(20, x))))

    def predict(self, sig):
        f = extract(sig)
        z = self.b + sum(w * x for w, x in zip(self.w, f))
        p = self._sig(z)
        if self.safeguards:
            p = max(PRED_MIN, min(PRED_MAX, p))   # clamp
        return p

    def update(self, sig, outcome, recency=1.0):
        f   = extract(sig)
        p   = self._sig(self.b + sum(w * x for w, x in zip(self.w, f)))
        err = (p - outcome) * recency
        self.b -= LR * err
        for i in range(len(self.w)):
            self.w[i] -= LR * err * f[i]
            if self.safeguards:
                self.w[i] *= (1 - L2_DECAY)   # L2 weight decay

    def record_outcome(self, pred_prob, outcome):
        """Feed into calibration monitor."""
        if not self.safeguards:
            return
        self.calib_preds.append(pred_prob)
        self.calib_outcomes.append(outcome)
        if len(self.calib_outcomes) >= CALIB_WIN:
            actual_wr  = sum(self.calib_outcomes) / len(self.calib_outcomes)
            avg_pred   = sum(self.calib_preds)    / len(self.calib_preds)
            self.miscalibrated = abs(avg_pred - actual_wr) > CALIB_TOL

    def record_loss(self, won):
        """Update consecutive-loss circuit breaker."""
        if not self.safeguards:
            return
        if self.cb_cooldown > 0:
            self.cb_cooldown -= 1
            return
        if won:
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            if self.consec_losses >= CB_LOSSES:
                self.cb_cooldown   = CB_COOLDOWN
                self.consec_losses = 0

    def effective_mult(self, prob):
        """Return sizing multiplier, respecting safeguard overrides."""
        if not self.safeguards:
            return size_mult(prob)
        if self.cb_cooldown > 0 or self.miscalibrated:
            return 1.0   # flat sizing during fallback / recalibration
        return size_mult(prob)

    def should_trade(self, sig, prob):
        if self.cb_cooldown > 0:
            return static_passes(sig)    # fall back to static rules
        return prob >= THRESHOLD


# ── Run one full simulation ───────────────────────────────────────────────────
def run_simulation(use_safeguards, rng_signals, rng_outcomes):
    model = Model(use_safeguards=use_safeguards)

    # Build regime schedule: normal for first 2/3, shifted for last 1/3
    regime_schedule = [1.0] * (TOTAL * 2 // 3) + [0.72] * (TOTAL - TOTAL * 2 // 3)

    taken_total, won_total, pnl_total = 0, 0, 0.0
    cb_events, miscalib_events        = 0, 0
    phase_results                     = {
        "warmup":   {"taken": 0, "won": 0, "pnl": 0.0},
        "learn":    {"taken": 0, "won": 0, "pnl": 0.0},
        "validate": {"taken": 0, "won": 0, "pnl": 0.0},
    }

    for i in range(TOTAL):
        sig    = generate_signal(rng_signals)
        regime = regime_schedule[i]
        won    = simulate_outcome(sig, rng_outcomes, regime)

        phase = "warmup" if i < N_WARMUP else (
                "learn"  if i < N_WARMUP + N_LEARN else "validate")

        prob = model.predict(sig)

        # Warmup: take all trades; learn/validate: use model filter
        if phase == "warmup":
            take = True
        else:
            take = model.should_trade(sig, prob)

        prev_cb    = model.cb_cooldown > 0
        prev_misc  = model.miscalibrated

        if take:
            mult = 1.0 if phase == "warmup" else model.effective_mult(prob)
            risk = BASE_RISK * mult
            pnl  = risk * RR if won else -risk

            phase_results[phase]["taken"] += 1
            phase_results[phase]["pnl"]   += pnl
            if won:
                phase_results[phase]["won"] += 1

            taken_total += 1
            pnl_total   += pnl
            if won:
                won_total += 1

            model.record_outcome(prob, 1 if won else 0)
            model.record_loss(won)

        # Online learning in warmup + learn phases
        if phase != "validate":
            rec = 0.3 + 0.7 * min(1, (i + 1) / ROLL_WIN)
            model.update(sig, 1 if won else 0, recency=rec)

        # Count safeguard activations
        if use_safeguards:
            if not prev_cb    and model.cb_cooldown > 0: cb_events    += 1
            if not prev_misc  and model.miscalibrated:   miscalib_events += 1

    return phase_results, model.w, cb_events, miscalib_events


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print("  ADAPTIVE LEARNING PROTOTYPE v3 — OVERCONFIDENCE SAFEGUARDS")
    print("=" * 72)
    print(f"\n  Safeguards:")
    print(f"  1. L2 regularisation   — decay={L2_DECAY} per update")
    print(f"  2. Prediction clamp    — probabilities capped at [{PRED_MIN}, {PRED_MAX}]")
    print(f"  3. Calibration monitor — revert to 1× if predicted vs actual gap > {CALIB_TOL:.0%}")
    print(f"  4. Circuit breaker     — {CB_LOSSES} consecutive losses → {CB_COOLDOWN} trades on static rules")
    print(f"\n  Regime shift: market becomes 28% harder at trade {TOTAL * 2 // 3}")
    print()

    results = {}
    for label, use_sg in [("Without safeguards", False), ("With safeguards", True)]:
        rng_s = random.Random(SEED)
        rng_o = random.Random(SEED + 1)
        phases, weights, cb_ev, misc_ev = run_simulation(use_sg, rng_s, rng_o)
        results[label] = {
            "phases": phases, "weights": weights,
            "cb_events": cb_ev, "misc_events": misc_ev,
        }

    # ── Phase comparison ──────────────────────────────────────────────────────
    print("  PHASE-BY-PHASE COMPARISON")
    print("  " + "─" * 68)
    print(f"  {'Phase':<12}  {'Metric':<14}  {'No safeguards':>16}  {'With safeguards':>16}")
    print("  " + "─" * 68)

    for phase in ["warmup", "learn", "validate"]:
        tag = "  ← OOS" if phase == "validate" else ""
        for metric in ["Win rate", "P&L"]:
            ns = results["Without safeguards"]["phases"][phase]
            sg = results["With safeguards"]["phases"][phase]

            def fmt(ph, m):
                t, w, p = ph["taken"], ph["won"], ph["pnl"]
                if m == "Win rate":
                    return f"{w/t*100:.1f}%" if t else "—"
                return f"${p:+,.0f}"

            row_phase = phase.capitalize() + tag if metric == "Win rate" else ""
            print(f"  {row_phase:<12}  {metric:<14}  {fmt(ns, metric):>16}  {fmt(sg, metric):>16}")
        print()

    # ── Safeguard events ──────────────────────────────────────────────────────
    sg = results["With safeguards"]
    print("  SAFEGUARD ACTIVATIONS (with safeguards only)")
    print("  " + "─" * 68)
    print(f"  Circuit breaker trips       : {sg['cb_events']:>4}  "
          f"(each = {CB_COOLDOWN} trades reverted to static)")
    print(f"  Calibration flags raised    : {sg['misc_events']:>4}  "
          f"(each = sizing held at 1× until recalibrated)")

    # ── Weight comparison ─────────────────────────────────────────────────────
    print()
    print("  FINAL WEIGHTS — WITH vs WITHOUT SAFEGUARDS")
    print("  " + "─" * 68)
    print(f"  {'Feature':<20}  {'No safeguards':>16}  {'With safeguards':>16}  {'Diff':>8}")
    print("  " + "─" * 68)
    w_no  = results["Without safeguards"]["weights"]
    w_yes = results["With safeguards"]["weights"]
    for i, name in enumerate(FEATURE_NAMES):
        diff = w_yes[i] - w_no[i]
        print(f"  {name:<20}  {w_no[i]:>+16.3f}  {w_yes[i]:>+16.3f}  {diff:>+8.3f}")

    print()
    print("  WHY WEIGHTS DIFFER:")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  L2 decay shrinks large weights toward zero on every update.")
    print("  This prevents any single feature from dominating after a lucky")
    print("  streak — which is the primary cause of overconfidence in live")
    print("  trading. Smaller weights = smaller, more measured size changes.")

    # ── Regime shift impact ───────────────────────────────────────────────────
    print()
    print("  REGIME SHIFT IMPACT (validate phase is 28% harder)")
    print("  ─────────────────────────────────────────────────────────────────")
    for label in ["Without safeguards", "With safeguards"]:
        v  = results[label]["phases"]["validate"]
        wr = v["won"] / v["taken"] * 100 if v["taken"] else 0
        pf = (v["won"] * RR) / (v["taken"] - v["won"]) if v["taken"] != v["won"] else float("inf")
        print(f"  {label:<22}  validate WR={wr:.1f}%  PF={pf:.2f}  P&L=${v['pnl']:>+,.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("  OVERALL SUMMARY")
    print("  " + "─" * 68)
    for label in ["Without safeguards", "With safeguards"]:
        phases = results[label]["phases"]
        total_pnl   = sum(p["pnl"]   for p in phases.values())
        total_won   = sum(p["won"]   for p in phases.values())
        total_taken = sum(p["taken"] for p in phases.values())
        total_pf    = (total_won * RR) / (total_taken - total_won) \
                      if total_taken != total_won else float("inf")
        print(f"  {label:<22}  trades={total_taken}  "
              f"WR={total_won/total_taken*100:.1f}%  "
              f"PF={total_pf:.2f}  P&L=${total_pnl:>+,.0f}")

    print()
    results_out = {
        k: {
            "phases": v["phases"],
            "weights": dict(zip(FEATURE_NAMES, v["weights"])),
            "cb_events": v["cb_events"],
            "misc_events": v["misc_events"],
        }
        for k, v in results.items()
    }
    with open("prototype-adaptive/results_v3.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("  Full results → prototype-adaptive/results_v3.json")
    print()


if __name__ == "__main__":
    run()
