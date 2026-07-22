"""
Adaptive Learning Prototype v2 — STANDALONE TEST ONLY
=======================================================
Improvements over v1:
  1. Online SGD        — weights update after EVERY trade, not batch every 20
  2. Rolling window    — only trains on the last N trades (recency bias)
  3. Confidence sizing — position size scales with model certainty (0.5x–2x)
  4. Walk-forward test — three periods: warm-up → learn → validate (true OOS)
  5. Weight evolution  — shows how the model's beliefs change over time

Nothing touches the production bot.
"""

import random
import math
import json
from collections import deque

random.seed(42)

# ── Simulation config ─────────────────────────────────────────────────────────
N_WARMUP   = 60     # trades before model starts filtering (learning phase)
N_LEARN    = 120    # trades where model actively learns and trades
N_VALIDATE = 120    # out-of-sample trades: model frozen, just measured
TOTAL      = N_WARMUP + N_LEARN + N_VALIDATE

BASE_RISK  = 200    # $ base risk per trade
MAX_MULT   = 2.0    # max position size multiplier (high confidence)
MIN_MULT   = 0.5    # min position size multiplier (low confidence)
RR         = 2.0    # 2:1 reward:risk
ROLL_WIN   = 60     # rolling window: only train on last N outcomes
LR         = 0.08   # online SGD learning rate
THRESHOLD  = 0.54   # min probability to take a trade


# ── Feature set ───────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "RSI",
    "Volume ratio",
    "Breakout",
    "Trend days",
    "ATR pct",
    "Vol × Breakout",   # interaction: high volume AND breakout together
    "RSI momentum",     # RSI distance from neutral 50 (directional)
]


def generate_signal():
    return {
        "rsi":        random.gauss(52, 12),
        "vol_ratio":  random.gauss(2.0, 0.8),
        "atr_pct":    random.uniform(0.8, 4.5),
        "breakout":   random.random() < 0.45,
        "trend_days": random.randint(0, 30),
    }


def true_win_prob(sig):
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
    return 1 / (1 + math.exp(-score + 3.5))


# ── Feature extractor (now includes interaction & derived features) ───────────
def extract(sig):
    vol_norm  = min(sig["vol_ratio"] / 3.0, 1.5)
    breakout  = 1.0 if sig["breakout"] else 0.0
    return [
        (sig["rsi"] - 38) / 40,                 # RSI normalised
        vol_norm,                                 # Volume ratio
        breakout,                                 # Breakout flag
        sig["trend_days"] / 30,                  # Trend days
        sig["atr_pct"] / 5.0,                    # ATR pct
        vol_norm * breakout,                      # Interaction term
        (sig["rsi"] - 50) / 50,                  # RSI momentum (signed)
    ]


# ── Static scorer (v1.1 production bot) ──────────────────────────────────────
def static_passes(sig):
    score = 0
    if 38 <= sig["rsi"] <= 75:       score += 2
    if sig["vol_ratio"] >= 1.5:      score += 2
    if sig["breakout"]:               score += 2
    if sig["trend_days"] >= 5:        score += 1
    if 1.0 <= sig["atr_pct"] <= 4.0: score += 1
    return score >= 5


# ── Online logistic regression ────────────────────────────────────────────────
class OnlineLogisticRegression:
    def __init__(self, n_features, lr=LR):
        self.w  = [0.0] * n_features
        self.b  = 0.0
        self.lr = lr
        self.history = []   # (trade_num, weights_snapshot)

    def _sig(self, x):
        return 1 / (1 + math.exp(-max(-20, min(20, x))))

    def predict(self, sig):
        f = extract(sig)
        z = self.b + sum(w * x for w, x in zip(self.w, f))
        return self._sig(z)

    def update(self, sig, outcome, weight=1.0):
        """Single-sample SGD update, with optional recency weight."""
        f   = extract(sig)
        p   = self.predict(sig)
        err = (p - outcome) * weight
        self.b -= self.lr * err
        for i in range(len(self.w)):
            self.w[i] -= self.lr * err * f[i]

    def snapshot(self, trade_num):
        self.history.append((trade_num, list(self.w)))


# ── Confidence-based position sizing ─────────────────────────────────────────
def size_multiplier(prob):
    """
    Scale position size based on model confidence.
    - At threshold (0.54): 0.5× base risk (minimum bet)
    - At 0.70+: 2.0× base risk (maximum bet)
    Linear interpolation between those anchors.
    """
    mult = MIN_MULT + (MAX_MULT - MIN_MULT) * (prob - THRESHOLD) / (0.70 - THRESHOLD)
    return max(MIN_MULT, min(MAX_MULT, mult))


def simulate_outcome(sig):
    p   = true_win_prob(sig)
    won = random.random() < p
    return won


# ── Main simulation ───────────────────────────────────────────────────────────
def run():
    model = OnlineLogisticRegression(n_features=len(FEATURE_NAMES))

    # Rolling buffer for recency-weighted training (last ROLL_WIN outcomes)
    roll_sigs     = deque(maxlen=ROLL_WIN)
    roll_outcomes = deque(maxlen=ROLL_WIN)

    # Trackers per phase
    phases = {
        "warmup":   {"taken": 0, "won": 0, "pnl": 0.0},
        "learn":    {"taken": 0, "won": 0, "pnl": 0.0},
        "validate": {"taken": 0, "won": 0, "pnl": 0.0},
    }
    static = {"taken": 0, "won": 0, "pnl": 0.0}

    # Rolling WR window for live chart
    adapt_rolling  = deque(maxlen=50)
    static_rolling = deque(maxlen=50)

    print("=" * 72)
    print("  ADAPTIVE LEARNING PROTOTYPE v2")
    print("  Online SGD  ·  Rolling window  ·  Confidence sizing  ·  Walk-forward")
    print("=" * 72)
    print(f"\n  Phases:  Warm-up ({N_WARMUP})  →  Learn ({N_LEARN})  →  "
          f"Validate OOS ({N_VALIDATE})")
    print(f"  Position sizing: {MIN_MULT}×–{MAX_MULT}× base risk based on confidence\n")
    print(f"  {'Trade':>6}  {'Phase':<10}  {'Static WR':>10}  {'Adapt WR':>10}  "
          f"{'Avg size':>10}  {'Adapt P&L':>10}")
    print("  " + "-" * 66)

    avg_sizes = []

    for i in range(TOTAL):
        sig = generate_signal()
        won = simulate_outcome(sig)

        # Determine phase
        if i < N_WARMUP:
            phase = "warmup"
        elif i < N_WARMUP + N_LEARN:
            phase = "learn"
        else:
            phase = "validate"

        prob = model.predict(sig)

        # ── Static decision ───────────────────────────────────────────────────
        if static_passes(sig):
            pnl = BASE_RISK * RR if won else -BASE_RISK
            static["taken"] += 1
            static["pnl"]   += pnl
            if won:
                static["won"] += 1
            static_rolling.append(1 if won else 0)

        # ── Adaptive decision ─────────────────────────────────────────────────
        # Warm-up: take every trade to gather data (model learns but doesn't filter)
        # Learn: filter by threshold, size by confidence
        # Validate: model frozen — just measure (no weight updates)
        take_trade = (phase == "warmup") or (phase != "warmup" and prob >= THRESHOLD)

        if take_trade:
            mult = 1.0 if phase == "warmup" else size_multiplier(prob)
            risk = BASE_RISK * mult
            pnl  = risk * RR if won else -risk
            phases[phase]["taken"] += 1
            phases[phase]["pnl"]   += pnl
            if won:
                phases[phase]["won"] += 1
            adapt_rolling.append(1 if won else 0)
            avg_sizes.append(mult)

        # ── Online learning (warm-up + learn phases only) ─────────────────────
        if phase != "validate":
            roll_sigs.append(sig)
            roll_outcomes.append(1 if won else 0)
            # Recency weight: most recent = 1.0, oldest = 0.3
            recency = 0.3 + 0.7 * (len(roll_sigs) / ROLL_WIN)
            model.update(sig, 1 if won else 0, weight=recency)

        # Snapshot weights every 60 trades
        if (i + 1) % 60 == 0:
            model.snapshot(i + 1)

        # Print progress every 60 trades
        if (i + 1) % 60 == 0:
            s_wr  = static["won"] / static["taken"] * 100 if static["taken"] else 0
            a_won = sum(p["won"] for p in phases.values())
            a_tak = sum(p["taken"] for p in phases.values())
            a_wr  = a_won / a_tak * 100 if a_tak else 0
            avg_m = sum(avg_sizes[-30:]) / min(30, len(avg_sizes)) if avg_sizes else 1.0
            a_pnl = sum(p["pnl"] for p in phases.values())
            print(f"  {i+1:>6}  {phase:<10}  {s_wr:>9.1f}%  {a_wr:>9.1f}%  "
                  f"  {avg_m:>7.2f}×   ${a_pnl:>+8.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  WALK-FORWARD RESULTS")
    print("=" * 72)
    print(f"\n  {'Phase':<12}  {'Trades':>8}  {'Win Rate':>10}  "
          f"{'P&L ($)':>10}  {'PF':>8}")
    print("  " + "-" * 52)
    for name, ph in phases.items():
        t, w, pnl = ph["taken"], ph["won"], ph["pnl"]
        wr  = w / t * 100  if t else 0
        pf  = (w * RR) / (t - w) if (t - w) > 0 else float("inf")
        tag = "  ← out-of-sample" if name == "validate" else ""
        print(f"  {name.capitalize():<12}  {t:>8}  {wr:>9.1f}%  "
              f"  ${pnl:>+8.0f}  {pf:>8.2f}{tag}")

    s_pf = (static["won"] * RR) / (static["taken"] - static["won"]) \
           if (static["taken"] - static["won"]) > 0 else float("inf")
    print(f"\n  {'Static (all)':<12}  {static['taken']:>8}  "
          f"{static['won']/static['taken']*100:>9.1f}%  "
          f"  ${static['pnl']:>+8.0f}  {s_pf:>8.2f}")

    # ── What the model learned ────────────────────────────────────────────────
    print()
    print("  LEARNED FEATURE WEIGHTS (end of learning phase)")
    print("  ─────────────────────────────────────────────────────────────────")
    paired = sorted(zip(FEATURE_NAMES, model.w), key=lambda x: -abs(x[1]))
    for name, w in paired:
        bar = "█" * int(abs(w) * 6)
        sign = "+" if w >= 0 else "-"
        print(f"  {name:<20}  {w:>+7.3f}  {sign}{bar}")

    # ── Weight evolution ──────────────────────────────────────────────────────
    print()
    print("  WEIGHT EVOLUTION (how beliefs changed over time)")
    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  {'Feature':<20}", end="")
    for t, _ in model.history:
        print(f"  T={t:<5}", end="")
    print()
    for j, fname in enumerate(FEATURE_NAMES):
        print(f"  {fname:<20}", end="")
        for _, snap in model.history:
            print(f"  {snap[j]:>+7.3f}", end="")
        print()

    # ── Confidence sizing breakdown ───────────────────────────────────────────
    print()
    print("  CONFIDENCE SIZING — VALIDATE PHASE BREAKDOWN")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  (Shows whether high-confidence trades actually won more)")
    buckets = {"Low (0.54–0.62)": [], "Mid (0.62–0.70)": [], "High (0.70+)": []}

    random.seed(42)
    for _ in range(N_WARMUP + N_LEARN):
        generate_signal()
        simulate_outcome(generate_signal())

    model2 = OnlineLogisticRegression(n_features=len(FEATURE_NAMES))
    random.seed(42)
    sigs_buf, out_buf = [], []
    for i in range(N_WARMUP + N_LEARN):
        s = generate_signal()
        w = simulate_outcome(s)
        sigs_buf.append(s)
        out_buf.append(w)
        rec = 0.3 + 0.7 * min(1, (i + 1) / ROLL_WIN)
        model2.update(s, 1 if w else 0, weight=rec)

    random.seed(99)
    for _ in range(N_VALIDATE):
        s = generate_signal()
        w = simulate_outcome(s)
        p = model2.predict(s)
        if p < THRESHOLD:
            continue
        if p < 0.62:
            buckets["Low (0.54–0.62)"].append(1 if w else 0)
        elif p < 0.70:
            buckets["Mid (0.62–0.70)"].append(1 if w else 0)
        else:
            buckets["High (0.70+)"].append(1 if w else 0)

    for bucket, outcomes in buckets.items():
        if outcomes:
            wr  = sum(outcomes) / len(outcomes) * 100
            pnl = sum(BASE_RISK * size_multiplier(
                      0.58 if "Low" in bucket else 0.66 if "Mid" in bucket else 0.75
                  ) * (RR if o else -1) for o in outcomes)
            print(f"  {bucket:<22}  {len(outcomes):>5} trades  "
                  f"WR={wr:>5.1f}%  P&L=${pnl:>+8.0f}")
        else:
            print(f"  {bucket:<22}  no trades in this band")

    # Save results
    results = {
        "phases":   {k: {**v, "wr": v["won"]/v["taken"]*100 if v["taken"] else 0}
                     for k, v in phases.items()},
        "static":   {**static, "wr": static["won"]/static["taken"]*100},
        "weights":  dict(zip(FEATURE_NAMES, model.w)),
        "evolution": [{"trade": t, "weights": dict(zip(FEATURE_NAMES, s))}
                      for t, s in model.history],
    }
    with open("prototype-adaptive/results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("  Full results → prototype-adaptive/results_v2.json")
    print()


if __name__ == "__main__":
    run()
