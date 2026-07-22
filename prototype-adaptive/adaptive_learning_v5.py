"""
Adaptive Learning Prototype v5 — STANDALONE TEST ONLY
=======================================================
Fixes v4's noisy regime detector, then runs a clean 3-way comparison:

  A) STATIC       — current production bot (fixed rules, no learning)
  B) ADAPTIVE     — single logistic regression, all v3 safeguards
  C) REGIME-AWARE — per-regime models, fixed detector, all safeguards

Detector fixes from v4:
  - Window: 20 → 40 bars  (more data before classifying)
  - Hysteresis: must hold new classification for 10 bars before switching
    (prevents flip-flopping on a single noisy week)

All runs use identical signals and outcomes (same RNG seeds).
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
CALIB_WIN    = 50
CALIB_TOL    = 0.18
CB_LOSSES    = 4
CB_COOLDOWN  = 10

# Fixed detector params
REGIME_WIN     = 40    # v5 fix: was 20 — wider window
REGIME_CONFIRM = 10    # v5 fix: new — hysteresis bars before switching
BULL_THRESH    =  0.003
BEAR_THRESH    = -0.003
WARM_BLEND     = 0.50

FEATURE_NAMES = ["RSI", "Volume ratio", "Breakout",
                 "Trend days", "ATR pct", "Vol×Breakout", "RSI momentum"]
REGIMES = ["BULL", "BEAR", "SIDEWAYS"]


# ── Market & signal generation ────────────────────────────────────────────────
def build_market(rng, n):
    series, segments = [], [
        (n // 3,           "BULL",     0.006, 0.012),
        (n // 4,           "BEAR",    -0.005, 0.015),
        (n - n//3 - n//4,  "SIDEWAYS", 0.000, 0.010),
    ]
    for length, label, mu, sigma in segments:
        for _ in range(length):
            series.append((rng.gauss(mu, sigma), label))
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
    score = 0.0
    vol   = min(sig["vol_ratio"] / 2.5, 1.0) * 2.5
    score += vol * (0.6 if regime == "BEAR" else 1.0)
    if sig["breakout"]:
        score += {"BULL": 2.0, "BEAR": 0.3, "SIDEWAYS": 1.0}[regime]
    rsi_s = max(0, 1.0 - abs(sig["rsi"] - 52) / 30)
    score += rsi_s * {"BULL": 1.2, "BEAR": 1.6, "SIDEWAYS": 2.2}[regime]
    trend = max(0, 1.0 - abs(sig["trend_days"] - 13) / 15)
    score += trend * {"BULL": 1.5, "BEAR": 0.3, "SIDEWAYS": 0.8}[regime]
    score += max(0, 1.0 - abs(sig["atr_pct"] - 2.2) / 2.5) * 0.8
    bias  = {"BULL": 3.2, "BEAR": 4.2, "SIDEWAYS": 3.7}[regime]
    return 1 / (1 + math.exp(-score + bias))


def simulate_outcome(sig, rng, regime):
    return rng.random() < true_win_prob(sig, regime)


def extract(sig):
    v = min(sig["vol_ratio"] / 3.0, 1.5)
    b = 1.0 if sig["breakout"] else 0.0
    return [(sig["rsi"] - 38) / 40, v, b, sig["trend_days"] / 30,
            sig["atr_pct"] / 5.0, v * b, (sig["rsi"] - 50) / 50]


# ── Static scorer (production bot) ───────────────────────────────────────────
def static_passes(sig):
    s = 0
    if 38 <= sig["rsi"] <= 75:       s += 2
    if sig["vol_ratio"] >= 1.5:      s += 2
    if sig["breakout"]:               s += 2
    if sig["trend_days"] >= 5:        s += 1
    if 1.0 <= sig["atr_pct"] <= 4.0: s += 1
    return s >= 5


def size_mult(prob):
    return max(MIN_MULT, min(MAX_MULT,
        MIN_MULT + (MAX_MULT - MIN_MULT) * (prob - THRESHOLD) / (0.70 - THRESHOLD)))


# ── Logistic regression ───────────────────────────────────────────────────────
class LogReg:
    def __init__(self):
        self.w = [0.0] * len(FEATURE_NAMES)
        self.b = 0.0

    def _s(self, x): return 1 / (1 + math.exp(-max(-20, min(20, x))))

    def predict(self, sig):
        z = self.b + sum(w * x for w, x in zip(self.w, extract(sig)))
        return max(PRED_MIN, min(PRED_MAX, self._s(z)))

    def update(self, sig, y, rec=1.0):
        f   = extract(sig)
        err = (self._s(self.b + sum(w*x for w,x in zip(self.w,f))) - y) * rec
        self.b -= LR * err
        for i in range(len(self.w)):
            self.w[i] -= LR * err * f[i]
            self.w[i] *= (1 - L2_DECAY)

    def blend(self, other, alpha=WARM_BLEND):
        for i in range(len(self.w)):
            self.w[i] = alpha * other.w[i] + (1 - alpha) * self.w[i]
        self.b = alpha * other.b + (1 - alpha) * self.b


# ── Regime detector (fixed) ───────────────────────────────────────────────────
class RegimeDetector:
    def __init__(self):
        self.buf          = deque(maxlen=REGIME_WIN)
        self.current      = "SIDEWAYS"
        self.candidate    = "SIDEWAYS"
        self.confirm_cnt  = 0
        self.transitions  = []

    def update(self, ret, i):
        self.buf.append(ret)
        if len(self.buf) < REGIME_WIN // 2:
            return self.current
        avg = sum(self.buf) / len(self.buf)
        raw = ("BULL" if avg > BULL_THRESH else
               "BEAR" if avg < BEAR_THRESH else "SIDEWAYS")
        # Hysteresis: only switch after REGIME_CONFIRM consecutive bars in new regime
        if raw == self.candidate:
            self.confirm_cnt += 1
        else:
            self.candidate   = raw
            self.confirm_cnt = 1
        if self.confirm_cnt >= REGIME_CONFIRM and raw != self.current:
            self.transitions.append((i, self.current, raw))
            self.current     = raw
            self.confirm_cnt = 0
        return self.current


# ── Safeguard mixin ───────────────────────────────────────────────────────────
class Safeguards:
    def __init__(self):
        self.cp = deque(maxlen=CALIB_WIN)   # predicted probs
        self.co = deque(maxlen=CALIB_WIN)   # outcomes
        self.miscal    = False
        self.consec    = 0
        self.cooldown  = 0
        self.cb_total  = 0
        self.cal_total = 0

    def observe(self, prob, won):
        self.cp.append(prob); self.co.append(int(won))
        if len(self.co) >= CALIB_WIN:
            was = self.miscal
            self.miscal = abs(sum(self.cp)/len(self.cp) -
                              sum(self.co)/len(self.co)) > CALIB_TOL
            if not was and self.miscal: self.cal_total += 1

    def loss_event(self, won):
        if self.cooldown > 0:
            self.cooldown -= 1; return
        self.consec = 0 if won else self.consec + 1
        if self.consec >= CB_LOSSES:
            self.cooldown = CB_COOLDOWN; self.consec = 0; self.cb_total += 1

    def mult(self, prob):
        return 1.0 if (self.cooldown > 0 or self.miscal) else size_mult(prob)

    def gate(self, sig, prob):
        return static_passes(sig) if self.cooldown > 0 else prob >= THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Simulation engines
# ─────────────────────────────────────────────────────────────────────────────

def run_static(market, rng_s, rng_o):
    """Pure static scorer — no learning."""
    phases = _blank_phases()
    for i, (_, regime) in enumerate(market):
        sig = generate_signal(rng_s)
        won = simulate_outcome(sig, rng_o, regime)
        ph  = _phase(i)
        if static_passes(sig):
            _record(phases[ph], won, BASE_RISK, regime)
    return phases, {}, [], 0, 0


def run_adaptive(market, rng_s, rng_o):
    """Single logistic regression + safeguards (v3 baseline)."""
    model = LogReg()
    sg    = Safeguards()
    phases = _blank_phases()
    roll_s, roll_o = deque(maxlen=ROLL_WIN), deque(maxlen=ROLL_WIN)

    for i, (_, regime) in enumerate(market):
        sig = generate_signal(rng_s)
        won = simulate_outcome(sig, rng_o, regime)
        ph  = _phase(i)
        prob = model.predict(sig)

        take = (ph == "warmup") or sg.gate(sig, prob)
        if take:
            mult = 1.0 if ph == "warmup" else sg.mult(prob)
            _record(phases[ph], won, BASE_RISK * mult, regime)
            sg.observe(prob, won); sg.loss_event(won)

        if ph != "validate":
            roll_s.append(sig); roll_o.append(int(won))
            model.update(sig, int(won), 0.3 + 0.7 * min(1, (i+1)/ROLL_WIN))

    weights = {"ALL": dict(zip(FEATURE_NAMES, model.w))}
    return phases, weights, [], sg.cb_total, sg.cal_total


def run_regime(market, rng_s, rng_o):
    """Per-regime models + fixed detector + safeguards (v5)."""
    models   = {r: LogReg()      for r in REGIMES}
    sgs      = {r: Safeguards()  for r in REGIMES}
    detector = RegimeDetector()
    phases   = _blank_phases()
    roll_s, roll_o = deque(maxlen=ROLL_WIN), deque(maxlen=ROLL_WIN)
    cb_total = cal_total = 0

    for i, (ret, regime) in enumerate(market):
        sig      = generate_signal(rng_s)
        won      = simulate_outcome(sig, rng_o, regime)
        ph       = _phase(i)
        prev     = detector.current
        detected = detector.update(ret, i)

        # Warm-transfer on regime flip
        if detected != prev:
            models[detected].blend(models[prev])

        m   = models[detected]
        sg  = sgs[detected]
        prob = m.predict(sig)

        take = (ph == "warmup") or sg.gate(sig, prob)
        if take:
            mult = 1.0 if ph == "warmup" else sg.mult(prob)
            _record(phases[ph], won, BASE_RISK * mult, detected)
            sg.observe(prob, won); sg.loss_event(won)

        if ph != "validate":
            roll_s.append(sig); roll_o.append(int(won))
            m.update(sig, int(won), 0.3 + 0.7 * min(1, (i+1)/ROLL_WIN))

    for sg in sgs.values():
        cb_total  += sg.cb_total
        cal_total += sg.cal_total
    weights = {r: dict(zip(FEATURE_NAMES, models[r].w)) for r in REGIMES}
    return phases, weights, detector.transitions, cb_total, cal_total


# ── Helpers ───────────────────────────────────────────────────────────────────
def _blank_phases():
    return {ph: {"taken":0,"won":0,"pnl":0.0,
                 "by_regime":{r:{"taken":0,"won":0,"pnl":0.0} for r in REGIMES}}
            for ph in ["warmup","learn","validate"]}


def _phase(i):
    return ("warmup"   if i < N_WARMUP else
            "learn"    if i < N_WARMUP + N_LEARN else "validate")


def _record(ph, won, risk, regime):
    pnl = risk * RR if won else -risk
    ph["taken"] += 1; ph["pnl"] += pnl
    if won: ph["won"] += 1
    r = ph["by_regime"][regime]
    r["taken"] += 1; r["pnl"] += pnl
    if won: r["won"] += 1


def _pf(won, taken):
    l = taken - won
    return round((won * RR) / l, 2) if l > 0 else float("inf")


def _wr(won, taken):
    return won / taken * 100 if taken else 0.0


# ── Report ────────────────────────────────────────────────────────────────────
def run():
    rng_mkt = random.Random(SEED)
    market  = build_market(rng_mkt, TOTAL)

    # Count true regime composition
    regime_counts = {}
    for _, r in market:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    print("=" * 74)
    print("  ADAPTIVE LEARNING v5 — 3-WAY COMPARISON")
    print("  Static  vs  Single Adaptive  vs  Regime-Aware Adaptive")
    print("=" * 74)
    print(f"\n  True market composition ({TOTAL} trades):")
    for r in REGIMES:
        c   = regime_counts.get(r, 0)
        bar = "█" * (c // 4)
        print(f"    {r:<10}  {c:>4} trades  {c/TOTAL*100:.0f}%  {bar}")

    runners = [
        ("Static (no learning)",   run_static),
        ("Single adaptive (v3)",   run_adaptive),
        ("Regime-aware (v5)",      run_regime),
    ]
    results = {}
    for label, fn in runners:
        rng_s = random.Random(SEED + 10)
        rng_o = random.Random(SEED + 20)
        phases, weights, transitions, cb, calib = fn(market, rng_s, rng_o)
        results[label] = dict(phases=phases, weights=weights,
                              transitions=transitions, cb=cb, calib=calib)

    # ── Regime detector quality ───────────────────────────────────────────────
    ra = results["Regime-aware (v5)"]
    print(f"\n  REGIME DETECTOR (v5 fixed — window={REGIME_WIN}, hysteresis={REGIME_CONFIRM})")
    print("  " + "─" * 70)
    print(f"  Transitions detected: {len(ra['transitions'])}  "
          f"(v4 had 24 — fixed by wider window + hysteresis)")
    for t, prev, new in ra["transitions"]:
        ph = _phase(t)
        print(f"    Trade {t:>3}  [{ph:<8}]  {prev} → {new}")

    # ── Phase-by-phase table ──────────────────────────────────────────────────
    labels = list(results.keys())
    print(f"\n  PHASE-BY-PHASE RESULTS")
    print("  " + "─" * 74)
    header = f"  {'Phase':<12}  {'Metric':<10}"
    for lb in labels:
        header += f"  {lb[:20]:>20}"
    print(header)
    print("  " + "─" * 74)

    for phase in ["warmup", "learn", "validate"]:
        oos = "  ← out-of-sample" if phase == "validate" else ""
        for metric in ["Trades", "Win rate", "PF", "P&L ($)"]:
            row = f"  {(phase.capitalize()+oos if metric=='Trades' else ''):<12}  {metric:<10}"
            for lb in labels:
                ph = results[lb]["phases"][phase]
                t, w, p = ph["taken"], ph["won"], ph["pnl"]
                if metric == "Trades":   cell = f"{t}"
                elif metric == "Win rate": cell = f"{_wr(w,t):.1f}%" if t else "—"
                elif metric == "PF":     cell = f"{_pf(w,t):.2f}" if t else "—"
                else:                    cell = f"${p:>+,.0f}"     if t else "—"
                row += f"  {cell:>20}"
            print(row)
        print()

    # ── Per-regime validate breakdown ─────────────────────────────────────────
    print("  VALIDATE PHASE — BY REGIME")
    print("  " + "─" * 74)
    print(f"  {'Regime':<10}  {'Model':<24}  {'Trades':>7}  {'Win rate':>10}  "
          f"{'PF':>6}  {'P&L':>10}")
    print("  " + "─" * 74)
    for regime in REGIMES:
        for lb in labels:
            br = results[lb]["phases"]["validate"]["by_regime"][regime]
            t, w, p = br["taken"], br["won"], br["pnl"]
            if not t:
                print(f"  {regime:<10}  {lb:<24}  {'—':>7}")
            else:
                print(f"  {regime:<10}  {lb:<24}  {t:>7}  "
                      f"{_wr(w,t):>9.1f}%  {_pf(w,t):>6.2f}  ${p:>+9,.0f}")
        print()

    # ── Feature weights comparison ────────────────────────────────────────────
    print("  LEARNED WEIGHTS")
    print("  " + "─" * 74)
    single_w = results["Single adaptive (v3)"]["weights"].get("ALL", {})
    regime_w = results["Regime-aware (v5)"]["weights"]
    print(f"  {'Feature':<20}  {'Single':>8}", end="")
    for r in REGIMES:
        print(f"  {r:>10}", end="")
    print()
    print("  " + "─" * 74)
    for fname in FEATURE_NAMES:
        sv = single_w.get(fname, 0)
        print(f"  {fname:<20}  {sv:>+8.3f}", end="")
        for r in REGIMES:
            rv   = regime_w.get(r, {}).get(fname, 0)
            diff = rv - sv
            mk   = " ↑" if diff > 0.05 else (" ↓" if diff < -0.05 else "  ")
            print(f"  {rv:>+8.3f}{mk}", end="")
        print()

    print()
    print("  ↑/↓ = regime model weight diverged > 0.05 from single model")

    # ── Safeguards ────────────────────────────────────────────────────────────
    print()
    print("  SAFEGUARD ACTIVATIONS")
    print("  " + "─" * 74)
    print(f"  {'Model':<28}  {'CB trips':>10}  {'Calib flags':>12}")
    print("  " + "─" * 74)
    for lb in labels[1:]:   # static has none
        print(f"  {lb:<28}  {results[lb]['cb']:>10}  {results[lb]['calib']:>12}")

    # ── Overall summary ───────────────────────────────────────────────────────
    print()
    print("  OVERALL SUMMARY")
    print("  " + "─" * 74)
    print(f"  {'Model':<28}  {'Trades':>8}  {'Win rate':>10}  {'PF':>6}  {'P&L':>12}")
    print("  " + "─" * 74)
    for lb in labels:
        phases      = results[lb]["phases"]
        tot_pnl     = sum(p["pnl"]   for p in phases.values())
        tot_won     = sum(p["won"]   for p in phases.values())
        tot_taken   = sum(p["taken"] for p in phases.values())
        print(f"  {lb:<28}  {tot_taken:>8}  "
              f"{_wr(tot_won, tot_taken):>9.1f}%  "
              f"{_pf(tot_won, tot_taken):>6.2f}  ${tot_pnl:>+10,.0f}")

    print()
    print("  KEY FINDING:")
    pnl_static = sum(p["pnl"] for p in results["Static (no learning)"]["phases"].values())
    pnl_single = sum(p["pnl"] for p in results["Single adaptive (v3)"]["phases"].values())
    pnl_regime = sum(p["pnl"] for p in results["Regime-aware (v5)"]["phases"].values())
    best = max([("Static", pnl_static), ("Single adaptive", pnl_single),
                ("Regime-aware", pnl_regime)], key=lambda x: x[1])
    print(f"  {best[0]} produced the highest P&L: ${best[1]:>+,.0f}")
    gain_ra_vs_s  = pnl_regime - pnl_static
    gain_ra_vs_si = pnl_regime - pnl_single
    print(f"  Regime-aware vs static:          {gain_ra_vs_s:>+,.0f}")
    print(f"  Regime-aware vs single adaptive: {gain_ra_vs_si:>+,.0f}")
    print()

    with open("prototype-adaptive/results_v5.json", "w") as f:
        json.dump({k: {"phases": v["phases"], "weights": v["weights"],
                       "cb": v["cb"], "calib": v["calib"],
                       "transitions": [(a,b,c) for a,b,c in v["transitions"]]}
                   for k, v in results.items()}, f, indent=2, default=str)
    print("  Full results → prototype-adaptive/results_v5.json")
    print()


if __name__ == "__main__":
    run()
