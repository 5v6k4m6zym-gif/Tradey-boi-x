"""
Standalone backtest runner — no Streamlit required.
Run: python3 _run_test.py

Uses the same universe as the live bot (build_universe) so results reflect
what the dashboard Bot Simulation would show.
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)

import db.database as db; db.init_db()

from datetime import date
from backtest.engine import run_backtest, parameter_sweep
from scanner.universe import build_universe

tickers = build_universe(markets=["ASX", "US"], apply_liquidity=False)

TEST_START = date(2024, 1, 1)
TEST_END   = date(2025, 12, 31)

base_params = {
    "min_score":             5,
    "min_prob":              0.50,
    "risk_pct":              2.0,
    "max_positions":         5,
    "hold_days":             15,
    "min_hold_days":         2,
    "brokerage":             2.0,
    "sl_mult_hi":            0.8,
    "sl_mult_mid":           0.6,
    "sl_mult_lo":            0.5,
    "target_hi":             15.0,
    "target_mid":            10.0,
    "target_lo":             7.0,
    "be_trigger_r":          0.5,
    "trail_trigger_r":       1.5,
    "trail_dist_r":          0.7,
    "cb_consecutive_losses": 3,
    "cb_pause_days":         7,
    "use_regime_filter":     True,
    "min_expected_r":        1.5,
}

print("=" * 65)
print("TRADEY BOI PRO — BOT SIMULATION BACKTEST")
print(f"Tickers : {len(tickers)} (live bot universe — ASX + US quality stocks)")
print(f"Period  : {TEST_START} → {TEST_END}")
print(f"Params  : score≥{base_params['min_score']}  prob≥{base_params['min_prob']}")
print(f"          sl={base_params['sl_mult_hi']}/{base_params['sl_mult_mid']}/{base_params['sl_mult_lo']}×ATR  hold={base_params['hold_days']}d")
print(f"          BE={base_params['be_trigger_r']}R  Trail={base_params['trail_trigger_r']}R/{base_params['trail_dist_r']}R")
print("=" * 65)

_pl = [""]
def progress(done, total, msg):
    line = f"\r[{int(done/max(total,1)*100):3d}%] {msg}"
    print(line + " " * max(0, len(_pl[0])-len(line)), end="", flush=True)
    _pl[0] = line

result = run_backtest(
    tickers=tickers,
    test_start=TEST_START,
    test_end=TEST_END,
    initial_capital=10_000.0,
    params=base_params,
    progress_cb=progress,
)
print()

def _report(result, label="RESULTS"):
    m  = result["metrics"]
    er = m.get("exit_reasons", {})
    tc = m["trade_count"]
    pf = m["profit_factor"]
    roi = m["roi_pct"]

    pf_flag  = "✅ PASS"           if pf >= 1.4 else ("⚠  MARGINAL" if pf >= 1.0 else "❌ FAIL")
    roi_flag = "✅"                 if 20 <= roi <= 40 else ("📈 above target" if roi > 40 else "")

    print(f"\n{'─'*65}")
    print(f" {label}")
    print(f"{'─'*65}")
    print(f" Trades        : {tc}")
    print(f" Win Rate      : {m['win_rate']*100:.1f}%")
    print(f" Profit Factor : {pf:.3f}  {pf_flag}")
    print(f" ROI           : {roi:+.2f}%  {roi_flag}")
    print(f" Avg Win       : ${m['avg_win']:,.2f}")
    print(f" Avg Loss      : ${m['avg_loss']:,.2f}")
    if m['avg_loss']:
        print(f" Win:Loss R    : {m['avg_win']/m['avg_loss']:.2f}×")
    print(f" Avg Hold      : {m['avg_hold_days']:.1f}d")
    print(f" Max Drawdown  : {m['max_drawdown']*100:.1f}%")
    print(f" Sharpe        : {m['sharpe']:.3f}")

    if tc == 0:
        print(" ⚠  Zero trades — check filters")
        return

    print()
    print(" EXIT BREAKDOWN:")
    for reason, count in sorted(er.items(), key=lambda x: -x[1]):
        subset  = [t for t in result["trades"] if t.exit_reason == reason]
        avg_pnl = sum(t.pnl for t in subset) / len(subset) if subset else 0
        wins    = sum(1 for t in subset if t.pnl >= 0)
        print(f"   {reason:<22} {count:3d} ({count/tc*100:4.0f}%)  "
              f"win={wins/count*100:.0f}%  avg=${avg_pnl:+,.0f}")

_report(result, "BASELINE RESULTS")

m  = result["metrics"]
tc = m["trade_count"]
pf = m["profit_factor"]

if tc == 0:
    print("\nNo trades — need to loosen filters.")
    sys.exit(1)

_raw_sigs = result.get("_precomputed_signals") or {}
if _raw_sigs:
    _tiers: dict = {}
    for v in _raw_sigs.values():
        t = v.get("tier", "?")
        _tiers[t] = _tiers.get(t, 0) + 1
    print(f"\n Prescan generated {len(_raw_sigs)} total signal-days:")
    for _t, _n in sorted(_tiers.items(), key=lambda x: -x[1]):
        print(f"   {_t}: {_n}")

pre_data = result.pop("_preloaded_data", None)
pre_reg  = result.pop("_preloaded_regimes", None)
pre_sig  = result.pop("_precomputed_signals", None)

print(f"\n Running 27-combo BE/trail sweep (reuses cached data)…")

sweep_combos = [
    {**base_params, "be_trigger_r": be, "trail_trigger_r": tt, "trail_dist_r": td}
    for be in [0.5, 1.0, 1.5]
    for tt in [1.5, 2.0, 2.5]
    for td in [0.5, 0.7, 1.0]
]

_sp = [""]
def sw_cb(done, total, msg):
    line = f"\r  [{int(done/max(total,1)*100):3d}%] {msg}"
    print(line + " " * max(0, len(_sp[0])-len(line)), end="", flush=True)
    _sp[0] = line

if pre_data:
    sw = parameter_sweep(
        tickers=[], test_start=TEST_START, test_end=TEST_END,
        sweep=sweep_combos, initial_capital=10_000.0, progress_cb=sw_cb,
        preloaded_data=pre_data, preloaded_regimes=pre_reg, precomputed_signals=pre_sig,
    )
    print()

    print("\n SWEEP TOP 10:")
    print(f"  {'BE':>5}  {'Trig':>5}  {'Dist':>5}  {'PF':>7}  {'WR%':>5}  {'AvgW':>7}  {'ROI%':>7}  {'TC':>4}")
    print(f"  {'─'*62}")
    for r in sw[:10]:
        p, sm = r["params"], r["metrics"]
        flag = " ◀" if r is sw[0] else ""
        print(f"  {p['be_trigger_r']:>5.1f}  {p['trail_trigger_r']:>5.1f}  "
              f"{p['trail_dist_r']:>5.1f}  {sm['profit_factor']:>7.3f}  "
              f"{sm['win_rate']*100:>4.0f}%  ${sm['avg_win']:>6,.0f}  "
              f"{sm['roi_pct']:>+6.1f}%  {sm['trade_count']:>4}{flag}")

    best = sw[0]
    bp, bm = best["params"], best["metrics"]
    print(f"\n WINNER: BE={bp['be_trigger_r']}R  Trail={bp['trail_trigger_r']}R/{bp['trail_dist_r']}R"
          f"  →  PF {bm['profit_factor']:.3f}  WR={bm['win_rate']*100:.0f}%  ROI={bm['roi_pct']:+.1f}%")

    import json
    winner = {
        "min_score":       base_params["min_score"],
        "sl_mult_hi":      base_params["sl_mult_hi"],
        "sl_mult_mid":     base_params["sl_mult_mid"],
        "sl_mult_lo":      base_params["sl_mult_lo"],
        "hold_days":       base_params["hold_days"],
        "be_trigger_r":    bp["be_trigger_r"],
        "trail_trigger_r": bp["trail_trigger_r"],
        "trail_dist_r":    bp["trail_dist_r"],
        "profit_factor":   bm["profit_factor"],
        "win_rate":        bm["win_rate"],
        "roi_pct":         bm["roi_pct"],
        "trade_count":     bm["trade_count"],
    }
    with open("stop_sweep_winner.json", "w") as f:
        json.dump(winner, f, indent=2)
    print(f"\n Saved winner to stop_sweep_winner.json")
else:
    print("  (no cached data for sweep — skipping)")

print()
