"""
Capital growth simulation — NOT a trading-logic change, read-only analysis.

Simulates a real account contributing $250 every fortnight, sizing each
approved trade as a fixed % of current equity risked (1R), using a
synthetic-but-CALIBRATED trade sequence.

WHY SYNTHETIC INSTEAD OF REPLAYING THE RAW BACKTEST LOG: the per-trade
outcomes from full_pipeline_live_gating_validation.py are written to an
ephemeral /tmp checkpoint cache that does not survive process crashes/
restarts in this environment, and re-running the full 407-ticker pipeline
(2y history download + model train + 3-layer gating replay) repeatedly
crashed this sandbox. Instead, this script reconstructs a trade sequence
that is mathematically consistent with the ALREADY-VALIDATED aggregate
stats from that run (see .agents/memory/tradey-boi-x-backtest.md):

    win_rate      = 45.1%
    profit_factor = 1.181
    expectancy_r  = +0.10R
    trade_count   = 122 over the out-of-sample window 2025-12-19 -> 2026-06-19

Solving for a standard R-multiple convention (losers = -1R by definition,
winners = +W R) that reproduces all three numbers simultaneously:
    expectancy_r  = win_rate*W - (1-win_rate)*1        => W ~= 1.44R
    profit_factor = (win_rate*W) / ((1-win_rate)*1)    => 1.182 (matches 1.181)
This is not a new/invented result — it's the unique R-multiple pair implied
by the numbers already produced by the real backtest. Win magnitudes are
sampled with realistic spread around 1.44R (not a flat constant) using a
fixed random seed for reproducibility. Trade timing is sampled uniformly
over business days in the real out-of-sample window.

Results are illustrative of what the ALREADY-MEASURED edge could translate
to with real money and contributions — not a promise of future returns.

Run with: python3 tests/capital_growth_simulation.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta

WIN_RATE = 0.451
PROFIT_FACTOR = 1.181
EXPECTANCY_R = 0.10
N_TRADES = 122
WINDOW_START = date(2025, 12, 19)
WINDOW_END = date(2026, 6, 19)

# EXTRAPOLATION NOTE: only 2025-12-19 -> 2026-06-19 (6 months, 122 trades) has
# actually been backtested/validated. A 12-month projection below assumes the
# SAME trade rate/edge repeats for a second, not-yet-observed 6-month period
# back-to-back — this is an assumption for illustration, not a second
# validated data point. Treat the 1-year numbers with proportionally more
# skepticism than the 6-month numbers.
PROJECTION_MONTHS = 12
N_TRADES_PROJECTED = round(N_TRADES * PROJECTION_MONTHS / 6)

FORTNIGHT_CONTRIBUTION = 250.0
CONTRIBUTION_INTERVAL_DAYS = 14
RISK_PCT_PER_TRADE = 0.02   # 1R = 2% of current equity risked per trade
PRED_DAYS_CALENDAR = 14     # ~10 trading days holding period
RNG_SEED = 42

AVG_WIN_R = (EXPECTANCY_R + (1 - WIN_RATE)) / WIN_RATE
_check_pf = (WIN_RATE * AVG_WIN_R) / ((1 - WIN_RATE) * 1.0)
assert abs(_check_pf - PROFIT_FACTOR) < 0.01, f"R-multiple solve inconsistent: {_check_pf}"


def generate_trades(rng: random.Random, window_start: date, window_end: date, n_trades: int) -> list[dict]:
    total_days = (window_end - window_start).days
    business_days = [window_start + timedelta(days=d) for d in range(total_days)
                      if (window_start + timedelta(days=d)).weekday() < 5]
    signal_dates = sorted(rng.sample(business_days, min(n_trades, len(business_days))))

    trades = []
    for d in signal_dates:
        is_win = rng.random() < WIN_RATE
        if is_win:
            r_multiple = max(0.1, rng.gauss(AVG_WIN_R, AVG_WIN_R * 0.5))
        else:
            r_multiple = -1.0
        trades.append({"signal_date": d, "r_multiple": r_multiple})
    return trades


def simulate(trades: list[dict], window_start: date, risk_pct_per_trade: float = RISK_PCT_PER_TRADE) -> dict:
    end = trades[-1]["signal_date"] + timedelta(days=PRED_DAYS_CALENDAR)
    by_date: dict[date, list[dict]] = {}
    for t in trades:
        by_date.setdefault(t["signal_date"], []).append(t)

    cash = 0.0
    open_positions: list[dict] = []
    total_contributed = 0.0
    n_trades_taken = 0
    n_trades_skipped = 0
    equity_curve: list[float] = []
    next_contribution = window_start

    d = window_start
    while d <= end:
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= d:
                cash += pos["risked_amount"] * (1 + pos["r_multiple"])
            else:
                still_open.append(pos)
        open_positions = still_open

        if d >= next_contribution:
            cash += FORTNIGHT_CONTRIBUTION
            total_contributed += FORTNIGHT_CONTRIBUTION
            next_contribution += timedelta(days=CONTRIBUTION_INTERVAL_DAYS)

        for t in by_date.get(d, []):
            equity = cash + sum(p["risked_amount"] for p in open_positions)
            risked = min(risk_pct_per_trade * equity, cash)
            if risked < 5.0:
                n_trades_skipped += 1
                continue
            cash -= risked
            open_positions.append({
                "exit_date": d + timedelta(days=PRED_DAYS_CALENDAR),
                "risked_amount": risked,
                "r_multiple": t["r_multiple"],
            })
            n_trades_taken += 1

        equity_curve.append(cash + sum(p["risked_amount"] for p in open_positions))
        d += timedelta(days=1)

    final_balance = cash + sum(p["risked_amount"] * (1 + p["r_multiple"]) for p in open_positions)

    peak, max_dd = 0.0, 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    # Modified Dietz method: robust money-weighted return for periodic
    # contributions, no root-finding needed. weight_i = fraction of the
    # total period that contribution i was invested for.
    total_period_days = (end - window_start).days
    contribution_dates = []
    cd = window_start
    while cd <= end:
        contribution_dates.append(cd)
        cd += timedelta(days=CONTRIBUTION_INTERVAL_DAYS)

    weighted_cf_sum = 0.0
    for cdate in contribution_dates:
        days_invested = (end - cdate).days
        weight = days_invested / total_period_days if total_period_days > 0 else 0
        weighted_cf_sum += FORTNIGHT_CONTRIBUTION * weight

    dietz_return = (final_balance - total_contributed) / (0 + weighted_cf_sum) if weighted_cf_sum > 0 else 0.0
    years = total_period_days / 365.25
    annualized_return = (1 + dietz_return) ** (1 / years) - 1 if years > 0 else 0.0

    return {
        "window": f"{window_start} -> {end}",
        "years_simulated": round((end - window_start).days / 365.25, 2),
        "total_contributed": round(total_contributed, 2),
        "final_balance": round(final_balance, 2),
        "net_gain": round(final_balance - total_contributed, 2),
        "trades_taken": n_trades_taken,
        "trades_skipped_insufficient_cash": n_trades_skipped,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "money_weighted_annualized_return_pct": round(annualized_return * 100, 2),
    }


N_RUNS = 20


def run_batch(window_start: date, window_end: date, n_trades: int,
              risk_pct_per_trade: float, n_runs: int = N_RUNS) -> list[dict]:
    results = []
    for seed in range(RNG_SEED, RNG_SEED + n_runs):
        rng = random.Random(seed)
        trades = generate_trades(rng, window_start, window_end, n_trades)
        results.append(simulate(trades, window_start, risk_pct_per_trade))
    return results


def summarize(label: str, results: list[dict]) -> dict:
    avg = lambda key: round(sum(r[key] for r in results) / len(results), 2)
    n = len(results)
    n_losing = sum(1 for r in results if r["net_gain"] < 0)
    print(f"\n{label}")
    print(f"  avg total_contributed  : {avg('total_contributed')}")
    print(f"  avg final_balance      : {avg('final_balance')}")
    print(f"  avg net_gain           : {avg('net_gain')}")
    print(f"  best / worst net_gain  : {max(r['net_gain'] for r in results)} / "
          f"{min(r['net_gain'] for r in results)}")
    print(f"  avg max_drawdown_pct   : {avg('max_drawdown_pct')}")
    print(f"  runs that lost money   : {n_losing}/{n} ({round(100*n_losing/n)}%)")
    return {"label": label, "avg_final_balance": avg("final_balance"), "avg_net_gain": avg("net_gain"),
            "avg_max_drawdown_pct": avg("max_drawdown_pct"), "loss_rate_pct": round(100 * n_losing / n)}


def main():
    print(f"Calibration check: avg winning trade solved as {AVG_WIN_R:.2f}R "
          f"(losers fixed at -1R), reproducing win_rate={WIN_RATE}, "
          f"profit_factor={PROFIT_FACTOR} (got {_check_pf:.3f}), expectancy={EXPECTANCY_R}R")
    print(f"{N_RUNS} runs per scenario (different random trade-outcome orderings)\n")

    # --- 1) 6-month window (the actually-validated period) at 2% risk/trade ---
    print("=" * 70)
    print(f"6-MONTH RESULT (validated window {WINDOW_START} -> {WINDOW_END}, "
          f"{N_TRADES} trades, 2% risk/trade)")
    print("=" * 70)
    results_6mo = run_batch(WINDOW_START, WINDOW_END, N_TRADES, 0.02)
    summarize("6-month", results_6mo)

    # --- 2) 12-month projection (EXTRAPOLATED, not independently validated) ---
    projected_end = WINDOW_START + timedelta(days=365)
    print("\n" + "=" * 70)
    print(f"12-MONTH PROJECTION (EXTRAPOLATED — assumes the same edge/trade rate "
          f"repeats for a second, unobserved 6-month stretch, {N_TRADES_PROJECTED} trades, "
          f"2% risk/trade)")
    print("=" * 70)
    results_12mo = run_batch(WINDOW_START, projected_end, N_TRADES_PROJECTED, 0.02)
    summarize("12-month (extrapolated)", results_12mo)

    # --- 3) Risk-per-trade sensitivity, over the 12-month projection ---
    print("\n" + "=" * 70)
    print("RISK-PER-TRADE SENSITIVITY (12-month extrapolated window)")
    print("=" * 70)
    sensitivity = []
    for risk_pct in (0.01, 0.02, 0.03, 0.05):
        results = run_batch(WINDOW_START, projected_end, N_TRADES_PROJECTED, risk_pct)
        sensitivity.append(summarize(f"{risk_pct*100:.0f}% risk/trade", results))

    print("\n" + "-" * 70)
    print("SENSITIVITY SUMMARY (12-month, contributing $250/fortnight = $6,500 total):")
    print(f"  {'risk/trade':12s} {'avg final $':>12s} {'avg net gain':>14s} {'avg max DD':>12s} {'loss rate':>10s}")
    for s in sensitivity:
        print(f"  {s['label']:12s} {s['avg_final_balance']:>12.0f} {s['avg_net_gain']:>14.0f} "
              f"{s['avg_max_drawdown_pct']:>11.1f}% {s['loss_rate_pct']:>9}%")


if __name__ == "__main__":
    main()
