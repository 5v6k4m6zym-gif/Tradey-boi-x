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

FORTNIGHT_CONTRIBUTION = 250.0
CONTRIBUTION_INTERVAL_DAYS = 14
RISK_PCT_PER_TRADE = 0.02   # 1R = 2% of current equity risked per trade
PRED_DAYS_CALENDAR = 14     # ~10 trading days holding period
RNG_SEED = 42

AVG_WIN_R = (EXPECTANCY_R + (1 - WIN_RATE)) / WIN_RATE
_check_pf = (WIN_RATE * AVG_WIN_R) / ((1 - WIN_RATE) * 1.0)
assert abs(_check_pf - PROFIT_FACTOR) < 0.01, f"R-multiple solve inconsistent: {_check_pf}"


def generate_trades(rng: random.Random) -> list[dict]:
    total_days = (WINDOW_END - WINDOW_START).days
    business_days = [WINDOW_START + timedelta(days=d) for d in range(total_days)
                      if (WINDOW_START + timedelta(days=d)).weekday() < 5]
    signal_dates = sorted(rng.sample(business_days, min(N_TRADES, len(business_days))))

    trades = []
    for d in signal_dates:
        is_win = rng.random() < WIN_RATE
        if is_win:
            r_multiple = max(0.1, rng.gauss(AVG_WIN_R, AVG_WIN_R * 0.5))
        else:
            r_multiple = -1.0
        trades.append({"signal_date": d, "r_multiple": r_multiple})
    return trades


def simulate(trades: list[dict]) -> dict:
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
    next_contribution = WINDOW_START

    d = WINDOW_START
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
            risked = min(RISK_PCT_PER_TRADE * equity, cash)
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
    total_period_days = (end - WINDOW_START).days
    contribution_dates = []
    cd = WINDOW_START
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
        "window": f"{WINDOW_START} -> {end}",
        "years_simulated": round((end - WINDOW_START).days / 365.25, 2),
        "total_contributed": round(total_contributed, 2),
        "final_balance": round(final_balance, 2),
        "net_gain": round(final_balance - total_contributed, 2),
        "trades_taken": n_trades_taken,
        "trades_skipped_insufficient_cash": n_trades_skipped,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "money_weighted_annualized_return_pct": round(annualized_return * 100, 2),
    }


def main():
    print(f"Calibration check: avg winning trade solved as {AVG_WIN_R:.2f}R "
          f"(losers fixed at -1R), reproducing win_rate={WIN_RATE}, "
          f"profit_factor={PROFIT_FACTOR} (got {_check_pf:.3f}), expectancy={EXPECTANCY_R}R")
    print(f"Risk per trade: {RISK_PCT_PER_TRADE*100:.0f}% of current equity (1R)")
    print()

    results = []
    for seed in range(RNG_SEED, RNG_SEED + 5):
        rng = random.Random(seed)
        trades = generate_trades(rng)
        results.append(simulate(trades))

    print("=" * 70)
    print(f"CAPITAL SIMULATION — ${FORTNIGHT_CONTRIBUTION:.0f}/fortnight, "
          f"{N_TRADES} trades over {WINDOW_START} -> {WINDOW_END}")
    print("(5 runs with different random trade-outcome orderings/win magnitudes)")
    print("=" * 70)
    for i, r in enumerate(results):
        print(f"\nRun {i+1}:")
        for k, v in r.items():
            print(f"  {k:38s}: {v}")

    avg = lambda key: round(sum(r[key] for r in results) / len(results), 2)
    print("\n" + "-" * 70)
    print("AVERAGE ACROSS RUNS:")
    print(f"  total_contributed                    : {avg('total_contributed')}")
    print(f"  final_balance                         : {avg('final_balance')}")
    print(f"  net_gain                              : {avg('net_gain')}")
    print(f"  max_drawdown_pct                       : {avg('max_drawdown_pct')}")
    print(f"  money_weighted_annualized_return_pct   : {avg('money_weighted_annualized_return_pct')}")


if __name__ == "__main__":
    main()
