"""
Overnight scanner — runs every hour during non-market hours.
Works through a large universe of ASX + US stocks in rotating batches,
so the full list is covered across a night of offline hours.
Cursor position is saved to overnight_cursor.json between runs.
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytz

import engine
from engine import (
    MAX_ALERTS, WATCHLIST, CORRELATION_GROUPS, get_data, train_model, decide,
    send_alert, log_signal, mark_alerted, big_mover_check, send_mover_alert,
    resolve_outcomes,
)
try:
    from opportunity.trade_evaluator import process_trade_signal
    from opportunity.config import ENABLE_TRADE_EVALUATOR, SHADOW_MODE
    _TRADE_EVAL_AVAILABLE = True
except ImportError:
    _TRADE_EVAL_AVAILABLE = False
    ENABLE_TRADE_EVALUATOR = False
    SHADOW_MODE = True

try:
    from opportunity.adaptive_core import process_trade_signal as process_adaptive_trade_signal
    from opportunity.config import ENABLE_ADAPTIVE_CORE
    _ADAPTIVE_CORE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_CORE_AVAILABLE = False
    ENABLE_ADAPTIVE_CORE = False

try:
    from opportunity.audit_engine import audit_trade
    from opportunity.config import ENABLE_AUDIT_ENGINE
    _AUDIT_ENGINE_AVAILABLE = True
except ImportError:
    _AUDIT_ENGINE_AVAILABLE = False
    ENABLE_AUDIT_ENGINE = False

try:
    from opportunity.strategy_optimizer import process_trade_signal as process_strategy_signal
    from opportunity.config import ENABLE_STRATEGY_OPTIMIZER
    _STRATEGY_OPTIMIZER_AVAILABLE = True
except ImportError:
    _STRATEGY_OPTIMIZER_AVAILABLE = False
    ENABLE_STRATEGY_OPTIMIZER = False

BASE_DIR    = Path(__file__).parent
CURSOR_FILE = BASE_DIR / "overnight_cursor.json"
BATCH_SIZE  = 250   # tickers per overnight run (~750 s at 3 s/ticker, fits 45 min)

US_TZ  = pytz.timezone("America/New_York")
ASX_TZ = pytz.timezone("Australia/Sydney")

# ─── OVERNIGHT UNIVERSE (extras) ──────────────────────────────────────────────
# Additional stocks beyond the market-hours WATCHLIST. At runtime these are
# merged with WATCHLIST (see _full_overnight_universe) so the overnight scan
# covers everything the bot tracks, not just this extra list.
# Any delisted/invalid tickers are silently skipped (get_data returns empty df).
OVERNIGHT_UNIVERSE = [
    # ── US — more tech & growth ───────────────────────────────────────────
    "SNOW", "ZS", "NET", "ANET", "DDOG", "MDB", "HUBS", "INTU", "TEAM",
    "ZM", "OKTA", "TWLO", "WDAY", "VEEV", "TTD", "ROKU", "ABNB", "DASH",
    "COIN", "SQ", "PYPL", "RBLX", "HOOD", "APP", "BILL", "GTLB",
    # ── US — more healthcare ──────────────────────────────────────────────
    "TMO", "ABT", "BSX", "MDT", "SYK", "ISRG", "ELV", "CI", "HCA",
    "BMY", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "IQV", "DGX", "LH",
    # ── US — more finance ────────────────────────────────────────────────
    "SPGI", "MCO", "ICE", "CME", "CBOE", "TRV", "AON", "MMC",
    "PRU", "MET", "AFL", "PGR", "COF", "SCHW", "USB", "TFC", "PNC",
    # ── US — more consumer ───────────────────────────────────────────────
    "PG", "KO", "PEP", "PM", "MO", "SBUX", "TGT", "LOW",
    "ULTA", "RH", "NKE", "F", "GM",
    # ── US — more industrials & transport ────────────────────────────────
    "UNP", "CSX", "NSC", "UPS", "FDX", "WM", "RSG",
    "EMR", "ETN", "PH", "AME", "ROK", "VRSK",
    "RTX", "LMT", "NOC", "GD", "HII",
    # ── US — more energy ─────────────────────────────────────────────────
    "VLO", "MPC", "PSX", "EOG", "DVN", "HAL", "SLB", "APA", "MRO",
    # ── US — materials ───────────────────────────────────────────────────
    "LIN", "APD", "ECL", "SHW", "PPG", "FCX", "NEM", "ALB", "MP",
    "NUE", "STLD", "AA",
    # ── US — REITs & utilities ───────────────────────────────────────────
    "AMT", "PLD", "EQIX", "WELL", "VTR", "EQR", "AVB",
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "PEG",

    # ── ASX — more gold miners ────────────────────────────────────────────
    "CMM.AX", "DCN.AX", "PRU.AX", "RED.AX", "RSG.AX", "ALK.AX",
    "RMS.AX", "SBM.AX", "TIE.AX", "OGC.AX", "BGL.AX", "MML.AX",
    # ── ASX — more lithium & battery metals ──────────────────────────────
    "EUR.AX", "LKE.AX", "INR.AX", "AGY.AX", "ADD.AX",
    "CPM.AX",
    # ── ASX — more uranium ────────────────────────────────────────────────
    "AGE.AX", "CAM.AX", "LOT.AX", "ERA.AX", "MEY.AX", "TOE.AX",
    # ── ASX — more copper & nickel ───────────────────────────────────────
    "CTM.AX", "AIS.AX", "NIC.AX", "MCR.AX",
    # ── ASX — more oil & gas ─────────────────────────────────────────────
    "CVN.AX", "HZN.AX", "VEA.AX", "ALD.AX",
    # ── ASX — more tech & software ───────────────────────────────────────
    "APX.AX", "TYR.AX", "AD8.AX", "PWR.AX", "RWC.AX",
    "EML.AX", "SLC.AX", "SDR.AX", "SKO.AX", "TNE.AX",
    # ── ASX — more healthcare & biotech ──────────────────────────────────
    "PNV.AX", "IMM.AX", "MSB.AX", "OPT.AX", "AVH.AX",
    "PHL.AX", "CUV.AX", "OSL.AX", "TLX.AX", "ANP.AX",
    # ── ASX — more consumer & retail ─────────────────────────────────────
    "DTL.AX", "BBN.AX", "OML.AX", "HLO.AX", "EVT.AX",
    "KGN.AX", "ELD.AX", "GUD.AX", "CWY.AX", "A2M.AX",
    # ── ASX — more finance & insurance ───────────────────────────────────
    "CGF.AX", "HUB.AX", "PPT.AX", "JHG.AX", "GQG.AX",
    "HMC.AX", "MFG.AX", "PTM.AX", "AMP.AX", "GFL.AX",
    # ── ASX — more REITs & property ──────────────────────────────────────
    "CIP.AX", "ARF.AX", "CLW.AX", "HDN.AX", "NSR.AX",
    "VCX.AX", "HCW.AX", "DXS.AX", "CQR.AX", "URW.AX",
    # ── ASX — more industrials & services ────────────────────────────────
    "NWH.AX", "GNG.AX", "MLD.AX", "MAH.AX", "SRG.AX",
    "CIM.AX", "SVW.AX", "DOW.AX", "AZJ.AX", "WOR.AX",
    # ── ASX — more diversified & small-cap resources ─────────────────────
    "MLX.AX", "PAN.AX", "IVZ.AX", "CHN.AX",
    "LRS.AX", "WRM.AX", "PGM.AX", "GRR.AX", "CIA.AX",
    # ── ASX — more diversified ───────────────────────────────────────────
    "WEB.AX", "ARX.AX", "NWL.AX", "HLS.AX", "DHG.AX",
    "SPK.AX", "NEC.AX", "OFX.AX", "NHF.AX", "GNE.AX",
]


def _markets_closed() -> bool:
    """True only when BOTH US and ASX markets are closed."""
    def _is_open(tz, oh, om, ch, cm):
        now = datetime.now(tz)
        if now.weekday() >= 5:
            return False
        o = now.replace(hour=oh, minute=om, second=0, microsecond=0)
        c = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
        return o <= now < c
    us_open  = _is_open(US_TZ,  9, 30, 16, 0)
    asx_open = _is_open(ASX_TZ, 10,  0, 16, 0)
    return not (us_open or asx_open)


def _load_cursor() -> int:
    try:
        return int(json.loads(CURSOR_FILE.read_text()).get("position", 0))
    except Exception:
        return 0


def _save_cursor(pos: int):
    tmp = CURSOR_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"position": pos}))
    tmp.replace(CURSOR_FILE)


def _corr_group(ticker: str) -> int | None:
    """Return the index of this ticker's correlation group, or None if ungrouped.
    Mirrors scanner.py's _corr_group exactly — same shared CORRELATION_GROUPS."""
    for i, group in enumerate(CORRELATION_GROUPS):
        if ticker in group:
            return i
    return None


def _full_overnight_universe() -> list:
    """Main WATCHLIST + the overnight-only extras, deduplicated, order preserved."""
    seen = set()
    combined = []
    for ticker in WATCHLIST + OVERNIGHT_UNIVERSE:
        if ticker not in seen:
            seen.add(ticker)
            combined.append(ticker)
    return combined


def run_overnight_scan(model) -> int:
    universe = _full_overnight_universe()
    total    = len(universe)
    pos      = _load_cursor()

    batch = universe[pos: pos + BATCH_SIZE]
    next_pos = pos + BATCH_SIZE
    if next_pos >= total:
        next_pos = 0   # wrap around — start fresh next run

    pct_done = round((pos + len(batch)) / total * 100)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Overnight scan — "
          f"batch {pos}–{pos + len(batch) - 1} of {total} ({pct_done}% through universe)")

    fired          = 0
    alerted_count  = 0
    alerted_groups: set = set()   # correlation guard — mirrors scanner.py, one alert per group per batch

    for ticker in batch:
        try:
            df = get_data(ticker, "6mo")
            if df.empty:
                continue

            res = decide(ticker, df, model)

            if res["alert"] and fired < MAX_ALERTS:
                # Correlation guard — skip if a correlated ticker already alerted this batch
                group_id = _corr_group(ticker)
                if group_id is not None and group_id in alerted_groups:
                    print(f"  ⏭ {ticker}: correlation guard — similar ticker already alerted")
                    continue

                price = float(df.iloc[-1]["Close"])

                # ── Trade Evaluation & Filtering Layer (Phase 8) ──────────
                # Mirrors scanner.py — same shared gate, same SHADOW_MODE
                # switch. Purely additive; never touches decide()/send_alert().
                if ENABLE_TRADE_EVALUATOR:
                    try:
                        params = engine._trade_params(ticker, res, price, df)
                        trade  = {
                            "ticker":      ticker,
                            "direction":   "LONG",
                            "entry":       price,
                            "stop_loss":   params["stop_loss"],
                            "take_profit": params["target_price"],
                            "probability": res.get("prob", 0.0),
                            "expected_r":  res.get("expected_r"),
                        }
                        approved = process_trade_signal(trade, df)
                        if not SHADOW_MODE and approved is None:
                            print(f"  🧪 {ticker}: rejected by trade evaluator (see logs/trade_evaluations.jsonl)")
                            continue
                    except Exception as _te:
                        print(f"  ⚠️  {ticker}: trade evaluator error ({_te}) — proceeding unaffected")

                # ── Adaptive Trading Core v4 (stacked ABOVE Phase 8) ──────
                if ENABLE_ADAPTIVE_CORE:
                    try:
                        params = engine._trade_params(ticker, res, price, df)
                        trade  = {
                            "ticker":      ticker,
                            "direction":   "LONG",
                            "entry":       price,
                            "stop_loss":   params["stop_loss"],
                            "take_profit": params["target_price"],
                            "probability": res.get("prob", 0.0),
                            "expected_r":  res.get("expected_r"),
                        }
                        adaptive_approved = process_adaptive_trade_signal(trade, df)
                        if not SHADOW_MODE and adaptive_approved is None:
                            print(f"  🧬 {ticker}: rejected by adaptive core (see logs/adaptive_core_decisions.jsonl)")
                            continue
                    except Exception as _ac:
                        print(f"  ⚠️  {ticker}: adaptive core error ({_ac}) — proceeding unaffected")

                # ── Self-Optimising Strategy Engine (SAFE MODE) ───────────
                if ENABLE_STRATEGY_OPTIMIZER:
                    try:
                        params = engine._trade_params(ticker, res, price, df)
                        trade  = {
                            "ticker":      ticker,
                            "direction":   "LONG",
                            "entry":       price,
                            "stop_loss":   params["stop_loss"],
                            "take_profit": params["target_price"],
                            "probability": res.get("prob", 0.0),
                            "expected_r":  res.get("expected_r"),
                            "why":         res.get("why", []),
                            "rsi":         res.get("rsi"),
                            "edge_score":  res.get("prob", 0.0),
                        }
                        strategy_approved = process_strategy_signal(trade, df)
                        if not SHADOW_MODE and strategy_approved is None:
                            print(f"  🧭 {ticker}: rejected by strategy optimiser (see logs/strategy_optimizer_decisions.jsonl)")
                            continue
                    except Exception as _so:
                        print(f"  ⚠️  {ticker}: strategy optimiser error ({_so}) — proceeding unaffected")

                # ── Full System Audit Suite — logging-only, never gates ───
                if ENABLE_AUDIT_ENGINE:
                    try:
                        params = engine._trade_params(ticker, res, price, df)
                        trade  = {
                            "ticker":      ticker,
                            "direction":   "LONG",
                            "entry":       price,
                            "stop_loss":   params["stop_loss"],
                            "take_profit": params["target_price"],
                            "probability": res.get("prob", 0.0),
                            "expected_r":  res.get("expected_r"),
                        }
                        audit_trade(trade, df)
                    except Exception as _ae:
                        print(f"  ⚠️  {ticker}: audit engine error ({_ae}) — proceeding unaffected")

                sent  = send_alert(ticker, res, price, df)
                if sent:
                    mark_alerted(ticker)
                    log_signal(ticker, price, res["signal"],
                               score=res.get("score", 0),
                               prob=res.get("prob", 0.0),
                               features={
                                   "regime":        res.get("regime", ""),
                                   "quality_score": res.get("quality_score", 0),
                                   "rsi":           res.get("rsi", 0),
                                   "multibagger":   bool(res.get("multibagger")),
                               })
                    if group_id is not None:
                        alerted_groups.add(group_id)
                    print(f"  ✅ {ticker}: {res['label']} (quality {res.get('quality_score',0)}/100)")
                    fired += 1
            else:
                mover = big_mover_check(ticker, df, model=model)
                if mover:
                    tier = mover["tier"]
                    sent = send_mover_alert(ticker, mover, df=df)
                    if sent:
                        alerted_count += 1
                        detail = (f"+{mover['daily_ret']*100:.1f}% | {mover['vol_r']:.1f}× vol"
                                  if tier == "ACTIVE" else
                                  f"ai={mover.get('ai_prob',0)*100:.0f}%")
                        print(f"  [{tier}] {ticker}: {detail} ✅")

        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    _save_cursor(next_pos)
    print(f"Overnight batch done. {fired + alerted_count} alert(s). "
          f"Cursor → {next_pos} (next run starts there).")
    return fired + alerted_count


def main(budget_seconds: float | None = None):
    """Run the overnight batch loop. If budget_seconds is set, keep scanning
    batches back-to-back until that much wall-clock time has elapsed, then
    return cleanly — this lets a single GitHub Actions job cover a whole
    overnight window instead of depending on GitHub's cron scheduler to fire
    reliably every hour (it does not, under load — same issue as scanner.py's
    hourly cron, see workflow scheduling notes)."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Overnight scanner starting…")
    if budget_seconds:
        print(f"  Time budget: {budget_seconds // 60:.0f} min (will exit cleanly after)")

    if not _markets_closed():
        print("Markets still open — overnight scanner only runs when both markets are closed.")
        return

    print("Training AI model…")
    model = train_model()
    print("Model ready.\n")

    try:
        resolve_outcomes()
    except Exception as e:
        print(f"  ⚠️  resolve_outcomes: {e}")

    _start = time.monotonic()
    while True:
        if not _markets_closed():
            print(f"[{datetime.now().strftime('%H:%M')}] A market has opened — stopping overnight scan.")
            return

        run_overnight_scan(model)

        if budget_seconds is None:
            return

        elapsed = time.monotonic() - _start
        remaining = budget_seconds - elapsed
        if remaining <= 0:
            print(f"[{datetime.now().strftime('%H:%M')}] Time budget reached — exiting cleanly.")
            return

        print(f"  {remaining // 60:.0f} min left in this session's budget — starting next batch.\n")


if __name__ == "__main__":
    if "--minutes" in sys.argv:
        # Bounded continuous-loop mode — keeps scanning overnight batches for
        # up to N minutes, then exits cleanly. Used by GitHub Actions so one
        # job covers a whole overnight window (ASX close → US open, or US
        # close → ASX open) instead of relying on an hourly cron.
        _idx = sys.argv.index("--minutes")
        _n_minutes = float(sys.argv[_idx + 1])
        main(budget_seconds=_n_minutes * 60)
    else:
        main()
