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

from engine import (
    MAX_ALERTS, get_data, train_model, decide, send_alert,
    log_signal, mark_alerted, big_mover_check, send_mover_alert,
    resolve_outcomes,
)

BASE_DIR    = Path(__file__).parent
CURSOR_FILE = BASE_DIR / "overnight_cursor.json"
BATCH_SIZE  = 250   # tickers per overnight run (~750 s at 3 s/ticker, fits 45 min)

US_TZ  = pytz.timezone("America/New_York")
ASX_TZ = pytz.timezone("Australia/Sydney")

# ─── OVERNIGHT UNIVERSE ───────────────────────────────────────────────────────
# Stocks NOT already in the market-hours WATCHLIST.
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


def run_overnight_scan(model) -> int:
    universe = OVERNIGHT_UNIVERSE
    total    = len(universe)
    pos      = _load_cursor()

    batch = universe[pos: pos + BATCH_SIZE]
    next_pos = pos + BATCH_SIZE
    if next_pos >= total:
        next_pos = 0   # wrap around — start fresh next run

    pct_done = round((pos + len(batch)) / total * 100)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Overnight scan — "
          f"batch {pos}–{pos + len(batch) - 1} of {total} ({pct_done}% through universe)")

    fired         = 0
    alerted_count = 0

    for ticker in batch:
        try:
            df = get_data(ticker, "6mo")
            if df.empty:
                continue

            res = decide(ticker, df, model)

            if res["alert"] and fired < MAX_ALERTS:
                price = float(df.iloc[-1]["Close"])
                sent  = send_alert(ticker, res, price, df)
                if sent:
                    mark_alerted(ticker)
                    log_signal(ticker, price, res["signal"],
                               score=res.get("score", 0),
                               prob=res.get("prob", 0.0))
                    print(f"  ✅ {ticker}: {res['label']} (score {res['score']}/14)")
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


BATCH_INTERVAL_SECONDS = 60 * 60  # one batch per hour while markets are closed


def _seconds_until_markets_closed_again() -> int:
    """When markets are open, sleep until the sooner of the two closes."""
    def _next_close(tz, ch, cm):
        now = datetime.now(tz)
        c = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
        if c <= now:
            c += timedelta(days=1)
        return (c - now).total_seconds()
    return int(min(_next_close(US_TZ, 16, 0), _next_close(ASX_TZ, 16, 0))) + 60


def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Overnight scanner starting…")
    print(f"Universe: {len(OVERNIGHT_UNIVERSE)} tickers (not on the main WATCHLIST) | "
          f"batch size {BATCH_SIZE} | runs once/hour while both markets are closed")

    print("\nTraining AI model…")
    model = train_model()
    print("Model ready.\n")

    while True:
        if _markets_closed():
            try:
                resolve_outcomes()
            except Exception as e:
                print(f"  ⚠️  resolve_outcomes: {e}")

            run_overnight_scan(model)
            print(f"Next overnight batch in {BATCH_INTERVAL_SECONDS // 60} min.\n")
            time.sleep(BATCH_INTERVAL_SECONDS)
        else:
            wait = _seconds_until_markets_closed_again()
            wake = datetime.now() + timedelta(seconds=wait)
            print(f"[{datetime.now().strftime('%H:%M')}] A market is open — overnight "
                  f"scanner pauses until both are closed again "
                  f"(~{wake.strftime('%Y-%m-%d %H:%M')}, {wait // 3600}h {(wait % 3600) // 60}m).")
            time.sleep(wait)


if __name__ == "__main__":
    main()
