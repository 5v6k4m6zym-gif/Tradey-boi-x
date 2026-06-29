"""
Background scanner — runs automatically every hour.
Trains the model once, then scans the full watchlist
on a loop and fires Discord alerts for qualifying signals.
No browser or dashboard needed.
"""
import time
from datetime import datetime

from engine import (
    WATCHLIST, MAX_ALERTS, PREDICTION_DAYS,
    get_data, train_model, decide, send_alert,
    log_signal, mark_alerted, resolve_outcomes,
)

SCAN_INTERVAL_SECONDS = 3600   # 1 hour

def run_scan(model) -> int:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Scanning {len(WATCHLIST)} tickers…")
    fired = 0

    for ticker in WATCHLIST:
        try:
            df  = get_data(ticker, "6mo")
            if df.empty:
                continue
            res = decide(ticker, df, model)

            if res["alert"] and fired < MAX_ALERTS:
                price = float(df.iloc[-1]["Close"])
                sent  = send_alert(ticker, res, price)
                if sent:
                    mark_alerted(ticker)
                    log_signal(ticker, price, res["signal"])
                    print(f"  ✅ Alert sent: {ticker} {res['label']} (score {res['score']}/14)")
                    fired += 1
            else:
                status = res["label"] if res["signal"] != "GATED" else "🚫 GATED"
                print(f"  — {ticker}: {status}")

        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    # Check if any past signals have matured and update outcomes
    resolve_outcomes()

    print(f"Scan done. {fired} alert(s) sent. Next scan in {SCAN_INTERVAL_SECONDS // 60} min.")
    return fired

def main():
    print("=" * 50)
    print("  TRADEY BOI X — Auto Scanner")
    print(f"  Scanning every {SCAN_INTERVAL_SECONDS // 60} minutes")
    print(f"  Watchlist: {', '.join(WATCHLIST)}")
    print("=" * 50)

    print("\nTraining AI model…")
    model = train_model()
    print("Model ready.\n")

    while True:
        run_scan(model)
        time.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
