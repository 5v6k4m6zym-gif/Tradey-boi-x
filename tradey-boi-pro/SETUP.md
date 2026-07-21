# Tradey Boi Pro — Setup Guide

## What this is

Tradey Boi Pro is a **fully independent** autonomous trading platform that runs on your own PC.

| | Tradey Boi X | Tradey Boi Pro |
|---|---|---|
| Where it runs | GitHub Actions (cloud) | Your PC / VPS (local) |
| Scan frequency | Once per day | Every 15 minutes (during market hours) |
| Stocks scanned | ~400 ASX | ASX top 200 + S&P 500 + custom |
| Output | Discord alerts | Real/paper trades via IBKR |
| Touches X's code | N/A | Never |

Pro has its own scanner. Tradey Boi X signals are an **optional bonus** — Pro works completely without them.

---

## Requirements

- Python 3.10 or newer
- Interactive Brokers account (paper or live)
- IB Gateway or Trader Workstation (TWS) installed and running

---

## Step 1 — Install IB Gateway

1. Go to https://www.interactivebrokers.com.au
2. Download **IB Gateway** (lighter than TWS)
3. Log into your **paper trading** account
4. **Configure → Settings → API → Enable ActiveX and Socket Clients** ✅
5. Socket port: **7497** (paper) — leave everything else default

---

## Step 2 — Install dependencies

Open a terminal in the `tradey-boi-pro/` folder:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Start the dashboard

```bash
python start_pro.py
```

Opens at: **http://localhost:8502**

---

## Step 4 — Connect & configure

1. **Dashboard tab** → click Connect (host 127.0.0.1, port 7497, Paper Trading)
2. **Scanner tab** → choose which markets to scan (ASX, US, or both)
3. **Settings tab** → review risk settings (defaults are conservative)
4. Click **▶ Start Bot** in the sidebar

---

## How Pro scans

- Checks market hours (ASX 10am–4pm AEST, US 9:30am–4pm ET)
- Downloads OHLCV data for your entire watchlist in batches via yfinance
- Applies breakout detection (same logic proven in Tradey Boi X):
  - Price breaks above 20-day high
  - Volume surge ≥ 1.5× average
  - Price above 50-day EMA (uptrend filter)
  - RSI 50–80 (momentum without extreme overbought)
  - Score 0–10; only trades score ≥ 7 (configurable)
- Places bracket orders automatically (entry + stop-loss + take-profit)
- Monitors positions every scan cycle; exits at stop, target, or max hold days

---

## Watchlist (Scanner tab)

**Default:**
- ASX: ~200 tickers (ASX top 200 by market cap)
- US: ~100 tickers (S&P 500 sample)

**To customise:**
- Scanner tab → Watchlist Management
- Add or remove tickers by typing them (e.g. `CBA.AX` for ASX, `AAPL` for US)
- Custom tab for one-off additions

---

## Risk defaults (Settings tab)

| Setting | Default | Description |
|---|---|---|
| Risk per trade | 2% | % of account risked per trade |
| Max positions | 5 | Maximum simultaneous open trades |
| Max exposure | 30% | Max % of account in open positions |
| Daily loss limit | 3% | Pauses bot if today's losses exceed this |
| Hold days | 15 | Auto-exits positions after 15 days |
| Scan interval | 15 min | How often to scan during market hours |
| Circuit breaker | 3 losses | Pauses for 7 days after 3 consecutive stops |

---

## Paper Trading → Live Trading

Paper trade for **at least 2–3 months** before going live. When ready:

1. Open IB Gateway with your **live** account, port **7496**
2. Dashboard → Settings → change port to 7496, reconnect
3. The dashboard will confirm you're switching to LIVE

---

## Keeping it running 24/7

**Recommended: small VPS (~$5–10/month)**
1. DigitalOcean, AWS Lightsail, or Vultr
2. Install Python + IB Gateway on the VPS
3. Run in a `tmux` or `screen` session:
   ```bash
   tmux new -s pro
   python start_pro.py
   ```
4. Access the dashboard from anywhere via the VPS IP: `http://YOUR_IP:8502`

---

## Tradey Boi X relationship

Pro runs 100% independently. X keeps running on GitHub Actions and sending Discord alerts exactly as always. If X produces a STRONG BUY in its `signal_log.json`, Pro will optionally pick that up as well — but it doesn't need to.

---

## Troubleshooting

**"Connection refused"** → IB Gateway is not running or API is not enabled

**"No signals found after scan"** → Markets may be closed, or no stocks met the quality gates today (score ≥ 7). This is intentional selectivity — not every day has setups.

**Scan takes a long time** → Normal for 300+ tickers. First scan after start downloads 90 days of data per ticker. Subsequent scans are faster (yfinance caches).

**Bot placed no trades** → Check Health tab for circuit breaker / exposure limit / errors.
