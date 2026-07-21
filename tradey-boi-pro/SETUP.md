# Tradey Boi Pro — Setup Guide

## What this is

Tradey Boi Pro is the autonomous execution layer on top of Tradey Boi X.
- **Tradey Boi X** → scans the market, sends Discord alerts (unchanged)
- **Tradey Boi Pro** → reads those same signals and places real trades automatically

---

## Requirements

- Python 3.10 or newer
- Interactive Brokers account (paper or live)
- IB Gateway or Trader Workstation (TWS) installed

---

## Step 1 — Install IB Gateway

1. Go to https://www.interactivebrokers.com.au
2. Download **IB Gateway** (free, lighter than TWS)
3. Install and log into your **paper trading** account first
4. Go to **Configure → Settings → API → Enable ActiveX and Socket Clients** ✅
5. Set **Socket Port** to **7497** (paper) — leave everything else default
6. Click **OK**

---

## Step 2 — Install Tradey Boi Pro

Open a terminal in the `tradey-boi-pro` folder and run:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Start the dashboard

```bash
python start_pro.py
```

Then open your browser to: **http://localhost:8502**

---

## Step 4 — Connect IBKR

In the dashboard:
1. Confirm **Host** is `127.0.0.1` and **Port** is `7497`
2. Select **Paper Trading**
3. Click **Connect**

If it connects you'll see your account balance appear in the sidebar.

---

## Step 5 — Start paper trading

1. Go to **Settings** tab → review risk settings (default: 2% per trade, max 5 positions)
2. Click **Save Settings**
3. Click **▶ Start Bot** in the sidebar

The bot will:
- Check Tradey Boi X signals every 60 minutes
- Automatically place bracket orders (entry + stop + target) for STRONG BUY signals
- Monitor open positions and close them at target, stop, or after 15 days
- Pause automatically if the circuit breaker trips (3 consecutive losses)

---

## Paper Trading → Live Trading

Run paper trading for **at least 2–3 months** before going live. When ready:

1. Open IB Gateway and log into your **live** account
2. Change the port in IB Gateway to **7496**
3. In the dashboard **Settings**, change the port to **7496** and reconnect
4. The dashboard will prompt you to confirm you're switching to LIVE mode

---

## Risk defaults (change in Settings tab)

| Setting | Default | Description |
|---|---|---|
| Risk per trade | 2% | % of account risked per trade |
| Max positions | 5 | Maximum simultaneous trades |
| Max exposure | 30% | Max % of account in open positions |
| Daily loss limit | 3% | Pauses trading if day's losses exceed this |
| Hold days | 15 | Auto-exits after this many days |
| Circuit breaker | 3 losses | Pauses for 7 days after 3 consecutive stops |

---

## Keeping it running 24/7

**Home PC:** Works, but PC must stay on. Set power settings to never sleep.

**Better option — VPS (~$5–10/month):**
1. Get a Linux VPS (DigitalOcean, AWS Lightsail, Vultr)
2. Install Python + IB Gateway on it
3. Run `python start_pro.py` in a `screen` or `tmux` session
4. Access the dashboard from your browser via the VPS IP

---

## Troubleshooting

**"Connection refused"**
→ Make sure IB Gateway is running and API is enabled (Step 1)

**"No pending signals"**
→ Tradey Boi X scanner on GitHub Actions hasn't found any STRONG BUY setups yet. This is normal — the system is selective.

**Bot placed no trades today**
→ Check the Health tab for errors. Could be: no signals, circuit breaker active, or exposure limit reached.

**Tradey Boi X still works as normal?**
→ Yes. Pro only reads from X's signal log. X keeps running on GitHub Actions exactly as before.
