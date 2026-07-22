"""
Tradey Boi Pro — quality trading universe.

~280 tickers total: ASX top-150 (large + liquid mid cap) + US top-130 (S&P 500 quality).
Small/speculative caps removed — they produce noisy signals and hurt win rate.
Apply liquidity filter = True for live scanning; False for backtests (faster).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import db.database as db

log = logging.getLogger("Universe")

# ── ASX — top ~150 liquid stocks (ASX 200 quality names) ─────────────────────
# Large cap + liquid mid cap only. Min ~$1 price, min ~500k daily volume.
# No speculative miners, no pre-revenue biotech, no penny stocks.
ASX_UNIVERSE: list[str] = [
    # ── Mega / Large Cap (ASX 50) ─────────────────────────────────────────────
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX", "MQG.AX",
    "TLS.AX", "RIO.AX", "FMG.AX", "GMG.AX", "WOW.AX", "TCL.AX", "REA.AX", "SHL.AX",
    "COH.AX", "ALL.AX", "RMD.AX", "QBE.AX", "IAG.AX", "SUN.AX", "ORG.AX", "AGL.AX",
    "AMP.AX", "ASX.AX", "BXB.AX", "CPU.AX", "CWY.AX", "DXS.AX", "GPT.AX", "ILU.AX",
    "IPL.AX", "MGR.AX", "MPL.AX", "MTS.AX", "NST.AX", "ORA.AX", "SCG.AX",
    "SGP.AX", "STO.AX", "TAH.AX", "TWE.AX", "VCX.AX", "XRO.AX", "MIN.AX", "EVN.AX",
    # ── Quality Mid Cap (ASX 100–200) ─────────────────────────────────────────
    "A2M.AX", "ALQ.AX", "ALU.AX", "AMC.AX", "ANN.AX", "APA.AX", "APE.AX",
    "ARB.AX", "AZJ.AX", "BAP.AX", "BGA.AX", "BKW.AX", "BLD.AX", "BOQ.AX",
    "BPT.AX", "BWP.AX", "CAR.AX", "CCP.AX", "CGF.AX", "CHC.AX", "CLW.AX",
    "CMM.AX", "COF.AX", "CTD.AX", "DCN.AX", "DHG.AX", "DMP.AX", "DOW.AX",
    "EBO.AX", "ELD.AX", "EVN.AX", "GNC.AX", "GOR.AX", "GUD.AX", "GWA.AX",
    "HLS.AX", "HMC.AX", "HVN.AX", "IEL.AX", "IGO.AX", "JBH.AX", "LLC.AX",
    "LYC.AX", "MGX.AX", "NEC.AX", "NHF.AX", "NUF.AX", "NXT.AX", "PDN.AX",
    "PGH.AX", "PLS.AX", "PME.AX", "PMV.AX", "PPT.AX", "QAN.AX", "RHC.AX",
    "SAR.AX", "SFR.AX", "SGM.AX", "SPK.AX", "SUL.AX", "SWM.AX", "TPG.AX",
    "VEA.AX", "WAF.AX", "WEB.AX", "WHC.AX", "WOR.AX", "WTC.AX", "YAL.AX",
    "FLT.AX", "SEK.AX", "CAM.AX", "CQR.AX", "REG.AX", "SXL.AX", "TRS.AX",
    "CXO.AX", "GDF.AX", "OFX.AX", "RWC.AX", "SRG.AX",
]

# ── US — S&P 500 quality + select high-momentum growth ───────────────────────
# No pre-revenue speculative stocks, no meme stocks, no EV startups.
# Established businesses with real earnings or clear growth trajectory.
US_UNIVERSE: list[str] = [
    # ── S&P 500 Core ──────────────────────────────────────────────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "COST", "AVGO", "WMT", "TMO", "MCD", "ACN", "CSCO", "ABT", "NEE", "DHR",
    "BMY", "LIN", "ADBE", "NKE", "TXN", "PM", "CRM", "QCOM", "UPS", "HON",
    "AMGN", "RTX", "LOW", "INTU", "SPGI", "GS", "BLK", "CAT", "ELV", "AMAT",
    "ISRG", "NOW", "AMD", "BKNG", "DE", "AXP", "MDLZ", "ADI", "CB", "GILD",
    "REGN", "ZTS", "LMT", "GE", "C", "MO", "VZ", "PFE", "CVS", "SYK",
    "MU", "BSX", "PLD", "AMT", "CCI", "EQIX", "AON", "ICE", "CME", "MCO",
    "CTAS", "CDNS", "KLAC", "MCHP", "PAYX", "VRSK", "ODFL", "FAST",
    "BIIB", "VRTX", "DXCM", "IQV", "MTD", "WST",
    # ── Financials & Insurance ────────────────────────────────────────────────
    "BAC", "WFC", "BK", "STT", "SCHW", "AMP", "PFG", "PRU", "MET", "AFL",
    "ALL", "TRV", "HIG", "ACGL",
    # ── Healthcare / Biopharma (profitable) ──────────────────────────────────
    "MCK", "ABC", "CAH", "CI", "HUM", "MOH", "CNC", "EW", "IDXX", "ROP",
    # ── Quality Tech / Software ───────────────────────────────────────────────
    "PANW", "CRWD", "NET", "ZS", "DDOG", "SNOW", "PLTR", "FTNT", "OKTA",
    "MDB", "GTLB", "HUBS", "WDAY", "VEEV", "SHOP", "MELI",
    # ── High-momentum profitable growth ──────────────────────────────────────
    "CELH", "ELF", "DUOL", "IBKR", "LPLA", "LULU", "ROST", "TJX",
    "CMG", "TXRH", "DRI", "NFLX", "UBER",
    # ── Industrials / Transport ───────────────────────────────────────────────
    "SAIA", "XPO", "JBHT", "CHRW", "EXPD", "GWW", "MSM", "SWK", "SNA",
    "HEICO", "TDG", "ITT", "IDEX", "ROP",
]

# Deduplicate both lists
US_UNIVERSE  = list(dict.fromkeys(US_UNIVERSE))
ASX_UNIVERSE = list(dict.fromkeys(ASX_UNIVERSE))


# ── Liquidity filter ───────────────────────────────────────────────────────────

_liquidity_cache: dict[str, bool]     = {}
_liquidity_cache_ts: datetime | None  = None
_CACHE_TTL_HOURS = 24


def filter_by_liquidity(
    tickers: list[str],
    min_avg_volume: int   = 200_000,
    min_price:      float = 0.50,
) -> list[str]:
    """
    Remove tickers that don't meet minimum liquidity thresholds.
    Results are cached for 24h to avoid hammering yfinance.
    Runs a fast check: just needs 5-day data.
    """
    global _liquidity_cache, _liquidity_cache_ts

    now = datetime.utcnow()
    if _liquidity_cache_ts and (now - _liquidity_cache_ts).total_seconds() < _CACHE_TTL_HOURS * 3600:
        return [t for t in tickers if _liquidity_cache.get(t, True)]

    try:
        import yfinance as yf
        log.info(f"Running liquidity filter on {len(tickers)} tickers…")
        batch_size = 100
        new_cache  = {}
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            try:
                raw = yf.download(
                    " ".join(batch), period="5d", interval="1d",
                    auto_adjust=True, progress=False,
                    group_by="ticker", threads=True
                )
                for t in batch:
                    try:
                        df = raw[t] if len(batch) > 1 else raw
                        df = df.dropna(how="all")
                        df.columns = [c.title() if isinstance(c, str) else c for c in df.columns]
                        if df.empty:
                            new_cache[t] = False
                            continue
                        avg_vol    = float(df["Volume"].mean())
                        last_price = float(df["Close"].iloc[-1])
                        new_cache[t] = (avg_vol >= min_avg_volume and last_price >= min_price)
                    except Exception:
                        new_cache[t] = True
            except Exception:
                for t in batch:
                    new_cache[t] = True
        _liquidity_cache    = new_cache
        _liquidity_cache_ts = now
        kept    = [t for t in tickers if new_cache.get(t, True)]
        removed = len(tickers) - len(kept)
        log.info(f"Liquidity filter: kept {len(kept)}, removed {removed} illiquid tickers")
        return kept
    except Exception as e:
        log.error(f"Liquidity filter error: {e}")
        return tickers


def build_universe(
    markets:         list[str] | None = None,
    apply_liquidity: bool = True,
    custom_tickers:  list[str] | None = None,
) -> list[str]:
    """
    Build the full scan universe for the given markets.
    markets: list of "ASX", "US". Defaults to both.
    """
    if markets is None:
        markets = ["ASX", "US"]

    from db.database import get_setting
    tickers: list[str] = []
    for mkt in markets:
        try:
            stored = get_setting(f"watchlist_{mkt.upper()}")
        except Exception:
            stored = None
        if stored is not None:
            tickers.extend(stored)
        else:
            if mkt.upper() == "ASX":
                tickers.extend(ASX_UNIVERSE)
            elif mkt.upper() == "US":
                tickers.extend(US_UNIVERSE)

    if custom_tickers:
        tickers.extend(custom_tickers)

    seen   = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    if apply_liquidity:
        deduped = filter_by_liquidity(deduped)

    log.info(f"Universe built: {len(deduped)} tickers ({', '.join(markets)})")
    return deduped
