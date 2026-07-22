"""
Tradey Boi Pro — trading universe.

Full universe: ~900 tickers (ASX + US) scanned for signals.
Quality universe: ~288 liquid large/mid-cap stocks that trade at normal min_score.
Extended universe: remaining ~600 stocks that only trade on ELITE signals.

This lets the bot capture rare breakout opportunities on smaller stocks
without letting their noisy signals drag down overall win rate.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import db.database as db

log = logging.getLogger("Universe")

# ── QUALITY universe — large/liquid mid-cap only (~288 stocks) ────────────────
# These stocks trade at the normal min_score threshold (STRONG BUY or ELITE).
# Clean trends, high liquidity, reliable signal quality.

QUALITY_ASX: list[str] = [
    # Mega / Large Cap
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX", "MQG.AX",
    "TLS.AX", "RIO.AX", "FMG.AX", "GMG.AX", "WOW.AX", "TCL.AX", "REA.AX", "SHL.AX",
    "COH.AX", "ALL.AX", "RMD.AX", "QBE.AX", "IAG.AX", "SUN.AX", "ORG.AX", "AGL.AX",
    "AMP.AX", "ASX.AX", "BXB.AX", "CPU.AX", "CWY.AX", "DXS.AX", "GPT.AX", "ILU.AX",
    "IPL.AX", "MGR.AX", "MPL.AX", "MTS.AX", "NST.AX", "ORA.AX", "SCG.AX",
    "SGP.AX", "STO.AX", "TAH.AX", "TWE.AX", "VCX.AX", "XRO.AX", "MIN.AX", "EVN.AX",
    # Quality Mid Cap
    "A2M.AX", "ALQ.AX", "ALU.AX", "AMC.AX", "ANN.AX", "APA.AX", "APE.AX",
    "ARB.AX", "AZJ.AX", "BAP.AX", "BGA.AX", "BKW.AX", "BLD.AX", "BOQ.AX",
    "BPT.AX", "BWP.AX", "CAR.AX", "CCP.AX", "CGF.AX", "CHC.AX", "CLW.AX",
    "CMM.AX", "COF.AX", "CTD.AX", "DCN.AX", "DHG.AX", "DMP.AX", "DOW.AX",
    "EBO.AX", "ELD.AX", "GNC.AX", "GOR.AX", "GUD.AX", "GWA.AX",
    "HLS.AX", "HMC.AX", "HVN.AX", "IEL.AX", "IGO.AX", "JBH.AX", "LLC.AX",
    "LYC.AX", "MGX.AX", "NEC.AX", "NHF.AX", "NUF.AX", "NXT.AX", "PDN.AX",
    "PGH.AX", "PLS.AX", "PME.AX", "PMV.AX", "PPT.AX", "QAN.AX", "RHC.AX",
    "SAR.AX", "SFR.AX", "SGM.AX", "SPK.AX", "SUL.AX", "SWM.AX", "TPG.AX",
    "VEA.AX", "WAF.AX", "WEB.AX", "WHC.AX", "WOR.AX", "WTC.AX", "YAL.AX",
    "FLT.AX", "SEK.AX", "CAM.AX", "CQR.AX", "SXL.AX", "TRS.AX",
    "OFX.AX", "RWC.AX", "SRG.AX",
]

QUALITY_US: list[str] = [
    # S&P 500 Core
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
    # Financials & Insurance
    "BAC", "WFC", "BK", "STT", "SCHW", "AMP", "PFG", "PRU", "MET", "AFL",
    "ALL", "TRV", "HIG", "ACGL",
    # Healthcare
    "MCK", "ABC", "CAH", "CI", "HUM", "MOH", "CNC", "EW", "IDXX", "ROP",
    # Quality Tech / Software
    "PANW", "CRWD", "NET", "ZS", "DDOG", "SNOW", "PLTR", "FTNT", "OKTA",
    "MDB", "GTLB", "HUBS", "WDAY", "VEEV", "SHOP", "MELI",
    # Profitable Growth
    "CELH", "ELF", "DUOL", "IBKR", "LPLA", "LULU", "ROST", "TJX",
    "CMG", "TXRH", "DRI", "NFLX", "UBER",
    # Industrials / Transport
    "SAIA", "XPO", "JBHT", "CHRW", "EXPD", "GWW", "MSM", "SWK", "SNA",
    "HEICO", "TDG", "IDEX",
]

QUALITY_ASX = list(dict.fromkeys(QUALITY_ASX))
QUALITY_US  = list(dict.fromkeys(QUALITY_US))

# Set for O(1) membership checks
QUALITY_SET: frozenset[str] = frozenset(QUALITY_ASX + QUALITY_US)


def is_quality_ticker(ticker: str) -> bool:
    """Return True if ticker is in the quality (large/liquid mid-cap) universe."""
    return ticker in QUALITY_SET


# ── FULL universe — everything scanned (~900 tickers) ────────────────────────
# Extended stocks (not in QUALITY_SET) are only traded on ELITE signals.

ASX_UNIVERSE: list[str] = QUALITY_ASX + [
    # ── Mid Cap extension ─────────────────────────────────────────────────────
    "ABB.AX", "ABC.AX", "ACL.AX", "ADH.AX", "AEF.AX", "AFL.AX", "AHX.AX",
    "AIZ.AX", "AKE.AX", "ALK.AX", "AOF.AX",
    "ARF.AX", "ATL.AX", "AX1.AX",
    "BAP.AX", "BGA.AX", "BPT.AX",
    "CNU.AX", "CNI.AX",
    "CXO.AX", "DEG.AX", "DGL.AX", "DRR.AX", "EDV.AX", "EHE.AX",
    "EML.AX", "EMR.AX", "GDF.AX", "GEM.AX", "GNX.AX",
    "GTK.AX", "HSN.AX", "HMD.AX", "IDX.AX",
    "INA.AX", "IRI.AX", "JLG.AX", "KAR.AX", "LFG.AX",
    "LGI.AX", "LNK.AX", "MLD.AX", "MNF.AX",
    "MPB.AX", "MPW.AX", "MRM.AX", "MSB.AX", "MVF.AX", "NEA.AX",
    "NIC.AX", "OML.AX", "PAC.AX",
    "PRG.AX", "PTM.AX", "RBL.AX", "SLC.AX",
    "SSM.AX", "SYD.AX", "THL.AX",
    "TUA.AX", "VGI.AX", "WSA.AX",
    # ── Small / Growth Cap — ELITE-only ───────────────────────────────────────
    "4DS.AX", "29M.AX", "360.AX", "ABY.AX", "ACQ.AX", "ADA.AX", "AFT.AX", "AGE.AX",
    "AGG.AX", "AIR.AX", "AIS.AX", "ALT.AX", "AMI.AX", "ANP.AX", "AQZ.AX",
    "ARU.AX", "ASB.AX", "ASG.AX", "ATG.AX", "AUB.AX", "AUC.AX", "AWC.AX",
    "AZS.AX", "BCB.AX", "BEX.AX", "BFG.AX", "BGL.AX", "BIO.AX", "BKL.AX", "BMN.AX",
    "BNL.AX", "BPH.AX", "BRE.AX", "BSL.AX", "BTH.AX", "CAN.AX", "CAT.AX",
    "CAZ.AX", "CBR.AX", "CCX.AX", "CDA.AX", "CDX.AX", "CEZ.AX", "CGC.AX", "CGL.AX",
    "CGS.AX", "CLH.AX", "CLT.AX", "CML.AX", "CMW.AX", "CNB.AX", "COE.AX", "COG.AX",
    "CRD.AX", "CRN.AX", "CST.AX", "CTP.AX", "CUP.AX", "CVN.AX", "CXL.AX", "CYC.AX",
    "DAL.AX", "DAM.AX", "DDR.AX", "DGO.AX", "DLM.AX", "DOR.AX", "DRO.AX", "DSK.AX",
    "DUG.AX", "DVP.AX", "DWS.AX", "EBR.AX", "ECF.AX", "ECX.AX", "EFE.AX", "EGH.AX",
    "EGN.AX", "ELA.AX", "ELO.AX", "EMN.AX", "EMV.AX", "ENR.AX", "EPD.AX", "EQT.AX",
    "ERA.AX", "ERD.AX", "ERX.AX", "ESS.AX", "ETL.AX", "EUC.AX", "EXP.AX", "FAR.AX",
    "FBU.AX", "FEX.AX", "FLN.AX", "FNP.AX", "FOR.AX", "FPH.AX",
    "G1A.AX", "GBT.AX", "GDG.AX", "GEL.AX", "GLB.AX", "GMA.AX", "GMD.AX", "GMV.AX",
    "GNM.AX", "GOZ.AX", "GRR.AX", "GS1.AX", "GSW.AX", "GTN.AX", "GXY.AX",
    "HAV.AX", "HCW.AX", "HDN.AX", "HFR.AX", "HLA.AX", "HLO.AX", "HPI.AX",
    "HRL.AX", "HUO.AX", "HXG.AX", "IAA.AX", "ICQ.AX", "IFM.AX", "IFN.AX",
    "IGL.AX", "IIL.AX", "ILQ.AX", "IMB.AX", "ING.AX", "INR.AX", "IOO.AX", "IRM.AX",
    "ISG.AX", "ITD.AX", "IVZ.AX", "JAN.AX", "JMS.AX", "KAM.AX", "KGD.AX", "KGN.AX",
    "KIN.AX", "KME.AX", "KMT.AX", "LM8.AX", "LME.AX", "LPD.AX", "LRK.AX", "LSF.AX",
    "LYL.AX", "MAF.AX", "MBK.AX", "MCP.AX", "MDR.AX", "MEB.AX", "MEZ.AX", "MFG.AX",
    "MGL.AX", "MHJ.AX", "MI1.AX", "MIO.AX", "MIX.AX", "MMS.AX", "MNY.AX",
    "MOY.AX", "MQ1.AX", "MRF.AX", "MTR.AX", "MWY.AX", "MXI.AX", "MYX.AX", "NAC.AX",
    "NBL.AX", "NCK.AX", "NDO.AX", "NHC.AX", "NMT.AX", "NTO.AX", "NVX.AX", "NWL.AX",
    "NWS.AX", "NYR.AX", "OBL.AX", "OCL.AX", "OPH.AX", "OSL.AX", "PAR.AX", "PEK.AX",
    "PEN.AX", "PIL.AX", "PLY.AX", "PNI.AX", "PPH.AX", "PRN.AX", "PRX.AX", "PSQ.AX",
    "PXA.AX", "PYC.AX", "QIP.AX", "RCW.AX", "RDY.AX", "RFG.AX", "RLE.AX",
    "RNO.AX", "RPL.AX", "RSG.AX", "RVS.AX", "SCR.AX", "SDG.AX",
    "SER.AX", "SFX.AX", "SHJ.AX", "SHV.AX", "SIO.AX", "SIS.AX", "SKI.AX",
    "SMP.AX", "SOM.AX", "SPN.AX", "SRK.AX", "STB.AX", "STG.AX", "STP.AX",
    "STX.AX", "SVR.AX", "SWP.AX", "SXE.AX", "SYA.AX", "TAM.AX",
    "TBI.AX", "TBR.AX", "TCG.AX", "TDO.AX", "TEM.AX", "TIE.AX", "TIG.AX", "TKM.AX",
    "TLX.AX", "TMX.AX", "TNT.AX", "TON.AX", "TPW.AX", "TRF.AX", "TSI.AX",
    "TUL.AX", "TVN.AX", "UNI.AX", "USB.AX", "VAL.AX", "VAN.AX",
    "VEN.AX", "VHT.AX", "VML.AX", "VOC.AX", "VR1.AX", "VSR.AX", "WAM.AX",
    "WGX.AX", "WIA.AX", "WIN.AX", "WLD.AX", "WNR.AX", "WPR.AX", "WRK.AX",
    "WWI.AX", "XF1.AX", "Z1P.AX", "ZEL.AX", "ZIP.AX", "ZNO.AX",
]

US_UNIVERSE: list[str] = QUALITY_US + [
    # Extended US — ELITE-only
    "UBER", "LYFT", "ABNB", "DASH", "RIVN", "LCID", "NIO", "XPEV", "LI", "BLNK",
    "SQ", "PYPL", "HOOD", "SOFI", "AFRM", "UPST", "LC", "OPEN", "OPFI",
    "SE", "GRAB", "GOTU", "WIX", "BILL",
    "ZM", "ZI", "S", "TENB", "QLYS", "VRNS", "SAIL",
    "ESTC", "CFLT", "DOCN", "DOMO", "BOX", "SMAR", "ASAN", "MNDY",
    "RBLX", "U", "APP", "IRBT", "FIGS", "UA", "VFC",
    "PVH", "RL", "TPR", "CPRI", "HBI", "URBN",
    "BURL", "FIVE", "OLLI", "DLTR", "DG", "GO", "SFM",
    "WEN", "QSR", "CAKE", "BJRI", "RRGB", "DENN",
    "NFLX", "WBD", "PARA", "AMCX", "AMC", "CNK", "IMAX",
    "INTC", "SWKS", "QRVO", "LSCC", "MPWR", "SITM", "COHU", "ONTO", "ACLS",
    "UCTT", "ICHR", "MKSI", "CCMP", "ENTG",
    "PHM", "DHI", "LEN", "TOL", "NVR", "TMHC", "MDC", "LGIH", "GRBK", "MHO",
    "WH", "HLT", "MAR", "H", "PK", "HST", "RHP",
    "NLY", "AGNC", "TWO", "DX", "PMT", "MFA", "RWT", "RITM",
    "WBD", "FOXA", "FOX", "PARA", "WMG", "SPOT", "LYV",
    "F", "T", "MMM",
    "CELH", "HIMS", "ACMR", "IOT", "TDC", "GWRE",
    "PCVX", "GMED", "IRTC", "LIVN", "RCM",
    "NXST", "SBGI", "GTN", "TGNA",
    "PRTH", "RPAY", "EVERI", "EVTC", "CASS", "WEX", "FLYW", "GPN", "FIS",
    "FI", "NCR", "JKHY", "ACI", "PAY", "XPO", "WERN", "ECHO", "FWRD", "MRTN",
    "LPLA", "SSNC", "BEN", "IVZ", "VRTS", "TROW", "NTRS",
    "O", "NNN", "SPG", "PSA", "EQR", "MAA", "UDR", "AVB", "ESS",
    "EXR", "IRM", "WELL", "ARE", "DLR", "VTR",
]

# Deduplicate
ASX_UNIVERSE = list(dict.fromkeys(ASX_UNIVERSE))
US_UNIVERSE  = list(dict.fromkeys(US_UNIVERSE))


# ── Liquidity filter ───────────────────────────────────────────────────────────

_liquidity_cache: dict[str, bool]     = {}
_liquidity_cache_ts: datetime | None  = None
_CACHE_TTL_HOURS = 24


def filter_by_liquidity(
    tickers: list[str],
    min_avg_volume: int   = 200_000,
    min_price:      float = 0.50,
) -> list[str]:
    """Remove tickers below minimum liquidity thresholds. Results cached 24h."""
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
    Quality stocks trade at normal min_score; extended stocks trade ELITE-only.
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

    seen    = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    if apply_liquidity:
        deduped = filter_by_liquidity(deduped)

    log.info(f"Universe built: {len(deduped)} tickers ({', '.join(markets)})")
    return deduped
