"""
Dynamic stock universe builder for Tradey Boi Pro.

Covers ~900 tickers (ASX + US) — far broader than Tradey Boi X's fixed watchlist.
Applies liquidity filtering so only tradeable stocks are scanned.
Universe is refreshed periodically; custom additions persist in SQLite.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import db.database as db

log = logging.getLogger("Universe")

# ── ASX — comprehensive liquid universe (~450 tickers) ────────────────────────
# Covers ASX 300 + midcap growth + resources + tech
ASX_UNIVERSE: list[str] = [
    # ── Mega/Large Cap ────────────────────────────────────────────────────────
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","MQG.AX",
    "TLS.AX","RIO.AX","FMG.AX","GMG.AX","WOW.AX","TCL.AX","REA.AX","SHL.AX",
    "COH.AX","ALL.AX","RMD.AX","QBE.AX","IAG.AX","SUN.AX","ORG.AX","AGL.AX",
    "AMP.AX","ASX.AX","BXB.AX","CPU.AX","CWY.AX","DXS.AX","GPT.AX","ILU.AX",
    "IPL.AX","MGR.AX","MPL.AX","MTS.AX","NCM.AX","NST.AX","ORA.AX","SCG.AX",
    "SGP.AX","STO.AX","TAH.AX","TWE.AX","VCX.AX","WPL.AX","XRO.AX",
    # ── Mid Cap ───────────────────────────────────────────────────────────────
    "A2M.AX","ABB.AX","ABC.AX","ACL.AX","ADH.AX","AEF.AX","AFL.AX","AHX.AX",
    "AIZ.AX","AKE.AX","ALK.AX","ALQ.AX","ALU.AX","AMC.AX","ANN.AX","AOF.AX",
    "APA.AX","APE.AX","APX.AX","ARB.AX","ARF.AX","ATL.AX","AVZ.AX","AX1.AX",
    "AZJ.AX","BAP.AX","BGA.AX","BKW.AX","BLD.AX","BOQ.AX","BPT.AX","BWP.AX",
    "CAR.AX","CCP.AX","CGF.AX","CHC.AX","CLW.AX","CMM.AX","CNI.AX","CNU.AX",
    "COF.AX","CQR.AX","CTD.AX","CXO.AX","DCN.AX","DEG.AX","DGL.AX","DHG.AX",
    "DMP.AX","DNX.AX","DOW.AX","DRR.AX","EBO.AX","EDV.AX","EHE.AX","ELD.AX",
    "EML.AX","EMR.AX","EVN.AX","GDF.AX","GEM.AX","GNC.AX","GNX.AX","GOR.AX",
    "GTK.AX","GUD.AX","GWA.AX","HLS.AX","HMC.AX","HSN.AX","HVN.AX","IDX.AX",
    "IEL.AX","IGO.AX","INA.AX","IRI.AX","JBH.AX","JLG.AX","KAR.AX","LFG.AX",
    "LGI.AX","LLC.AX","LOT.AX","LNK.AX","LYC.AX","MGX.AX","MLD.AX","MNF.AX",
    "MPB.AX","MPW.AX","MRM.AX","MSB.AX","MVF.AX","NEA.AX","NEC.AX","NHF.AX",
    "NIC.AX","NUF.AX","NXT.AX","OFX.AX","OML.AX","OPY.AX","OZL.AX","PAC.AX",
    "PDN.AX","PGH.AX","PLS.AX","PME.AX","PMV.AX","PPT.AX","PRG.AX","PTM.AX",
    "QAN.AX","RBL.AX","RHC.AX","SAR.AX","SFR.AX","SGM.AX","SKC.AX","SLR.AX",
    "SPK.AX","SRG.AX","SSM.AX","STO.AX","SWM.AX","SXL.AX","SYD.AX","THL.AX",
    "TPG.AX","TUA.AX","VEA.AX","VGI.AX","WAF.AX","WEB.AX","WHC.AX","WSA.AX",
    # ── Small/Growth Cap ──────────────────────────────────────────────────────
    "4DS.AX","29M.AX","360.AX","ABY.AX","ACQ.AX","ADA.AX","AFT.AX","AGE.AX",
    "AGG.AX","AIR.AX","AIS.AX","AJY.AX","ALT.AX","AMI.AX","ANP.AX","AQZ.AX",
    "ARU.AX","ASB.AX","ASG.AX","ATG.AX","AUB.AX","AUC.AX","AWC.AX","AXP.AX",
    "AZS.AX","BCB.AX","BEX.AX","BFG.AX","BGL.AX","BIO.AX","BKL.AX","BMN.AX",
    "BNL.AX","BPH.AX","BRE.AX","BSL.AX","BTH.AX","CAM.AX","CAN.AX","CAT.AX",
    "CAZ.AX","CBR.AX","CCX.AX","CDA.AX","CDX.AX","CEZ.AX","CGC.AX","CGL.AX",
    "CGS.AX","CLH.AX","CLT.AX","CML.AX","CMW.AX","CNB.AX","COE.AX","COG.AX",
    "CRD.AX","CRN.AX","CST.AX","CTP.AX","CUP.AX","CVN.AX","CXL.AX","CYC.AX",
    "DAL.AX","DAM.AX","DDR.AX","DGO.AX","DLM.AX","DOR.AX","DRO.AX","DSK.AX",
    "DUG.AX","DVP.AX","DWS.AX","EBR.AX","ECF.AX","ECX.AX","EFE.AX","EGH.AX",
    "EGN.AX","ELA.AX","ELO.AX","EMN.AX","EMV.AX","ENR.AX","EPD.AX","EQT.AX",
    "ERA.AX","ERD.AX","ERX.AX","ESS.AX","ETL.AX","EUC.AX","EXP.AX","FAR.AX",
    "FBU.AX","FEX.AX","FLN.AX","FLT.AX","FMG.AX","FNP.AX","FOR.AX","FPH.AX",
    "G1A.AX","GBT.AX","GDG.AX","GEL.AX","GLB.AX","GMA.AX","GMD.AX","GMV.AX",
    "GNM.AX","GOZ.AX","GPT.AX","GRR.AX","GS1.AX","GSW.AX","GTN.AX","GXY.AX",
    "HAV.AX","HCW.AX","HDN.AX","HFR.AX","HLA.AX","HLO.AX","HMD.AX","HPI.AX",
    "HRL.AX","HUO.AX","HVN.AX","HXG.AX","IAA.AX","ICQ.AX","IFM.AX","IFN.AX",
    "IGL.AX","IIL.AX","ILQ.AX","IMB.AX","ING.AX","INR.AX","IOO.AX","IRM.AX",
    "ISG.AX","ITD.AX","IVZ.AX","JAN.AX","JMS.AX","KAM.AX","KGD.AX","KGN.AX",
    "KIN.AX","KME.AX","KMT.AX","LM8.AX","LME.AX","LPD.AX","LRK.AX","LSF.AX",
    "LYL.AX","MAF.AX","MBK.AX","MCP.AX","MDR.AX","MEB.AX","MEZ.AX","MFG.AX",
    "MGL.AX","MHJ.AX","MI1.AX","MIN.AX","MIO.AX","MIX.AX","MMS.AX","MNY.AX",
    "MOY.AX","MQ1.AX","MRF.AX","MTR.AX","MWY.AX","MXI.AX","MYX.AX","NAC.AX",
    "NBL.AX","NCK.AX","NDO.AX","NHC.AX","NMT.AX","NTO.AX","NVX.AX","NWL.AX",
    "NWS.AX","NYR.AX","OBL.AX","OCL.AX","OPH.AX","OSL.AX","PAR.AX","PEK.AX",
    "PEN.AX","PIL.AX","PLY.AX","PNI.AX","PPH.AX","PRN.AX","PRX.AX","PSQ.AX",
    "PXA.AX","PYC.AX","QIP.AX","RCW.AX","RDY.AX","RFG.AX","RIO.AX","RLE.AX",
    "RNO.AX","RPL.AX","RSG.AX","RVS.AX","RWC.AX","SCR.AX","SDG.AX","SEK.AX",
    "SER.AX","SFX.AX","SHJ.AX","SHV.AX","SIO.AX","SIS.AX","SKI.AX","SLC.AX",
    "SMP.AX","SOM.AX","SPN.AX","SRK.AX","STB.AX","STG.AX","STO.AX","STP.AX",
    "STX.AX","SUL.AX","SUN.AX","SVR.AX","SWP.AX","SXE.AX","SYA.AX","TAM.AX",
    "TBI.AX","TBR.AX","TCG.AX","TDO.AX","TEM.AX","TIE.AX","TIG.AX","TKM.AX",
    "TLX.AX","TMX.AX","TNT.AX","TON.AX","TPW.AX","TRF.AX","TRS.AX","TSI.AX",
    "TUA.AX","TUL.AX","TVN.AX","UNI.AX","USB.AX","VAL.AX","VAN.AX","VCX.AX",
    "VEN.AX","VGI.AX","VHT.AX","VML.AX","VOC.AX","VR1.AX","VSR.AX","WAM.AX",
    "WGX.AX","WIA.AX","WIN.AX","WLD.AX","WNR.AX","WOR.AX","WPR.AX","WRK.AX",
    "WTC.AX","WWI.AX","XF1.AX","YAL.AX","Z1P.AX","ZEL.AX","ZIP.AX","ZNO.AX",
]

# ── US — S&P 500 + Russell 1000 extension (~600 tickers) ─────────────────────
US_UNIVERSE: list[str] = [
    # S&P 500 core
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","UNH","XOM","JNJ",
    "JPM","V","PG","MA","HD","CVX","MRK","ABBV","PEP","KO","COST","AVGO",
    "WMT","TMO","MCD","DIS","ACN","CSCO","ABT","NEE","DHR","BMY","LIN",
    "ADBE","NKE","TXN","PM","CRM","QCOM","UPS","HON","AMGN","RTX","LOW",
    "INTU","SPGI","GS","BLK","CAT","BA","ELV","AMAT","ISRG","NOW","AMD",
    "BKNG","DE","AXP","MDLZ","ADI","CB","GILD","REGN","ZTS","LMT","GE",
    "MMM","C","USB","MO","F","T","VZ","PFE","CVS","SYK","MU","BSX",
    "PLD","AMT","CCI","EQIX","AON","ICE","CME","NDAQ","EW","A","IDXX",
    "ROP","MCO","IT","CTAS","CDNS","KLAC","MCHP","PAYX","VRSK","FTNT",
    "ODFL","FAST","BIIB","VRTX","ILMN","MRNA","DXCM","IQV","MTD","WST",
    "ALGN","PODD","TDY","TECH","NTRA","BMRN","ALNY","PTGX","SGEN","EXAS",
    "EXR","IRM","WELL","ARE","DLR","VTR","O","NNN","SPG","PSA","EQR",
    "MAA","UDR","AVB","ESS","CPT","AIV","SUI","ELS","UE","REG","FRT",
    "BXP","KIM","NLY","AGNC","TWO","DX","PMT","NYMT","MFA","RWT","RITM",
    "JPM","BAC","WFC","C","GS","MS","BK","STT","SCHW","AMP","PFG","PRU",
    "MET","AFL","ALL","TRV","HIG","CNA","EG","WRB","RLI","ACGL","RE",
    "MCK","ABC","CAH","CVS","CI","HUM","MOH","CNC","WBD","FOXA","FOX",
    "PARA","WMG","SPOT","LYV","SIX","MTN","VAIL","SKY","PHM","DHI","LEN",
    "TOL","NVR","TMHC","MDC","LGIH","GRBK","MHO","STR","WH","HLT","MAR",
    "H","PK","HST","RHP","SHO","APLE","PEB","XHR","CLDT","BRAEMAR","INN",
    # Growth / Tech
    "UBER","LYFT","ABNB","DASH","RIVN","LCID","NIO","XPEV","LI","BLNK",
    "SQ","PYPL","HOOD","SOFI","AFRM","UPST","LEND","LC","OPEN","OPFI",
    "SHOP","MELI","SE","GRAB","GOTU","WIX","WDAY","VEEV","HUBS","BILL",
    "ZM","ZI","DDOG","SNOW","PLTR","CRWD","NET","ZS","OKTA","FTNT","S",
    "PANW","CYLC","TENB","QLYS","VRNS","SAIL","RDWR","CYBE","OSPN","SCWX",
    "MDB","ESTC","CFLT","GTLB","DOCN","DOMO","BOX","SMAR","ASAN","MNDY",
    "RBLX","U","UNITY","APP","IRBT","FIGS","LULU","SKX","NKE","UA","VFC",
    "PVH","RL","TPR","CPRI","KORS","KATE","COH","TIF","HBI","URBN","ROST",
    "TJX","BURL","FIVE","OLLI","DLTR","DG","GO","SFM","CHEF","FRSH","EAT",
    "DRI","CMG","YUM","MCD","WEN","QSR","CAKE","TXRH","BJRI","RRGB","DENN",
    "NFLX","WBD","PARA","AMCX","AMC","CNK","IMAX","LGF-A","SONY","NTDOY",
    "AAPL","MSFT","GOOGL","META","AMZN","NVDA","AMD","INTC","QCOM","AVGO",
    "TXN","MCHP","SWKS","QRVO","LSCC","MPWR","SITM","COHU","ONTO","ACLS",
    "UCTT","ICHR","MKSI","CCMP","ENTG","AXTA","PPG","SHW","RPM","AZEK",
    "BLD","BECN","IBP","BLDR","LBX","GMS","TREX","FBHS","MAS","AWI","CSL",
    "DOOR","PGTI","MTRN","PATK","SSD","AMWD","NCI","APOG","WMS","REXN",
    "GTLS","FLOW","IDEX","RXN","GWW","MSM","FAST","WSO","AIT","DGX","SWK",
    "SNA","KMT","TKR","ATU","ROLL","WTS","CFX","GNSS","TGI","HAYN","CW",
    "HEICO","TDG","SPR","KTOS","AJRD","MRCY","CACI","KEYW","PSN","ESNT",
    "NMIH","PFSI","GHLD","UWM","RATE","RDFN","Z","ZG","OPEN","OPAD","EXPI",
    # High-momentum / growth
    "CELH","ELF","DUOL","IBKR","LPLA","SCHW","MORN","ENV","TROW","NTRS",
    "SSNC","BEN","IVZ","VRTS","CLOU","HIMS","ACMR","IOT","TDC","GWRE",
    "PCVX","GMED","IRTC","LIVN","RCM","HMSY","NXST","SBGI","GTN","TGNA",
    "PRTH","RPAY","PAYA","EVERI","EVTC","CASS","WEX","FLYW","GPN","FIS",
    "FI","NCR","JKHY","ACI","PAY","FORM","CCRD","FOUR","XPO","SAIA","ODFL",
    "JBHT","WERN","CHRW","EXPD","ECHO","FWRD","MRTN","USFC","HUBG","TFII",
]

# Deduplicate
US_UNIVERSE  = list(dict.fromkeys(US_UNIVERSE))
ASX_UNIVERSE = list(dict.fromkeys(ASX_UNIVERSE))


# ── Liquidity filter ───────────────────────────────────────────────────────────

_liquidity_cache: dict[str, bool]     = {}
_liquidity_cache_ts: datetime | None  = None
_CACHE_TTL_HOURS = 24


def filter_by_liquidity(
    tickers: list[str],
    min_avg_volume: int = 100_000,
    min_price:      float = 0.20,
) -> list[str]:
    """
    Remove tickers that don't meet minimum liquidity thresholds.
    Results are cached for 24h to avoid hammering yfinance.
    Runs a fast check: just needs 5-day data.
    """
    global _liquidity_cache, _liquidity_cache_ts

    # Cache validity
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
                        if df.empty:
                            new_cache[t] = False
                            continue
                        avg_vol = float(df["Volume"].mean())
                        last_price = float(df["Close"].iloc[-1])
                        new_cache[t] = (avg_vol >= min_avg_volume and last_price >= min_price)
                    except Exception:
                        new_cache[t] = True  # keep on error
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

    # Start from custom DB overrides, fall back to built-in lists
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

    # Deduplicate
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    if apply_liquidity:
        deduped = filter_by_liquidity(deduped)

    log.info(f"Universe built: {len(deduped)} tickers ({', '.join(markets)})")
    return deduped
