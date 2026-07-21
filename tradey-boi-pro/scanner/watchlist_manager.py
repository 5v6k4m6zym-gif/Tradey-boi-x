"""
Watchlist manager for Tradey Boi Pro.
Manages ASX, US, and custom ticker lists. Much larger universe than Tradey Boi X.
All stored in SQLite so user can edit from the dashboard.
"""
from __future__ import annotations
import db.database as db

# ── Built-in watchlists ────────────────────────────────────────────────────────

ASX_TOP200 = [
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","MQG.AX",
    "TLS.AX","RIO.AX","FMG.AX","GMG.AX","WOW.AX","TCL.AX","REA.AX","SHL.AX",
    "COH.AX","ALL.AX","RMD.AX","QBE.AX","IAG.AX","SUN.AX","ORG.AX","AGL.AX",
    "AMP.AX","ASX.AX","BOQ.AX","BXB.AX","CAR.AX","CCP.AX","CGF.AX","CHC.AX",
    "CPU.AX","CWY.AX","DXS.AX","ELD.AX","EVN.AX","GPT.AX","HVN.AX","IEL.AX",
    "ILU.AX","IPL.AX","JBH.AX","LNK.AX","MGR.AX","MIN.AX","MPL.AX","MTS.AX",
    "NCM.AX","NHF.AX","NST.AX","NUF.AX","NXT.AX","ORA.AX","OZL.AX","PLS.AX",
    "PME.AX","PTM.AX","RHC.AX","SAR.AX","SCG.AX","SFR.AX","SGM.AX","SGP.AX",
    "STO.AX","TAH.AX","TWE.AX","VCX.AX","VEA.AX","WHC.AX","WPL.AX","WSA.AX",
    "XRO.AX","29M.AX","360.AX","A2M.AX","ABB.AX","ABC.AX","ACF.AX","ACL.AX",
    "ADH.AX","AEF.AX","AEI.AX","AFL.AX","AGG.AX","AHX.AX","AIZ.AX","AKE.AX",
    "ALK.AX","ALQ.AX","ALU.AX","AMC.AX","ANN.AX","AOF.AX","APA.AX","APE.AX",
    "APX.AX","ARB.AX","ARF.AX","ARU.AX","ASB.AX","ASG.AX","ATL.AX","AVZ.AX",
    "AX1.AX","AZJ.AX","BAP.AX","BGA.AX","BKW.AX","BLD.AX","BPT.AX","BWP.AX",
    "CAM.AX","CGF.AX","CLW.AX","CMM.AX","CNU.AX","COF.AX","CQR.AX","CTD.AX",
    "CXO.AX","DCN.AX","DEG.AX","DGL.AX","DHG.AX","DMP.AX","DOW.AX","DRR.AX",
    "DUG.AX","EBO.AX","EDV.AX","EHE.AX","EML.AX","EMR.AX","GDF.AX","GEM.AX",
    "GNC.AX","GNX.AX","GOR.AX","GTK.AX","GUD.AX","GWA.AX","HLS.AX","HMC.AX",
    "HPI.AX","HSN.AX","IDX.AX","IGO.AX","INA.AX","ING.AX","IRI.AX","IVZ.AX",
    "JAN.AX","JLG.AX","KAR.AX","LFG.AX","LGI.AX","LLC.AX","LOT.AX","LYC.AX",
    "MGX.AX","MLD.AX","MNF.AX","MRM.AX","MSB.AX","MVF.AX","MYX.AX","NEA.AX",
    "NEC.AX","NIC.AX","NTO.AX","NVX.AX","NWL.AX","OBL.AX","OCL.AX","OFX.AX",
    "OML.AX","OPY.AX","PAC.AX","PAR.AX","PDN.AX","PEK.AX","PGH.AX","PIL.AX",
    "PMV.AX","PPT.AX","PRG.AX","PXA.AX","QAN.AX","QIP.AX","RBL.AX","RDY.AX",
]

SP500_SAMPLE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","UNH","XOM","JNJ",
    "JPM","V","PG","MA","HD","CVX","MRK","ABBV","PEP","KO","COST","AVGO",
    "WMT","TMO","MCD","DIS","ACN","CSCO","ABT","NEE","DHR","BMY","LIN",
    "ADBE","NKE","TXN","PM","CRM","QCOM","UPS","HON","AMGN","RTX","LOW",
    "INTU","SPGI","GS","BLK","CAT","BA","ELV","AMAT","ISRG","NOW","AMD",
    "BKNG","DE","AXP","MDLZ","ADI","CB","GILD","REGN","ZTS","LMT","GE",
    "MMM","C","USB","MO","F","T","VZ","PFE","CVS","SYK","MU","BSX",
    "PLD","AMT","CCI","EQIX","AON","ICE","CME","NDAQ","UBER","SQ","PYPL",
    "SHOP","ROKU","ZM","SPOT","COIN","SNAP","RBLX","HOOD","SOFI","PLTR",
    "CRWD","DDOG","NET","SNOW","U","GTLB","MDB","ESTC","ZS","OKTA","HUBS",
    "TTD","BILL","DOCU","DOCN","CFLT","MNDY","IOT","WIX","AMPL","APP",
    "ELF","CELH","DUOL","SMAR","PCVX","GMED","IRTC","TREX","LULU","SKX",
]


def _db_key(market: str) -> str:
    return f"watchlist_{market.upper()}"


def get_watchlist(market: str) -> list[str]:
    stored = db.get_setting(_db_key(market))
    if stored is not None:
        return stored
    if market.upper() == "ASX":
        return ASX_TOP200
    if market.upper() == "US":
        return SP500_SAMPLE
    return []


def set_watchlist(market: str, tickers: list[str]):
    clean = [t.strip().upper() for t in tickers if t.strip()]
    db.set_setting(_db_key(market), clean)


def add_tickers(market: str, tickers: list[str]):
    current  = get_watchlist(market)
    combined = list(dict.fromkeys(current + [t.strip().upper() for t in tickers]))
    set_watchlist(market, combined)


def remove_tickers(market: str, tickers: list[str]):
    to_remove = {t.strip().upper() for t in tickers}
    set_watchlist(market, [t for t in get_watchlist(market) if t not in to_remove])


def get_all_active_tickers() -> list[str]:
    enabled = db.get_setting("enabled_markets") or ["ASX", "US"]
    tickers = []
    for mkt in enabled:
        tickers.extend(get_watchlist(mkt))
    tickers.extend(get_watchlist("CUSTOM"))
    seen = set()
    out  = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def set_enabled_markets(markets: list[str]):
    db.set_setting("enabled_markets", [m.upper() for m in markets])
