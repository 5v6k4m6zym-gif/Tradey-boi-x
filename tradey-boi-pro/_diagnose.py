"""Quick signal-funnel diagnostic."""
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

import db.database as db; db.init_db()
import yfinance as yf, pandas as pd
from scanner.market_scanner import _compute_x_features, _normalize_columns

TICKERS = [
    "BHP.AX","CBA.AX","CSL.AX","WBC.AX","ANZ.AX","JBH.AX","XRO.AX","ALL.AX","REA.AX","GMG.AX",
    "AAPL","MSFT","NVDA","META","GOOGL","AMZN","V","MA","UNH","AVGO",
]

print("Downloading 20 tickers 2025-01-01 to 2026-06-30 ...")
raw = yf.download(
    " ".join(TICKERS), start="2025-01-01", end="2026-06-30",
    interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True,
)
print("Done.\n")

c = dict(ema=0, macd=0, rsi72=0, rising=0,
         vol12=0, vol13=0, vol15=0, candle=0, brk=0, score6=0, score8=0)

for ticker in TICKERS:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = _normalize_columns(raw[ticker].dropna(how="all"))
        else:
            df = _normalize_columns(raw.dropna(how="all"))
        feat = _compute_x_features(df)
        if feat is None or len(feat) < 2:
            continue

        for i in range(1, len(feat)):
            row  = feat.iloc[i]
            prev = feat.iloc[i - 1]

            e20 = float(row.get("ema20", 0)); e50 = float(row.get("ema50", 0))
            if e20 <= e50: continue
            pe20 = float(prev.get("ema20", 0)); pe50 = float(prev.get("ema50", 0))
            if pe20 <= pe50: continue
            c["ema"] += 1

            md = row.get("macd_diff"); pmd = prev.get("macd_diff")
            if not pd.isna(md) and float(md) <= 0: continue
            if not pd.isna(pmd) and float(pmd) <= 0: continue
            c["macd"] += 1

            rsi = float(row.get("rsi", 0)) if not pd.isna(row.get("rsi")) else 0
            if rsi >= 72 or rsi <= 25: continue
            c["rsi72"] += 1

            cl = float(row.get("Close", 0)); pcl = float(prev.get("Close", 0))
            if cl <= pcl: continue
            if float(row.get("ema20", 0)) <= float(prev.get("ema20", 0)): continue
            c["rising"] += 1

            vr = float(row.get("vol_ratio", 0)) if not pd.isna(row.get("vol_ratio")) else 0
            if vr >= 1.2: c["vol12"] += 1
            if vr >= 1.3: c["vol13"] += 1
            if vr >= 1.5: c["vol15"] += 1

            # From here require vol >= 1.2
            if vr < 1.2: continue

            h = float(row.get("High", cl)); l = float(row.get("Low", cl))
            rng = h - l
            if rng > 0 and (cl - l) / rng >= 0.60:
                c["candle"] += 1

            brk = bool(int(row.get("breakout", 0)))
            if brk: c["brk"] += 1

            # Heuristic score
            score = 0
            prob = 0.65
            if prob >= 0.70: score += 2
            elif prob >= 0.60: score += 1
            if brk:      score += 3
            if vr > 1.5: score += 2
            if 35 <= rsi <= 65: score += 2
            elif rsi < 70:      score += 1
            score += 1  # ema20 > ema50 always

            if score >= 6: c["score6"] += 1
            if score >= 8: c["score8"] += 1

    except Exception as exc:
        print(f"  {ticker}: {exc}")

print("Signal funnel (20 tickers, ~18 months):")
print(f"  EMA20>EMA50 both days        : {c['ema']:4d}")
print(f"  +MACD>0 both days            : {c['macd']:4d}")
print(f"  +RSI 25-72                   : {c['rsi72']:4d}")
print(f"  +Price & EMA rising          : {c['rising']:4d}")
print(f"    of which vol>=1.2          : {c['vol12']:4d}")
print(f"    of which vol>=1.3          : {c['vol13']:4d}")
print(f"    of which vol>=1.5          : {c['vol15']:4d}")
print(f"  (vol>=1.2) + candle quality  : {c['candle']:4d}")
print(f"  is_breakout (52-wk high)     : {c['brk']:4d}")
print(f"  heuristic score >= 6         : {c['score6']:4d}")
print(f"  heuristic score >= 8         : {c['score8']:4d}")
print()
print("Key insight: score>=8 requires breakout(+3)+prob>=0.70(+2)+vol>1.5(+2)+ema(+1)=8")
print("  or breakout(+3)+prob>=0.60(+1)+vol>1.5(+2)+RSI35-65(+2)=8")
