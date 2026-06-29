import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# ----------------------------
# DISCORD (SECURE)
# ----------------------------
DISCORD_WEBHOOK = os.getenv("discordwebhook")

last_alerts = {}  # cooldown tracker (in-memory)

def send_discord(message):
    if not DISCORD_WEBHOOK:
        print("Webhook missing")
        return

    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message})
    except Exception as e:
        print("Discord error:", e)

# ----------------------------
# WATCHLIST
# ----------------------------
WATCHLIST = [
    "BHP.AX", "CBA.AX", "FMG.AX", "RIO.AX",
    "NST.AX", "CXO.AX", "LTR.AX", "PDN.AX",
    "TSLA", "AAPL", "NVDA", "MSFT"
]

# ----------------------------
# FEATURES
# ----------------------------
def make_features(df):
    df = df.copy()

    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()

    df["vol_change"] = df["Volume"].pct_change(5)

    df = df.dropna()
    return df

# ----------------------------
# TRAIN AI MODEL
# ----------------------------
def train_model(ticker="AAPL"):
    df = yf.Ticker(ticker).history(period="1y")
    df = make_features(df)

    df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
    df["target"] = (df["future_return"] > 0.05).astype(int)

    features = ["ret_5", "ret_10", "ma_10", "ma_30", "vol_change"]

    df = df.dropna()

    X = df[features]
    y = df["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    return model, features

# ----------------------------
# PREDICT
# ----------------------------
def predict(model, features, row):
    return model.predict_proba([row])[0][1]

# ----------------------------
# TREND FILTER
# ----------------------------
def trend_filter(df):
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]

    return ma20 > ma50  # bullish only

# ----------------------------
# SIGNAL CLASSIFIER
# ----------------------------
def classify(prob, trend_ok):
    if not trend_ok:
        return "IGNORE"

    if prob >= 0.75:
        return "BUY"
    elif prob >= 0.60:
        return "WATCH"
    else:
        return "IGNORE"

# ----------------------------
# COOLDOWN CHECK
# ----------------------------
def cooldown_ok(ticker):
    now = datetime.now()

    if ticker not in last_alerts:
        return True

    last_time = last_alerts[ticker]
    return now - last_time > timedelta(hours=6)

# ----------------------------
# SCAN MARKET
# ----------------------------
def scan(model, features):
    results = []

    for ticker in WATCHLIST:
        try:
            df = yf.Ticker(ticker).history(period="6mo")
            df = make_features(df)

            if len(df) < 60:
                continue

            if not trend_filter(df):
                continue

            latest = df.iloc[-1]

            row = [
                latest["ret_5"],
                latest["ret_10"],
                latest["ma_10"],
                latest["ma_30"],
                latest["vol_change"]
            ]

            prob = predict(model, features, row)

            results.append({
                "ticker": ticker,
                "price": round(latest["Close"], 2),
                "prob": round(prob, 3)
            })

        except Exception as e:
            print(f"{ticker} error: {e}")

    results.sort(key=lambda x: x["prob"], reverse=True)
    return results

# ----------------------------
# MAIN ENGINE
# ----------------------------
def main():
    print("\n=======================")
    print(" TRADEY BOI X AI v2 ")
    print("=======================\n")

    print("Training model...\n")
    model, features = train_model("AAPL")

    print("Scanning market...\n")
    results = scan(model, features)

    for r in results[:10]:

        signal = classify(r["prob"], True)

        msg = f"{r['ticker']} | Price: {r['price']} | Prob: {r['prob']*100:.1f}% | {signal}"

        print(msg)

        # ALERT SYSTEM
        if signal in ["BUY", "WATCH"] and r["prob"] >= 0.60 and cooldown_ok(r["ticker"]):

            send_discord("🚨 TRADEY BOI X AI SIGNAL\n" + msg)

            last_alerts[r["ticker"]] = datetime.now()

    print("\nDone.\n")

if __name__ == "__main__":
    main()