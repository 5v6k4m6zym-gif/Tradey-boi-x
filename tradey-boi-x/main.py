import yfinance as yf


def get_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period="1d")

    print(f"\n=== {ticker.upper()} ===")
    print(f"Name:          {info.get('longName', 'N/A')}")
    print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
    print(f"Open:          ${info.get('open', 'N/A')}")
    print(f"Day High:      ${info.get('dayHigh', 'N/A')}")
    print(f"Day Low:       ${info.get('dayLow', 'N/A')}")
    print(f"Volume:        {info.get('volume', 'N/A'):,}")
    print(f"Market Cap:    ${info.get('marketCap', 'N/A'):,}")
    print(f"52W High:      ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"52W Low:       ${info.get('fiftyTwoWeekLow', 'N/A')}")


def main():
    print("Welcome to Tradey Boi X")
    print("=======================")

    tickers = ["AAPL", "TSLA", "MSFT"]

    for ticker in tickers:
        get_stock_info(ticker)


if __name__ == "__main__":
    main()
