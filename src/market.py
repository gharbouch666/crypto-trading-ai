import ccxt

exchange = ccxt.binance()

def get_price(symbol="BTC/USDT"):
    ticker = exchange.fetch_ticker(symbol)
    return ticker["last"]

if __name__ == "__main__":
    print(f"BTC/USDT: {get_price():,.2f}")
