import ccxt
import pandas as pd

exchange = ccxt.binance()

def get_ohlcv(symbol="BTC/USDT", timeframe="5m", limit=100):
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    return pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])

if __name__ == "__main__":
    df = get_ohlcv()
    print(df.tail())
