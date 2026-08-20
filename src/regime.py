import ccxt
import pandas as pd

exchange = ccxt.binance({"enableRateLimit": True})

def regime(symbol="BTC/USDT"):
    candles = exchange.fetch_ohlcv(symbol, "1h", limit=100)
    closes = pd.Series([c[4] for c in candles], dtype=float)

    ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
    change24h = (closes.iloc[-1] / closes.iloc[-25] - 1) * 100

    if closes.iloc[-1] > ema20 > ema50 and change24h > 1:
        return "RISK-ON"
    if closes.iloc[-1] < ema20 < ema50 and change24h < -1:
        return "RISK-OFF"
    return "NEUTRAL"

if __name__ == "__main__":
    print("BTC MARKET REGIME:", regime())
