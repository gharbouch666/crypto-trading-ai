import ccxt
import pandas as pd
import numpy as np

exchange = ccxt.binance({"enableRateLimit": True})
markets = exchange.load_markets()
tickers = exchange.fetch_tickers()

rows = []

for symbol, market in markets.items():
    if not symbol.endswith("/USDT") or not market["spot"] or not market["active"]:
        continue

    ticker = tickers.get(symbol)
    if not ticker or not ticker.get("quoteVolume") or ticker["quoteVolume"] < 10_000_000:
        continue

    try:
        candles = exchange.fetch_ohlcv(symbol, "5m", limit=50)
        if len(candles) < 50:
            continue

        df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume"])

        price_change = (df["close"].iloc[-1] / df["close"].iloc[-12] - 1) * 100
        volatility = df["close"].pct_change().std() * 100
        avg_volume = df["volume"].iloc[-21:-1].mean()
        current_volume = df["volume"].iloc[-1]
        relative_volume = current_volume / avg_volume if avg_volume else 0

        rows.append({
            "symbol": symbol,
            "24h_volume": ticker["quoteVolume"],
            "5h_change": price_change,
            "relative_volume": relative_volume,
            "volatility": volatility
        })

    except Exception:
        continue

df = pd.DataFrame(rows)

df = df.sort_values("5h_change", ascending=False)

print("\nMARKET DATA - TOP 25 BY 5H MOMENTUM\n")
print(df.head(25).to_string(index=False))

df.to_csv("data/market_snapshot.csv", index=False)

print(f"\nScanned: {len(df)} liquid USDT spot pairs")
print("Saved: data/market_snapshot.csv")

