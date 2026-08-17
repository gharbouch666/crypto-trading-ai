import ccxt

exchange = ccxt.binance({"enableRateLimit": True})

markets = exchange.load_markets()
tickers = exchange.fetch_tickers()

results = []

for symbol, market in markets.items():
    if not symbol.endswith("/USDT") or not market["spot"] or not market["active"]:
        continue
    ticker = tickers.get(symbol)
    if not ticker or not ticker.get("quoteVolume"):
        continue
    volume_24h = ticker["quoteVolume"]
    if volume_24h < 5_000_000:
        continue
    change = ticker.get("percentage")
    if change is None:
        continue
    results.append((symbol, change, volume_24h))

results.sort(key=lambda x: x[1], reverse=True)

print("\nTOP LIQUID BINANCE MOMENTUM COINS\n")
for i, (symbol, change, volume) in enumerate(results[:20], 1):
    print(f"{i:2}. {symbol:15} {change:+7.2f}%   24h Volume: ${volume:,.0f}")
