import ccxt

exchange = ccxt.binance({"enableRateLimit": True})

markets = exchange.load_markets()
symbols = [s for s in markets if s.endswith("/USDT") and markets[s]["spot"] and markets[s]["active"]]

results = []

for symbol in symbols:
    try:
        candles = exchange.fetch_ohlcv(symbol, "5m", limit=6)
        if len(candles) < 6:
            continue
        old_price = candles[0][4]
        current_price = candles[-1][4]
        volume = sum(c[5] for c in candles)
        change = ((current_price - old_price) / old_price) * 100
        results.append((symbol, change, volume))
    except Exception:
        continue

results.sort(key=lambda x: x[1], reverse=True)

print("\nTOP 10 MOMENTUM COINS\n")
for i, (symbol, change, volume) in enumerate(results[:10], 1):
    print(f"{i:2}. {symbol:15} {change:+7.2f}%   Volume: {volume:,.2f}")
