import ccxt
import pandas as pd
import numpy as np

exchange = ccxt.binance({"enableRateLimit": True})

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
TIMEFRAME = "5m"
LIMIT = 1000
RR = 2.0


def get_data(symbol):
    data = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)

    return pd.DataFrame(
        data,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )


def backtest(symbol):

    df = get_data(symbol)

    trades = []

    for i in range(60, len(df) - 20):

        current = df.iloc[i]

        price = current["close"]

        # Trend
        closes = df["close"].iloc[:i+1]

        ema20 = closes.ewm(
            span=20,
            adjust=False
        ).mean().iloc[-1]

        ema50 = closes.ewm(
            span=50,
            adjust=False
        ).mean().iloc[-1]

        trend_up = price > ema20 > ema50

        # Momentum
        move1h = (
            price / df["close"].iloc[i-12] - 1
        ) * 100

        move4h = (
            price / df["close"].iloc[i-48] - 1
        ) * 100

        # Relative volume
        avg_volume = df["volume"].iloc[i-20:i].mean()

        rv = (
            current["volume"] / avg_volume
            if avg_volume > 0
            else 0
        )

        # Recent high
        recent_high = df["high"].iloc[i-12:i].max()

        breakout = price > recent_high

        # Entry
        signal = (
            trend_up
            and move1h > 0
            and move4h > 0
            and rv >= 1.2
            and breakout
        )

        if not signal:
            continue

        entry = price

        recent_low = df["low"].iloc[i-12:i].min()

        stop = recent_low

        risk = entry - stop

        if risk <= 0:
            continue

        target = entry + risk * RR

        result = "OPEN"

        for j in range(i + 1, min(i + 20, len(df))):

            future = df.iloc[j]

            if future["low"] <= stop:
                result = "LOSS"
                break

            if future["high"] >= target:
                result = "WIN"
                break

        if result != "OPEN":
            trades.append(result)

    wins = trades.count("WIN")
    losses = trades.count("LOSS")

    total = wins + losses

    if total == 0:
        return symbol, 0, 0, 0, 0

    winrate = wins / total * 100

    # At 2R:
    # Win = +2R
    # Loss = -1R
    expectancy = (
        (wins * 2) - losses
    ) / total

    return (
        symbol,
        total,
        wins,
        losses,
        winrate,
        expectancy
    )


print("\n==============================")
print("STRATEGY BACKTEST")
print("==============================\n")

results = []

for symbol in SYMBOLS:

    try:
        result = backtest(symbol)
        results.append(result)

        print(
            f"{symbol:12} "
            f"Trades:{result[1]:4} "
            f"Wins:{result[2]:4} "
            f"Losses:{result[3]:4} "
            f"WinRate:{result[4]:6.2f}% "
            f"Expectancy:{result[5]:+.3f}R"
        )

    except Exception as e:
        print(symbol, "ERROR:", e)

print("\nBacktest complete.")
