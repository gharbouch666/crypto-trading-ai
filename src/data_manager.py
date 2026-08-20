import ccxt
import pandas as pd
import os
import time

EXCHANGE = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 30000
})

DATA_DIR = "data/history"
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT"
]

TIMEFRAME = "15m"
DAYS = 180
LIMIT = 1000


def filename(symbol):
    return symbol.replace("/", "_") + f"_{TIMEFRAME}.csv"


def download_symbol(symbol):
    path = os.path.join(DATA_DIR, filename(symbol))

    # USE EXISTING DATA
    if os.path.exists(path):
        df = pd.read_csv(path)

        if len(df) > 1000:
            print(f"{symbol}: using cached data ({len(df):,} candles)")
            return df

    print(f"{symbol}: downloading {DAYS} days...")

    since = EXCHANGE.milliseconds() - DAYS * 24 * 60 * 60 * 1000

    candles = []

    while since < EXCHANGE.milliseconds():
        try:
            batch = EXCHANGE.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=since,
                limit=LIMIT
            )

            if not batch:
                break

            candles.extend(batch)

            since = batch[-1][0] + 1

            print(
                f"{symbol}: {len(candles):,} candles",
                end="\r"
            )

            time.sleep(0.15)

        except Exception as e:
            print(f"\n{symbol}: retry:", type(e).__name__)
            time.sleep(2)

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp")

    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"\n{symbol}: SAVED {len(df):,} candles")
    print(f"File: {path}")

    return df


def main():

    print("=" * 50)
    print("CRYPTO HISTORICAL DATA MANAGER")
    print("=" * 50)
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Period: {DAYS} days")
    print("=" * 50)

    for symbol in SYMBOLS:

        df = download_symbol(symbol)

        print(
            f"{symbol:<10} "
            f"{len(df):>7,} candles"
        )

    print("\n" + "=" * 50)
    print("DATA READY")
    print("=" * 50)
    print(f"Stored in: {DATA_DIR}/")


if __name__ == "__main__":
    main()
