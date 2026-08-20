# src/merge_regimes.py

import os
import pandas as pd
import numpy as np

FEATURE_DIR = "data/features"
OUT_DIR = "data/features"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
]


def safe(symbol):
    return symbol.replace("/", "_")


def find_datetime(df):

    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "Date",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def make_regime(df):

    # Use existing regime if already present
    if "regime" in df.columns:
        return df["regime"].astype(str)

    if "market_regime" in df.columns:
        return df["market_regime"].astype(str)

    # Required indicators
    required = [
        "atr_percent",
        "volume",
        "volume_ma_20",
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:
        print("Missing:", missing)
        return pd.Series(
            ["UNKNOWN"] * len(df),
            index=df.index
        )

    # -----------------------------
    # VOLATILITY
    # -----------------------------

    atr = df["atr_percent"]

    vol_median = (
        atr.rolling(96, min_periods=96)
        .median()
    )

    high_vol = atr > vol_median

    # -----------------------------
    # TREND
    # -----------------------------

    if "ema_20" in df.columns:
        ema20 = df["ema_20"]
    else:
        ema20 = df["close"].ewm(
            span=20,
            adjust=False
        ).mean()

    if "ema_50" in df.columns:
        ema50 = df["ema_50"]
    else:
        ema50 = df["close"].ewm(
            span=50,
            adjust=False
        ).mean()

    if "ema_200" in df.columns:
        ema200 = df["ema_200"]
    else:
        ema200 = df["close"].ewm(
            span=200,
            adjust=False
        ).mean()

    trend_strength = (
        abs(ema20 - ema50) / df["close"]
    )

    trend_threshold = (
        trend_strength
        .rolling(96, min_periods=96)
        .median()
    )

    trending = (
        trend_strength > trend_threshold
    )

    # -----------------------------
    # VOLUME EXPANSION
    # -----------------------------

    volume_expansion = (
        df["volume"] >
        df["volume_ma_20"] * 1.5
    )

    # -----------------------------
    # REGIME
    # -----------------------------

    regime = pd.Series(
        "TRANSITION",
        index=df.index,
        dtype="object"
    )

    regime.loc[
        trending & ~high_vol
    ] = "TREND_LOW_VOL"

    regime.loc[
        trending & high_vol
    ] = "TREND_HIGH_VOL"

    regime.loc[
        ~trending & ~high_vol
    ] = "CHOP_LOW_VOL"

    regime.loc[
        ~trending & high_vol
    ] = "CHOP_HIGH_VOL"

    # Give volume expansion priority
    regime.loc[
        volume_expansion
    ] = "VOLUME_EXPANSION"

    return regime


def main():

    print("=" * 70)
    print("REGIME + FEATURE MERGER")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    for symbol in SYMBOLS:

        filename = safe(symbol) + "_15m.csv"

        path = os.path.join(
            FEATURE_DIR,
            filename
        )

        print()
        print("Processing:", symbol)

        if not os.path.exists(path):
            print("FILE NOT FOUND:", path)
            continue

        df = pd.read_csv(path)

        print("Rows:", len(df))

        df["regime"] = make_regime(df)

        counts = df["regime"].value_counts()

        print()
        print("REGIMES:")

        for regime, count in counts.items():

            print(
                f"  {regime:<20} "
                f"{count:,}"
            )

        df.to_csv(
            path,
            index=False
        )

        print()
        print("UPDATED:", path)

    print()
    print("=" * 70)
    print("REGIME MERGE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
