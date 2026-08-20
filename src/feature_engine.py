import os
import pandas as pd
import numpy as np

DATA_DIR = "data/history"
FEATURE_DIR = "data/features"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT"
]


def add_features(df):

    df = df.copy()

    # -----------------------------
    # RETURNS / MOMENTUM
    # -----------------------------

    df["ret_15m"] = df["close"].pct_change(1) * 100
    df["ret_1h"] = df["close"].pct_change(4) * 100
    df["ret_4h"] = df["close"].pct_change(16) * 100
    df["ret_12h"] = df["close"].pct_change(48) * 100
    df["ret_24h"] = df["close"].pct_change(96) * 100

    # -----------------------------
    # EMA TREND
    # -----------------------------

    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_100"] = df["close"].ewm(span=100).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()

    df["ema20_distance"] = (
        (df["close"] / df["ema_20"]) - 1
    ) * 100

    df["ema50_distance"] = (
        (df["close"] / df["ema_50"]) - 1
    ) * 100

    # -----------------------------
    # TRUE RANGE / ATR
    # -----------------------------

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - previous_close)
    tr3 = abs(df["low"] - previous_close)

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr_14"] = df["tr"].rolling(14).mean()

    df["atr_percent"] = (
        df["atr_14"] / df["close"]
    ) * 100

    # -----------------------------
    # RSI
    # -----------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # -----------------------------
    # VOLUME
    # -----------------------------

    df["volume_ma_20"] = (
        df["volume"].rolling(20).mean()
    )

    df["relative_volume"] = (
        df["volume"] /
        df["volume_ma_20"]
    )

    # -----------------------------
    # VOLATILITY
    # -----------------------------

    df["volatility_20"] = (
        df["ret_15m"]
        .rolling(20)
        .std()
    )

    df["volatility_96"] = (
        df["ret_15m"]
        .rolling(96)
        .std()
    )

    # -----------------------------
    # PRICE STRUCTURE
    # -----------------------------

    df["high_20"] = (
        df["high"]
        .rolling(20)
        .max()
    )

    df["low_20"] = (
        df["low"]
        .rolling(20)
        .min()
    )

    df["high_96"] = (
        df["high"]
        .rolling(96)
        .max()
    )

    df["low_96"] = (
        df["low"]
        .rolling(96)
        .min()
    )

    df["breakout_up"] = (
        df["close"] >
        df["high_20"].shift(1)
    ).astype(int)

    df["breakout_down"] = (
        df["close"] <
        df["low_20"].shift(1)
    ).astype(int)

    # -----------------------------
    # CANDLE STRUCTURE
    # -----------------------------

    df["body"] = abs(
        df["close"] - df["open"]
    )

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, np.nan)
    )

    # -----------------------------
    # TREND SCORE
    # -----------------------------

    df["trend_score"] = 0

    df.loc[
        df["close"] > df["ema_20"],
        "trend_score"
    ] += 1

    df.loc[
        df["ema_20"] > df["ema_50"],
        "trend_score"
    ] += 1

    df.loc[
        df["ema_50"] > df["ema_100"],
        "trend_score"
    ] += 1

    df.loc[
        df["ema_100"] > df["ema_200"],
        "trend_score"
    ] += 1

    # -----------------------------
    # FUTURE TARGETS
    #
    # IMPORTANT:
    # These are NOT used as inputs.
    # They are what the ML model
    # will eventually try to predict.
    # -----------------------------

    df["future_4h"] = (
        df["close"].shift(-16) /
        df["close"] - 1
    ) * 100

    df["future_12h"] = (
        df["close"].shift(-48) /
        df["close"] - 1
    ) * 100

    df["future_24h"] = (
        df["close"].shift(-96) /
        df["close"] - 1
    ) * 100

    # Remove incomplete rows
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    return df


def main():

    os.makedirs(FEATURE_DIR, exist_ok=True)

    print("=" * 55)
    print("CRYPTO FEATURE ENGINE")
    print("=" * 55)

    for symbol in SYMBOLS:

        filename = (
            symbol.replace("/", "_")
            + "_15m.csv"
        )

        source = os.path.join(
            DATA_DIR,
            filename
        )

        output = os.path.join(
            FEATURE_DIR,
            filename
        )

        print(f"\nProcessing: {symbol}")

        df = pd.read_csv(source)

        df = add_features(df)

        df.to_csv(
            output,
            index=False
        )

        print(
            f"Features: {len(df):,} rows"
        )

        print(
            f"Saved: {output}"
        )

    print("\n" + "=" * 55)
    print("FEATURE ENGINE COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
