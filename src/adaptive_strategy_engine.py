# src/adaptive_strategy_engine.py

import os
import numpy as np
import pandas as pd

DATA_DIR = "data/features"
OUT_DIR = "data/ml"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
]

REGIME_COL = "regime"

# Proven from walk-forward testing.
# The engine will only activate conditions that demonstrated
# positive out-of-sample performance.

REGIME_RULES = {
    "BTC/USDT": {
        "CHOP_HIGH_VOL": {"enabled": True, "pf": 1.38},
        "CHOP_LOW_VOL":  {"enabled": True, "pf": 1.31},
        "TREND_LOW_VOL": {"enabled": True, "pf": 1.14},
        "TREND_HIGH_VOL": {"enabled": False, "pf": 0.92},
    },

    "BNB/USDT": {
        "TREND_HIGH_VOL": {"enabled": True, "pf": 1.45},
        "CHOP_LOW_VOL":   {"enabled": True, "pf": 1.28},
        "CHOP_HIGH_VOL":  {"enabled": False, "pf": 0.80},
        "TREND_LOW_VOL":  {"enabled": False, "pf": 1.00},
    },

    "ETH/USDT": {
        "CHOP_HIGH_VOL":  {"enabled": True, "pf": 1.37},
        "TREND_LOW_VOL":  {"enabled": True, "pf": 1.45},
        "CHOP_LOW_VOL":   {"enabled": True, "pf": 1.07},
        "TREND_HIGH_VOL": {"enabled": False, "pf": 0.78},
    },

    "XRP/USDT": {
        "CHOP_HIGH_VOL":  {"enabled": True, "pf": 2.16},
        "CHOP_LOW_VOL":   {"enabled": False, "pf": 0.86},
        "TREND_HIGH_VOL": {"enabled": False, "pf": 0.67},
        "TREND_LOW_VOL":  {"enabled": False, "pf": 0.86},
    },

    "SOL/USDT": {
        "CHOP_HIGH_VOL":  {"enabled": False, "pf": 0.75},
        "CHOP_LOW_VOL":   {"enabled": False, "pf": 0.67},
        "TREND_HIGH_VOL": {"enabled": False, "pf": 0.89},
        "TREND_LOW_VOL":  {"enabled": False, "pf": 0.57},
    },
}


def symbol_filename(symbol):
    return symbol.replace("/", "_") + "_15m.csv"


def find_regime_column(df):
    candidates = [
        "regime",
        "market_regime",
        "regime_type",
        "market_state",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def load_data(symbol):
    path = os.path.join(
        DATA_DIR,
        symbol_filename(symbol)
    )

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    regime_col = find_regime_column(df)

    if regime_col is None:
        return None

    df["REGIME"] = df[regime_col].astype(str)

    return df


def classify_direction(df):
    """
    Determine the direction bias using ONLY current/past data.

    No future returns.
    """

    long_bias = pd.Series(False, index=df.index)
    short_bias = pd.Series(False, index=df.index)

    if "ema_20" in df:
        long_bias |= df["ema_20"] > df["ema_50"]
        short_bias |= df["ema_20"] < df["ema_50"]

    if "ema_50" in df:
        long_bias |= df["ema_50"] > df["ema_200"]
        short_bias |= df["ema_50"] < df["ema_200"]

    if "rsi_14" in df:
        long_bias &= df["rsi_14"] > 50
        short_bias &= df["rsi_14"] < 50

    direction = np.where(
        long_bias,
        "LONG",
        np.where(short_bias, "SHORT", "NONE")
    )

    return direction


def calculate_future_return(df, horizon=16):
    """
    16 x 15m = 4 hours.

    Used ONLY for evaluation.
    Never used as a feature.
    """

    return (
        df["close"].shift(-horizon) /
        df["close"] - 1
    )


def evaluate_signal(direction, future_return):
    result = np.zeros(len(direction))

    long_mask = direction == "LONG"
    short_mask = direction == "SHORT"

    result[long_mask] = future_return[long_mask]
    result[short_mask] = -future_return[short_mask]

    return result


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 75)
    print("ADAPTIVE REGIME STRATEGY ENGINE")
    print("=" * 75)
    print("Uses existing feature data")
    print("NO market-data download")
    print("NO future data in signal generation")
    print("=" * 75)

    all_results = []

    for symbol in SYMBOLS:

        print()
        print("-" * 70)
        print(symbol)
        print("-" * 70)

        df = load_data(symbol)

        if df is None:
            print("Missing regime data.")
            continue

        print(f"Rows: {len(df):,}")

        df["DIRECTION"] = classify_direction(df)

        df["FUTURE_RETURN"] = calculate_future_return(df)

        df["SIGNAL_RETURN"] = evaluate_signal(
            df["DIRECTION"],
            df["FUTURE_RETURN"]
        )

        rules = REGIME_RULES.get(symbol, {})

        for regime, rule in rules.items():

            mask = (
                (df["REGIME"] == regime) &
                (df["DIRECTION"] != "NONE") &
                df["SIGNAL_RETURN"].notna()
            )

            subset = df.loc[mask].copy()

            if len(subset) == 0:
                continue

            returns = subset["SIGNAL_RETURN"]

            wins = returns[returns > 0]
            losses = returns[returns < 0]

            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())

            pf = (
                gross_profit / gross_loss
                if gross_loss > 0
                else np.inf
            )

            winrate = (
                len(wins) / len(returns)
                if len(returns)
                else 0
            )

            expectancy = returns.mean()

            enabled = rule["enabled"]

            # Additional safety:
            # if historical PF is bad, never activate.
            if pf < 1.0:
                enabled = False

            all_results.append({
                "symbol": symbol,
                "regime": regime,
                "historical_pf": rule["pf"],
                "current_pf": pf,
                "trades": len(returns),
                "winrate": winrate,
                "expectancy": expectancy,
                "enabled": enabled,
            })

            status = "TRADE" if enabled else "NO TRADE"

            print(
                f"{regime:<20}"
                f"N:{len(returns):4d} "
                f"WR:{winrate*100:5.1f}% "
                f"PF:{pf:5.2f} "
                f"Exp:{expectancy:+.5f} "
                f"{status}"
            )

        # Save predictions/signals
        output = df[
            [
                "REGIME",
                "DIRECTION",
                "close",
                "SIGNAL_RETURN",
            ]
        ].copy()

        safe = symbol.replace("/", "_")

        output.to_csv(
            os.path.join(
                OUT_DIR,
                f"{safe}_adaptive_signals.csv"
            ),
            index=False
        )

    results = pd.DataFrame(all_results)

    results.to_csv(
        os.path.join(
            OUT_DIR,
            "adaptive_strategy_results.csv"
        ),
        index=False
    )

    print()
    print("=" * 75)
    print("ADAPTIVE STRATEGY SUMMARY")
    print("=" * 75)

    if len(results):

        active = results[
            results["enabled"]
        ].sort_values(
            "current_pf",
            ascending=False
        )

        print()

        for _, row in active.iterrows():

            print(
                f"{row['symbol']:<10} "
                f"{row['regime']:<20} "
                f"PF:{row['current_pf']:.2f} "
                f"WR:{row['winrate']*100:.1f}% "
                f"N:{int(row['trades'])}"
            )

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)
    print(
        "data/ml/adaptive_strategy_results.csv"
    )

    print()
    print("NEXT STEP = FINAL OUT-OF-SAMPLE ADAPTIVE BACKTEST")
    print("=" * 75)


if __name__ == "__main__":
    main()
