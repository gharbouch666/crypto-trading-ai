import os
import pandas as pd
import numpy as np

FEATURE_DIR = "data/features"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT"
]


def classify_regime(row):

    trend = row["trend_score"]
    vol = row["volatility_20"]
    atr = row["atr_percent"]
    rv = row["relative_volume"]

    if trend >= 3 and vol > 0.25:
        return "TREND_HIGH_VOL"

    if trend >= 3:
        return "TREND_LOW_VOL"

    if trend <= 1 and vol > 0.25:
        return "CHOP_HIGH_VOL"

    if trend <= 1:
        return "CHOP_LOW_VOL"

    if rv >= 1.5:
        return "VOLUME_EXPANSION"

    return "TRANSITION"


def analyze_regime(df, regime):

    data = df[df["regime"] == regime].copy()

    if len(data) < 100:
        return None

    future = data["future_4h"].dropna()

    if len(future) == 0:
        return None

    positive = future[future > 0]
    negative = future[future < 0]

    winrate_long = (
        len(positive) / len(future) * 100
    )

    winrate_short = (
        len(negative) / len(future) * 100
    )

    avg_move = future.mean()

    median_move = future.median()

    best_move = future.quantile(0.90)
    worst_move = future.quantile(0.10)

    return {
        "samples": len(future),
        "long_winrate": winrate_long,
        "short_winrate": winrate_short,
        "avg_4h": avg_move,
        "median_4h": median_move,
        "best_10pct": best_move,
        "worst_10pct": worst_move
    }


def analyze_symbol(symbol):

    filename = (
        symbol.replace("/", "_")
        + "_15m.csv"
    )

    path = os.path.join(
        FEATURE_DIR,
        filename
    )

    df = pd.read_csv(path)

    df["regime"] = df.apply(
        classify_regime,
        axis=1
    )

    results = []

    regimes = df["regime"].unique()

    for regime in regimes:

        result = analyze_regime(
            df,
            regime
        )

        if result is None:
            continue

        result["symbol"] = symbol
        result["regime"] = regime

        results.append(result)

    return results


def main():

    print("=" * 70)
    print("MARKET REGIME DISCOVERY ENGINE")
    print("=" * 70)

    all_results = []

    for symbol in SYMBOLS:

        print(f"\nAnalyzing: {symbol}")

        results = analyze_symbol(symbol)

        all_results.extend(results)

        for r in results:

            print(
                f"{r['regime']:<20} "
                f"N:{r['samples']:5} "
                f"LONG:{r['long_winrate']:6.2f}% "
                f"SHORT:{r['short_winrate']:6.2f}% "
                f"AVG:{r['avg_4h']:+7.3f}% "
                f"MED:{r['median_4h']:+7.3f}%"
            )

    result_df = pd.DataFrame(all_results)

    os.makedirs("data/analysis", exist_ok=True)

    output = (
        "data/analysis/"
        "regime_analysis.csv"
    )

    result_df.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print("BEST HISTORICAL CONDITIONS")
    print("=" * 70)

    if len(result_df) > 0:

        ranked = result_df.sort_values(
            "avg_4h",
            ascending=False
        )

        for _, r in ranked.head(10).iterrows():

            print(
                f"{r['symbol']:<10} "
                f"{r['regime']:<20} "
                f"AVG:{r['avg_4h']:+.3f}% "
                f"LONG:{r['long_winrate']:.1f}% "
                f"SHORT:{r['short_winrate']:.1f}% "
                f"N:{int(r['samples'])}"
            )

    print("\nSaved:")
    print(output)

    print("\n" + "=" * 70)
    print("REGIME DISCOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
