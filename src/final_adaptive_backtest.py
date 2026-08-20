# src/final_adaptive_backtest.py

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

TRAIN_BARS = 7000
TEST_BARS = 1000
STEP = 1000

RR = 2.0
ATR_MULT = 1.5

FEE = 0.001
SLIPPAGE = 0.0005

RISK_PER_TRADE = 1.0

CONFIDENCE = 0.0


def filename(symbol):
    return symbol.replace("/", "_") + "_15m.csv"


def load(symbol):

    path = os.path.join(
        DATA_DIR,
        filename(symbol)
    )

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    required = [
        "open",
        "high",
        "low",
        "close",
        "atr_percent",
        "regime",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        print("Missing columns:", missing)
        return None

    return df.reset_index(drop=True)


def direction_signal(df, i):

    row = df.iloc[i]

    long_signal = True
    short_signal = True

    if "ema_20" in df:
        long_signal &= row["ema_20"] > row["ema_50"]
        short_signal &= row["ema_20"] < row["ema_50"]

    if "ema_50" in df:
        long_signal &= row["ema_50"] > row["ema_200"]
        short_signal &= row["ema_50"] < row["ema_200"]

    if "rsi_14" in df:
        long_signal &= row["rsi_14"] > 50
        short_signal &= row["rsi_14"] < 50

    if long_signal:
        return "LONG"

    if short_signal:
        return "SHORT"

    return "NONE"


def historical_regime_score(
    df,
    start,
    end,
    regime
):

    subset = df.iloc[start:end].copy()

    subset = subset[
        subset["regime"] == regime
    ]

    if len(subset) < 50:
        return 0.0

    returns = []

    for i in subset.index:

        if i + 16 >= len(df):
            continue

        direction = direction_signal(
            df,
            i
        )

        if direction == "NONE":
            continue

        entry = df.iloc[i]["close"]
        future = df.iloc[i + 16]["close"]

        ret = (
            future / entry - 1
        )

        if direction == "SHORT":
            ret = -ret

        returns.append(ret)

    if len(returns) < 50:
        return 0.0

    returns = np.array(returns)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(losses) == 0:
        return 0.0

    pf = (
        wins.sum() /
        abs(losses.sum())
    )

    return float(pf)


def get_allowed_regimes(
    df,
    train_start,
    train_end
):

    regimes = df["regime"].dropna().unique()

    scores = {}

    for regime in regimes:

        pf = historical_regime_score(
            df,
            train_start,
            train_end,
            regime
        )

        scores[regime] = pf

    # Require actual historical evidence.
    allowed = {
        regime
        for regime, pf in scores.items()
        if pf >= 1.05
    }

    return allowed, scores


def simulate_trade(
    df,
    i,
    direction
):

    entry = df.iloc[i]["close"]

    atr_percent = (
        df.iloc[i]["atr_percent"]
    )

    if not np.isfinite(atr_percent):
        return None

    atr_distance = (
        entry *
        atr_percent *
        ATR_MULT
    )

    if atr_distance <= 0:
        return None

    if direction == "LONG":

        stop = entry - atr_distance
        target = (
            entry +
            atr_distance * RR
        )

        entry_price = (
            entry *
            (1 + SLIPPAGE)
        )

    else:

        stop = entry + atr_distance
        target = (
            entry -
            atr_distance * RR
        )

        entry_price = (
            entry *
            (1 - SLIPPAGE)
        )

    max_forward = min(
        i + 16,
        len(df) - 1
    )

    result_r = None

    for j in range(
        i + 1,
        max_forward + 1
    ):

        high = df.iloc[j]["high"]
        low = df.iloc[j]["low"]

        if direction == "LONG":

            hit_stop = low <= stop
            hit_target = high >= target

        else:

            hit_stop = high >= stop
            hit_target = low <= target

        # Conservative assumption:
        # if both are touched in the same candle,
        # assume STOP happened first.

        if hit_stop:

            result_r = -1.0
            break

        if hit_target:

            result_r = RR
            break

    if result_r is None:

        exit_price = df.iloc[
            max_forward
        ]["close"]

        if direction == "LONG":
            raw_return = (
                exit_price /
                entry_price - 1
            )
        else:
            raw_return = (
                entry_price /
                exit_price - 1
            )

        risk_distance = (
            atr_distance /
            entry_price
        )

        result_r = (
            raw_return /
            risk_distance
        )

    # Fees on both sides
    fee_cost = (
        2 * FEE
    )

    risk_distance = (
        atr_distance /
        entry_price
    )

    fee_r = (
        fee_cost /
        risk_distance
    )

    result_r -= fee_r

    return result_r


def max_drawdown(values):

    equity = np.cumsum(values)

    peak = np.maximum.accumulate(
        equity
    )

    dd = equity - peak

    return dd.min()


def stats(trades):

    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "winrate": 0,
            "pf": 0,
            "expectancy": 0,
            "net": 0,
            "dd": 0,
        }

    arr = np.array(trades)

    wins = arr[arr > 0]
    losses = arr[arr < 0]

    gross_profit = (
        wins.sum()
        if len(wins)
        else 0
    )

    gross_loss = abs(
        losses.sum()
    ) if len(losses) else 0

    pf = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "trades": len(arr),
        "wins": len(wins),
        "winrate": len(wins) / len(arr),
        "pf": pf,
        "expectancy": arr.mean(),
        "net": arr.sum(),
        "dd": max_drawdown(arr),
    }


def run_symbol(symbol):

    df = load(symbol)

    if df is None:
        return []

    print()
    print("=" * 70)
    print("ADAPTIVE WALK-FORWARD:", symbol)
    print("=" * 70)

    trades = []
    regime_results = []

    window = 0

    train_start = 0

    while (
        train_start +
        TRAIN_BARS +
        TEST_BARS
        <= len(df)
    ):

        train_end = (
            train_start +
            TRAIN_BARS
        )

        test_end = (
            train_end +
            TEST_BARS
        )

        print(
            f"Window {window + 1}: "
            f"train={train_start}:{train_end} "
            f"test={train_end}:{test_end}"
        )

        allowed, scores = (
            get_allowed_regimes(
                df,
                train_start,
                train_end
            )
        )

        print(
            "Allowed:",
            ", ".join(
                sorted(allowed)
            ) if allowed else "NONE"
        )

        # Test period
        i = train_end

        while i < test_end:

            regime = df.iloc[i]["regime"]

            if regime not in allowed:
                i += 1
                continue

            direction = direction_signal(
                df,
                i
            )

            if direction == "NONE":
                i += 1
                continue

            result = simulate_trade(
                df,
                i,
                direction
            )

            if result is not None:

                trades.append(result)

                regime_results.append({
                    "symbol": symbol,
                    "window": window + 1,
                    "regime": regime,
                    "direction": direction,
                    "result_r": result,
                })

                # One trade at a time.
                i += 16
            else:
                i += 1

        window += 1
        train_start += STEP

    return trades, regime_results


def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("FINAL ADAPTIVE WALK-FORWARD BACKTEST")
    print("=" * 70)
    print("Train:", TRAIN_BARS)
    print("Test:", TEST_BARS)
    print("Step:", STEP)
    print("RR:", RR)
    print("ATR:", ATR_MULT)
    print("Fee:", FEE)
    print("Slippage:", SLIPPAGE)
    print("=" * 70)

    summary = []
    all_regimes = []

    for symbol in SYMBOLS:

        result = run_symbol(symbol)

        if not result:
            continue

        trades, regime_results = result

        s = stats(trades)

        summary.append({
            "symbol": symbol,
            **s,
        })

        all_regimes.extend(
            regime_results
        )

        print()
        print("RESULT")
        print("-" * 40)
        print(
            f"Trades:       {s['trades']}"
        )
        print(
            f"Win rate:     {s['winrate']*100:.2f}%"
        )
        print(
            f"Profit factor:{s['pf']:.2f}"
        )
        print(
            f"Expectancy:   {s['expectancy']:+.4f}R"
        )
        print(
            f"Net:          {s['net']:+.2f}R"
        )
        print(
            f"Max drawdown: {s['dd']:+.2f}R"
        )

    summary_df = pd.DataFrame(
        summary
    )

    regime_df = pd.DataFrame(
        all_regimes
    )

    summary_path = os.path.join(
        OUT_DIR,
        "FINAL_adaptive_walkforward.csv"
    )

    regime_path = os.path.join(
        OUT_DIR,
        "FINAL_adaptive_regimes.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    regime_df.to_csv(
        regime_path,
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if len(summary_df):

        summary_df = summary_df.sort_values(
            "net",
            ascending=False
        )

        for _, r in summary_df.iterrows():

            print(
                f"{r['symbol']:<10} "
                f"Trades:{int(r['trades']):4d} "
                f"Win:{r['winrate']*100:5.2f}% "
                f"PF:{r['pf']:5.2f} "
                f"Exp:{r['expectancy']:+.4f}R "
                f"Net:{r['net']:+.2f}R "
                f"DD:{r['dd']:+.2f}R"
            )

    print()
    print("Saved:")
    print(summary_path)
    print(regime_path)

    print()
    print("=" * 70)
    print("FINAL ADAPTIVE BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
