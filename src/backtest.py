import ccxt
import pandas as pd
import numpy as np
import time

exchange = ccxt.binance({
    "enableRateLimit": True
})

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT"
]

TIMEFRAME = "5m"
DAYS = 180
LIMIT = 1000

RR = 2.0
FEE = 0.001
SLIPPAGE = 0.0005

ATR_PERIOD = 14
ATR_STOP = 1.5

MIN_RV = 1.2
MAX_HOLD = 36
COOLDOWN = 12


def fetch_history(symbol):

    now = exchange.milliseconds()

    since = now - DAYS * 24 * 60 * 60 * 1000

    candles = []

    while since < now:

        try:

            batch = exchange.fetch_ohlcv(
                symbol,
                TIMEFRAME,
                since=since,
                limit=LIMIT
            )

            if not batch:
                break

            candles.extend(batch)

            new_since = batch[-1][0] + 1

            if new_since <= since:
                break

            since = new_since

            if len(batch) < LIMIT:
                break

        except Exception as e:

            print(
                f"{symbol} retry:",
                type(e).__name__
            )

            time.sleep(3)

    df = pd.DataFrame(
        candles,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df = (
        df
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema20"] = (
        close
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ema50"] = (
        close
        .ewm(span=50, adjust=False)
        .mean()
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        tr
        .rolling(ATR_PERIOD)
        .mean()
    )

    df["avg_volume"] = (
        df["volume"]
        .rolling(20)
        .mean()
        .shift(1)
    )

    df["rv"] = (
        df["volume"] /
        df["avg_volume"]
    )

    df["mom1h"] = (
        close /
        close.shift(12)
        - 1
    ) * 100

    df["mom4h"] = (
        close /
        close.shift(48)
        - 1
    ) * 100

    df["range_high"] = (
        high
        .rolling(12)
        .max()
        .shift(1)
    )

    df["range_low"] = (
        low
        .rolling(12)
        .min()
        .shift(1)
    )

    return df


def backtest(df):

    trades = []

    i = 60

    while i < len(df) - MAX_HOLD - 2:

        row = df.iloc[i]

        price = row["close"]
        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        long_signal = (
            price > row["ema20"]
            and row["ema20"] > row["ema50"]
            and row["mom1h"] > 0
            and row["mom4h"] > 0
            and row["rv"] >= MIN_RV
            and price > row["range_high"]
        )

        short_signal = (
            price < row["ema20"]
            and row["ema20"] < row["ema50"]
            and row["mom1h"] < 0
            and row["mom4h"] < 0
            and row["rv"] >= MIN_RV
            and price < row["range_low"]
        )

        if not long_signal and not short_signal:
            i += 1
            continue

        # IMPORTANT:
        # Signal is detected on candle i.
        # Entry happens on candle i+1.
        entry_candle = df.iloc[i + 1]

        if long_signal:

            entry = (
                entry_candle["open"] *
                (1 + SLIPPAGE)
            )

            stop = entry - ATR_STOP * atr

            target = (
                entry +
                RR * (entry - stop)
            )

            direction = "LONG"

        else:

            entry = (
                entry_candle["open"] *
                (1 - SLIPPAGE)
            )

            stop = entry + ATR_STOP * atr

            target = (
                entry -
                RR * (stop - entry)
            )

            direction = "SHORT"

        risk = abs(entry - stop)

        if risk <= 0:
            i += 1
            continue

        result_r = None

        exit_index = None

        for j in range(
            i + 2,
            min(i + 2 + MAX_HOLD, len(df))
        ):

            future = df.iloc[j]

            if direction == "LONG":

                hit_stop = future["low"] <= stop
                hit_target = future["high"] >= target

                # Conservative assumption:
                # if both happen in same candle,
                # stop is considered first.
                if hit_stop and hit_target:
                    result_r = -1.0
                    exit_index = j
                    break

                if hit_stop:
                    result_r = -1.0
                    exit_index = j
                    break

                if hit_target:
                    result_r = RR
                    exit_index = j
                    break

            else:

                hit_stop = future["high"] >= stop
                hit_target = future["low"] <= target

                if hit_stop and hit_target:
                    result_r = -1.0
                    exit_index = j
                    break

                if hit_stop:
                    result_r = -1.0
                    exit_index = j
                    break

                if hit_target:
                    result_r = RR
                    exit_index = j
                    break

        # Time exit
        if result_r is None:

            exit_index = min(
                i + 1 + MAX_HOLD,
                len(df) - 1
            )

            exit_price = df.iloc[exit_index]["close"]

            if direction == "LONG":

                pnl = (
                    exit_price - entry
                ) / risk

            else:

                pnl = (
                    entry - exit_price
                ) / risk

            result_r = pnl

        # Round-trip transaction cost
        entry_cost = FEE + SLIPPAGE
        exit_cost = FEE + SLIPPAGE

        total_cost_pct = (
            entry_cost +
            exit_cost
        )

        risk_pct = risk / entry

        cost_r = (
            total_cost_pct /
            risk_pct
        )

        net_r = result_r - cost_r

        trades.append({
            "r": net_r,
            "direction": direction
        })

        # Do not immediately re-enter
        i = exit_index + COOLDOWN

    if not trades:
        return None

    r = np.array(
        [x["r"] for x in trades]
    )

    wins = r[r > 0]
    losses = r[r < 0]

    total = len(r)

    win_rate = (
        len(wins) /
        total *
        100
    )

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    profit_factor = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else np.inf
    )

    expectancy = r.mean()

    equity = np.cumsum(r)

    peaks = np.maximum.accumulate(equity)

    drawdown = equity - peaks

    max_drawdown = drawdown.min()

    long_count = sum(
        x["direction"] == "LONG"
        for x in trades
    )

    short_count = sum(
        x["direction"] == "SHORT"
        for x in trades
    )

    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "net_r": r.sum(),
        "max_drawdown": max_drawdown,
        "longs": long_count,
        "shorts": short_count
    }


print("\n========================================")
print("REALISTIC CRYPTO STRATEGY BACKTEST")
print("========================================")
print("Period: 180 days")
print("Timeframe: 5m")
print("RR:", RR)
print("ATR stop:", ATR_STOP)
print("Fee:", FEE)
print("Slippage:", SLIPPAGE)
print("========================================\n")

results = []

for symbol in SYMBOLS:

    print("Downloading", symbol)

    try:

        df = fetch_history(symbol)

        print(
            f"{symbol}: "
            f"{len(df):,} candles"
        )

        if len(df) < 10000:

            print("Not enough data.\n")
            continue

        df = indicators(df)

        result = backtest(df)

        if result is None:

            print("NO TRADES\n")
            continue

        results.append(
            (symbol, result)
        )

        print(
            f"Trades:       {result['trades']}"
        )

        print(
            f"Win rate:     {result['win_rate']:.2f}%"
        )

        print(
            f"Profit factor:{result['profit_factor']:.2f}"
        )

        print(
            f"Expectancy:   {result['expectancy']:+.3f}R"
        )

        print(
            f"Net:          {result['net_r']:+.2f}R"
        )

        print(
            f"Max drawdown: {result['max_drawdown']:+.2f}R"
        )

        print(
            f"Longs:        {result['longs']}"
        )

        print(
            f"Shorts:       {result['shorts']}"
        )

        print()

    except Exception as e:

        print(
            f"{symbol} ERROR:",
            type(e).__name__,
            str(e)
        )


print("\n========================================")
print("FINAL RESULTS")
print("========================================\n")

for symbol, r in results:

    print(
        f"{symbol:10} "
        f"Trades:{r['trades']:4} "
        f"Win:{r['win_rate']:6.2f}% "
        f"PF:{r['profit_factor']:5.2f} "
        f"Exp:{r['expectancy']:+.3f}R "
        f"Net:{r['net_r']:+8.2f}R "
        f"DD:{r['max_drawdown']:+8.2f}R"
    )

print("\nBACKTEST COMPLETE")
