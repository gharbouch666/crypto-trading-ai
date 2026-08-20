from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/ml")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
STRATEGIES = ["TREND", "MOMENTUM", "MEAN_REVERSION"]
REGIMES = [
    "CHOP_HIGH_VOL",
    "CHOP_LOW_VOL",
    "TREND_HIGH_VOL",
    "TREND_LOW_VOL",
    "VOLUME_EXPANSION",
]

RR_VALUES = [1.5, 2.0, 2.5, 3.0]
ATR_VALUES = [1.0, 1.5, 2.0, 2.5]

TRAIN_SIZE = 7000
TEST_SIZE = 1000
STEP = 1000

MIN_TRAIN_TRADES = 100
MIN_TRAIN_PF = 1.05
MIN_EXPECTANCY = 0.0
COST_R = 0.0008


def profit_factor(returns):
    returns = np.asarray(returns, dtype=float)

    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return np.inf if wins > 0 else 0.0

    return wins / losses


def expectancy(returns):
    if len(returns) == 0:
        return 0.0

    return float(np.mean(returns))


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0

    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak

    return float(dd.min())


def score_strategy(pf, exp, trades, dd):
    if trades < MIN_TRAIN_TRADES:
        return -np.inf

    if pf < MIN_TRAIN_PF:
        return -np.inf

    if exp <= MIN_EXPECTANCY:
        return -np.inf

    dd_penalty = abs(dd) * 0.10

    return (
        exp * 100.0
        + np.log1p(max(pf, 0.0))
        + np.log1p(trades) * 0.05
        - dd_penalty
    )


def classify_regime(df):
    out = df.copy()

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    out["atr"] = true_range.rolling(14).mean()
    out["atr_pct"] = out["atr"] / close

    out["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    out["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    out["trend_strength"] = (
        (out["ema20"] - out["ema50"]).abs()
        / close
    )

    out["volume_ma"] = volume.rolling(20).mean()
    out["volume_ratio"] = (
        volume / out["volume_ma"]
    )

    vol_median = (
        out["atr_pct"]
        .rolling(100)
        .median()
    )

    trend_threshold = 0.003

    out["regime"] = "CHOP_LOW_VOL"

    out.loc[
        out["atr_pct"] > vol_median,
        "regime"
    ] = "CHOP_HIGH_VOL"

    out.loc[
        (out["trend_strength"] >= trend_threshold)
        & (out["atr_pct"] <= vol_median),
        "regime"
    ] = "TREND_LOW_VOL"

    out.loc[
        (out["trend_strength"] >= trend_threshold)
        & (out["atr_pct"] > vol_median),
        "regime"
    ] = "TREND_HIGH_VOL"

    out.loc[
        out["volume_ratio"] >= 1.8,
        "regime"
    ] = "VOLUME_EXPANSION"

    return out


def generate_signal(row, strategy):
    close = row["close"]
    ema20 = row["ema20"]
    ema50 = row["ema50"]

    if strategy == "TREND":

        if ema20 > ema50 and close > ema20:
            return 1

        if ema20 < ema50 and close < ema20:
            return -1

    elif strategy == "MOMENTUM":

        if row["close"] > row["high_prev"]:
            return 1

        if row["close"] < row["low_prev"]:
            return -1

    elif strategy == "MEAN_REVERSION":

        if row["atr"] <= 0:
            return 0

        distance = (
            (close - ema20)
            / row["atr"]
        )

        if distance < -1.0:
            return 1

        if distance > 1.0:
            return -1

    return 0


def simulate(df, strategy, rr, atr_mult):
    trades = []
    n = len(df)
    i = 2

    while i < n - 1:

        row = df.iloc[i]

        if pd.isna(row["atr"]) or row["atr"] <= 0:
            i += 1
            continue

        signal = generate_signal(row, strategy)

        if signal == 0:
            i += 1
            continue

        entry = float(df.iloc[i + 1]["open"])

        atr = float(row["atr"])
        risk = atr * atr_mult

        if risk <= 0:
            i += 1
            continue

        if signal == 1:
            stop = entry - risk
            target = entry + risk * rr
        else:
            stop = entry + risk
            target = entry - risk * rr

        result = None
        exit_j = None

        for j in range(i + 1, n):

            candle = df.iloc[j]

            h = float(candle["high"])
            l = float(candle["low"])

            if signal == 1:

                hit_sl = l <= stop
                hit_tp = h >= target

                if hit_sl:
                    result = -1.0 - COST_R
                    exit_j = j
                    break

                if hit_tp:
                    result = rr - COST_R
                    exit_j = j
                    break

            else:

                hit_sl = h >= stop
                hit_tp = l <= target

                if hit_sl:
                    result = -1.0 - COST_R
                    exit_j = j
                    break

                if hit_tp:
                    result = rr - COST_R
                    exit_j = j
                    break

        if result is not None:
            trades.append(result)
            i = exit_j + 1
        else:
            i += 1

    return np.asarray(
        trades,
        dtype=float
    )


def prepare(df):
    df = df.copy()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for c in required:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    df["high_prev"] = df["high"].shift(1)
    df["low_prev"] = df["low"].shift(1)

    df = classify_regime(df)

    return df.dropna(
        subset=[
            "atr",
            "ema20",
            "ema50",
            "high_prev",
            "low_prev"
        ]
    ).reset_index(drop=True)


def load_symbol(symbol):

    filename = (
        symbol.replace("/", "_")
        + ".csv"
    )

    candidates = [
        Path("data/history") / filename.replace(".csv", "_15m.csv"),
        DATA_DIR / filename,
        DATA_DIR / "raw" / filename,
        Path("data") / filename,
        Path("data/raw") / filename,
    ]

    for path in candidates:

        if path.exists():

            print(
                f"Loading: {path}"
            )

            return prepare(
                pd.read_csv(path)
            )

    raise FileNotFoundError(
        f"\nNo dataset found for {symbol}.\n"
        f"Expected one of:\n"
        + "\n".join(
            str(x) for x in candidates
        )
    )


def evaluate_regime(
    train,
    regime,
    strategy,
    rr,
    atr_mult
):

    subset = train[
        train["regime"] == regime
    ].copy()

    if len(subset) < 100:
        return None

    returns = simulate(
        subset,
        strategy,
        rr,
        atr_mult
    )

    if len(returns) < MIN_TRAIN_TRADES:
        return None

    pf = profit_factor(returns)
    exp = expectancy(returns)
    dd = max_drawdown(returns)

    score = score_strategy(
        pf,
        exp,
        len(returns),
        dd
    )

    if not np.isfinite(score):
        return None

    return {
        "strategy": strategy,
        "rr": rr,
        "atr": atr_mult,
        "pf": pf,
        "expectancy": exp,
        "trades": len(returns),
        "drawdown": dd,
        "score": score
    }


def select_strategy(train):

    selected = {}

    for regime in REGIMES:

        candidates = []

        for strategy in STRATEGIES:

            for rr in RR_VALUES:

                for atr in ATR_VALUES:

                    result = evaluate_regime(
                        train,
                        regime,
                        strategy,
                        rr,
                        atr
                    )

                    if result is not None:
                        candidates.append(
                            result
                        )

        if not candidates:

            selected[regime] = None

            print(
                f"{regime:<20}"
                "NO VALID STRATEGY"
            )

            continue

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        best = candidates[0]

        selected[regime] = best

        print(
            f"{regime:<20}"
            f"{best['strategy']:<17}"
            f"RR:{best['rr']:<3.1f} "
            f"ATR:{best['atr']:<3.1f} "
            f"PF:{best['pf']:.2f} "
            f"Exp:{best['expectancy']:+.4f} "
            f"N:{best['trades']}"
        )

    return selected


def run_oos(test, selected):

    results = []

    for regime, config in selected.items():

        if config is None:
            continue

        subset = test[
            test["regime"] == regime
        ].copy()

        if len(subset) == 0:
            continue

        returns = simulate(
            subset,
            config["strategy"],
            config["rr"],
            config["atr"]
        )

        if len(returns) == 0:
            continue

        for r in returns:

            results.append({
                "regime": regime,
                "strategy": config["strategy"],
                "rr": config["rr"],
                "atr": config["atr"],
                "R": float(r)
            })

    return pd.DataFrame(results)


def main():

    print("=" * 78)
    print(
        "AI STRATEGY SELECTION LAB — "
        "PORTFOLIO ENGINE"
    )
    print("MULTI-ASSET")
    print("REGIME-SPECIFIC")
    print("MULTI-STRATEGY")
    print("MULTI-RR / ATR")
    print("COST-AWARE")
    print("STRICT WALK-FORWARD")
    print("NO FUTURE DATA")
    print("NON-OVERLAPPING POSITIONS (FIXED)")
    print("=" * 78)

    all_summary = []
    all_trades = []

    for symbol in SYMBOLS:

        print("\n" + "=" * 70)
        print(symbol)
        print("=" * 70)

        df = load_symbol(symbol)

        print(
            f"Rows: {len(df)}"
        )

        if len(df) < (
            TRAIN_SIZE + TEST_SIZE
        ):

            print(
                "SKIPPED: insufficient data"
            )

            continue

        symbol_trades = []
        window = 0
        train_start = 0

        while (
            train_start
            + TRAIN_SIZE
            + TEST_SIZE
            <= len(df)
        ):

            window += 1

            train_end = (
                train_start
                + TRAIN_SIZE
            )

            test_end = (
                train_end
                + TEST_SIZE
            )

            train = df.iloc[
                train_start:train_end
            ].copy()

            test = df.iloc[
                train_end:test_end
            ].copy()

            print(
                f"\nWindow {window}: "
                f"train={train_start}:{train_end} "
                f"test={train_end}:{test_end}"
            )

            selected = select_strategy(
                train
            )

            oos = run_oos(
                test,
                selected
            )

            if len(oos):

                oos["symbol"] = symbol
                oos["window"] = window
                oos["test_start"] = train_end
                oos["test_end"] = test_end

                symbol_trades.append(oos)
                all_trades.append(oos)

                print(
                    f"\nOOS: "
                    f"trades={len(oos)} "
                    f"net={oos['R'].sum():+.3f}R"
                )

            else:

                print(
                    "\nOOS: trades=0 "
                    "net=0.000R"
                )

            train_start += STEP

        if symbol_trades:

            trades_df = pd.concat(
                symbol_trades,
                ignore_index=True
            )

            net_r = trades_df["R"].sum()

            print("\n" + "-" * 70)
            print(
                f"{symbol} COMPLETE"
            )
            print("-" * 70)

            print(
                f"OOS trades: "
                f"{len(trades_df)}"
            )

            print(
                f"OOS net: "
                f"{net_r:+.3f}R"
            )

            all_summary.append({
                "symbol": symbol,
                "windows": window,
                "trades": len(trades_df),
                "net_R": net_r,
                "PF": profit_factor(
                    trades_df["R"].values
                ),
                "expectancy": expectancy(
                    trades_df["R"].values
                ),
                "max_DD_R": max_drawdown(
                    trades_df["R"].values
                )
            })

    if not all_summary:

        print("NO RESULTS")
        return

    summary = pd.DataFrame(
        all_summary
    ).sort_values(
        "net_R",
        ascending=False
    )

    trades = pd.concat(
        all_trades,
        ignore_index=True
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trades.to_csv(
        DATA_DIR / "oos_trades.csv",
        index=False
    )

    summary.to_csv(
        DATA_DIR / "oos_portfolio_summary.csv",
        index=False
    )

    print("\n")
    print("=" * 78)
    print("FINAL PORTFOLIO RESULTS")
    print("=" * 78)

    print(
        summary.to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("PORTFOLIO")
    print("=" * 78)

    total_r = trades["R"].sum()

    print(
        f"Total OOS trades: "
        f"{len(trades)}"
    )

    print(
        f"Total OOS net: "
        f"{total_r:+.3f}R"
    )

    print(
        f"OOS PF: "
        f"{profit_factor(trades['R'].values):.3f}"
    )

    print(
        f"OOS expectancy: "
        f"{expectancy(trades['R'].values):+.5f}R"
    )

    print(
        f"OOS max DD: "
        f"{max_drawdown(trades['R'].values):+.3f}R"
    )

    print("\nSaved:")
    print(
        DATA_DIR / "oos_trades.csv"
    )
    print(
        DATA_DIR / "oos_portfolio_summary.csv"
    )


if __name__ == "__main__":
    main()
