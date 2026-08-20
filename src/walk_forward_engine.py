import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# WALK-FORWARD ADAPTIVE CRYPTO ENGINE
# ============================================================

DATA_DIR = "data/features"
OUT_DIR = "data/ml"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
]

# 15m candles
BARS_4H = 16

# Walk-forward configuration
TRAIN_BARS = 7000
VALIDATION_BARS = 2000
TEST_BARS = 1000
STEP_BARS = 1000

# Trading
RR = 2.0
ATR_MULT = 1.5
FEE = 0.001
SLIPPAGE = 0.0005

# Only trade when model has enough confidence
MIN_CONFIDENCE = 0.70

# Regime parameters
VOL_WINDOW = 96
TREND_WINDOW = 96

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def filename(symbol):
    return symbol.replace("/", "_") + "_15m.csv"


def load_data(symbol):
    path = os.path.join(DATA_DIR, filename(symbol))

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def build_target(df):
    """
    Predict direction over next 4 hours.
    """
    close = df["close"]

    future = close.shift(-BARS_4H)

    target = (future > close).astype(int)

    target[future.isna()] = np.nan

    return target


def detect_regime(df):
    """
    Simple causal market regime.
    Uses only information available at the current candle.
    """

    close = df["close"]

    ret = close.pct_change()

    volatility = (
        ret.rolling(VOL_WINDOW)
        .std()
    )

    vol_median = volatility.rolling(
        2000,
        min_periods=500
    ).median()

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    trend_strength = (
        (ema50 - ema200).abs()
        / close
    )

    trend_threshold = trend_strength.rolling(
        2000,
        min_periods=500
    ).median()

    regimes = []

    for i in range(len(df)):

        if pd.isna(volatility.iloc[i]):
            regimes.append("UNKNOWN")
            continue

        high_vol = (
            volatility.iloc[i]
            > vol_median.iloc[i]
        )

        strong_trend = (
            trend_strength.iloc[i]
            > trend_threshold.iloc[i]
        )

        if strong_trend and high_vol:
            regime = "TREND_HIGH_VOL"

        elif strong_trend and not high_vol:
            regime = "TREND_LOW_VOL"

        elif not strong_trend and high_vol:
            regime = "CHOP_HIGH_VOL"

        else:
            regime = "CHOP_LOW_VOL"

        regimes.append(regime)

    return pd.Series(
        regimes,
        index=df.index
    )


def prepare_features(df):

    target = build_target(df)

    feature_cols = []

    forbidden = [
        "target",
        "future_4h",
        "future_12h",
        "future_24h",
        "timestamp",
        "datetime",
        "date",
    ]

    for col in df.columns:

        if col in forbidden:
            continue

        name = col.lower()

        if (
            "future" in name
            or "target" in name
            or "label" in name
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    X = df[feature_cols].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X, target, feature_cols


def calculate_atr(df):

    if "atr" in df.columns:
        return df["atr"]

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(14).mean()


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    direction,
    atr_value,
):
    """
    Conservative OHLC simulation.

    Entry happens at next candle open.

    If SL and TP are both touched inside
    the same candle, SL wins.
    """

    if entry_index + 1 >= len(df):
        return None

    entry_row = df.iloc[entry_index + 1]

    entry = float(entry_row["open"])

    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    stop_distance = atr_value * ATR_MULT
    target_distance = stop_distance * RR

    if direction == 1:

        stop = entry - stop_distance
        target = entry + target_distance

    else:

        stop = entry + stop_distance
        target = entry - target_distance

    entry_cost = FEE + SLIPPAGE

    # Scan next 48 hours.
    max_bars = min(
        192,
        len(df) - entry_index - 1
    )

    for j in range(
        entry_index + 1,
        entry_index + max_bars
    ):

        row = df.iloc[j]

        high = float(row["high"])
        low = float(row["low"])

        if direction == 1:

            # Conservative: SL first.
            if low <= stop:

                return {
                    "r": -1.0 - entry_cost,
                    "exit": "SL",
                    "bars": j - entry_index,
                }

            if high >= target:

                return {
                    "r": RR - entry_cost,
                    "exit": "TP",
                    "bars": j - entry_index,
                }

        else:

            if high >= stop:

                return {
                    "r": -1.0 - entry_cost,
                    "exit": "SL",
                    "bars": j - entry_index,
                }

            if low <= target:

                return {
                    "r": RR - entry_cost,
                    "exit": "TP",
                    "bars": j - entry_index,
                }

    # Time exit.
    final_row = df.iloc[
        entry_index + max_bars - 1
    ]

    final_close = float(final_row["close"])

    if direction == 1:

        raw_r = (
            final_close - entry
        ) / stop_distance

    else:

        raw_r = (
            entry - final_close
        ) / stop_distance

    # Approximate round-trip cost.
    raw_r -= (
        FEE * 2
        + SLIPPAGE * 2
    )

    return {
        "r": raw_r,
        "exit": "TIME",
        "bars": max_bars,
    }


# ============================================================
# MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_i = imputer.fit_transform(
        X_train
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_i,
        y_train
    )

    return model, imputer


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_symbol(symbol):

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD:", symbol)
    print("=" * 70)

    df = load_data(symbol)

    print("Candles:", len(df))

    X, y, features = prepare_features(df)

    regimes = detect_regime(df)

    atr = calculate_atr(df)

    # Align.
    valid = (
        X.notna().sum(axis=1) > 0
        & y.notna()
        & atr.notna()
    )

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    regimes = regimes.loc[valid].reset_index(drop=True)
    atr = atr.loc[valid].reset_index(drop=True)

    df = df.loc[valid].reset_index(drop=True)

    n = len(df)

    all_trades = []

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    window_number = 0

    start = TRAIN_BARS

    while start + TEST_BARS < n:

        window_number += 1

        train_start = start - TRAIN_BARS
        train_end = start

        test_start = start
        test_end = min(
            start + TEST_BARS,
            n
        )

        X_train = X.iloc[
            train_start:train_end
        ]

        y_train = y.iloc[
            train_start:train_end
        ]

        X_test = X.iloc[
            test_start:test_end
        ]

        print(
            f"Window {window_number}: "
            f"train={train_start}:{train_end} "
            f"test={test_start}:{test_end}"
        )

        # Train.
        model, imputer = train_model(
            X_train,
            y_train
        )

        X_test_i = imputer.transform(
            X_test
        )

        probabilities = model.predict_proba(
            X_test_i
        )

        predictions = probabilities.argmax(
            axis=1
        )

        confidence = probabilities.max(
            axis=1
        )

        # ----------------------------------------------------
        # TEST PERIOD
        # ----------------------------------------------------

        last_trade_index = -999

        for local_i, global_i in enumerate(
            range(test_start, test_end)
        ):

            # Avoid overlapping trades.
            if global_i <= last_trade_index:
                continue

            conf = float(
                confidence[local_i]
            )

            if conf < MIN_CONFIDENCE:
                continue

            direction = int(
                predictions[local_i]
            )

            regime = regimes.iloc[
                global_i
            ]

            if regime == "UNKNOWN":
                continue

            atr_value = float(
                atr.iloc[global_i]
            )

            trade = simulate_trade(
                df,
                global_i,
                direction,
                atr_value,
            )

            if trade is None:
                continue

            r = trade["r"]

            equity += r

            peak = max(
                peak,
                equity
            )

            dd = equity - peak

            max_dd = min(
                max_dd,
                dd
            )

            all_trades.append(
                {
                    "symbol": symbol,
                    "window": window_number,
                    "index": global_i,
                    "direction": (
                        "LONG"
                        if direction == 1
                        else "SHORT"
                    ),
                    "prediction": direction,
                    "confidence": conf,
                    "regime": regime,
                    "r": r,
                    "exit": trade["exit"],
                    "bars": trade["bars"],
                }
            )

            last_trade_index = (
                global_i
                + trade["bars"]
            )

        start += STEP_BARS

    # ========================================================
    # RESULTS
    # ========================================================

    trades = pd.DataFrame(
        all_trades
    )

    if trades.empty:

        print("NO TRADES")

        return None

    wins = (
        trades["r"] > 0
    ).sum()

    losses = (
        trades["r"] <= 0
    ).sum()

    gross_profit = trades.loc[
        trades["r"] > 0,
        "r"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["r"] <= 0,
            "r"
        ].sum()
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    expectancy = trades["r"].mean()

    winrate = (
        wins / len(trades)
    )

    returns = trades["r"]

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(len(returns))
        if returns.std() > 0
        else 0
    )

    print("\nRESULT")
    print("-" * 40)
    print("Trades:", len(trades))
    print("Wins:", wins)
    print("Losses:", losses)
    print(f"Win rate:     {winrate:.2%}")
    print(f"Profit factor:{pf:.2f}")
    print(f"Expectancy:   {expectancy:+.4f}R")
    print(f"Net:          {trades['r'].sum():+.2f}R")
    print(f"Max drawdown: {max_dd:+.2f}R")
    print(f"Sharpe:       {sharpe:.2f}")

    # --------------------------------------------------------
    # REGIME BREAKDOWN
    # --------------------------------------------------------

    print("\nREGIME PERFORMANCE")

    regime_results = []

    for regime, group in trades.groupby(
        "regime"
    ):

        gp = group.loc[
            group["r"] > 0,
            "r"
        ].sum()

        gl = abs(
            group.loc[
                group["r"] <= 0,
                "r"
            ].sum()
        )

        regime_pf = (
            gp / gl
            if gl > 0
            else np.inf
        )

        regime_results.append(
            {
                "symbol": symbol,
                "regime": regime,
                "trades": len(group),
                "winrate": (
                    group["r"] > 0
                ).mean(),
                "pf": regime_pf,
                "expectancy": group["r"].mean(),
                "net": group["r"].sum(),
            }
        )

        print(
            f"{regime:20s} "
            f"N:{len(group):4d} "
            f"Win:{(group['r'] > 0).mean():.1%} "
            f"PF:{regime_pf:.2f} "
            f"Exp:{group['r'].mean():+.4f} "
            f"Net:{group['r'].sum():+.2f}R"
        )

    # Save.
    trade_path = os.path.join(
        OUT_DIR,
        symbol.replace("/", "_")
        + "_walkforward_trades.csv"
    )

    trades.to_csv(
        trade_path,
        index=False
    )

    regime_path = os.path.join(
        OUT_DIR,
        symbol.replace("/", "_")
        + "_walkforward_regimes.csv"
    )

    pd.DataFrame(
        regime_results
    ).to_csv(
        regime_path,
        index=False
    )

    return {
        "symbol": symbol,
        "trades": len(trades),
        "winrate": winrate,
        "pf": pf,
        "expectancy": expectancy,
        "net": trades["r"].sum(),
        "drawdown": max_dd,
        "sharpe": sharpe,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADAPTIVE WALK-FORWARD CRYPTO ENGINE")
    print("=" * 70)
    print("Existing feature data")
    print("No market-data download")
    print(f"RR: {RR}")
    print(f"ATR stop: {ATR_MULT}")
    print(f"Fee: {FEE}")
    print(f"Slippage: {SLIPPAGE}")
    print(f"Confidence: {MIN_CONFIDENCE}")
    print("=" * 70)

    results = []

    for symbol in SYMBOLS:

        try:

            result = walk_forward_symbol(
                symbol
            )

            if result:
                results.append(result)

        except Exception as e:

            print(
                f"\nERROR {symbol}: "
                f"{type(e).__name__}: {e}"
            )

    if results:

        summary = pd.DataFrame(
            results
        )

        summary = summary.sort_values(
            "net",
            ascending=False
        )

        path = os.path.join(
            OUT_DIR,
            "walkforward_summary.csv"
        )

        summary.to_csv(
            path,
            index=False
        )

        print("\n")
        print("=" * 70)
        print("FINAL WALK-FORWARD RESULTS")
        print("=" * 70)

        for _, r in summary.iterrows():

            print(
                f"{r['symbol']:10s} "
                f"Trades:{int(r['trades']):5d} "
                f"Win:{r['winrate']:.2%} "
                f"PF:{r['pf']:.2f} "
                f"Exp:{r['expectancy']:+.4f}R "
                f"Net:{r['net']:+.2f}R "
                f"DD:{r['drawdown']:+.2f}R "
                f"Sharpe:{r['sharpe']:.2f}"
            )

        print("\nSaved:", path)

    else:

        print("\nNO RESULTS")

    print("\n")
    print("=" * 70)
    print("WALK-FORWARD ENGINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
