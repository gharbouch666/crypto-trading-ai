import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURE_DIR = "data/features"
OUTPUT_DIR = "data/ml"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT"
]

FEATURES = [
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "ret_12h",
    "ret_24h",
    "ema20_distance",
    "ema50_distance",
    "atr_percent",
    "rsi_14",
    "relative_volume",
    "volatility_20",
    "volatility_96",
    "body_ratio",
    "trend_score",
    "breakout_up",
    "breakout_down"
]


def create_target(df):
    """
    Predict whether the next 4 hours finish UP or DOWN.

    1 = UP
    0 = DOWN

    We deliberately remove neutral moves so the
    model learns meaningful directional movement.
    """

    df = df.copy()

    future = df["future_4h"]

    df["target"] = np.where(
        future > 0.10,
        1,
        np.where(
            future < -0.10,
            0,
            np.nan
        )
    )

    return df.dropna(subset=["target"])


def train_symbol(symbol):

    filename = (
        symbol.replace("/", "_")
        + "_15m.csv"
    )

    path = os.path.join(
        FEATURE_DIR,
        filename
    )

    df = pd.read_csv(path)

    df = create_target(df)

    df = df.dropna(
        subset=FEATURES
    )

    # ---------------------------------
    # CHRONOLOGICAL SPLIT
    # ---------------------------------

    split = int(len(df) * 0.70)

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[FEATURES]
    y_train = train["target"].astype(int)

    X_test = test[FEATURES]
    y_test = test["target"].astype(int)

    # ---------------------------------
    # MODEL
    # ---------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------
    # PREDICTIONS
    # ---------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ---------------------------------
    # HIGH-CONFIDENCE SIGNALS
    # ---------------------------------

    high_conf = (
        (probabilities >= 0.60) |
        (probabilities <= 0.40)
    )

    hc_accuracy = accuracy_score(
        y_test[high_conf],
        predictions[high_conf]
    ) if high_conf.sum() > 0 else 0

    # ---------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    # ---------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------

    test["probability_up"] = probabilities
    test["prediction"] = predictions

    test["signal"] = np.where(
        probabilities >= 0.60,
        "LONG",
        np.where(
            probabilities <= 0.40,
            "SHORT",
            "NO_TRADE"
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    prediction_file = os.path.join(
        OUTPUT_DIR,
        symbol.replace("/", "_")
        + "_predictions.csv"
    )

    importance_file = os.path.join(
        OUTPUT_DIR,
        symbol.replace("/", "_")
        + "_importance.csv"
    )

    test.to_csv(
        prediction_file,
        index=False
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    return {
        "symbol": symbol,
        "train": len(train),
        "test": len(test),
        "accuracy": accuracy,
        "high_conf_signals": int(high_conf.sum()),
        "high_conf_accuracy": hc_accuracy,
        "long_predictions": int(
            (probabilities >= 0.60).sum()
        ),
        "short_predictions": int(
            (probabilities <= 0.40).sum()
        ),
        "importance": importance
    }


def main():

    print("=" * 70)
    print("CRYPTO ML RESEARCH ENGINE")
    print("=" * 70)

    print("Model: Random Forest")
    print("Target: next 4H direction")
    print("Train/Test: 70% / 30% chronological")
    print("=" * 70)

    results = []

    for symbol in SYMBOLS:

        print(f"\n{'=' * 50}")
        print(f"TRAINING: {symbol}")
        print(f"{'=' * 50}")

        result = train_symbol(symbol)

        results.append(result)

        print(
            f"Train samples:      {result['train']:,}"
        )

        print(
            f"Test samples:       {result['test']:,}"
        )

        print(
            f"Accuracy:            "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"High-confidence:     "
            f"{result['high_conf_signals']:,}"
        )

        print(
            f"High-conf accuracy:  "
            f"{result['high_conf_accuracy'] * 100:.2f}%"
        )

        print(
            f"LONG predictions:    "
            f"{result['long_predictions']:,}"
        )

        print(
            f"SHORT predictions:   "
            f"{result['short_predictions']:,}"
        )

        print("\nTOP FEATURES:")

        for _, row in result["importance"].head(7).iterrows():

            print(
                f"  {row['feature']:<22}"
                f"{row['importance']:.4f}"
            )

    # ---------------------------------
    # SUMMARY
    # ---------------------------------

    print("\n" + "=" * 70)
    print("ML SUMMARY")
    print("=" * 70)

    for r in results:

        print(
            f"{r['symbol']:<10}"
            f" Accuracy:{r['accuracy'] * 100:6.2f}%"
            f" | HC:{r['high_conf_accuracy'] * 100:6.2f}%"
            f" | Signals:{r['high_conf_signals']:5}"
        )

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print("\nSaved models' predictions to:")
    print("data/ml/")


if __name__ == "__main__":
    main()
