import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

DATA_DIR = "data/features"
OUT_DIR = "data/ml"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
]

CONFIDENCES = [0.55, 0.60, 0.65, 0.70, 0.75]

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
GAP = 96

os.makedirs(OUT_DIR, exist_ok=True)


def filename(symbol):
    return symbol.replace("/", "_") + "_15m.csv"


def load_data(symbol):
    path = os.path.join(DATA_DIR, filename(symbol))
    return pd.read_csv(path)


def find_target(df):
    candidates = [
        "target",
        "direction",
        "future_direction",
        "target_4h",
        "direction_4h",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Build target ourselves if feature file has close.
    if "close" in df.columns:
        future = df["close"].shift(-16)
        target = (future > df["close"]).astype(int)
        return "__generated_target__"

    raise ValueError("No target column found")


def prepare(df):
    df = df.copy()

    target_col = find_target(df)

    if target_col == "__generated_target__":
        future = df["close"].shift(-16)
        df["__target__"] = (future > df["close"]).astype(int)
        target_col = "__target__"

    # Absolutely remove future-looking columns.
    forbidden = []

    for col in df.columns:
        name = col.lower()

        if (
            "future" in name
            or "target" in name
            or "forward" in name
            or "label" in name
        ):
            if col != target_col:
                forbidden.append(col)

    forbidden += [
        "__target__",
        "timestamp",
        "datetime",
        "date",
    ]

    forbidden = list(set(forbidden))

    y = df[target_col].copy()

    feature_cols = [
        c for c in df.columns
        if c not in forbidden
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].copy()

    # Remove rows where target is unavailable.
    valid = y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    # Remove inf.
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, feature_cols


def train_symbol(symbol):
    print("\n" + "=" * 60)
    print("TRAINING", symbol)
    print("=" * 60)

    df = load_data(symbol)

    print("Rows:", len(df))

    X, y, feature_cols = prepare(df)

    n = len(X)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Gap between train/validation/test prevents temporal contamination.
    train_end = min(train_end, n - GAP * 2)
    val_start = train_end + GAP
    test_start = val_end + GAP

    if test_start >= n:
        raise ValueError("Dataset too small for requested gap.")

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[val_start:val_end]
    y_val = y.iloc[val_start:val_end]

    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    print("Features:", len(feature_cols))
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    imputer = SimpleImputer(strategy="median")

    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")

    model.fit(X_train_i, y_train)

    val_pred = model.predict(X_val_i)
    test_pred = model.predict(X_test_i)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    baseline = max(y_test.mean(), 1 - y_test.mean())
    edge = test_acc - baseline

    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"TEST accuracy:       {test_acc:.2%}")
    print(f"Majority baseline:   {baseline:.2%}")
    print(f"Model edge:          {edge:+.2%}")

    # Probabilities.
    probabilities = model.predict_proba(X_test_i)

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)

    print("\nCONFIDENCE")

    for threshold in CONFIDENCES:
        mask = confidence >= threshold

        if mask.sum() == 0:
            print(f"{threshold:.2f} Signals: 0")
            continue

        acc = accuracy_score(
            y_test.iloc[np.where(mask)[0]],
            predictions[mask]
        )

        print(
            f"{threshold:.2f} "
            f"Signals:{mask.sum():5d} "
            f"Accuracy:{acc:.2%}"
        )

    # --------------------------------------------------
    # IMPORTANT:
    # Build ONE dataframe from arrays of identical length.
    # This fixes the "All arrays must be of the same length" bug.
    # --------------------------------------------------

    result = pd.DataFrame({
        "index": X_test.index.to_numpy(),
        "actual": y_test.to_numpy(),
        "prediction": predictions,
        "prob_long": probabilities[:, 1],
        "prob_short": probabilities[:, 0],
        "confidence": confidence,
    })

    result["correct"] = (
        result["actual"] == result["prediction"]
    ).astype(int)

    result["symbol"] = symbol

    output = os.path.join(
        OUT_DIR,
        symbol.replace("/", "_") + "_predictions.csv"
    )

    result.to_csv(output, index=False)

    print("Predictions saved:", output)

    # Feature importance.
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nTOP FEATURES")

    for _, row in importance.head(10).iterrows():
        print(
            f"{row['feature']:25s} "
            f"{row['importance']:.4f}"
        )

    return {
        "symbol": symbol,
        "rows": len(df),
        "train": len(X_train),
        "test": len(X_test),
        "accuracy": test_acc,
        "baseline": baseline,
        "edge": edge,
    }


def main():

    print("=" * 70)
    print("CLEAN ML + MARKET REGIME ENGINE")
    print("=" * 70)
    print("Existing feature data")
    print("No market-data download")
    print("Target: NEXT 4H direction")
    print("Chronological split")
    print("Gap:", GAP)
    print("=" * 70)

    results = []

    for symbol in SYMBOLS:

        try:
            result = train_symbol(symbol)
            results.append(result)

        except Exception as e:
            print(
                f"\nERROR: {symbol} {type(e).__name__}: {e}"
            )

    if results:

        summary = pd.DataFrame(results)

        summary = summary.sort_values(
            "edge",
            ascending=False
        )

        path = os.path.join(
            OUT_DIR,
            "clean_ml_summary.csv"
        )

        summary.to_csv(path, index=False)

        print("\n")
        print("=" * 70)
        print("FINAL OUT-OF-SAMPLE RESULTS")
        print("=" * 70)

        for _, r in summary.iterrows():

            print(
                f"{r['symbol']:10s} "
                f"Accuracy:{r['accuracy']:.2%} "
                f"Baseline:{r['baseline']:.2%} "
                f"Edge:{r['edge']:+.2%}"
            )

        print("\nSaved:", path)

    print("\nML ENGINE COMPLETE")


if __name__ == "__main__":
    main()
