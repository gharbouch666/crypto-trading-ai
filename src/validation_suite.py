from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_strategy_selection as base

np.random.seed(42)

DISJOINT_TRAIN = 4000
DISJOINT_TEST = 1000


def run_disjoint(symbol, signal_fn):
    original_fn = base.generate_signal
    base.generate_signal = signal_fn
    try:
        df = base.load_symbol(symbol)
        n = len(df)
        block = DISJOINT_TRAIN + DISJOINT_TEST

        window = 0
        train_start = 0
        symbol_trades = []

        while train_start + block <= n:
            window += 1
            train_end = train_start + DISJOINT_TRAIN
            test_end = train_end + DISJOINT_TEST

            train = df.iloc[train_start:train_end].copy()
            test = df.iloc[train_end:test_end].copy()

            selected = {}
            for regime in base.REGIMES:
                candidates = []
                for strategy in base.STRATEGIES:
                    for rr in base.RR_VALUES:
                        for atr in base.ATR_VALUES:
                            result = base.evaluate_regime(train, regime, strategy, rr, atr)
                            if result is not None:
                                candidates.append(result)
                if candidates:
                    candidates.sort(key=lambda x: x["score"], reverse=True)
                    selected[regime] = candidates[0]
                else:
                    selected[regime] = None

            oos = base.run_oos(test, selected)
            if len(oos):
                oos["symbol"] = symbol
                oos["window"] = window
                symbol_trades.append(oos)

            train_start = test_end

        if symbol_trades:
            return pd.concat(symbol_trades, ignore_index=True), window
        return pd.DataFrame(), window
    finally:
        base.generate_signal = original_fn


def random_signal(row, strategy, p_trade=0.15):
    if np.random.random() > p_trade:
        return 0
    return 1 if np.random.random() < 0.5 else -1


def summarize(trades_df, label):
    if len(trades_df) == 0:
        print(f"{label:<12} NO TRADES")
        return None
    r = trades_df["R"].values
    net = r.sum()
    exp = r.mean()
    pf = base.profit_factor(r)
    print(f"{label:<12} N={len(r):<6} net_R={net:>10.3f}  exp_R={exp:>8.4f}  PF={pf:>6.3f}")
    return {"net_R": net, "exp_R": exp, "PF": pf, "N": len(r)}


def main():
    print("=" * 78)
    print("VALIDATION SUITE")
    print("1) Disjoint (non-overlapping) walk-forward -- real signals")
    print("2) Same disjoint windows -- RANDOM direction baseline")
    print("=" * 78)

    real_all = []
    random_all = []

    for symbol in base.SYMBOLS:
        print(f"\n--- {symbol} ---")

        real_trades, n_windows = run_disjoint(symbol, base.generate_signal)
        print(f"  ({n_windows} disjoint windows available)")
        real_res = summarize(real_trades, "REAL")
        if len(real_trades):
            real_trades["symbol"] = symbol
            real_all.append(real_trades)

        rand_trades, _ = run_disjoint(symbol, random_signal)
        rand_res = summarize(rand_trades, "RANDOM")
        if len(rand_trades):
            rand_trades["symbol"] = symbol
            random_all.append(rand_trades)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    if real_all:
        real_df = pd.concat(real_all, ignore_index=True)
        r = real_df["R"].values
        print(f"REAL   pooled: N={len(r):<6} net_R={r.sum():>10.3f}  exp_R={r.mean():>8.4f}  PF={base.profit_factor(r):>6.3f}")
    else:
        print("REAL   pooled: NO TRADES (not enough disjoint windows -- need more history)")

    if random_all:
        rand_df = pd.concat(random_all, ignore_index=True)
        rr = rand_df["R"].values
        print(f"RANDOM pooled: N={len(rr):<6} net_R={rr.sum():>10.3f}  exp_R={rr.mean():>8.4f}  PF={base.profit_factor(rr):>6.3f}")
    else:
        print("RANDOM pooled: NO TRADES")

    print("\nHOW TO READ THIS:")
    print("- REAL should clearly beat RANDOM on exp_R. If they're close, the edge")
    print("  is coming from the RR/cost structure, not from real directional signal.")
    print("- Disjoint window count will be small (~3-4) given ~17k rows of history --")
    print("  treat this as a sanity check, not a statistically powerful test on its own.")


if __name__ == "__main__":
    main()
