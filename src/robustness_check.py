from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/ml")
N_BOOTSTRAP = 5000
np.random.seed(42)


def bootstrap_ci(returns, n=N_BOOTSTRAP):
    returns = np.asarray(returns, dtype=float)
    means = np.empty(n)
    for k in range(n):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        means[k] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    pct_negative = (means < 0).mean() * 100
    return lo, hi, pct_negative


def main():
    trades = pd.read_csv(DATA_DIR / "oos_trades.csv")

    print("=" * 78)
    print("BOOTSTRAP ROBUSTNESS CHECK (5000 resamples, 95% CI)")
    print("=" * 78)

    print("\nPER SYMBOL:")
    print(f"{'symbol':<10}{'N':>6}{'exp_R':>10}{'CI_low':>10}{'CI_high':>10}{'%<0':>8}")

    for symbol, grp in trades.groupby("symbol"):
        r = grp["R"].values
        lo, hi, pct_neg = bootstrap_ci(r)
        print(f"{symbol:<10}{len(r):>6}{r.mean():>10.4f}{lo:>10.4f}{hi:>10.4f}{pct_neg:>7.1f}%")

    print("\nPORTFOLIO (all trades pooled):")
    r_all = trades["R"].values
    lo, hi, pct_neg = bootstrap_ci(r_all)
    print(f"N={len(r_all)}  mean_R={r_all.mean():.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  P(mean<0)={pct_neg:.2f}%")

    print("\nPER-WINDOW correlation across symbols (net_R by window):")
    pivot = trades.groupby(["symbol", "window"])["R"].sum().unstack(level=0)
    print(pivot.corr().round(2).to_string())

    print("\nInterpretation:")
    print("- If a symbol's CI_low is comfortably above 0, the edge is unlikely to be pure noise.")
    print("- If %<0 is high (>5%), that symbol's edge is fragile.")
    print("- High correlation (>0.6) between symbols' per-window returns means they are")
    print("  NOT independent bets -- true portfolio risk is higher than summing them naively.")


if __name__ == "__main__":
    main()
