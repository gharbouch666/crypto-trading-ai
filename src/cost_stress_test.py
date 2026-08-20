from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/ml")


def profit_factor(returns):
    returns = np.asarray(returns, dtype=float)
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if wins > 0 else 0.0
    return wins / losses


def main():
    trades = pd.read_csv(DATA_DIR / "oos_trades.csv")

    base_cost = 0.0008
    multipliers = [1, 2, 3, 5, 8, 10]

    print("=" * 78)
    print("COST / SLIPPAGE STRESS TEST")
    print("=" * 78)
    print(f"{'cost_mult':>10}{'cost_R':>10}{'net_R':>12}{'exp_R':>10}{'PF':>8}")

    for mult in multipliers:
        extra_cost = base_cost * (mult - 1)
        adjusted = trades["R"] - extra_cost
        net = adjusted.sum()
        exp = adjusted.mean()
        pf = profit_factor(adjusted.values)
        flag = "  <-- BREAKS EVEN/NEGATIVE" if net <= 0 else ""
        print(f"{mult:>9}x{base_cost*mult:>10.4f}{net:>12.2f}{exp:>10.4f}{pf:>8.2f}{flag}")

    print("\nPer-symbol at 3x cost (realistic slippage estimate):")
    extra_cost = base_cost * 2
    trades["R_adj"] = trades["R"] - extra_cost
    summary = trades.groupby("symbol")["R_adj"].agg(["sum", "mean", "count"])
    summary.columns = ["net_R", "exp_R", "trades"]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
