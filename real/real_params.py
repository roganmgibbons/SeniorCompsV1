# estimates the same parameters the generator draws, but from real stocks,
# and puts them next to the fake ones
#
#   stock return = alpha + beta * market return + noise
#   beta  = slope, alpha = intercept, stock-specific vol = spread of the noise

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------- real returns
stocks = pd.read_csv(DATA_DIR / "real_prices.csv", index_col=0, parse_dates=True)
spy = pd.read_csv(DATA_DIR / "real_prices_spy.csv", index_col=0, parse_dates=True)["SPY"]

this_month = pd.Timestamp.today().to_period("M").to_timestamp()
prices = stocks.join(spy, how="inner")
prices = prices[prices.index < this_month]           # drop the unfinished month
returns = prices.pct_change().dropna()

market = returns.pop("SPY")                          # S&P 500 = "the market"
print(f"{returns.shape[1]} stocks, {len(returns)} months, "
      f"{returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y}\n")

# ---------------------------------------------------------------- fit one line per stock
rows = {}
for ticker in returns.columns:
    r = returns[ticker].values
    beta, intercept = np.polyfit(market.values, r, 1)       # slope, intercept
    noise = r - (intercept + beta * market.values)
    rows[ticker] = {
        "beta": beta,
        "alpha": intercept * 12,                            # annual
        "idio_vol": noise.std() * np.sqrt(12),              # annual
        "total_vol": r.std() * np.sqrt(12),                 # annual
    }
real = pd.DataFrame(rows).T

real_market_vol = market.std() * np.sqrt(12)
corr = returns.corr().values
real_avg_corr = corr[np.triu_indices(len(corr), k=1)].mean()

# ---------------------------------------------------------------- the fake ones (kept markets)
d = np.load(DATA_DIR / "train.npz")
beta_f, idio_f, alpha_f = d["param_beta"], d["param_idio_vol"], d["param_alpha"]
mvol_f = d["param_market_vol"]
total_f = np.sqrt(beta_f**2 * mvol_f[:, None]**2 + idio_f**2)

# ---------------------------------------------------------------- side by side
def summary(x):
    x = np.asarray(x).ravel()
    return [x.mean(), x.std(), *np.percentile(x, [10, 50, 90])]

TABLE = [
    ("beta",                      "num", beta_f,  real["beta"]),
    ("alpha (annual)",            "pct", alpha_f, real["alpha"]),
    ("stock-specific volatility", "pct", idio_f,  real["idio_vol"]),
    ("total stock volatility",    "pct", total_f, real["total_vol"]),
]

def f(x, kind):
    return f"{x:.1%}" if kind == "pct" else f"{x:.2f}"

cols = ["mean", "sd", "p10", "p50", "p90"]
print(f"{'parameter':28} {'':6}" + "".join(f"{c:>9}" for c in cols))
print("-" * 80)
for name, kind, fake, rl in TABLE:
    for label, x in [("fake", fake), ("real", rl)]:
        print(f"{name if label == 'fake' else '':28} {label:6}"
              + "".join(f"{f(v, kind):>9}" for v in summary(x)))
    print()

print(f"market volatility        fake: mean {mvol_f.mean():.1%} (range {mvol_f.min():.0%}-{mvol_f.max():.0%})"
      f"   real S&P 500: {real_market_vol:.1%}")
print(f"avg correlation          real 25 stocks: {real_avg_corr:.2f}  (fake median was 0.23)")

print("\nreal estimates per stock:")
print(real.sort_values("beta").to_string(float_format=lambda v: f"{v:.2f}"))