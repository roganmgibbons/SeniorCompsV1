# how different are the 100,000 generated markets from each other?
# describes what came OUT of the generator (after the leverage filter)

from pathlib import Path

import numpy as np

from src.optimize import GAMMA

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

d = np.load(DATA_DIR / "train.npz")
mu = d["market_mu"]            # (100000, 25)      true monthly expected returns
S = d["market_Sigma"]          # (100000, 25, 25)  true monthly covariance
w_star = d["y"][::3]           # (100000, 25)      perfect portfolio, one per market
                               # (every market has 3 rows with the same answer key)

# --- per stock ---
vol = np.sqrt(np.einsum("mii->mi", S)) * np.sqrt(12)    # annual volatility of each stock
exp_ret = mu * 12                                       # annual expected return of each stock

# --- per market ---
# average correlation between the 25 stocks (the 300 pairs)
inv = 1 / np.sqrt(np.einsum("mii->mi", S))
corr_sum = np.einsum("mi,mij,mj->m", inv, S, inv)       # sum of all 625 correlations
avg_corr = (corr_sum - 25) / (25 * 24)                  # drop the 25 ones on the diagonal

leverage = np.abs(w_star).sum(axis=1)                   # how extreme the perfect portfolio is
best_ce = (np.einsum("mi,mi->m", w_star, mu)
           - GAMMA / 2 * np.einsum("mi,mij,mj->m", w_star, S, w_star)) * 12

# --- print the spread of each ---
rows = {
    "stock volatility (annual)":   (vol.ravel(), "{:.1%}"),
    "stock expected return":       (exp_ret.ravel(), "{:.1%}"),
    "avg correlation in a market": (avg_corr, "{:.2f}"),
    "perfect portfolio leverage":  (leverage, "{:.2f}"),
    "best possible CE (annual)":   (best_ce, "{:.1%}"),
}

qs = [0, 10, 50, 90, 100]
print(f"{len(mu):,} markets\n")
print(f"{'':30}" + "".join(f"{'p' + str(q):>9}" for q in qs) + f"{'mean':>9}")
for name, (x, fmt) in rows.items():
    vals = [fmt.format(np.percentile(x, q)) for q in qs] + [fmt.format(x.mean())]
    print(f"{name:30}" + "".join(f"{v:>9}" for v in vals))