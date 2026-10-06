"""How unstable are Markowitz weights, as a function of sample size?

Sweeps the number of months of data and the random draw, and reports how
leveraged the resulting portfolio is. Comparing to the oracle and known results we've previously implemented.
"""

import numpy as np

from src.artificial_market import factor_model_known
from src.sample import draw_returns
from src.optimize import markowitz, oracle_weights, equal_weight

N_ASSETS = 25
N_TRIALS = 200
MONTHS = [24, 60, 120, 240, 600]

mu, Sigma = factor_model_known(n_assets=N_ASSETS, seed=42)
w_star = oracle_weights(mu, Sigma)

print(f"N = {N_ASSETS} assets, {N_TRIALS} trials per row")
print(f"oracle gross leverage: {np.abs(w_star).sum():.2f}\n")

print(f"{'months':>7} {'years':>6} {'gross lev':>11} {'worst short':>12} {'% w/ shorts':>12}")
print("-" * 52)

for T in MONTHS:
    gross, worst = [], []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(1000 * trial + T)
        R = draw_returns(mu, Sigma, T, rng)
        w = markowitz(R)
        gross.append(np.abs(w).sum())
        worst.append(w.min())
    gross, worst = np.array(gross), np.array(worst)
    print(f"{T:>7} {T/12:>6.0f} {gross.mean():>11.2f} "
          f"{worst.mean():>11.1%} {(worst < 0).mean():>11.0%}")