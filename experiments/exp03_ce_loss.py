"""What does estimation error actually cost, in return?"""

import numpy as np

from src.artificial_market import factor_model_known
from src.sample import draw_returns
from src.optimize import markowitz, oracle_weights, equal_weight
from src.metrics import ce_loss, certainty_equivalent

N_ASSETS, N_TRIALS = 25, 300
MONTHS = [60, 120, 240, 600]

mu, Sigma = factor_model_known(n_assets=N_ASSETS, seed=42)
w_star = oracle_weights(mu, Sigma)

print(f"oracle CE: {certainty_equivalent(w_star, mu, Sigma):.5f}/month "
      f"({certainty_equivalent(w_star, mu, Sigma)*12:.2%}/yr)\n")

print(f"{'months':>7} {'Markowitz loss':>16} {'1/N loss':>12} {'annualized':>12}")
print("-" * 52)

for T in MONTHS:
    mk, eq = [], []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(1000 * trial + T)
        R = draw_returns(mu, Sigma, T, rng)
        mk.append(ce_loss(markowitz(R), w_star, mu, Sigma))
        eq.append(ce_loss(equal_weight(R), w_star, mu, Sigma))
    mk, eq = np.array(mk), np.array(eq)
    print(f"{T:>7} {mk.mean():>16.5f} {eq.mean():>12.5f} {mk.mean()*12:>11.2%}")

print(f"\nnegative losses (should be 0): {(mk < -1e-12).sum()}")