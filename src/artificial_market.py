import numpy as np

from src.config import PERIODS_PER_YEAR


# function expanding on "known.py" - moving from the 3 asset test to 25
# most asset managers run between 25-35 assets per portfolio

# n assets means the number of assets in the market contained in our portfolio
# seed fixes the random draw
# annual premium refers to extra return an asset earns per unit of market exposure
    # essentially equity risk + premium or historical value
    # shortcut for our simulation

# market vol annual refers to annual volatility in the market (based on S&P)
# idio vol refers to average volatility of each assets individual noise
# rng is the seeded random generator
# betas is each assets market exposure


def factor_model_known(n_assets, seed,
                       annual_premium = 0.06,
                       market_vol_annual = 0.16,
                       idio_vol_annual = 0.22):

    rng = np.random.default_rng(seed)
    # 1. betas - each asset's market exposure
    betas = rng.normal(1.0, 0.35, size=n_assets)
    betas = np.clip(betas, 0.2, 2.2)

    # 2. each asset's own volatility -> monthly VARIANCE
    idio_vol_ann = rng.normal(idio_vol_annual, 0.08, size=n_assets)
    idio_vol_ann = np.clip(idio_vol_ann, 0.08, 0.60)
    idio_var = (idio_vol_ann ** 2) / PERIODS_PER_YEAR

    # 3. market variance, monthly
    market_var = (market_vol_annual ** 2) / PERIODS_PER_YEAR

    # 4. assemble - same np.outer trick as make_truth
    Sigma = market_var * np.outer(betas, betas) + np.diag(idio_var)

    # 5. expected returns: paid for market exposure, plus noise
    mu = betas * (annual_premium / PERIODS_PER_YEAR)
    mu = mu + rng.normal(0, 0.03 / PERIODS_PER_YEAR, size=n_assets)

    # 6. kill floating-point asymmetry
    Sigma = 0.5 * (Sigma + Sigma.T)

    return mu, Sigma




