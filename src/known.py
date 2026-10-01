# setting the parameters of the simulated market
# we choose these, basically setting the "expected" returns as
# something we know and can "grade" our algorithms on

import numpy as np

from src.config import PERIODS_PER_YEAR

# 3 assets, dictated with a NP array
# annual returns, volatility, and "correlation"

ANNUAL_RETURNS = np.array([0.12,0.04,0.09])
ANNUAL_VOLS    = np.array([0.22, 0.06, 0.28])
CORRELATIONS = np.array([
    [1.00, 0.10, 0.45],
    [0.10, 1.00, 0.05],
    [0.45, 0.05, 1.00],
])
ASSET_NAMES = ["Tech", "Bond", "Energy"]

def make_truth():

    # convert expected returns to monthly
    monthly = ANNUAL_RETURNS / PERIODS_PER_YEAR

    # make the volatilities monthly

    monthly_vol = ANNUAL_VOLS / np.sqrt(PERIODS_PER_YEAR)

    #covariance matrix to calculate portfolio weights
    Sigma = np.outer(monthly_vol, monthly_vol) * CORRELATIONS

    return monthly, Sigma



