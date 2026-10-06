# drawing returns from our known market data
# in normal, real world investing, we do not know
import numpy as np

def draw_returns(mu, Sigma, n_months, rng):
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n_months)

