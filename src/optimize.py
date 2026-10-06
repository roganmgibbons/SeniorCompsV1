# ground truths for my project stem from the optimizer which generates portfolio weights
# here is where we implement the markowitz formula


import numpy as np

from src.estimate import sample_mean, sample_covariance

GAMMA = 3.0    # variable for risk aversion


def solve(Sigma, b):


    return np.linalg.solve(Sigma, b)




def mean_variance_weights(mu, Sigma, gamma=GAMMA):

# weights for our mean variance equation
        ones = np.ones(len(mu))
        a    = solve(Sigma, mu)       # Sigma^-1 mu
        b    = solve(Sigma, ones)     # Sigma^-1 1
        lam  = (ones @ a - gamma) / (ones @ b)
        w    = (a - lam * b) / gamma
        return w


def markowitz(R, gamma=GAMMA):
    """Markowitz weights from a return sample"""
    mu_hat = sample_mean(R)
    Sigma_hat = sample_covariance(R)
    return mean_variance_weights(mu_hat, Sigma_hat, gamma)


def oracle_weights(mu, Sigma, gamma=GAMMA):
    """The portfolio you'd hold with perfect information, benchmark"""
    return mean_variance_weights(mu, Sigma, gamma)



def equal_weight(R):
    n_assets = R.shape[1]
    return np.full(n_assets, 1.0 / n_assets)