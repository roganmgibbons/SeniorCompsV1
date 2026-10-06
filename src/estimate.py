# estimation of expected returns, basically where we think the returns will be
# implementing estimation error

import numpy as np

def sample_mean(R):
    return np.mean(R, axis=0)

def sample_covariance(R):
    #Covariance matrix. (T, N) in -> (N, N) out
    T = R.shape[0]
    if T < 2:
        raise ValueError("need at least 2 observations")

    mu_hat = sample_mean(R)  # (N,) each asset's average
    D = R - mu_hat  # (T, N) how far each month was from that average
    return D.T @ D / (T - 1)
