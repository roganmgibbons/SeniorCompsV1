"""Scoring a portfolio against the truth."""

import numpy as np

GAMMA = 3.0


def certainty_equivalent(w, mu, Sigma, gamma=GAMMA):
    """How good is this portfolio? One number.

        CE(w) = w'mu - (gamma/2) * w'Sigma w

    Expected return minus a penalty for risk. Higher is better. Units are
    monthly return, so 0.004 means "as good as a guaranteed 0.4% a month."

    mu and Sigma MUST be the true parameters. Scoring a portfolio with the same
    wrong numbers that built it is how backtests lie.
    """
    return w @ mu - (gamma / 2) * (w @ Sigma @ w)


def ce_loss(w_hat, w_star, mu, Sigma, gamma=GAMMA):
    """How much was given up by estimating instead of knowing.

        loss = CE(w_star) - CE(w_hat)

    Always >= 0, because the oracle is optimal over the same constraint set.
    A negative value means a bug, not a discovery.

    This is the number Phase 2 trains on.
    """
    return (certainty_equivalent(w_star, mu, Sigma, gamma)
            - certainty_equivalent(w_hat, mu, Sigma, gamma))