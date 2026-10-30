# builds the training data for the model
# each row pairs one noisy sample of returns with the oracle weights
# for the market that sample came from
#
# produces two files:
#   train.npz   normal markets, for training and the held-out test
#   stress.npz  crisis markets, never trained on, for stress testing

from pathlib import Path

import numpy as np

from src.artificial_market import factor_model_known
from src.sample import draw_returns
from src.estimate import sample_mean, sample_covariance
from src.optimize import oracle_weights, markowitz

N_ASSETS = 25
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# turns a (T, N) return sample into one flat row of 351 numbers
def make_features(R):
    mu_hat = sample_mean(R)                  # (25,)
    S = sample_covariance(R)                 # (25, 25)

    vols = np.sqrt(np.diag(S))               # (25,)
    corr = S / np.outer(vols, vols)          # (25, 25)

    # upper triangle only, k=1 skips the diagonal since it is always 1
    iu = np.triu_indices(N_ASSETS, k=1)
    corr_flat = corr[iu]                     # (300,)

    n_months = R.shape[0]

    return np.concatenate([mu_hat, vols, corr_flat, [n_months]])


def generate(n_markets, samples_per_market, seed,
             market_vol_range=(0.10, 0.30),
             idio_vol_range=(0.12, 0.45),
             max_oracle_leverage=4.0,
             label="markets"):

    rng = np.random.default_rng(seed)

    X, y, market_id = [], [], []
    market_mu, market_Sigma, w_markowitz = [], [], []
    rejected = 0

    for m in range(n_markets):

        # some parameter combinations make the 25 assets near duplicates of
        # each other, and then even the perfect portfolio needs absurd
        # leverage, no investor could hold that so we throw the market away
        # and draw new parameters
        attempts = 0
        while True:
            market_vol = rng.uniform(*market_vol_range)
            idio_vol = rng.uniform(*idio_vol_range)

            mu, Sigma = factor_model_known(
                n_assets=N_ASSETS,
                seed=int(rng.integers(2 ** 32)),
                market_vol_annual=market_vol,
                idio_vol_annual=idio_vol,
            )
            w_star = oracle_weights(mu, Sigma)

            if np.abs(w_star).sum() <= max_oracle_leverage:
                break

            rejected += 1
            attempts += 1
            if attempts > 300:
                raise RuntimeError(f"market {m}: no draw under leverage {max_oracle_leverage} after 300 tries")

        # keep the truth so we can score ce loss later
        market_mu.append(mu)
        market_Sigma.append(Sigma)

        for s in range(samples_per_market):

            # vary how much history the sample contains, 5 to 50 years
            n_months = int(rng.integers(60, 601))

            draw_rng = np.random.default_rng(int(rng.integers(2 ** 32)))
            R = draw_returns(mu, Sigma, n_months, draw_rng)

            X.append(make_features(R))
            y.append(w_star)
            market_id.append(m)

            # markowitz on the same sample, so we can compare later
            w_markowitz.append(markowitz(R))

        if (m + 1) % 2000 == 0:
            print(f"  {label}: {m + 1} / {n_markets}")

    print(f"  {label}: rejected {rejected} markets for excess oracle leverage")

    return (
        np.array(X),
        np.array(y),
        np.array(market_id),
        np.array(market_mu),
        np.array(market_Sigma),
        np.array(w_markowitz),
    )


def report_and_save(out, filename):
    X, y, market_id, market_mu, market_Sigma, w_mk = out
    gross = np.abs(y).sum(axis=1)

    print(f"\n  X: {X.shape}   y: {y.shape}   markets: {market_id.max() + 1}")
    print(f"  oracle leverage: median {np.median(gross):.2f}  max {gross.max():.2f}")
    print(f"  any nan or infinite values? {not np.all(np.isfinite(X))}")

    np.savez_compressed(
        DATA_DIR / filename,
        X=X,
        y=y,
        market_id=market_id,
        market_mu=market_mu,
        market_Sigma=market_Sigma,
        w_markowitz=w_mk,
    )
    print(f"  saved to {DATA_DIR / filename}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # normal markets
    # 20,000 markets with 3 samples each beats 5,000 with 10, same number of
    # rows but four times as many distinct answers to learn from
    print("generating normal markets...")
    report_and_save(
        generate(n_markets=20_000, samples_per_market=3, seed=0, label="normal"),
        "train.npz",
    )

    # crisis markets, never trained on
    # high market volatility plus low idiosyncratic volatility means
    # everything moves together, which is what a crisis looks like
    # capped at 8x so the target stays something an investor could approximate
    print("\ngenerating crisis markets...")
    report_and_save(
        generate(n_markets=2_000, samples_per_market=3, seed=999,
                 market_vol_range=(0.25, 0.45),
                 idio_vol_range=(0.08, 0.18),
                 max_oracle_leverage=8.0,
                 label="crisis"),
        "stress.npz",
    )