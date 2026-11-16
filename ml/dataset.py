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
    iu = np.triu_indices(R.shape[1], k=1)    # works for any number of stocks
    corr_flat = corr[iu]                     # (300,)

    n_months = R.shape[0]

    return np.concatenate([mu_hat, vols, corr_flat, [n_months]])


def generate(n_markets, samples_per_market, seed,
             market_vol_range=(0.10, 0.30),
             idio_vol_range=(0.12, 0.45),
             max_oracle_leverage=4.0,
             n_assets=N_ASSETS,
             label="markets"):

    rng = np.random.default_rng(seed)

    X, y, market_id = [], [], []
    market_mu, market_Sigma, w_markowitz = [], [], []
    rejected = 0

    # every parameter of every KEPT market, so it can be measured later
    kept = {"market_vol": [], "idio_level": [], "beta": [], "idio_vol": [], "alpha": []}
    # every parameter of EVERY draw (kept or rejected), to measure what the filter does
    draws = {"market_vol": [], "idio_level": [], "beta": [], "idio_vol": [], "alpha": [], "kept": []}

    for m in range(n_markets):

        # some parameter combinations make the 25 assets near duplicates of
        # each other, and then even the perfect portfolio needs absurd
        # leverage, no investor could hold that so we throw the market away
        # and draw new parameters
        attempts = 0
        while True:
            market_vol = rng.uniform(*market_vol_range)
            idio_vol = rng.uniform(*idio_vol_range)

            mu, Sigma, params = factor_model_known(
                n_assets=n_assets,
                seed=int(rng.integers(2**32)),
                market_vol_annual=market_vol,
                idio_vol_annual=idio_vol,
                return_params=True,
            )
            w_star = oracle_weights(mu, Sigma)
            ok = np.abs(w_star).sum() <= max_oracle_leverage

            # record this draw whether or not it survives the filter
            draws["market_vol"].append(market_vol)
            draws["idio_level"].append(idio_vol)
            for k in ("beta", "idio_vol", "alpha"):
                draws[k].append(params[k])
            draws["kept"].append(ok)

            if ok:
                break

            rejected += 1
            attempts += 1
            if attempts > 300:
                raise RuntimeError(f"market {m}: no draw under leverage {max_oracle_leverage} after 300 tries")

        # keep the truth so we can score ce loss later
        market_mu.append(mu)
        market_Sigma.append(Sigma)

        # keep the parameters of the market that survived
        kept["market_vol"].append(market_vol)
        kept["idio_level"].append(idio_vol)
        for k in ("beta", "idio_vol", "alpha"):
            kept[k].append(params[k])

        for s in range(samples_per_market):

            # vary how much history the sample contains, 5 to 50 years
            n_months = int(rng.integers(60, 601))

            draw_rng = np.random.default_rng(int(rng.integers(2**32)))
            R = draw_returns(mu, Sigma, n_months, draw_rng)

            X.append(make_features(R))
            y.append(w_star)
            market_id.append(m)

            # markowitz on the same sample, so we can compare later
            w_markowitz.append(markowitz(R))

        if (m + 1) % 2000 == 0:
            print(f"  {label}: {m + 1} / {n_markets}")

    print(f"  {label}: rejected {rejected} draws for excess oracle leverage")

    return (
        np.array(X),
        np.array(y),
        np.array(market_id),
        np.array(market_mu),
        np.array(market_Sigma),
        np.array(w_markowitz),
        {k: np.array(v) for k, v in kept.items()},
        {k: np.array(v) for k, v in draws.items()},
    )


def report_and_save(out, filename):
    X, y, market_id, market_mu, market_Sigma, w_mk, kept, draws = out
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
        # parameters of the kept markets, e.g. param_beta has shape (n_markets, 25)
        **{f"param_{k}": v for k, v in kept.items()},
        # parameters of every draw, kept or rejected, e.g. draw_beta has shape (n_draws, 25)
        **{f"draw_{k}": v for k, v in draws.items()},
    )
    print(f"  saved to {DATA_DIR / filename}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--markets", type=int, default=100_000)
    p.add_argument("--histories", type=int, default=3)
    p.add_argument("--assets", type=int, default=N_ASSETS)
    p.add_argument("--market-vol", type=float, nargs=2, default=[0.10, 0.30])
    p.add_argument("--idio-vol", type=float, nargs=2, default=[0.12, 0.45])
    p.add_argument("--out", default="train.npz")
    p.add_argument("--no-crisis", action="store_true", help="skip the crisis set")
    a = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    print(f"generating {a.markets:,} markets x {a.histories} histories, {a.assets} stocks, "
          f"market vol {a.market_vol}, stock-specific vol {a.idio_vol} -> {a.out}")
    report_and_save(
        generate(n_markets=a.markets, samples_per_market=a.histories, seed=0,
                 market_vol_range=tuple(a.market_vol), idio_vol_range=tuple(a.idio_vol),
                 n_assets=a.assets, label="normal"),
        a.out,
    )

    # crisis markets, never trained on
    if not a.no_crisis:
        print("\ngenerating crisis markets...")
        report_and_save(
            generate(n_markets=2_000, samples_per_market=3, seed=999,
                     market_vol_range=(0.25, 0.45),
                     idio_vol_range=(0.08, 0.18),
                     max_oracle_leverage=8.0,
                     n_assets=a.assets, label="crisis"),
            "stress.npz" if a.out == "train.npz" else a.out.replace(".npz", "_stress.npz"),
        )